"""Collect successful expert continuations from states visited by a frozen policy."""

import argparse
import hashlib
import json
import sys
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import torch

from me5418_harvesting.agents.dqn_agent import ObservationEncoder
from me5418_harvesting.evaluation.scene_feasibility import _copy_environment
from me5418_harvesting.planners.collision_free import NoCollisionFreeRoute, collision_free_actions

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "research_results" / "20261118_generalization"
TRAINING_SEEDS = range(5418400, 5418480)


def load_policy(checkpoint):
    sys.path.insert(0, str(ARCHIVE))
    try:
        from coordinate_policy import CoordinateAgent
    finally:
        sys.path.pop(0)
    return CoordinateAgent.load(checkpoint)


def first_intervention(agent, env, observation):
    """Find the first blockage, short loop, or contact likely to damage a branch."""
    actions, clean_steps = [], [0]
    recent_positions = deque([observation[:2].copy()], maxlen=6)
    previous_contacts = 0
    while env.step_count < env.max_steps:
        action = agent.act(observation, epsilon=0.0)
        observation, _, terminated, truncated, info = env.step(action)
        actions.append(action)
        repeated = any(
            np.linalg.norm(observation[:2] - previous) < 1e-4 for previous in recent_positions
        )
        contact = info["contact_steps"] > previous_contacts
        if info["damage"] or (contact and info["peak_normalized_deformation"] >= 0.85):
            reason = "damage_or_high_deformation"
        elif info["collision_blocked"]:
            reason = "blocked"
        elif repeated:
            reason = "position_repeated"
        else:
            reason = None
        if reason is not None and not info["success"]:
            intervention = len(actions) - int(terminated or truncated)
            snapshots = [intervention]
            earlier = next((step for step in reversed(clean_steps) if step < intervention), None)
            if earlier is not None:
                snapshots.insert(0, earlier)
            return actions, reason, snapshots
        if terminated or truncated:
            return actions, "success" if info["success"] else "finished_without_trigger", []
        if not contact:
            clean_steps.append(len(actions))
        recent_positions.append(observation[:2].copy())
        previous_contacts = info["contact_steps"]
    return actions, "finished_without_trigger", []


def expert_continuation(env, encoder):
    """Plan and execute a teacher rollout on a copy, retaining only safe completions."""
    probe = _copy_environment(env)
    try:
        try:
            actions = collision_free_actions(probe, include_current_shape=True)
        except NoCollisionFreeRoute:
            return [], "no_collision_free_route", 0
        observation = probe._observation()
        rows = []
        for action in actions:
            successor, reward, terminated, truncated, info = probe.step(action)
            rows.append((
                encoder(observation), encoder(successor), action, reward, terminated, truncated,
            ))
            observation = successor
            if terminated or truncated:
                if info["success"] and not info["damage"]:
                    return rows, "success", len(rows)
                reason = "damage" if info["damage"] else "budget_or_truncation"
                return [], reason, len(rows)
        return [], "route_exhausted", len(rows)
    finally:
        probe.close()


def collect_scene(seed, checkpoint):
    torch.set_num_threads(1)
    agent, env, _ = load_policy(checkpoint)
    env.noise_fraction = 0.0
    rows = []
    record = {"seed": seed, "probes": []}
    try:
        try:
            observation, reset_info = env.reset(seed=seed)
        except ValueError as error:
            record.update(trigger="scene_generation_failed", error=str(error))
            return rows, record
        initial = _copy_environment(env)
        try:
            with torch.inference_mode():
                actions, reason, snapshots = first_intervention(agent, env, observation)
            record.update(
                trigger=reason, learner_steps=len(actions),
                scene_generation_attempts=reset_info["scene_generation_attempts"],
                reference_route_attempts=reset_info.get("reference_route_attempts", 0),
            )
            encoder = ObservationEncoder(initial)
            for snapshot in snapshots:
                for action in actions[initial.step_count:snapshot]:
                    initial.step(action)
                accepted, result, steps = expert_continuation(initial, encoder)
                rows.extend((*row, seed) for row in accepted)
                record["probes"].append({
                    "start_step": snapshot, "result": result, "executed_steps": steps,
                    "accepted_transitions": len(accepted),
                })
        finally:
            initial.close()
    finally:
        env.close()
    return rows, record


def write_dataset(output, rows):
    observations, successors, actions, rewards, terminated, truncated, seeds = zip(*rows)
    with output.open("xb") as stream:
        np.savez_compressed(
            stream, observations=np.stack(observations), next_observations=np.stack(successors),
            actions=np.asarray(actions, dtype=np.int64),
            rewards=np.asarray(rewards, dtype=np.float32),
            terminated=np.asarray(terminated, dtype=np.bool_),
            truncated=np.asarray(truncated, dtype=np.bool_),
            scene_seeds=np.asarray(seeds, dtype=np.int64),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ARCHIVE / "models" / "prepared.pt")
    parser.add_argument("--seed", type=int, default=TRAINING_SEEDS.start)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seeds = range(args.seed, args.seed + args.count)
    if args.count <= 0 or seeds[0] not in TRAINING_SEEDS or seeds[-1] not in TRAINING_SEEDS:
        parser.error("Recovery seeds must stay within the assigned training interval 5418400–5418479.")
    if args.workers <= 0:
        parser.error("The number of workers must be positive.")
    manifest = args.output.with_suffix(".json")
    if args.output.suffix != ".npz" or args.output.exists() or manifest.exists():
        parser.error("The output must be a new .npz file with a new .json sidecar.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.resolve()
    worker = partial(collect_scene, checkpoint=checkpoint)
    rows, records = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for accepted, record in executor.map(worker, seeds):
            rows.extend(accepted)
            records.append(record)
            print(
                f"Seed {record['seed']}: {record['trigger']}; "
                f"{len(accepted)} accepted transitions",
                flush=True,
            )
    if rows:
        write_dataset(args.output, rows)
    report = {
        "method": "expert_continuations_from_greedy_learner_states",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "dataset_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest() if rows else None,
        "requested_scenes": len(records),
        "accepted_scenes": len({row[-1] for row in rows}),
        "accepted_transitions": len(rows),
        "records": records,
    }
    with manifest.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(f"Accepted {len(rows)} transitions from {report['accepted_scenes']} scenes.", flush=True)


if __name__ == "__main__":
    main()
