"""Test supervised fitting, return calibration and TD retention on three known scenes."""

import argparse
import hashlib
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from me5418_harvesting.agents.demonstrations import seed_replay_from_reference
from me5418_harvesting.agents.dqn import DQNConfig
from me5418_harvesting.agents.dqn_agent import DQNAgent
from me5418_harvesting.agents.multiscene_demonstrations import (
    build_multiscene_batch,
    slice_transition_batch,
)
from me5418_harvesting.agents.optimizer_transition import (
    begin_td_with_optimizer,
    optimizer_state_summary,
)
from me5418_harvesting.agents.replay_buffer import ReplayBuffer
from me5418_harvesting.agents.return_calibration import calibration_update
from me5418_harvesting.agents.run_record import begin_run, finish_run
from me5418_harvesting.agents.staged_learning import supervised_update
from me5418_harvesting.agents.training import read_mapping
from me5418_harvesting.envs.harvesting_env import HarvestingEnv

ROOT = Path(__file__).resolve().parents[3]
ADAM_RUNNER = ROOT / "docs/experiments/010_adam_state/run_adam_state.py"
SPEC = importlib.util.spec_from_file_location("multiscene_adam_reference", ADAM_RUNNER)
adam = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adam)
calibration = adam.calibration
staged = adam.staged
fit = adam.fit
SEEDS = (10000, 10001, 10002)
INIT_SEED = 0
ARMS = ("reset", "preserve")
PRETRAIN_UPDATES = 20000
CALIBRATION_UPDATES = 20000
TD_UPDATES = 2000
PRETRAIN_CHECKPOINTS = (0, 1000, 5000, 10000, 20000)
CALIBRATION_CHECKPOINTS = (0, 1, 10, 100, 1000, 5000, 10000, 20000)
TD_CHECKPOINTS = (*range(11), 20, 50, 100, 200, 500, 1000, 2000)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def collect_dataset(environment, agent, output, seeds=SEEDS):
    """Try each declared seed once and retain all collection failures without substitution."""
    collections, trajectories = {}, []

    def save_collection():
        record = {
            "requested_seeds": list(seeds), "collections": collections,
            "successful_episodes": sum(row["successful_episodes"] for row in collections.values()),
            "collection_environment_steps": sum(
                row["collection_environment_steps"] for row in collections.values()),
            "accepted_transitions": sum(row["accepted_transitions"] for row in collections.values()),
            "note": "Privileged reference routes are additional training data; planner/reset probes are excluded.",
        }
        fit.save_json(output / "demonstrations.json", record)
        return record

    for seed in seeds:
        replay = ReplayBuffer(agent.learner.config.replay_capacity)

        def progress(record, current_seed=seed):
            collections[str(current_seed)] = record
            save_collection()

        collection = seed_replay_from_reference(
            environment, agent.encoder, replay, 1, seed, progress=progress,
        )
        collections[str(seed)] = collection
        save_collection()
        if collection["successful_episodes"] != 1:
            raise RuntimeError(f"Declared training scene {seed} did not yield a successful demonstration.")
        trajectories.append(fit.complete_batch(replay))
    batch, returns, episodes = build_multiscene_batch(trajectories, seeds, agent.learner.config.gamma)
    path = output / "demonstration.npz"
    np.savez_compressed(path, **{
        name: getattr(batch, name).numpy() for name in batch.__dataclass_fields__
    })
    targets, scene_records = [], []
    for episode in episodes:
        scene = slice_transition_batch(batch, episode["start"], episode["stop"])
        _, check = calibration.compute_targets(scene, agent.learner.config.gamma)
        targets.extend(check["returns"])
        scene_records.append({
            **episode,
            "initial_observation_sha256": hashlib.sha256(scene.observations[0].numpy().tobytes()).hexdigest(),
            "action_label_counts": torch.bincount(scene.actions, minlength=agent.action_count).tolist(),
        })
    if targets != returns.tolist():
        raise RuntimeError("Concatenated return targets differ from independent per-episode calculations.")
    unique, inverse, counts = np.unique(batch.observations.numpy(), axis=0,
                                        return_inverse=True, return_counts=True)
    labels = batch.actions.numpy()
    conflicts = sum(len(np.unique(labels[inverse == index])) > 1
                    for index in range(len(unique)) if counts[index] > 1)
    dataset = {
        "sha256": fit.file_hash(path), "transitions": len(batch.actions),
        "effective_batch_size": len(batch.actions),
        "configured_batch_size_unused": agent.learner.config.batch_size,
        "observation_size": batch.observations.shape[1], "seeds": list(seeds),
        "episodes": scene_records, "full_batch_weighting": "equal_per_transition",
        "action_label_counts": torch.bincount(batch.actions, minlength=agent.action_count).tolist(),
        "unique_observations": len(unique), "duplicate_observation_rows": len(labels) - len(unique),
        "conflicting_duplicate_observation_groups": int(conflicts),
        "distinct_initial_observations": len({row["initial_observation_sha256"] for row in scene_records}),
        "scene_seed_in_network_input": False,
    }
    fit.save_json(output / "dataset.json", dataset)
    fit.save_json(output / "mc_targets.json", {
        "gamma": agent.learner.config.gamma, "returns": returns.tolist(),
        "rewards": batch.rewards.tolist(), "episodes": [
            {**episode, "numerical_check": calibration.compute_targets(
                slice_transition_batch(batch, episode["start"], episode["stop"]),
                agent.learner.config.gamma,
            )[1]} for episode in episodes
        ],
        "network_bootstrapping_used": False, "episode_boundaries_respected": True,
    })
    if dataset["distinct_initial_observations"] != len(seeds):
        raise RuntimeError("Declared training seeds did not generate distinct initial observations.")
    return batch, returns, episodes, dataset, save_collection()


def aggregate_scenes(scenes):
    return {
        "scene_count": len(scenes),
        "success_count": sum(bool(row["rollout"]["success"]) for row in scenes),
        "success_seeds": [row["seed"] for row in scenes if row["rollout"]["success"]],
        "damage_count": sum(bool(row["rollout"]["damage"]) for row in scenes),
        "budget_exhausted_count": sum(bool(row["rollout"]["budget_exhausted"]) for row in scenes),
        "evaluation_environment_steps": sum(row["rollout"]["steps"] for row in scenes),
    }


def observe(agent, env, environment, batch, returns, episodes, phase, step, prior, directory):
    """Evaluate every declared training scene, never replacing failures with a planner."""
    before = fit.parameter_hash(agent)
    rng_before = json.dumps(agent.rng.bit_generator.state)
    scenes = []
    for episode in episodes:
        scene = slice_transition_batch(batch, episode["start"], episode["stop"])
        scene_returns = returns[episode["start"]:episode["stop"]]
        scenes.append({
            "seed": episode["seed"], "fit": fit.fit_metrics(agent, scene, 0.8),
            "return_fit": calibration.return_fit(agent, scene, scene_returns),
            "rollout": fit.fixed_scene_rollout(agent, env, episode["seed"], scene),
        })
    row = {
        "update": step, "total_updates_on_branch": prior + step,
        "fit": fit.fit_metrics(agent, batch, 0.8),
        "return_fit": calibration.return_fit(agent, batch, returns),
        "value_metrics": staged.value_metrics(agent, batch),
        "optimizer_state": optimizer_state_summary(agent.learner),
        "scenes": scenes, "aggregate": aggregate_scenes(scenes),
    }
    if fit.parameter_hash(agent) != before or json.dumps(agent.rng.bit_generator.state) != rng_before:
        raise RuntimeError("Evaluation changed policy weights or exploration RNG state.")
    agent.save(directory / f"update_{step}.pt", environment, {
        "phase": "multiscene_retention_diagnostic", "objective": phase, "updates": step,
        "prior_updates_on_branch": prior, "total_updates_on_branch": prior + step,
        "training_scene_only": True, "scene_seeds": [item["seed"] for item in episodes],
        "fit": row["fit"], "aggregate": row["aggregate"],
    })
    details = ", ".join(f"{item['seed']}:{'success' if item['rollout']['success'] else 'damage' if item['rollout']['damage'] else 'budget'}"
                        f"/{item['rollout']['steps']}" for item in scenes)
    print(f"{phase}: update={step}, fit={row['fit']['correct']}/{len(batch.actions)}, "
          f"success={row['aggregate']['success_count']}/{len(episodes)}, {details}", flush=True)
    return row


def retention_metrics(observations, expected_start):
    """Separate retaining an acquired scene from recovering a previously failed scene."""
    eligible = {row["seed"]: bool(row["rollout"]["success"]) for row in expected_start["scenes"]}
    scenes = {}
    for seed, acquired in eligible.items():
        failures = [row["update"] for row in observations
                    if not next(item for item in row["scenes"] if item["seed"] == seed)["rollout"]["success"]]
        scenes[str(seed)] = {
            "retention_eligible": acquired, "failed_sampled_updates": failures,
            "first_sampled_failure_update": min(failures) if failures else None,
            "retained_at_all_sampled_checkpoints": acquired and not failures,
            "interpretation": "retention_of_acquired_scene" if acquired
            else "downstream_behavior_after_previously_failed_scene",
            "first_ten_updates_success_count": sum(
                next(item for item in row["scenes"] if item["seed"] == seed)["rollout"]["success"]
                for row in observations if 1 <= row["update"] <= 10),
            "first_ten_updates_observation_count": sum(1 <= row["update"] <= 10 for row in observations),
        }
    acquired_seeds = [seed for seed, acquired in eligible.items() if acquired]
    return {
        "eligible_seeds": acquired_seeds, "per_scene": scenes,
        "all_scene_retention_eligible": bool(eligible) and all(eligible.values()),
        "all_scenes_retained_at_all_sampled_checkpoints": bool(scenes) and all(
            item["retained_at_all_sampled_checkpoints"] for item in scenes.values()),
        "acquired_subset_retained_at_all_sampled_checkpoints": bool(acquired_seeds) and all(
            scenes[str(seed)]["retained_at_all_sampled_checkpoints"] for seed in acquired_seeds),
    }


def execute_phase(agent, env, environment, batch, returns, episodes, phase, updates,
                  checkpoints, output, prior_updates, expected_start=None):
    directory = output / phase
    directory.mkdir(parents=True, exist_ok=False)
    initial_hash = fit.parameter_hash(agent)
    observations, synchronizations = [], 0
    with (directory / "updates.jsonl").open("x", encoding="utf-8") as log:
        for step in range(updates + 1):
            if step:
                if phase == "pretraining":
                    metrics = supervised_update(agent.learner, batch, objective="margin_only", margin=0.8)
                elif phase == "calibration":
                    metrics = calibration_update(agent.learner, batch, returns)
                elif phase in ("reset/td", "preserve/td"):
                    metrics = asdict(agent.learner.update(batch, demonstration_weight=1.0,
                                                        demonstration_margin=0.8))
                else:
                    raise ValueError(f"Unknown multi-scene phase: {phase}")
                synchronizations += int(metrics["target_synchronized"])
                log.write(json.dumps({"update": step, **metrics}, allow_nan=False) + "\n")
            if step in checkpoints:
                row = observe(agent, env, environment, batch, returns, episodes, phase,
                              step, prior_updates, directory)
                observations.append(row)
                fit.save_json(directory / "observations.json", observations)
                if step == 0 and expected_start is not None:
                    previous = expected_start["final_observation"]
                    if any(row[key] != previous[key] for key in ("fit", "return_fit", "scenes", "aggregate")):
                        raise RuntimeError("The phase boundary changed the expected per-scene behavior.")
                log.flush()
            elif step % 100 == 0:
                log.flush()
    snapshot_path = directory / "training_state.pt"
    torch.save(adam.training_snapshot(agent, environment), snapshot_path)
    result = {
        "directory": phase, "initial_parameter_sha256": initial_hash,
        "final_parameter_sha256": fit.parameter_hash(agent),
        "optimizer_updates": agent.learner.update_count, "prior_updates_on_branch": prior_updates,
        "total_updates_on_branch": prior_updates + agent.learner.update_count,
        "target_synchronizations": synchronizations,
        "initial_observation": observations[0], "final_observation": observations[-1],
        "retention": retention_metrics(observations, expected_start["final_observation"])
        if expected_start is not None else None,
        "evaluation_environment_steps": sum(row["aggregate"]["evaluation_environment_steps"]
                                            for row in observations),
        "training_snapshot_file": f"{phase}/training_state.pt",
        "training_snapshot_sha256": fit.file_hash(snapshot_path),
        "checkpoint_sha256": {path.name: fit.file_hash(path) for path in sorted(directory.glob("*.pt"))},
    }
    fit.save_json(directory / "summary.json", result)
    return result


def start_phase(agent, preserve, expected_hash, prior_updates, path):
    before = optimizer_state_summary(agent.learner)
    record = begin_td_with_optimizer(agent.learner, preserve_optimizer=preserve)
    if (fit.parameter_hash(agent) != expected_hash or agent.learner.update_count != 0
            or record["optimizer_after"]["settings_sha256"] != before["settings_sha256"]
            or not all(torch.equal(online, target) for online, target in zip(
                agent.learner.online.parameters(), agent.learner.target.parameters()))):
        raise RuntimeError("The new phase must preserve weights/settings and synchronize the target clock.")
    record.update(policy_parameter_sha256=expected_hash, total_prior_updates_on_branch=prior_updates,
                  target_equals_online=True)
    fit.save_json(path, record)
    return record


def write_accounting(output):
    counts = {path.relative_to(output).as_posix(): len(path.read_text(encoding="utf-8").splitlines())
              for path in output.rglob("updates.jsonl")}
    evaluation = sum(row["aggregate"]["evaluation_environment_steps"]
                     for path in output.rglob("observations.json") for row in read_json(path))
    collection_path = output / "demonstrations.json"
    collection = read_json(collection_path)["collection_environment_steps"] if collection_path.exists() else 0
    record = {
        "update_records_per_log": counts, "optimizer_updates_total": sum(counts.values()),
        "td_optimizer_updates": sum(count for name, count in counts.items() if "/td/" in name),
        "demonstration_collection_steps": collection, "evaluation_environment_steps": evaluation,
        "inherited_optimizer_updates": 0, "online_training_steps": 0, "held_out_scenes_accessed": 0,
        "note": "Actual recorded completed updates/actions; planner search and reset probes are excluded.",
    }
    fit.save_json(output / "accounting.json", record)
    return record


def run(output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite a multi-scene experiment: {output}")
    environment_path = ROOT / "configs/environment_multiroute.yaml"
    dqn_path = ROOT / "configs/dqn_multiroute_pilot.yaml"
    environment = read_mapping(environment_path)
    config = DQNConfig(**read_mapping(dqn_path))
    helper_files = [ADAM_RUNNER, adam.CALIBRATION_RUNNER, calibration.REFERENCE_LOADER,
                    calibration.reference_module.STAGED_RUNNER, staged.FIT_RUNNER]
    protocol = {
        "kind": "multiscene_supervision_calibration_adam_retention", "scene_seeds": SEEDS,
        "initialization_seed": INIT_SEED, "fresh_initialization": True,
        "pretraining_updates": PRETRAIN_UPDATES, "pretraining_checkpoints": PRETRAIN_CHECKPOINTS,
        "calibration_updates": CALIBRATION_UPDATES, "calibration_checkpoints": CALIBRATION_CHECKPOINTS,
        "td_updates": TD_UPDATES, "td_checkpoints": TD_CHECKPOINTS, "arms": ARMS,
        "gamma": config.gamma, "learning_rate": config.learning_rate, "margin": 0.8,
        "demonstration_weight": 1.0, "batch_mode": "all_transitions_in_all_declared_episodes",
        "loss_weighting": "equal_per_transition_not_equal_per_scene", "seed_id_in_network_input": False,
        "pretraining_objective": "margin_only", "calibration_objective": "fixed_return_huber_plus_margin",
        "td_objective": "one_step_huber_td_plus_margin", "early_stopping": False,
        "calibration_boundary": "preserved_weights_fresh_adam_synced_target_zero_phase_clock",
        "td_boundary": "identical_calibrated_weights_settings_synced_targets_zero_phase_clock",
        "td_treatment_difference": "retain_or_reset_Adam_moments_and_internal_steps",
        "retention_eligibility": "Per-scene success at the preceding phase's final checkpoint.",
        "continue_after_unsuccessful_calibration": True,
        "script_sha256": fit.file_hash(Path(__file__)),
        "helper_sha256": {path.relative_to(ROOT).as_posix(): fit.file_hash(path) for path in helper_files},
        "online_training_steps": 0, "held_out_scenes_accessed": 0,
    }
    output.mkdir(parents=True, exist_ok=False)
    env = None
    try:
        begin_run(output, environment, asdict(config), protocol)
        fit.save_json(output / "input_manifest.json", {
            "config_sha256": {path.relative_to(ROOT).as_posix(): fit.file_hash(path)
                              for path in (environment_path, dqn_path)},
            "helper_sha256": protocol["helper_sha256"], "inherited_checkpoint_used": False,
        })
        torch.set_num_threads(1)
        torch.manual_seed(INIT_SEED)
        env = HarvestingEnv(environment)
        encoder_agent = DQNAgent(env, config, INIT_SEED)
        batch, returns, episodes, dataset, collection = collect_dataset(environment, encoder_agent, output, SEEDS)
        torch.manual_seed(INIT_SEED)
        agent = DQNAgent(env, config, INIT_SEED)
        initial_path = output / "initial_training_state.pt"
        torch.save(adam.training_snapshot(agent, environment), initial_path)
        fit.save_json(output / "initial_training_snapshot.json", {
            "file": initial_path.name, "sha256": fit.file_hash(initial_path),
            "parameter_sha256": fit.parameter_hash(agent),
            "optimizer_summary": optimizer_state_summary(agent.learner),
        })
        pretraining = execute_phase(agent, env, environment, batch, returns, episodes, "pretraining",
                                    PRETRAIN_UPDATES, PRETRAIN_CHECKPOINTS, output, 0)
        calibration_boundary = start_phase(agent, False, pretraining["final_parameter_sha256"],
                                           PRETRAIN_UPDATES, output / "calibration_boundary.json")
        calibrated = execute_phase(agent, env, environment, batch, returns, episodes, "calibration",
                                   CALIBRATION_UPDATES, CALIBRATION_CHECKPOINTS, output,
                                   PRETRAIN_UPDATES, pretraining)
        env.close()
        env = None
        snapshot_path = output / calibrated["training_snapshot_file"]
        arms, boundaries = {}, {}
        for branch in ARMS:
            (output / branch).mkdir()
            if fit.file_hash(snapshot_path) != calibrated["training_snapshot_sha256"]:
                raise RuntimeError("The shared calibrated snapshot changed between TD treatments.")
            payload = torch.load(snapshot_path, map_location="cpu", weights_only=True)
            agent, env = adam.restore_training_snapshot(payload)
            prior = PRETRAIN_UPDATES + CALIBRATION_UPDATES
            boundaries[branch] = start_phase(agent, branch == "preserve", calibrated["final_parameter_sha256"],
                                             prior, output / branch / "td_boundary.json")
            arms[branch] = execute_phase(agent, env, environment, batch, returns, episodes, f"{branch}/td",
                                        TD_UPDATES, TD_CHECKPOINTS, output, prior, calibrated)
            env.close()
            env = None
        summary = {
            "protocol": protocol, "dataset": dataset, "demonstration_collection": collection,
            "pretraining": pretraining, "calibration": calibrated, "arms": arms,
            "calibration_boundary": calibration_boundary, "td_boundaries": boundaries,
            **write_accounting(output),
            "limitations": [
                "Three known training scenes and one initialization; generalization remains untested.",
                "All phases use fixed privileged demonstrations rather than online exploration.",
                "Losses weight transitions equally; longer routes contribute more rows.",
                "Per-scene retention requires success before the corresponding phase.",
                "Full acquisition is not assumed; failures and recovery remain visible.",
                "Remaining-budget features may still support route memorization.",
            ],
        }
        fit.save_json(output / "summary.json", summary)
    except BaseException as exc:
        if (output / "run_status.json").exists():
            write_accounting(output)
            finish_run(output, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", exc)
        raise
    else:
        finish_run(output, "completed")
        return summary
    finally:
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.output_dir.resolve())


if __name__ == "__main__":
    main()
