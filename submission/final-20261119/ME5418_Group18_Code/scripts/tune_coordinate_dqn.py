"""Tune a coordinate-DQN checkpoint on new demonstrations and old-policy anchors."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from me5418_harvesting.agents.dqn_learning import (
    TransitionBatch,
    demonstration_margin_loss,
    td_loss,
)

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "research_results" / "20261118_generalization"
OLD_DATA = ROOT / "docs" / "experiments" / "029_training_coverage" / "evidence"

sys.path.insert(0, str(ARCHIVE))
from coordinate_policy import CoordinateAgent, transform_encoded


def load_demonstrations(path, agent, encode, metadata=None):
    with np.load(path, allow_pickle=False) as data:
        batch = TransitionBatch(
            torch.from_numpy(encode(data["observations"], agent)),
            torch.tensor(data["actions"], dtype=torch.long),
            torch.tensor(data["rewards"], dtype=torch.float32),
            torch.from_numpy(encode(data["next_observations"], agent)),
            torch.tensor(data["terminated"], dtype=torch.bool),
            torch.tensor(data["truncated"], dtype=torch.bool),
        )
        if metadata is None:
            seeds = np.asarray(data["scene_seeds"], dtype=np.int64)
        else:
            seeds = np.full(len(batch.actions), -1, dtype=np.int64)
            for episode in metadata["episodes"]:
                seeds[episode["start"]:episode["stop"]] = episode["seed"]
    if not len(seeds) or seeds.shape != batch.actions.shape or (seeds < 0).any():
        raise ValueError(f"Invalid or empty demonstration seed mapping: {path}")
    if any(len(column) != len(seeds) for column in (
        batch.observations, batch.next_observations, batch.rewards,
        batch.terminated, batch.truncated,
    )):
        raise ValueError(f"Demonstration columns have different lengths: {path}")
    input_size = agent.learner.online[0].in_features
    if any(column.shape != (len(seeds), input_size) for column in (
        batch.observations, batch.next_observations,
    )):
        raise ValueError(f"Demonstration observation shape does not match the agent: {path}")
    if not 0 <= int(batch.actions.min()) <= int(batch.actions.max()) < agent.action_count:
        raise ValueError(f"Demonstration action outside the action space: {path}")
    unique_seeds, counts = np.unique(seeds, return_counts=True)
    scene_weights = np.repeat(1.0 / counts, counts)
    # Advanced indexing writes the weight of each scene back to original row order.
    probabilities = np.empty(len(seeds), dtype=np.float64)
    probabilities[np.argsort(seeds, kind="stable")] = scene_weights / len(unique_seeds)
    return batch, probabilities, unique_seeds.tolist()


def sample(batch, probabilities, size, rng):
    rows = torch.from_numpy(rng.choice(len(probabilities), size, replace=True, p=probabilities))
    return TransitionBatch(
        batch.observations[rows], batch.actions[rows], batch.rewards[rows],
        batch.next_observations[rows], batch.terminated[rows], batch.truncated[rows],
    )


def encode_coordinates(values, agent):
    return transform_encoded(
        values, agent.relative, agent.encoder.base.branches, agent.encoder.base.nodes,
    )


def train(args):
    torch.set_num_threads(1)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    agent, env, _ = CoordinateAgent.load(args.checkpoint)
    old_path = OLD_DATA / "expanded_demonstrations.npz"
    old_manifest = json.loads((OLD_DATA / "dataset.json").read_text(encoding="utf-8"))
    try:
        encode = encode_coordinates
        if args.compact:
            from guided_coordinate_policy import CompactGuidedAgent

            original, _, _ = load_demonstrations(old_path, agent, encode, old_manifest)
            with torch.no_grad():
                compact_teacher_actions = agent.learner.online(original.observations).argmax(dim=1)
            agent = CompactGuidedAgent(env, agent.learner.config, args.seed, agent.relative)
            encode = encode_guided
        elif args.guided:
            from guided_coordinate_policy import GuidedCoordinateAgent

            agent = GuidedCoordinateAgent.from_baseline(agent, env)
            encode = encode_guided
        elif args.geometry:
            from geometric_coordinate_policy import GeometricCoordinateAgent, transform_for_agent

            agent = GeometricCoordinateAgent.from_baseline(agent, env)
            encode = transform_for_agent
        old, old_probabilities, old_seeds = load_demonstrations(
            old_path, agent, encode, old_manifest,
        )
        new = [load_demonstrations(path, agent, encode) for path in args.demos]
        if any(set(old_seeds).intersection(dataset[2]) for dataset in new):
            raise ValueError("New demonstration files must be disjoint from the old scenes.")
        environment_config = torch.load(
            args.checkpoint, weights_only=True, map_location="cpu",
        )["environment_config"]
        learner = agent.learner
        # Preserve the baseline as a frozen teacher while the DQN target follows the learner.
        with torch.no_grad():
            old_q = learner.online(old.observations)
        teacher_actions = compact_teacher_actions if args.compact else old_q.argmax(dim=1)
        teacher_advantages = old_q - old_q.mean(dim=1, keepdim=True)
        optimizer = torch.optim.Adam(learner.online.parameters(), lr=args.learning_rate)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        manifest = {
            "source": str(args.checkpoint),
            "source_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "old_data": {"path": str(old_path), "seeds": old_seeds},
            "new_data": [
                {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "seeds": seeds,
                    "transitions": len(batch.actions),
                }
                for path, (batch, _, seeds) in zip(args.demos, new, strict=True)
            ],
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "batch_size_per_group": args.batch_size,
            "old_weight": args.old_weight,
            "td_weight": args.td_weight,
            "margin": args.margin,
            "target_sync_interval": args.target_sync,
            "observation": next((
                mode for mode in ("compact", "guided", "geometry") if getattr(args, mode)
            ), "relative"),
            "supervision": args.supervision,
            "demo_weights": args.demo_weights,
            "optimizer_start": "fresh Adam, original optimizer state unavailable",
        }
        (args.output_dir / "training.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8",
        )
        learner.online.train()
        last_update = max(args.updates)
        source_weights = None
        if args.demo_weights is not None:
            weights = np.asarray(args.demo_weights, dtype=np.float64)
            source_weights = weights / weights.sum()
        for step in range(1, last_update + 1):
            source = ((step - 1) % len(new) if source_weights is None
                      else rng.choice(len(new), p=source_weights))
            batch, probabilities, _ = new[source]
            expert = sample(batch, probabilities, args.batch_size, rng)
            indices = torch.from_numpy(rng.choice(
                len(old_probabilities), args.batch_size, replace=True, p=old_probabilities,
            ))
            expert_q = learner.online(expert.observations)
            if args.supervision == "cross-entropy":
                expert_loss = F.cross_entropy(expert_q / 2.0, expert.actions)
            else:
                expert_loss = demonstration_margin_loss(expert_q, expert.actions, args.margin)
            anchor_q = learner.online(old.observations[indices])
            anchor_loss = demonstration_margin_loss(anchor_q, teacher_actions[indices], args.margin)
            if not args.compact:
                anchor_advantages = anchor_q - anchor_q.mean(dim=1, keepdim=True)
                anchor_loss += F.smooth_l1_loss(anchor_advantages, teacher_advantages[indices])
            bellman_loss = expert_loss.new_zeros(())
            if args.td_weight:
                bellman_loss, _, _ = td_loss(
                    learner.online, learner.target, expert, learner.config.gamma,
                )
            loss = expert_loss + args.old_weight * anchor_loss + args.td_weight * bellman_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite coordinate-DQN tuning loss.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                learner.online.parameters(), learner.config.gradient_clip_norm,
                error_if_nonfinite=True,
            )
            optimizer.step()
            if step % args.target_sync == 0:
                learner.target.load_state_dict(learner.online.state_dict())
            if step in args.updates:
                file = args.output_dir / f"update_{step:05}.pt"
                metrics = {
                    "updates": step,
                    "demo_margin_loss": float(expert_loss.detach()),
                    "anchor_loss": float(anchor_loss.detach()),
                    "td_loss": float(bellman_loss.detach()),
                }
                agent.save(file, environment_config, {**manifest, **metrics})
                print(json.dumps({"checkpoint": str(file), **metrics}), flush=True)
    finally:
        env.close()


def encode_guided(values, agent):
    return agent.encoder.from_absolute(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ARCHIVE / "models" / "prepared.pt")
    parser.add_argument("--demos", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--updates", nargs="+", type=int, default=[250, 750, 1500])
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--old-weight", type=float, default=1.0)
    parser.add_argument("--td-weight", type=float, default=0.05)
    parser.add_argument("--margin", type=float, default=0.8)
    parser.add_argument("--target-sync", type=int, default=200)
    parser.add_argument("--seed", type=int, default=5418)
    parser.add_argument("--demo-weights", nargs="+", type=float)
    parser.add_argument("--supervision", choices=["margin", "cross-entropy"], default="margin")
    encodings = parser.add_mutually_exclusive_group()
    encodings.add_argument("--geometry", action="store_true")
    encodings.add_argument("--guided", action="store_true")
    encodings.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    if min(args.batch_size, args.target_sync, *args.updates) < 1 or args.seed < 0:
        parser.error("Batch size, target sync and update counts must be positive; seed nonnegative.")
    settings = [args.learning_rate, args.margin, args.old_weight, args.td_weight]
    if not np.isfinite(settings).all() or min(settings[:2]) <= 0 or min(settings[2:]) < 0:
        parser.error("Learning rate and margin must be positive; loss weights nonnegative.")
    if args.demo_weights is not None and (
        len(args.demo_weights) != len(args.demos)
        or not np.isfinite(args.demo_weights).all() or min(args.demo_weights) <= 0
    ):
        parser.error("Supply one finite, positive weight per demonstration file.")
    train(args)


if __name__ == "__main__":
    main()
