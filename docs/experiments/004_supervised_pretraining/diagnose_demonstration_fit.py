"""Inspect a frozen pretrained policy on its own regenerated demonstration states.

This post-hoc diagnostic uses training data only. It measures action agreement,
not held-out episode success, and never updates the model or its optimizer.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from me5418_harvesting.agents.demonstrations import seed_replay_from_reference
from me5418_harvesting.agents.dqn_agent import DQNAgent
from me5418_harvesting.agents.replay_buffer import ReplayBuffer

ROOT = Path(__file__).resolve().parents[3]
ACTION_NAMES = (
    "north_small", "south_small", "east_small", "west_small",
    "north_large", "south_large", "east_large", "west_large",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagnose(run_dir: Path, output: Path) -> dict:
    run_dir, output = run_dir.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic evidence: {output}")
    settings_path = run_dir / "settings.json"
    demonstrations_path = run_dir / "demonstrations.json"
    provenance_path = run_dir / "provenance.json"
    checkpoint = run_dir / "pretrained.pt"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    expected_collection = json.loads(demonstrations_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    changed_sources = [
        name for name, digest in provenance["source_sha256"].items()
        if sha256(ROOT / name) != digest
    ]
    if changed_sources:
        raise ValueError(f"Run sources have changed since launch: {changed_sources}")

    torch.set_num_threads(1)
    checkpoint_hash = sha256(checkpoint)
    agent, checkpoint_env, metadata = DQNAgent.load(checkpoint)
    try:
        if metadata.get("phase") != "pretraining":
            raise ValueError("Expected the frozen post-pretraining checkpoint.")
        settings_training = settings["training"]
        replay = ReplayBuffer(settings["dqn"]["replay_capacity"])
        collection = seed_replay_from_reference(
            settings["environment"], agent.encoder, replay,
            settings_training["demonstration_episodes"], settings_training["demonstration_seed"],
        )
        if collection != expected_collection:
            raise ValueError("Regenerated demonstration records do not exactly match the run.")
        if not replay.records:
            raise ValueError("No accepted demonstration transitions are available.")

        observations = torch.from_numpy(np.stack([row[0] for row in replay.records]))
        actions = torch.tensor([row[1] for row in replay.records], dtype=torch.long)
        agent.learner.online.eval()
        margin = settings_training["demonstration_pretrain_margin"]
        with torch.no_grad():
            values = agent.learner.online(observations)
            if not torch.isfinite(values).all():
                raise FloatingPointError("The checkpoint produced non-finite Q values.")
            predicted = values.argmax(dim=1)
            demonstrated_q = values.gather(1, actions[:, None]).squeeze(1)
            other_values = values.clone()
            other_values.scatter_(1, actions[:, None], -float("inf"))
            highest_other_q = other_values.max(dim=1).values
            margin_violation = (highest_other_q + margin - demonstrated_q).clamp_min(0)
            margin_satisfied = demonstrated_q >= highest_other_q + margin
            correct = predicted == actions
            label_counts = torch.bincount(actions, minlength=agent.action_count)
            prediction_counts = torch.bincount(predicted, minlength=agent.action_count)
            per_action = []
            for index, name in enumerate(ACTION_NAMES):
                selected = actions == index
                count = int(selected.sum())
                count_correct = int(correct[selected].sum())
                per_action.append({
                    "action": index, "name": name, "label_count": count,
                    "correct": count_correct,
                    "accuracy": count_correct / count if count else None,
                })
            majority_action = int(label_counts.argmax())
            result = {
                "training_only": True,
                "post_hoc_diagnostic": True,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "run_dir": str(run_dir),
                "checkpoint": "pretrained.pt",
                "checkpoint_sha256": checkpoint_hash,
                "checkpoint_updates": metadata["updates"],
                "training_git_head": provenance["git_head"],
                "script_sha256": sha256(Path(__file__)),
                "settings_sha256": sha256(settings_path),
                "saved_demonstrations_sha256": sha256(demonstrations_path),
                "training_provenance_sha256": sha256(provenance_path),
                "sources_match_training_provenance": True,
                "collection_records_match_exactly": True,
                "dataset_transitions": len(replay),
                "demonstration_seed": settings_training["demonstration_seed"],
                "demonstration_episodes": settings_training["demonstration_episodes"],
                "action_order": list(ACTION_NAMES),
                "demonstration_action_counts": label_counts.tolist(),
                "majority_class_action": majority_action,
                "majority_class_baseline_accuracy": int(label_counts.max()) / len(replay),
                "predicted_action_counts": prediction_counts.tolist(),
                "greedy_action_agreement": float(correct.float().mean()),
                "per_action": per_action,
                "margin": margin,
                "mean_margin_violation": float(margin_violation.mean()),
                "margin_satisfied_fraction": float(margin_satisfied.float().mean()),
                "mean_demonstrated_action_q": float(demonstrated_q.mean()),
                "mean_max_action_q": float(values.max(dim=1).values.mean()),
                "diagnostic_regeneration_actions": collection["collection_environment_steps"],
                "optimizer_updates_performed": 0,
                "notes": [
                    "Full-dataset agreement is measured on the same states used for training.",
                    "No validation or final test scenes are accessed; this is not a task-success rate.",
                    "Regeneration actions are additional post-hoc diagnostic work, not online training.",
                    "Regeneration counts exclude planner search and reset-time feasibility probes.",
                    "No inference checkpoint, training log, model weight or optimizer is modified.",
                ],
            }
        if sha256(checkpoint) != checkpoint_hash:
            raise RuntimeError("The frozen checkpoint changed during the diagnostic.")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return result
    finally:
        checkpoint_env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = diagnose(args.run_dir, args.output)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
