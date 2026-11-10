"""Summarize a completed training run without changing weights or selecting test seeds."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def summarize(run_dir: Path) -> dict:
    """Write learning diagnostics from the exact episode and validation logs."""
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    settings = json.loads((run_dir / "settings.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    if len(records) != summary["episodes"]:
        raise ValueError("Episode log and completed-run summary disagree.")
    validations = [(row["episode"], row["validation"])
                   for row in records if "validation" in row]
    initial_path = run_dir / "initial_validation.json"
    initial = json.loads(initial_path.read_text(encoding="utf-8")) if initial_path.exists() else None
    if initial is not None:
        validations.insert(0, (0, initial))
    episodes = [row["episode"] for row in records]
    validation_episodes = [episode for episode, _ in validations]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    axes[0, 0].plot(episodes, [row["return"] for row in records], color="black", marker=".")
    axes[0, 0].set(title="Training return", ylabel="Return")
    for key, label, style in (("success_rate", "Success", "-o"),
                              ("damage_rate", "Damage", "--s")):
        axes[0, 1].plot(validation_episodes, [row[key] for _, row in validations],
                        style, label=label, color="black")
    axes[0, 1].set(title="Greedy validation", ylabel="Episode fraction", ylim=(-0.05, 1.05))
    axes[0, 1].legend()
    axes[1, 0].plot(episodes, [row.get("distance_to_target", float("nan")) for row in records],
                    color="0.6", label="Training")
    axes[1, 0].plot(validation_episodes,
                    [row.get("mean_distance_to_target", float("nan")) for _, row in validations],
                    "-o", color="black", label="Validation")
    axes[1, 0].set(title="Final distance to target", ylabel="Workspace units")
    axes[1, 0].legend()
    for key, label, style in (("mean_stationary_fraction", "Stationary", "-o"),
                              ("mean_blocked_fraction", "Blocked", "--s")):
        axes[1, 1].plot(validation_episodes,
                        [row.get(key, float("nan")) for _, row in validations],
                        style, color="black", label=label)
    axes[1, 1].set(title="Validation motion diagnostics", ylabel="Mean step fraction",
                   ylim=(-0.05, 1.05))
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.set_xlabel("Training episode (0 = before training)")
        axis.grid(alpha=0.2)
    fig.suptitle(f"DQN training diagnostics | seed {settings['training']['seed']}")
    fig.savefig(run_dir / "learning_diagnostics.png", dpi=160)
    plt.close(fig)
    result = {
        "training_episodes": len(records), "environment_steps": summary["environment_steps"],
        "optimizer_updates": summary["updates"],
        "training_successes": sum(row["success"] for row in records),
        "training_damage_episodes": sum(row["damage"] for row in records),
        "last_exploration_epsilon": records[-1]["epsilon"],
        "initial_validation": initial, "final_validation": summary["validation"],
        "interpretation": "Validation diagnostics from one training seed are not final test results.",
    }
    (run_dir / "analysis.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8",
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = summarize(args.run_dir)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
