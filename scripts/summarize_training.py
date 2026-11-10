"""Summarize a completed training run without changing weights or selecting test seeds."""

import argparse
import json
from pathlib import Path
from statistics import fmean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def trailing_mean(values: list[float], window: int = 50) -> list[float]:
    """Use the available prefix until a full trailing window is available."""
    return [fmean(values[max(0, index + 1 - window):index + 1])
            for index in range(len(values))]


def supervision_diagnostics(pretraining: dict, losses: list[dict]) -> dict | None:
    """Summarize sampled demonstration agreement without labeling it validation."""
    weight = pretraining.get("demonstration_weight", 0.0)
    if weight <= 0:
        return None
    if not losses:
        raise ValueError("Supervised pretraining needs per-update diagnostic records.")
    accuracies = [row["demonstration_action_accuracy"] for row in losses]
    if any(value is None or not 0 <= value <= 1 for value in accuracies):
        raise ValueError("Demonstration batch accuracy must be between zero and one.")
    return {
        "weight": weight,
        "margin": pretraining["demonstration_margin"],
        "mean_first_100_batch_accuracy": fmean(accuracies[:100]),
        "mean_last_100_batch_accuracy": fmean(accuracies[-100:]),
        "batch_accuracy_window_updates": min(100, len(accuracies)),
        "loss_plot_trailing_window_updates": 50,
        "batch_accuracy_note": (
            "Agreement on sampled training demonstrations before each update; "
            "not held-out task success. Windows use all available updates if fewer than 100."
        ),
    }


def plot_pretraining_run(
    run_dir: Path, records: list[dict], checkpoints: list[dict], seed: int,
    supervision: dict | None = None,
):
    """Keep pretraining updates, online episodes and validation stages on separate axes."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    episodes = [row["episode"] for row in records]
    labels = [row["label"] for row in checkpoints]
    positions = list(range(len(checkpoints)))
    validations = [row["validation"] for row in checkpoints]
    axes[0, 0].plot(episodes, [row["return"] for row in records], ".-", color="black")
    axes[0, 0].set(title="Online training return", ylabel="Return", xlabel="Online episode")
    axes[1, 0].plot(episodes, [row.get("distance_to_target", float("nan")) for row in records],
                    ".-", color="black")
    axes[1, 0].set(title="Online training final distance", ylabel="Workspace units",
                   xlabel="Online episode")
    loss_path = run_dir / "pretraining_metrics.jsonl"
    losses = [json.loads(line) for line in loss_path.read_text(encoding="utf-8").splitlines()
              if line.strip()] if loss_path.exists() else []
    if losses and supervision is not None:
        updates = [row["pretraining_update"] for row in losses]
        components = (
            ([row["loss_before"] for row in losses], "Total", "black", "-"),
            ([row["td_loss_before"] for row in losses], "TD", "0.55", "--"),
            ([supervision["weight"] * row["demonstration_loss_before"] for row in losses],
             "Weighted supervision", "black", ":"),
        )
        for values, label, color, style in components:
            axes[2, 0].plot(updates, trailing_mean(values), label=label,
                            color=color, linestyle=style, linewidth=1.1)
        axes[2, 0].set(title="Pretraining loss components (50-update mean)",
                       ylabel="Smoothed batch loss", xlabel="Pretraining optimizer update")
        axes[2, 0].legend(fontsize=8)
    elif losses:
        axes[2, 0].plot([row["pretraining_update"] for row in losses],
                        [row["loss_before"] for row in losses], color="black", linewidth=0.7)
    else:
        axes[2, 0].text(0.5, 0.5, "No per-update loss record", ha="center", va="center",
                        transform=axes[2, 0].transAxes)
    if supervision is None:
        axes[2, 0].set(title="Demonstration pretraining loss", ylabel="Batch Huber loss",
                       xlabel="Pretraining optimizer update")
    for key, label, style in (("success_rate", "Success", "-o"),
                              ("damage_rate", "Damage", "--s")):
        axes[0, 1].plot(positions, [row[key] for row in validations], style,
                        color="black", label=label)
    axes[0, 1].set(title="Greedy validation", ylabel="Episode fraction", ylim=(-0.05, 1.05))
    axes[0, 1].legend()
    axes[1, 1].plot(positions,
                    [row.get("mean_distance_to_target", float("nan")) for row in validations],
                    "-o", color="black")
    axes[1, 1].set(title="Validation final distance", ylabel="Workspace units")
    for key, label, style in (("mean_stationary_fraction", "Stationary", "-o"),
                              ("mean_blocked_fraction", "Blocked", "--s")):
        axes[2, 1].plot(positions, [row.get(key, float("nan")) for row in validations],
                        style, color="black", label=label)
    axes[2, 1].set(title="Validation motion diagnostics", ylabel="Mean step fraction",
                   ylim=(-0.05, 1.05))
    axes[2, 1].legend()
    for axis in axes[:, 1]:
        axis.set_xticks(positions, labels, rotation=15)
        axis.set_xlabel("Validation checkpoint (equally spaced stages)")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    fig.suptitle(f"DQN pretraining and online diagnostics | seed {seed}")
    fig.savefig(run_dir / "learning_diagnostics.png", dpi=160)
    plt.close(fig)


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
    pretraining_path = run_dir / "pretraining_validation.json"
    pretraining_validation = (
        json.loads(pretraining_path.read_text(encoding="utf-8"))
        if pretraining_path.exists() else None
    )
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
    if pretraining_validation is None:
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
    if pretraining_validation is not None:
        pretraining_updates = summary["pretraining_updates"]
        online_updates = summary["online_updates"]
        if pretraining_updates + online_updates != summary["updates"]:
            raise ValueError("Pretraining and online updates do not sum to total updates.")
        checkpoints = []
        if initial is not None:
            checkpoints.append({
                "label": "Initial", "phase": "initial", "online_episode": 0,
                "optimizer_updates": 0, "validation": initial,
            })
        checkpoints.append({
            "label": "Pretrained", "phase": "pretraining", "online_episode": 0,
            "optimizer_updates": pretraining_updates, "validation": pretraining_validation,
        })
        checkpoints.extend({
            "label": f"Ep {row['episode']}", "phase": "online",
            "online_episode": row["episode"], "optimizer_updates": row["updates"],
            "validation": row["validation"],
        } for row in records if "validation" in row)
        result.update({
            "pretraining_updates": pretraining_updates, "online_updates": online_updates,
            "pretraining_validation": pretraining_validation,
            "pretraining": summary["pretraining"], "validation_checkpoints": checkpoints,
        })
        loss_path = run_dir / "pretraining_metrics.jsonl"
        losses = [json.loads(line) for line in loss_path.read_text(encoding="utf-8").splitlines()
                  if line.strip()] if loss_path.exists() else []
        supervision = supervision_diagnostics(summary["pretraining"], losses)
        if supervision is not None:
            result["pretraining_supervision"] = supervision
        plot_pretraining_run(run_dir, records, checkpoints, settings["training"]["seed"],
                             supervision)
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
