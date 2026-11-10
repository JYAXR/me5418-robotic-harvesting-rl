"""Plot learning-rate retention controls with the historical baseline identified."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

SERIES = (
    ("historical", "3e-4 (historical 007)", ":", "^"),
    ("lr_3e_5", "3e-5", "--", "s"),
    ("lr_3e_6", "3e-6", "-", "o"),
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def outcome_text(rollout):
    if rollout["success"]:
        outcome = "Success"
    elif rollout["damage"]:
        outcome = "Damage"
    elif rollout["budget_exhausted"]:
        outcome = "Budget limit"
    else:
        outcome = "Other failure"
    return f"{outcome}\n{rollout['steps']} steps"


def plot(run_dir, output):
    """Read all completed observation snapshots; do not run any policy or select a checkpoint."""
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite a learning-rate figure: {output}")
    if output.suffix.lower() != ".png":
        raise ValueError("The retention figure output must have a .png extension.")
    if read_json(run_dir / "run_status.json")["status"] != "completed":
        raise ValueError("The learning-rate diagnostic must finish before plotting.")
    summary = read_json(run_dir / "summary.json")
    manifest = read_json(run_dir / "input_manifest.json")
    historical = read_json(run_dir / "historical_td_margin.json")
    if (not historical["historical_reference_only"]
            or historical != summary["historical_td_margin"]
            or manifest["historical_baseline_reexecuted"]
            or "run_status.json" not in manifest["reference_files_sha256"]):
        raise ValueError("Expected the recorded, unreexecuted historical reference from attempt 007.")
    protocol = summary["protocol"]
    checkpoints = tuple(protocol["phase_checkpoints"])
    if not checkpoints or tuple(sorted(set(checkpoints))) != checkpoints or checkpoints[0] != 0:
        raise ValueError("Expected a sorted, unique checkpoint schedule starting at zero.")
    if checkpoints[-1] != protocol["phase_updates"]:
        raise ValueError("The plot must include the full declared adaptation budget.")
    baseline = summary["pretrained_policy"]
    observations = {"historical": historical["observations"]}
    for name, arm in summary["arms"].items():
        observations[name] = read_json(run_dir / arm["directory"] / "observations.json")
    if set(observations) != {name for name, _, _, _ in SERIES}:
        raise ValueError("Expected both new rates and the historical reference.")
    for name, rows in observations.items():
        if tuple(row["update"] for row in rows) != checkpoints:
            raise ValueError(f"The scheduled retention observations are incomplete: {name}")
        if (rows[0]["fit"] != baseline["final_fit"]
                or rows[0]["rollout"] != baseline["final_rollout"]):
            raise ValueError(f"The rate comparison did not share one starting policy: {name}")
        if not all(row["rollout"]["training_scene_only"]
                   and row["rollout"]["seed"] == protocol["scene_seed"] for row in rows):
            raise ValueError("Expected the same known training scene for every observation.")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "black",
        "text.color": "black", "axes.labelcolor": "black", "figure.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False,
    })
    count = len(checkpoints)
    figure = plt.figure(figsize=(max(13.5, 1.3 * count + 2), 8.0))
    grid = figure.add_gridspec(2, 1, height_ratios=(1.25, 1.0), hspace=0.55,
                              left=0.075, right=0.98, top=0.79, bottom=0.13)
    accuracy_axis = figure.add_subplot(grid[0, 0])
    outcome_axis = figure.add_subplot(grid[1, 0])
    for name, label, line, marker in SERIES:
        accuracy_axis.plot(
            range(count), [100 * row["fit"]["action_agreement"] for row in observations[name]],
            color="black", linestyle=line, marker=marker, markersize=5,
            markerfacecolor="white", linewidth=1.5, label=label,
        )
    accuracy_axis.set_title("A. Agreement with the demonstrated actions", loc="left", pad=10)
    accuracy_axis.set_ylabel(f"Agreement on all {baseline['final_fit']['transitions']} states (%)")
    accuracy_axis.set_xticks(range(count), labels=[str(value) for value in checkpoints])
    accuracy_axis.set_xlabel("TD + margin updates (equally spaced sampled checkpoints)")
    accuracy_axis.set_ylim(-3, 104)
    accuracy_axis.set_yticks(range(0, 101, 20))
    accuracy_axis.grid(axis="y", color="0.85", linewidth=0.5)
    accuracy_axis.legend(frameon=False, loc="lower left", ncol=3, fontsize=9)

    outcome_axis.set_title("B. Greedy neural rollout on the same known training scene",
                           loc="left", pad=10)
    outcome_axis.set_axis_off()
    table = outcome_axis.table(
        cellText=[
            [label] + [outcome_text(row["rollout"]) for row in observations[name]]
            for name, label, _, _ in SERIES
        ],
        colLabels=["Adaptation learning rate"] + [str(value) for value in checkpoints],
        cellLoc="center", colLoc="center",
        colWidths=[2.0 / (count + 2)] + [1.0 / (count + 2)] * count,
        bbox=[0.0, 0.0, 1.0, 0.94],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.4)
    for (row, column), cell in table.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("black")
        cell.visible_edges = ""
        if row == 0:
            cell.visible_edges = "TB"
            cell.set_linewidth(0.8)
            cell.get_text().set_weight("bold")
        elif row == len(SERIES):
            cell.visible_edges = "B"
            cell.set_linewidth(0.8)
        if column == 0:
            cell.get_text().set_ha("left")

    figure.suptitle("Attempt 008: learning rate and retention after pretraining",
                   fontsize=15, y=0.965)
    figure.text(0.5, 0.91,
                f"Known multi-route training scene, seed {protocol['scene_seed']}; "
                f"{protocol['phase_updates']:,} TD + margin updates per rate; "
                f"same policy after {protocol['pretraining_updates']:,} supervised updates",
                ha="center", fontsize=10)
    figure.text(0.5, 0.865,
                "The 3e-4 control is historical evidence from attempt 007, not a new run. "
                "Both smaller rates start with fresh Adam and synchronized targets.",
                ha="center", fontsize=9.5)
    figure.text(0.075, 0.055,
                "All scheduled outcomes are shown; success between checkpoints is unverified. "
                "The demonstration data are fixed, with no online exploration.\n"
                "One-scene retention does not establish generalization or convergence; "
                "smaller steps may postpone later forgetting.",
                ha="left", va="center", fontsize=9, linespacing=1.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(plot(args.run_dir.resolve(), args.output.resolve()))


if __name__ == "__main__":
    main()
