"""Plot frozen-dataset agreement and neural rollouts for the fixed-scene diagnostic."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

ARMS = (
    ("td_margin", "TD + margin", "-", "o"),
    ("margin_only", "Margin only", "--", "s"),
    ("cross_entropy", "Cross-entropy only", ":", "^"),
)
CHECKPOINTS = (0, 500, 1000, 2000)


def outcome_text(rollout):
    if rollout["success"]:
        outcome = "Success"
    elif rollout["damage"]:
        outcome = "Fail: damage"
    elif rollout["budget_exhausted"]:
        outcome = "Fail: budget"
    else:
        outcome = "Fail: other"
    return f"{outcome}\n{rollout['steps']} steps"


def plot(run_dir: Path, output: Path):
    """Read completed observation snapshots only; never load or change model weights."""
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite the diagnostic figure: {output}")
    if output.suffix.lower() != ".png":
        raise ValueError("The diagnostic figure output must have a .png extension.")
    observations = {}
    for name, _, _, _ in ARMS:
        rows = json.loads((run_dir / name / "observations.json").read_text(encoding="utf-8"))
        if tuple(row["update"] for row in rows) != CHECKPOINTS:
            raise ValueError(f"All scheduled checkpoints must be complete before plotting: {name}")
        observations[name] = rows
    transition_counts = {
        row["fit"]["transitions"] for rows in observations.values() for row in rows
    }
    scene_seeds = {row["rollout"]["seed"] for rows in observations.values() for row in rows}
    if len(transition_counts) != 1 or len(scene_seeds) != 1:
        raise ValueError("The fitting arms must use one common dataset and scene seed.")
    if not all(row["rollout"]["training_scene_only"]
               for rows in observations.values() for row in rows):
        raise ValueError("Expected training-scene diagnostic rollouts only.")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "black",
        "text.color": "black", "axes.labelcolor": "black", "figure.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False,
    })
    figure = plt.figure(figsize=(13.2, 5.8))
    grid = figure.add_gridspec(1, 2, width_ratios=(1.0, 1.55), wspace=0.22,
                              left=0.065, right=0.98, top=0.76, bottom=0.20)
    accuracy_axis = figure.add_subplot(grid[0, 0])
    outcome_axis = figure.add_subplot(grid[0, 1])
    for name, label, line, marker in ARMS:
        rows = observations[name]
        accuracy_axis.plot(
            CHECKPOINTS, [100 * row["fit"]["action_agreement"] for row in rows],
            color="black", linestyle=line, marker=marker, markersize=5,
            markerfacecolor="white", linewidth=1.5, label=label,
        )
    accuracy_axis.set_title("A. Demonstration action agreement", loc="left", fontsize=11, pad=13)
    accuracy_axis.set_xlabel("Optimizer updates per arm")
    accuracy_axis.set_ylabel("Agreement on all demonstration states (%)")
    accuracy_axis.set_xticks(CHECKPOINTS)
    accuracy_axis.set_ylim(-3, 104)
    accuracy_axis.set_yticks(range(0, 101, 20))
    accuracy_axis.grid(axis="y", color="0.85", linewidth=0.5)
    accuracy_axis.legend(frameon=False, loc="upper left", fontsize=9)

    outcome_axis.set_title("B. Neural rollout on the same training scene", loc="left",
                           fontsize=11, pad=13)
    outcome_axis.set_axis_off()
    row_names = [label for _, label, _, _ in ARMS]
    cell_text = [
        [row_names[index]] + [outcome_text(row["rollout"]) for row in observations[name]]
        for index, (name, _, _, _) in enumerate(ARMS)
    ]
    table = outcome_axis.table(
        cellText=cell_text, colLabels=["Objective"] + [str(step) for step in CHECKPOINTS],
        cellLoc="center", colLoc="center", colWidths=[0.27, 0.18, 0.18, 0.18, 0.19],
        bbox=[0.0, 0.13, 1.0, 0.77],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.7)
    for (row, column), cell in table.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("black")
        cell.visible_edges = ""
        if row == 0:
            cell.visible_edges = "TB"
            cell.set_linewidth(0.8)
            cell.get_text().set_weight("bold")
        elif row == len(ARMS):
            cell.visible_edges = "B"
            cell.set_linewidth(0.8)
        if column == 0:
            cell.get_text().set_ha("left")
    outcome_axis.text(0.5, 0.96, "Columns show optimizer updates", ha="center",
                      transform=outcome_axis.transAxes, fontsize=9)

    count, seed = next(iter(transition_counts)), next(iter(scene_seeds))
    figure.suptitle("Attempt 005: fixed-scene fitting diagnostic", fontsize=15, y=0.96)
    figure.text(0.5, 0.885,
                f"One multi-route training scene (seed {seed}); {count} demonstration states; "
                "identical network initialization",
                ha="center", fontsize=10)
    figure.text(0.065, 0.075,
                "All points use the same complete training dataset. Rollouts use greedy neural "
                "actions without a planner fallback.\n"
                "Fixed-scene success can reflect memorization, including the remaining-budget "
                "input; it does not establish generalization.",
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
