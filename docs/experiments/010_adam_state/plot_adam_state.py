"""Plot every scheduled Adam-state outcome without selecting successful checkpoints."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

SERIES = (("reset", "Reset Adam", "--", "s"),
          ("preserve", "Preserve Adam", "-", "o"))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def outcome_text(rollout):
    if rollout["success"]:
        status = "Success"
    elif rollout["damage"]:
        status = "Damage"
    elif rollout["budget_exhausted"]:
        status = "Budget limit"
    else:
        status = "Other failure"
    return f"{status}\n{rollout['steps']} steps"


def draw_table(axis, observations, steps):
    axis.set_axis_off()
    lookup = {name: {row["update"]: row for row in rows}
              for name, rows in observations.items()}
    table = axis.table(
        cellText=[[label] + [outcome_text(lookup[name][step]["rollout"]) for step in steps]
                  for name, label, _, _ in SERIES],
        colLabels=["Optimizer boundary"] + [str(step) for step in steps],
        colWidths=[0.16] + [0.84 / len(steps)] * len(steps),
        cellLoc="center", colLoc="center", bbox=[0, 0, 1, 0.88],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row, column), cell in table.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("black")
        cell.set_linewidth(0.8)
        cell.visible_edges = "TB" if row == 0 else "B" if row == 2 else ""
        if row == 0:
            cell.get_text().set_weight("bold")
        if column == 0:
            cell.get_text().set_ha("left")


def plot(run_dir, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite the Adam-state figure: {output}")
    if output.suffix.lower() != ".png":
        raise ValueError("The figure output must have a .png extension.")
    if read_json(run_dir / "run_status.json")["status"] != "completed":
        raise ValueError("The Adam-state experiment must finish before plotting.")
    summary = read_json(run_dir / "summary.json")
    observations = {
        name: read_json(run_dir / summary["arms"][name]["directory"] / "observations.json")
        for name, _, _, _ in SERIES
    }
    steps = [*range(11), 20, 50, 100, 200, 500, 1000, 2000]
    for rows in observations.values():
        if [row["update"] for row in rows] != steps:
            raise ValueError("Missing predeclared early or later TD observations.")
        if any(not row["rollout"]["training_scene_only"] for row in rows):
            raise ValueError("Expected training-scene-only observations.")
    if observations["reset"][0]["rollout"] != observations["preserve"][0]["rollout"]:
        raise ValueError("The two branches did not start from the same rollout.")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "text.color": "black",
        "axes.edgecolor": "black", "axes.labelcolor": "black", "figure.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False,
    })
    figure = plt.figure(figsize=(17, 11))
    grid = figure.add_gridspec(3, 1, height_ratios=(1.3, 1, 1), hspace=0.60,
                              left=0.075, right=0.98, top=0.80, bottom=0.14)
    fit_axis = figure.add_subplot(grid[0])
    early_axis = figure.add_subplot(grid[1])
    later_axis = figure.add_subplot(grid[2])
    for name, label, line, marker in SERIES:
        fit_axis.plot(range(len(steps)), [row["fit"]["correct"] for row in observations[name]],
                      color="black", linestyle=line, marker=marker, markerfacecolor="white",
                      markersize=5, linewidth=1.5, label=label)
    fit_axis.set_title("A. Demonstrated action agreement during TD adaptation", loc="left", pad=12)
    fit_axis.set_xticks(range(len(steps)), labels=[str(step) for step in steps])
    fit_axis.set_ylim(0, 36)
    fit_axis.set_yticks([0, 10, 20, 30, 35])
    fit_axis.set_ylabel("Correct actions / 35 states")
    fit_axis.set_xlabel("TD updates (equally spaced sampled checkpoints)")
    fit_axis.legend(frameon=False)
    early_axis.set_title("B. Actual greedy episodes at every update from 0 to 10", loc="left", pad=12)
    draw_table(early_axis, observations, steps[:11])
    later_axis.set_title("C. Actual greedy episodes at all later scheduled checkpoints", loc="left", pad=12)
    draw_table(later_axis, observations, steps[11:])
    figure.suptitle("Attempt 010: Adam state at the calibrated TD boundary", fontsize=18, y=0.97)
    figure.text(0.5, 0.925, "Identical calibrated weights and fixed 35-transition demonstration; "
                "one known training scene (seed 10000).", ha="center", fontsize=11)
    figure.text(0.5, 0.885, "Only accumulated Adam state differs: reset versus preserved moments "
                "and step counter. Learning rate remains 0.0003.", ha="center", fontsize=11)
    figure.text(0.5, 0.85, "Both target networks start synchronized; "
                "both TD phase clocks restart at zero and synchronize every 200 updates.",
                ha="center", fontsize=11)
    figure.text(0.075, 0.06, "All 36 scheduled TD rollouts are shown, including failures. "
                "Action agreement alone does not establish task success.\n"
                "These are checkpoints from one known scene, not independent test episodes. "
                "No online exploration or held-out testing; generalization is untested.",
                va="center", fontsize=10, linespacing=1.6)
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
