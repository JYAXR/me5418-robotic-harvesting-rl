"""Plot every sampled retention outcome after a shared supervised pretraining phase."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

ARMS = (
    ("margin_only", "Margin-only continuation", "--", "s"),
    ("td_only", "TD only", "-", "o"),
    ("td_margin", "TD + margin", ":", "^"),
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
    """Read observations only; preserve intermediate failures and avoid policy inference."""
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite a retention figure: {output}")
    if output.suffix.lower() != ".png":
        raise ValueError("The retention figure output must have a .png extension.")
    if read_json(run_dir / "run_status.json")["status"] != "completed":
        raise ValueError("The diagnostic must finish before plotting its complete outcomes.")
    summary = read_json(run_dir / "summary.json")
    protocol = summary["protocol"]
    checkpoints = tuple(protocol["phase_checkpoints"])
    if not checkpoints or tuple(sorted(set(checkpoints))) != checkpoints or checkpoints[0] != 0:
        raise ValueError("Expected a sorted, unique checkpoint schedule starting at zero.")
    baseline = summary["pretraining"]
    observations = {}
    for name, _, _, _ in ARMS:
        rows = read_json(run_dir / name / "observations.json")
        if tuple(row["update"] for row in rows) != checkpoints:
            raise ValueError(f"The scheduled retention observations are incomplete: {name}")
        if (rows[0]["fit"] != baseline["final_fit"]
                or rows[0]["rollout"] != baseline["final_rollout"]):
            raise ValueError(f"The branch did not start with the common pretrained policy: {name}")
        if not all(row["rollout"]["training_scene_only"]
                   and row["rollout"]["seed"] == protocol["scene_seed"] for row in rows):
            raise ValueError("Expected a single known training scene for every retention check.")
        observations[name] = rows

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
    for name, label, line, marker in ARMS:
        accuracy_axis.plot(
            range(count), [100 * row["fit"]["action_agreement"] for row in observations[name]],
            color="black", linestyle=line, marker=marker, markersize=5,
            markerfacecolor="white", linewidth=1.5, label=label,
        )
    accuracy_axis.set_title("A. Agreement with the demonstrated actions", loc="left", pad=10)
    accuracy_axis.set_ylabel(f"Agreement on all {baseline['final_fit']['transitions']} states (%)")
    accuracy_axis.set_xticks(range(count), labels=[str(value) for value in checkpoints])
    accuracy_axis.set_xlabel("Updates after pretraining (equally spaced sampled checkpoints)")
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
            for name, label, _, _ in ARMS
        ],
        colLabels=["Continuation objective"] + [str(value) for value in checkpoints],
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
        elif row == len(ARMS):
            cell.visible_edges = "B"
            cell.set_linewidth(0.8)
        if column == 0:
            cell.get_text().set_ha("left")

    baseline_fit = baseline["final_fit"]["action_agreement"] * 100
    baseline_rollout = baseline["final_rollout"]
    baseline_result = "success" if baseline_rollout["success"] else "failure"
    figure.suptitle("Attempt 007: retention after supervised pretraining", fontsize=15, y=0.965)
    figure.text(0.5, 0.91,
                f"Known multi-route training scene, seed {protocol['scene_seed']}; "
                f"{protocol['pretraining_updates']:,} supervised updates before the branch point",
                ha="center", fontsize=10)
    figure.text(0.5, 0.865,
                f"Common starting policy: {baseline_fit:.0f}% action agreement; "
                f"{baseline_result} in {baseline_rollout['steps']} steps. "
                "All branches use fresh Adam and synchronized targets.",
                ha="center", fontsize=9.5)
    figure.text(0.075, 0.055,
                "TD updates use the fixed demonstration data; no online exploration or held-out "
                "scenes are used.\n"
                "Every scheduled outcome is shown. Retention between checkpoints is unverified; "
                "single-scene success does not establish generalization.",
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
