"""Show every unmodified and one-action-corrected episode without color coding."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def plot(run_dir):
    if read_json(run_dir / "run_status.json")["status"] != "completed":
        raise ValueError("Complete the diagnostic before plotting.")
    output = run_dir / "single_action_outcomes.png"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite figure: {output}")
    rows = []
    for update in (2009, 10000):
        for mode, label in (("greedy", "Original greedy"), ("counterfactual", "Step 21 corrected")):
            trace = read_json(run_dir / f"update_{update}_{mode}.json")
            rollout = trace["rollout"]
            outcome = "Success" if rollout["success"] else "Damage" if rollout["damage"] else (
                "Budget exhausted" if rollout["budget_exhausted"] else "Other failure")
            blocked = sum(record["collision_blocked"] for record in trace["records"])
            rows.append([str(update), label, outcome, str(rollout["steps"]),
                         f"{rollout['return']:.4f}", str(blocked), str(trace["forced_action_count"])])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "text.color": "black", "figure.facecolor": "white"})
    figure, axis = plt.subplots(figsize=(14, 5.4))
    figure.subplots_adjust(left=0.035, right=0.975, bottom=0.26, top=0.73)
    axis.set_axis_off()
    table = axis.table(
        cellText=rows, colLabels=["TD model", "Replay", "Outcome", "Actions", "Return", "Blocked", "Forced"],
        colWidths=[0.09, 0.24, 0.21, 0.10, 0.14, 0.11, 0.11],
        cellLoc="center", colLoc="center", bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (row, _), cell in table.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("black")
        cell.set_linewidth(0.9)
        cell.visible_edges = "TB" if row == 0 else "B" if row == len(rows) else ""
        if row == 0:
            cell.get_text().set_weight("bold")
    figure.suptitle("Attempt 015: replace only step 21 with north_small", fontsize=18, y=0.94)
    figure.text(0.5, 0.83, "Scene 10001; frozen TD 2009 and 10000 models; "
                "four episodes and zero training updates.", ha="center")
    figure.text(0.035, 0.10,
                "All other actions come from the same frozen neural policy, including every step after the intervention.\n"
                "Corrected episodes are assisted diagnostic trials. They do not change the original policy's recorded failures.",
                fontsize=10, linespacing=1.7)
    figure.savefig(output, dpi=170, facecolor="white")
    plt.close(figure)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(plot(args.run_dir.resolve()))


if __name__ == "__main__":
    main()
