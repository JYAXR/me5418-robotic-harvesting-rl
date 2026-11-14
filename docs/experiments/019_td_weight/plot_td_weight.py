"""Plot complete physical outcomes and seven dense Q-gap series for the TD-weight trial."""

import argparse
import importlib.util
import json
from math import isfinite
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "td019_plot_helpers", ROOT / "docs/experiments/016_local_supervision/plot_local_supervision.py")
previous = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(previous)
CHECKS = previous.CHECKS
STEPS = (6, 18, 20, 21, 22, 23, 24)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def dense_gaps(path, start=2000, end=10000):
    """Read all seven recorded states; never silently drop the earlier fragile decisions."""
    updates, gaps = [], {step: [] for step in STEPS}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["scene_seed"] != 10001 or row["extension_update"] != row["update"] - start:
                raise ValueError("Dense probes must preserve their declared scene and absolute clock.")
            if tuple(state["reference_step"] for state in row["states"]) != STEPS:
                raise ValueError("All seven fixed reference probes must be present once in their declared order.")
            updates.append(row["update"])
            for state in row["states"]:
                gap = state["signed_demonstration_gap"]
                if not isfinite(gap):
                    raise ValueError("Reference-state Q gaps must be finite.")
                gaps[state["reference_step"]].append(gap)
    if updates != list(range(start, end + 1)):
        raise ValueError("Every declared absolute update must be present exactly once.")
    return updates, gaps


def plot(run_dir):
    if read_json(run_dir / "run_status.json")["status"] != "completed":
        raise ValueError("Complete the TD-weight treatment before plotting.")
    control = Path(read_json(run_dir / "baseline_reference.json")["reference_run"])
    outcome_path, gap_path = run_dir / "td_weight_outcomes.png", run_dir / "neighbor_q_gaps.png"
    if outcome_path.exists() or gap_path.exists():
        raise FileExistsError("Refusing to overwrite the TD-weight figures.")
    sources = {"baseline": (control, "baseline/td"), "treatment": (run_dir, "treatment/td")}
    labels = {"baseline": "Historical TD weight 1", "treatment": "TD weight 0.5"}
    outcomes = {arm: previous.previous.load_observations(directory, phase, CHECKS)
                for arm, (directory, phase) in sources.items()}
    dense = {arm: dense_gaps(directory / phase / "sentinel_q.jsonl")
             for arm, (directory, phase) in sources.items()}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "text.color": "black", "figure.facecolor": "white"})
    entries = [(f"{labels[arm]} | {seed}", outcomes[arm], seed)
               for arm in sources for seed in previous.previous.SEEDS]
    figure = plt.figure(figsize=(18, 14))
    grid = figure.add_gridspec(3, 1, hspace=0.28, left=0.035, right=0.985, top=0.85, bottom=0.13)
    for index, (title, checks) in enumerate((
        ("A. Resume and each update from TD 2001 through 2010", CHECKS[:11]),
        ("B. Target synchronization and scheduled checks through TD 5000", CHECKS[11:19]),
        ("C. Remaining scheduled checks through TD 10000", CHECKS[19:]),
    )):
        axis = figure.add_subplot(grid[index])
        axis.set_title(title, loc="left", pad=10)
        previous.previous.draw_table(axis, entries, checks)
    figure.suptitle("Attempt 019: halve only the TD loss coefficient", fontsize=18, y=0.965)
    figure.text(0.5, 0.925, "Same full TD 2000 state, retained Adam and learning rate 0.0001. "
                "All margin weights stay 1; both arms have no MC objective.", ha="center")
    figure.text(0.5, 0.891, "87 new treatment episodes and 87 verified historical control episodes "
                "are shown. Evaluation actions are autonomous.", ha="center")
    figure.text(0.035, 0.06, "Three known training scenes and one initialization. "
                "Historical control training and evaluation are reused, not rerun.\n"
                "Physical checks do not establish retention between checkpoints or on unseen layouts.",
                fontsize=10, linespacing=1.7)
    figure.savefig(outcome_path, dpi=160, facecolor="white")
    plt.close(figure)

    figure, axes = plt.subplots(len(STEPS), 2, figsize=(16, 19), sharex=True, sharey="row")
    figure.subplots_adjust(left=0.07, right=0.985, top=0.91, bottom=0.09, hspace=0.48, wspace=0.10)
    for row_index, step in enumerate(STEPS):
        for column, arm in enumerate(sources):
            axis = axes[row_index, column]
            updates, gaps = dense[arm]
            axis.plot(updates, gaps[step], color="black", linewidth=0.6)
            axis.axhline(0, color="black", linestyle="--", linewidth=0.8)
            axis.set_title(f"{labels[arm]} | Reference step {step}", loc="left", fontsize=11)
            axis.spines[["top", "right"]].set_visible(False)
            axis.set_xlim(2000, 10000)
            if column == 0:
                axis.set_ylabel("Signed Q gap")
            if row_index == len(STEPS) - 1:
                axis.set_xlabel("Absolute TD update")
    figure.suptitle("Scene 10001: all seven recorded reference decisions", fontsize=17, y=0.98)
    figure.text(0.5, 0.946, "All 8001 records per series; vertical scales match within each row. "
                "The control series is reused from verified attempt 016.", ha="center")
    figure.text(0.07, 0.035, "Gap = Q(demonstrated action) - max Q(other actions). "
                "Negative values are reference-state ranking reversals.\n"
                "These probes execute no environment actions and do not measure episode failure.",
                fontsize=10, linespacing=1.7)
    figure.savefig(gap_path, dpi=160, facecolor="white")
    plt.close(figure)
    return outcome_path, gap_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in plot(args.run_dir.resolve()):
        print(path)


if __name__ == "__main__":
    main()
