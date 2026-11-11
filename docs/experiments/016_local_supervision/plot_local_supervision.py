"""Compare all scene outcomes and neighboring fixed-state Q gaps in black and white."""

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "local016_table_helpers", ROOT / "docs/experiments/011_multiscene_retention/plot_multiscene_retention.py",
)
previous = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(previous)
CHECKS = (2000, *range(2001, 2011), 2200, 2201, *range(2500, 10001, 500))
NEIGHBORS = (20, 21, 22, 23)
ARMS = ("baseline", "treatment")


def dense_gaps(path):
    updates, gaps = [], {step: [] for step in NEIGHBORS}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["scene_seed"] != 10001:
                raise ValueError("The dense probes must refer to scene 10001.")
            states = {state["reference_step"]: state for state in row["states"]}
            updates.append(row["update"])
            for step in NEIGHBORS:
                gaps[step].append(states[step]["signed_demonstration_gap"])
    if updates != list(range(2000, 10001)):
        raise ValueError("Every absolute update from 2000 to 10000 must be present.")
    return updates, gaps


def plot(run_dir):
    if previous.read_json(run_dir / "run_status.json")["status"] != "completed":
        raise ValueError("Complete both arms before plotting.")
    outcomes_path, gaps_path = run_dir / "local_supervision_outcomes.png", run_dir / "neighbor_q_gaps.png"
    if outcomes_path.exists() or gaps_path.exists():
        raise FileExistsError("Refusing to overwrite local-supervision figures.")
    outcomes = {arm: previous.load_observations(run_dir, f"{arm}/td", CHECKS) for arm in ARMS}
    dense = {arm: dense_gaps(run_dir / arm / "td/sentinel_q.jsonl") for arm in ARMS}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "text.color": "black", "figure.facecolor": "white"})
    entries = [(f"{arm.title()} | {seed}", outcomes[arm], seed) for arm in ARMS for seed in previous.SEEDS]
    figure = plt.figure(figsize=(18, 14))
    grid = figure.add_gridspec(3, 1, hspace=0.28, left=0.035, right=0.985, top=0.85, bottom=0.13)
    for index, (title, checks) in enumerate((
        ("A. Resume and every continuation update from 2001 through 2010", CHECKS[:11]),
        ("B. Next target synchronization and scheduled checks through TD 5000", CHECKS[11:19]),
        ("C. Remaining scheduled checkpoints through TD 10000", CHECKS[19:]),
    )):
        axis = figure.add_subplot(grid[index])
        axis.set_title(title, loc="left", pad=10)
        previous.draw_table(axis, entries, checks)
    figure.suptitle("Attempt 016: double only the step-21 sample's supervision", fontsize=18, y=0.965)
    figure.text(0.5, 0.925, "Same TD 2000 full state, learning rate 0.0001 and preserved Adam. "
                "All other supervised coefficients remain 1.", ha="center")
    figure.text(0.5, 0.891, "All 174 scheduled episodes are shown. Every evaluation action comes "
                "from the neural policy without correction.", ha="center")
    figure.text(0.035, 0.06, "Repeated checks of three known training scenes from one initialization; "
                "the selected sample was identified by earlier failure diagnosis.\n"
                "Checkpoint success does not establish success between checkpoints or on unseen scenes.",
                fontsize=10, linespacing=1.7)
    figure.savefig(outcomes_path, dpi=160, facecolor="white")
    plt.close(figure)

    figure, axes = plt.subplots(4, 2, figsize=(16, 12), sharex=True, sharey="row")
    figure.subplots_adjust(left=0.07, right=0.985, top=0.86, bottom=0.14, hspace=0.48, wspace=0.10)
    for row_index, step in enumerate(NEIGHBORS):
        for column, arm in enumerate(ARMS):
            axis = axes[row_index, column]
            updates, gaps = dense[arm]
            axis.plot(updates, gaps[step], color="black", linewidth=0.6)
            axis.axhline(0, color="black", linestyle="--", linewidth=0.8)
            axis.set_title(f"{arm.title()} | Reference step {step}", loc="left", fontsize=11)
            axis.spines[["top", "right"]].set_visible(False)
            axis.set_xlim(2000, 10000)
            if column == 0:
                axis.set_ylabel("Signed Q gap")
            if row_index == len(NEIGHBORS) - 1:
                axis.set_xlabel("Absolute TD update")
    figure.suptitle("Scene 10001: target decision and neighboring action rankings", fontsize=17, y=0.965)
    figure.text(0.5, 0.918, "Only step 21 receives extra supervision. Both columns use the same "
                "vertical scale within each row; all 8001 records per arm are shown.", ha="center")
    figure.text(0.07, 0.065, "Gap = Q(demonstrated action) - max Q(other actions). "
                "The dashed line marks zero. Negative gaps are reference-state rank reversals.\n"
                "These probes execute no environment actions and do not measure episode failure. "
                "Steps 6, 18 and 24 remain in the complete recorded analysis.", fontsize=10, linespacing=1.7)
    figure.savefig(gaps_path, dpi=160, facecolor="white")
    plt.close(figure)
    return outcomes_path, gaps_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in plot(args.run_dir.resolve()):
        print(path)


if __name__ == "__main__":
    main()
