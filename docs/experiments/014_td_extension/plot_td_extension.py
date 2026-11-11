"""Plot every scheduled outcome and all dense reference-state Q gaps."""

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "extension014_tables", ROOT / "docs/experiments/011_multiscene_retention/plot_multiscene_retention.py",
)
previous = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(previous)
CHECKS = (2000, *range(2001, 2011), 2200, 2201, *range(2500, 10001, 500))
STEPS = (6, 18, 22, 24)


def plot(run_dir):
    if previous.read_json(run_dir / "run_status.json")["status"] != "completed":
        raise ValueError("Complete the extension before plotting.")
    outcomes_path = run_dir / "extension_outcomes.png"
    gaps_path = run_dir / "sentinel_gaps.png"
    if outcomes_path.exists() or gaps_path.exists():
        raise FileExistsError("Refusing to overwrite existing extension figures.")
    outcomes = previous.load_observations(run_dir, "td", CHECKS)
    probes = [json.loads(line) for line in (run_dir / "td/sentinel_q.jsonl").read_text(
        encoding="utf-8").splitlines()]
    if [row["update"] for row in probes] != list(range(2000, 10001)):
        raise ValueError("The complete dense Q-probe sequence is required.")
    if any(row["scene_seed"] != 10001 or tuple(state["reference_step"] for state in row["states"])
           != STEPS for row in probes):
        raise ValueError("Every probe must contain the four fixed scene-10001 states.")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "text.color": "black", "figure.facecolor": "white"})
    figure = plt.figure(figsize=(18, 11))
    grid = figure.add_gridspec(3, 1, hspace=0.34, left=0.045, right=0.985, top=0.83, bottom=0.15)
    entries = [(str(seed), outcomes, seed) for seed in previous.SEEDS]
    for index, (title, checks) in enumerate((
        ("A. Resume and the first ten continuation updates", CHECKS[:11]),
        ("B. Next target synchronization and continuation through TD 5000", CHECKS[11:19]),
        ("C. Remaining scheduled checkpoints through TD 10000", CHECKS[19:]),
    )):
        axis = figure.add_subplot(grid[index])
        axis.set_title(title, loc="left", pad=10)
        previous.draw_table(axis, entries, checks)
    figure.suptitle("Attempt 014: extend preserved-Adam TD from 2000 to 10000 updates",
                     fontsize=18, y=0.965)
    figure.text(0.5, 0.916, "Learning rate 0.0001; demonstration weight 1; unchanged Adam state "
                "and 117 training transitions.", ha="center")
    figure.text(0.5, 0.879, "All 87 scheduled episodes are shown. Columns use absolute TD updates; "
                "the 2000 boundary is re-evaluated.", ha="center")
    figure.text(0.045, 0.065, "Repeated checks of three known training scenes from one initialization. "
                "No held-out evaluation or online exploration.\n"
                "Success at scheduled checkpoints does not establish success at every intervening "
                "update, on unseen scenes, or indefinitely.", va="center", fontsize=10, linespacing=1.65)
    figure.savefig(outcomes_path, dpi=160, facecolor="white")
    plt.close(figure)

    figure, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    figure.subplots_adjust(left=0.09, right=0.98, top=0.87, bottom=0.14, hspace=0.43)
    updates = [row["update"] for row in probes]
    for index, (axis, step) in enumerate(zip(axes, STEPS, strict=True)):
        gaps = [row["states"][index]["signed_demonstration_gap"] for row in probes]
        axis.plot(updates, gaps, color="black", linewidth=0.65)
        axis.axhline(0, color="black", linestyle="--", linewidth=0.8)
        axis.set_title(f"Reference step {step}", loc="left", fontsize=11)
        axis.set_ylabel("Signed Q gap")
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlim(2000, 10000)
    axes[-1].set_xlabel("Absolute TD update")
    figure.suptitle("Scene 10001: dense Q probes at fixed demonstration states", fontsize=17, y=0.965)
    figure.text(0.5, 0.918, "All 8001 records: restored TD 2000 and every subsequent update "
                "through TD 10000.", ha="center")
    figure.text(0.09, 0.055, "Gap = Q(demonstrated action) - max Q(other actions). Negative values "
                "indicate a reference-state ranking reversal.\n"
                "These probes execute no environment actions and are not measured episode failures. "
                "Step 24 is a known successful-deviation control.", fontsize=10, linespacing=1.65)
    figure.savefig(gaps_path, dpi=160, facecolor="white")
    plt.close(figure)
    return [outcomes_path, gaps_path]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in plot(args.run_dir.resolve()):
        print(path)


if __name__ == "__main__":
    main()
