"""Show all per-scene outcomes of staged learning and both TD state treatments."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

SEEDS = (10000, 10001, 10002)
PRETRAIN_CHECKS = (0, 1000, 5000, 10000, 20000)
CALIBRATION_CHECKS = (0, 1, 10, 100, 1000, 5000, 10000, 20000)
TD_CHECKS = (*range(11), 20, 50, 100, 200, 500, 1000, 2000)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def outcome_text(rollout):
    status = "Success" if rollout["success"] else "Damage" if rollout["damage"] else (
        "Budget limit" if rollout["budget_exhausted"] else "Other failure")
    return f"{status}\n{rollout['steps']} steps"


def load_observations(run_dir, directory, schedule):
    rows = read_json(run_dir / directory / "observations.json")
    if tuple(row["update"] for row in rows) != schedule:
        raise ValueError(f"The complete predeclared schedule is required: {directory}")
    for row in rows:
        if tuple(scene["seed"] for scene in row["scenes"]) != SEEDS:
            raise ValueError("Each checkpoint must contain all three ordered training scenes.")
        if any(not scene["rollout"]["training_scene_only"] for scene in row["scenes"]):
            raise ValueError("Expected training-only evaluation records.")
    return {row["update"]: {scene["seed"]: scene for scene in row["scenes"]} for row in rows}


def draw_table(axis, entries, steps):
    axis.set_axis_off()
    table = axis.table(
        cellText=[[label] + [outcome_text(observations[step][seed]["rollout"]) for step in steps]
                  for label, observations, seed in entries],
        colLabels=["Training scene"] + [str(step) for step in steps],
        colWidths=[0.17] + [0.83 / len(steps)] * len(steps),
        cellLoc="center", colLoc="center", bbox=[0, 0, 1, 0.90],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.4)
    for (row, column), cell in table.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("black")
        cell.set_linewidth(0.8)
        cell.visible_edges = "TB" if row == 0 else "B" if row == len(entries) else ""
        if row == 0:
            cell.get_text().set_weight("bold")
        if column == 0:
            cell.get_text().set_ha("left")


def plot(run_dir, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite the multi-scene figure: {output}")
    if output.suffix.lower() != ".png":
        raise ValueError("The figure output must have a .png extension.")
    if read_json(run_dir / "run_status.json")["status"] != "completed":
        raise ValueError("The multi-scene run must finish before plotting.")
    pretraining = load_observations(run_dir, "pretraining", PRETRAIN_CHECKS)
    calibration = load_observations(run_dir, "calibration", CALIBRATION_CHECKS)
    arms = {arm: load_observations(run_dir, f"{arm}/td", TD_CHECKS)
            for arm in ("reset", "preserve")}
    for arm in arms.values():
        for seed in SEEDS:
            if arm[0][seed]["rollout"] != calibration[20000][seed]["rollout"]:
                raise ValueError("TD must begin from unchanged calibrated behavior in each scene.")
    eligible = [seed for seed in SEEDS if calibration[20000][seed]["rollout"]["success"]]
    entries = [(f"{label} | {seed}", arms[arm], seed)
               for arm, label in (("reset", "Reset"), ("preserve", "Preserve")) for seed in SEEDS]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "text.color": "black", "figure.facecolor": "white"})
    figure = plt.figure(figsize=(17, 16))
    grid = figure.add_gridspec(4, 1, height_ratios=(1, 1, 1.5, 1.5), hspace=0.32,
                              left=0.055, right=0.98, top=0.86, bottom=0.10)
    panels = (
        ("A. Supervised pretraining: 20000 updates from fresh seed-0 weights",
         [(str(seed), pretraining, seed) for seed in SEEDS], PRETRAIN_CHECKS),
        ("B. Actual-return calibration: 20000 updates with fresh Adam",
         [(str(seed), calibration, seed) for seed in SEEDS], CALIBRATION_CHECKS),
        ("C. TD transition: every update from 0 to 10", entries, TD_CHECKS[:11]),
        ("D. TD adaptation: all later scheduled checkpoints", entries, TD_CHECKS[11:]),
    )
    for index, (title, rows, steps) in enumerate(panels):
        axis = figure.add_subplot(grid[index])
        axis.set_title(title, loc="left", pad=12)
        draw_table(axis, rows, steps)
    figure.suptitle("Attempt 011: staged learning across three known training scenes", fontsize=18, y=0.975)
    figure.text(0.5, 0.942, "One shared network; full-batch supervision, per-episode discounted-return "
                "calibration, then matched TD + margin treatments.", ha="center", fontsize=11)
    figure.text(0.5, 0.911, "Only Adam state differs between TD arms. "
                "Weights, data, learning rate and target synchronization schedule are matched.",
                ha="center", fontsize=11)
    figure.text(0.5, 0.882, "Columns show optimizer updates within each phase; "
                "all scene outcomes are retained, including failures.", ha="center", fontsize=11)
    eligible_text = ", ".join(map(str, eligible)) if eligible else "none"
    footer = (
        f"TD retention-eligible scenes at the calibration boundary: {eligible_text}. "
        "Later success in an ineligible scene is recovery/acquisition, not retention.\n"
        "All 147 scheduled neural episodes are shown. These are repeated training-scene checks, "
        "not independent test success rates.\n"
        "No held-out scenes or online exploration; broader generalization and final project readiness "
        "are not established."
    )
    figure.text(0.055, 0.048, footer, ha="left", va="center", fontsize=9.5, linespacing=1.6)
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
