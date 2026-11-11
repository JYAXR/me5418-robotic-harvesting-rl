"""Plot every scheduled outcome of the single-factor TD comparison."""

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "previous_retention_plot",
    ROOT / "docs/experiments/011_multiscene_retention/plot_multiscene_retention.py",
)
previous = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(previous)


def plot(run_dir, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite the comparison figure: {output}")
    if output.suffix.lower() != ".png":
        raise ValueError("Expected a .png output file.")
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    if status["status"] != "completed":
        raise ValueError("The comparison must finish before plotting.")
    protocol = json.loads((run_dir / "settings.json").read_text(encoding="utf-8"))["training"]
    arms = {
        arm: previous.load_observations(run_dir, f"{arm}/td", previous.TD_CHECKS)
        for arm in ("baseline", "treatment")
    }
    entries = [
        (f"{arm.title()} | {seed}", arms[arm], seed)
        for arm in arms for seed in previous.SEEDS
    ]
    for seed in previous.SEEDS:
        if arms["baseline"][0][seed]["rollout"] != arms["treatment"][0][seed]["rollout"]:
            raise ValueError("The initial policies must have identical trajectories.")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "text.color": "black", "figure.facecolor": "white"})
    figure = plt.figure(figsize=(18, 10))
    grid = figure.add_gridspec(2, 1, hspace=0.32, left=0.045, right=0.985,
                              top=0.82, bottom=0.16)
    for index, (title, steps) in enumerate((
        ("A. Every TD update from 0 through 10", previous.TD_CHECKS[:11]),
        ("B. All later scheduled TD checkpoints", previous.TD_CHECKS[11:]),
    )):
        axis = figure.add_subplot(grid[index])
        axis.set_title(title, loc="left", pad=12)
        previous.draw_table(axis, entries, steps)
    figure.suptitle("Attempt 012: one-factor TD control after first-divergence diagnosis",
                     fontsize=18, y=0.965)
    figure.text(0.5, 0.914,
                "Both arms reuse the same calibrated network, 117 transitions and accumulated Adam state.",
                ha="center")
    figure.text(0.5, 0.876,
                f"{protocol['factor'].replace('_', ' ').title()}: baseline {protocol['baseline_value']:g}, "
                f"treatment {protocol['treatment_value']:g}. All 108 scheduled episodes are shown; columns are TD updates.",
                ha="center")
    figure.text(0.045, 0.075,
                "These are repeated checks of three known training scenes from one network initialization.\n"
                "Checkpoint fractions are not independent test success rates. No held-out evaluation or online exploration.\n"
                "The first-divergence intervention is diagnostic only; every rollout here uses the unmodified greedy neural policy.",
                va="center", fontsize=10, linespacing=1.65)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, facecolor="white")
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
