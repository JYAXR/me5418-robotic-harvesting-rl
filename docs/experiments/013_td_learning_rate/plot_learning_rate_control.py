"""Show all scheduled scene outcomes for the preserved-Adam learning-rate control."""

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "lr013_table_helpers", ROOT / "docs/experiments/011_multiscene_retention/plot_multiscene_retention.py",
)
previous = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(previous)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def plot(run_dir, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite the learning-rate figure: {output}")
    if output.suffix.lower() != ".png":
        raise ValueError("Expected a .png output file.")
    if read_json(run_dir / "run_status.json")["status"] != "completed":
        raise ValueError("The learning-rate control must finish before plotting.")
    protocol = read_json(run_dir / "settings.json")["training"]
    if (protocol["factor"] != "learning_rate" or protocol["baseline_value"] != 3e-4
            or protocol["treatment_value"] != 1e-4
            or any(protocol[f"{arm}_settings"]["demonstration_weight"] != 1
                   for arm in ("baseline", "treatment"))):
        raise ValueError("The figure requires the predeclared learning-rate-only comparison.")
    arms = {arm: previous.load_observations(run_dir, f"{arm}/td", previous.TD_CHECKS)
            for arm in ("baseline", "treatment")}
    for seed in previous.SEEDS:
        if arms["baseline"][0][seed]["rollout"] != arms["treatment"][0][seed]["rollout"]:
            raise ValueError("The initial policies must have identical trajectories.")
    labels = {"baseline": "0.0003", "treatment": "0.0001"}
    entries = [(f"{labels[arm]} | {seed}", arms[arm], seed)
               for arm in arms for seed in previous.SEEDS]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "text.color": "black", "figure.facecolor": "white"})
    figure = plt.figure(figsize=(18, 10))
    grid = figure.add_gridspec(2, 1, hspace=0.32, left=0.045, right=0.985, top=0.82, bottom=0.16)
    for index, (title, steps) in enumerate((
        ("A. Every TD update from 0 through 10", previous.TD_CHECKS[:11]),
        ("B. All later scheduled TD checkpoints", previous.TD_CHECKS[11:]),
    )):
        axis = figure.add_subplot(grid[index])
        axis.set_title(title, loc="left", pad=12)
        previous.draw_table(axis, entries, steps)
    figure.suptitle("Attempt 013: lower TD learning rate with calibrated values and preserved Adam",
                     fontsize=18, y=0.965)
    figure.text(0.5, 0.914,
                "Only learning rate changes: 0.0003 versus 0.0001. Demonstration weight remains 1.",
                ha="center")
    figure.text(0.5, 0.876,
                "Same calibrated weights, Adam moments and 117 transitions. All 108 episodes are shown; columns are TD updates.",
                ha="center")
    figure.text(0.045, 0.075,
                "Repeated checks of three known training scenes from one initialization; no held-out evaluation or online exploration.\n"
                "Sampled retention does not establish success at every unsampled update or beyond the 2000-update horizon.\n"
                "Both arms perform real updates under the same budget; a lower rate may slow adaptation or postpone failure.",
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
