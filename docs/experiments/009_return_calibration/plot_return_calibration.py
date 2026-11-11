"""Plot fixed-return calibration errors and every scheduled neural rollout outcome."""

import argparse
import json
from math import isfinite
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

SERIES = (
    ("continued_margin", "Matched margin-only control", "--", "s"),
    ("calibrated", "Return calibration + margin", "-", "o"),
)


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


def draw_outcome_table(axis, rows, checkpoints, phase, eligibility):
    labels = []
    for name, label, _, _ in SERIES:
        if phase == "td" and not eligibility[name]:
            label += "\nNot a retention test*"
        labels.append(label)
    count = len(checkpoints)
    axis.set_axis_off()
    table = axis.table(
        cellText=[
            [labels[index]] + [outcome_text(row["rollout"]) for row in rows[name][phase]]
            for index, (name, _, _, _) in enumerate(SERIES)
        ],
        colLabels=["Pre-TD objective"] + [str(step) for step in checkpoints],
        colWidths=[0.23] + [0.77 / count] * count, cellLoc="center", colLoc="center",
        bbox=[0.0, 0.02, 1.0, 0.88],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.2)
    for (row, column), cell in table.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("black")
        cell.visible_edges = ""
        if row == 0:
            cell.visible_edges = "TB"
            cell.set_linewidth(0.8)
            cell.get_text().set_weight("bold")
        elif row == len(SERIES):
            cell.visible_edges = "B"
            cell.set_linewidth(0.8)
        if column == 0:
            cell.get_text().set_ha("left")


def plot(run_dir, output):
    """Read completed evidence only; never load models, update weights or run policies."""
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite the calibration figure: {output}")
    if output.suffix.lower() != ".png":
        raise ValueError("The calibration figure output must have a .png extension.")
    if read_json(run_dir / "run_status.json")["status"] != "completed":
        raise ValueError("The calibration experiment must finish before plotting.")
    summary = read_json(run_dir / "summary.json")
    protocol = summary["protocol"]
    targets = read_json(run_dir / "mc_targets.json")
    baseline = summary["pretrained_policy"]
    rows, eligibility = {}, {}
    if set(summary["arms"]) != {name for name, _, _, _ in SERIES}:
        raise ValueError("Expected the matched control and fixed-return calibration branches.")
    schedules = {phase: tuple(protocol[f"{phase}_checkpoints"])
                 for phase in ("calibration", "td")}
    for phase, checkpoints in schedules.items():
        if (not checkpoints or tuple(sorted(set(checkpoints))) != checkpoints
                or checkpoints[0] != 0 or checkpoints[-1] != protocol[f"{phase}_updates"]):
            raise ValueError(f"Invalid complete checkpoint schedule: {phase}")
    for name, _, _, _ in SERIES:
        arm = summary["arms"][name]
        rows[name] = {}
        eligibility[name] = arm["retention_eligible"]
        if eligibility[name] != arm["calibration"]["final_rollout"]["success"]:
            raise ValueError(f"Retention eligibility differs from calibration outcome: {name}")
        for phase in ("calibration", "td"):
            observations = read_json(run_dir / arm[phase]["directory"] / "observations.json")
            if tuple(row["update"] for row in observations) != schedules[phase]:
                raise ValueError(f"Missing sampled observations: {name}/{phase}")
            for row in observations:
                if (not row["rollout"]["training_scene_only"]
                        or row["rollout"]["seed"] != protocol["scene_seed"]
                        or row["return_fit"]["mc_targets"] != targets["returns"]
                        or not isfinite(row["return_fit"]["rmse"])
                        or row["return_fit"]["rmse"] < 0):
                    raise ValueError("Expected one common training scene and fixed return targets.")
            rows[name][phase] = observations
        if (rows[name]["calibration"][0]["fit"] != baseline["final_fit"]
                or rows[name]["calibration"][0]["rollout"] != baseline["final_rollout"]):
            raise ValueError("The calibration branches must begin with the same acquired policy.")
        if (rows[name]["td"][0]["fit"] != arm["calibration"]["final_fit"]
                or rows[name]["td"][0]["rollout"] != arm["calibration"]["final_rollout"]):
            raise ValueError("The TD handoff must preserve the corresponding calibrated policy.")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "text.color": "black",
        "axes.edgecolor": "black", "axes.labelcolor": "black", "figure.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False,
    })
    figure = plt.figure(figsize=(16.0, 11.8))
    grid = figure.add_gridspec(3, 1, height_ratios=(1.35, 1.0, 1.0), hspace=0.60,
                              left=0.075, right=0.98, top=0.82, bottom=0.15)
    error_axis = figure.add_subplot(grid[0, 0])
    calibration_axis = figure.add_subplot(grid[1, 0])
    td_axis = figure.add_subplot(grid[2, 0])
    calibration_steps = schedules["calibration"]
    for name, label, line, marker in SERIES:
        error_axis.plot(
            range(len(calibration_steps)),
            [row["return_fit"]["rmse"] for row in rows[name]["calibration"]],
            color="black", linestyle=line, marker=marker, markerfacecolor="white",
            markersize=5, linewidth=1.5, label=label,
        )
    error_axis.set_title("A. Error against fixed demonstrated discounted returns", loc="left", pad=12)
    error_axis.set_xticks(range(len(calibration_steps)), labels=[str(step) for step in calibration_steps])
    error_axis.set_xlabel("Pre-TD updates (equally spaced sampled checkpoints)")
    error_axis.set_ylabel("Demonstrated-action RMSE\n(symlog scale; linear below 1)")
    error_axis.set_yscale("symlog", linthresh=1.0)
    largest_error = max(row["return_fit"]["rmse"]
                        for phases in rows.values() for row in phases["calibration"])
    error_axis.set_ylim(0, max(1.0, 1.3 * largest_error))
    error_axis.grid(axis="y", color="0.85", linewidth=0.5)
    error_axis.legend(frameon=False, fontsize=10)
    calibration_axis.set_title("B. Neural rollout during the matched pre-TD phases",
                               loc="left", pad=11)
    draw_outcome_table(calibration_axis, rows, calibration_steps, "calibration", eligibility)
    td_axis.set_title("C. Neural rollout after switching both branches to TD + margin",
                     loc="left", pad=11)
    draw_outcome_table(td_axis, rows, schedules["td"], "td", eligibility)

    figure.suptitle("Attempt 009: fixed-return calibration before TD adaptation", fontsize=17, y=0.97)
    figure.text(0.5, 0.93,
                f"One known training scene (seed {protocol['scene_seed']}); "
                f"{targets['transitions']} demonstration states; "
                f"{protocol['inherited_pretraining_updates']:,} inherited supervised updates",
                ha="center", fontsize=11)
    figure.text(0.5, 0.895,
                f"Both branches receive {protocol['calibration_updates']:,} matched pre-TD updates, "
                f"then {protocol['td_updates']:,} identical TD + margin updates; "
                "fresh Adam and synchronized targets at each boundary.",
                ha="center", fontsize=10.5)
    error_summary = "; ".join(
        f"{label}: {summary['arms'][name]['calibration']['final_return_fit']['rmse']:.4f}"
        for name, label, _, _ in SERIES
    )
    footer = (
        f"Final pre-TD return RMSE — {error_summary}.\n"
        "Every scheduled rollout is shown. Fixed demonstration data; no online exploration or "
        "held-out scenes; generalization is untested.\n"
        "Recorded returns describe the demonstration continuation, not a known optimal Q function."
    )
    if not all(eligibility.values()):
        footer += ("\n*The branch entered TD after an unsuccessful pre-TD rollout; its later behavior "
                   "is not evidence of retaining a successful policy.")
    figure.text(0.075, 0.065, footer, ha="left", va="center", fontsize=10, linespacing=1.5)
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
