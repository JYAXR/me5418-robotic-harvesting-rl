"""Plot matching routes and the contact opening from recorded environment data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


def draw_branches(axes, frame, detail=False) -> None:
    branches = frame["branches"][:2] if detail else frame["branches"]
    for index, branch in enumerate(branches):
        rest = np.asarray(branch["rest"])
        current = np.asarray(branch["current"])
        axes.plot(*rest.T, "--", color="#9ca3af", linewidth=1.2)
        axes.plot(
            *current.T,
            "-o",
            color="#dc2626" if branch["contact"] else (
                "#92400e" if branch.get("stiffness", 0) >= 1e4 else "#2f855a"
            ),
            linewidth=2.2,
            markersize=3,
        )
        if not detail:
            axes.scatter(*current[0], marker="s", color="#111827", s=28)
            if index == 0:
                axes.plot([], [], "--", color="#9ca3af", label="Rest shape")
                axes.plot([], [], "-", color="#2f855a", label="Current branch")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-dir", type=Path, default=root / "results/demo")
    parser.add_argument("--output-dir", type=Path, default=root / "results/comparison")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--title", default=None)
    parser.add_argument("--contact-policy", choices=("push", "hybrid"), default="push")
    args = parser.parse_args()
    demo = args.demo_dir
    figure, axes = plt.subplots(1, 2, figsize=(11.6, 6.2), layout="constrained")
    for axis, policy in zip(axes, (args.contact_policy, "detour")):
        stem = f"{policy}_seed_{args.seed}"
        record = json.loads((demo / f"{stem}_trace.json").read_text())
        result = json.loads((demo / f"{stem}.json").read_text())
        frames = record["frames"]
        final = frames[-1]
        draw_branches(axis, final)
        if final.get("layout") == "mixed_canopy":
            for (first, second), kind in zip(final["passage_pairs"], final["passage_kinds"]):
                center = 0.5 * (
                    np.asarray(final["branches"][first]["rest"])[-1]
                    + np.asarray(final["branches"][second]["rest"])[-1]
                )
                axis.text(center[0] + 0.013, center[1] + 0.018, kind, fontsize=8)
        positions = np.asarray([frame["position"] for frame in frames])
        axis.plot(*positions.T, color="#2563eb", linewidth=2.1, label="Robot route")
        axis.add_patch(Circle(final["position"], record["robot_radius"], color="#2563eb"))
        axis.add_patch(Circle(final["target"], 0.05, color="#f59e0b", alpha=0.75))
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_aspect("equal")
        axis.set_xlabel("x position")
        axis.set_ylabel("y position")
        axis.grid(alpha=0.14)
        title = (
            "Contact through internal gaps" if policy == "push" else (
                "Hybrid: gap opening + detours" if policy == "hybrid" else "Collision-free A*"
            )
        )
        axis.set_title(title, fontsize=13, fontweight="bold", pad=12)
        axis.text(
            0.02,
            0.04,
            f"{result['steps']} actions | travel {result['travel_distance']:.3f}\n"
            f"Peak deformation / limit: {result['peak_normalized_deformation']:.3f}\n"
            + ("No grid route found" if result.get("planning_failure") else (
                "Safe success" if result["success"] else "Episode failed"
            )),
            transform=axis.transAxes,
            fontsize=10,
            color="#374151",
        )
        if policy in {"push", "hybrid"}:
            # Select the widest first-gate opening while the robot is there.
            gate_x = frames[0]["branches"][0]["rest"][-1][0]
            candidates = [
                frame for frame in frames
                if abs(frame["position"][0] - gate_x) < 0.055
            ]
            contact = max(
                candidates or frames,
                key=lambda frame: np.linalg.norm(
                    np.asarray(frame["branches"][1]["current"])[-1]
                    - np.asarray(frame["branches"][0]["current"])[-1]
                ),
            )
            detail = axis.inset_axes([0.035, 0.68, 0.31, 0.28])
            draw_branches(detail, contact, detail=True)
            detail.add_patch(Circle(contact["position"], record["robot_radius"], color="#2563eb"))
            center = 0.5 * (
                np.asarray(contact["branches"][0]["rest"])[-1]
                + np.asarray(contact["branches"][1]["rest"])[-1]
            )
            detail.set_xlim(center[0] - 0.09, center[0] + 0.09)
            detail.set_ylim(center[1] - 0.085, center[1] + 0.085)
            detail.set_aspect("equal")
            detail.set_title(f"Opening gate 1, step {contact['step']}", fontsize=8)
            detail.tick_params(labelsize=7)
            detail.grid(alpha=0.14)
        else:
            axis.legend(loc="upper left", fontsize=9)
    figure.suptitle(args.title or f"Same scene and actions, seed {args.seed}", fontsize=14)
    output = args.output_dir / f"routes_seed_{args.seed}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
