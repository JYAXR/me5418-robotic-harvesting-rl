"""Compare contact and collision-free demos on identical deterministic scenes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml
from run_demo import run_episode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("results/comparison"))
    parser.add_argument("--config", type=Path, default=Path("configs/environment.yaml"))
    parser.add_argument(
        "--policies", nargs="+", choices=("push", "hybrid", "detour", "random", "greedy"),
        default=("push", "detour"),
    )
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    results = []
    for seed in range(args.seed, args.seed + args.seeds):
        for policy in args.policies:
            result = run_episode(policy, seed, None, config)
            results.append(result)
            print(
                f"seed={seed} policy={policy:6s} success={result['success']} "
                f"damage={result['damage']} travel={result['travel_distance']:.3f} "
                f"peak_deformation={result['peak_normalized_deformation']:.3f} "
                f"planning_failure={bool(result['planning_failure'])}",
                flush=True,
            )
    (args.output_dir / "episodes.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "episodes.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved paired diagnostic results in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
