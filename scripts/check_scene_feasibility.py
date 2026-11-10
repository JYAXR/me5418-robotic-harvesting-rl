"""Record real-dynamics reference-route checks for every requested scene seed."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import yaml

from me5418_harvesting import HarvestingEnv
from me5418_harvesting.evaluation.scene_feasibility import verify_reference_route


def check_scenes(config: dict, seed: int, count: int) -> dict:
    """Check raw geometrically valid samples without filtering by reference success."""
    if seed < 0 or count <= 0:
        raise ValueError("The first seed must be nonnegative and count must be positive.")
    records = []
    # Disable optional reset-time certification to avoid conditioning this audit
    # on the very controller whose failures it is intended to measure.
    audit_config = deepcopy(config)
    audit_config["verify_reference_route"] = False
    env = HarvestingEnv(audit_config)
    try:
        for scene_seed in range(seed, seed + count):
            try:
                _, info = env.reset(seed=scene_seed)
            except ValueError as exc:
                records.append({
                    "seed": scene_seed, "generation_success": False,
                    "success": False, "generation_error": str(exc),
                })
                continue
            records.append({
                "seed": scene_seed, "generation_success": True,
                "generation_attempts": info["scene_generation_attempts"],
                **verify_reference_route(env),
            })
    finally:
        env.close()
    successes = sum(record["success"] for record in records)
    return {
        "first_seed": seed, "requested_scenes": count,
        "route_filter_disabled": True,
        "successful_reference_routes": successes,
        "reference_success_rate": successes / count,
        "all_reference_routes_succeeded": successes == count,
        "interpretation": (
            "Raw geometrically valid scenes are tested without reference-route filtering. "
            "Successful real-dynamics rollouts witness feasible routes. "
            "Failed reference checks do not prove that a scene is infeasible."
        ),
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--output", type=Path, help="Optional JSON result file.")
    args = parser.parse_args()
    try:
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise TypeError("The environment configuration must be a YAML mapping.")
        result = check_scenes(config, args.seed, args.count)
        serialized = json.dumps(result, indent=2, allow_nan=False)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized + "\n", encoding="utf-8")
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    print(serialized)


if __name__ == "__main__":
    main()
