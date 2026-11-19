"""Evaluate the bundled trained DQN without running training."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
RELEASED = ROOT / "models/compact_guided_generalization.pt"
RELEASED_SHA256 = "75d7d7ca728b1c85c8ba6fa226076f85f4f72bdce3cfe4c6c0177946e5fe51e6"

sys.path.insert(0, str(ROOT / "src"))
from evaluate_coordinate_models import KNOWN_DATASET, evaluate_models, evaluation_groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=RELEASED)
    parser.add_argument("--seed", type=int, default=5420000)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--known", action="store_true", help="Also run the 33 familiar layouts.")
    parser.add_argument("--output", type=Path, help="Save the full result to a new JSON file.")
    args = parser.parse_args()
    model = args.checkpoint.resolve()
    if args.output and args.output.exists():
        parser.error("Use a new output path.")
    try:
        with model.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        if model == RELEASED and digest != RELEASED_SHA256:
            raise ValueError("The bundled checkpoint does not match its published hash.")
        groups = evaluation_groups(args.seed, args.episodes, KNOWN_DATASET if args.known else None)
        result = evaluate_models(
            {"compact": {"path": str(model), "sha256": digest, "policy": "greedy"}},
            groups, args.workers,
        )
        if model == RELEASED and args.seed == 5420000:
            first = result["records"][0]
            if first["seed"] != 5420000 or not first["success"] or first["damage"] or first["steps"] != 46:
                raise ValueError("The published validation trajectory did not reproduce.")
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as out:
                json.dump(result, out, indent=2, allow_nan=False)
                out.write("\n")
        for group, metrics in result["summary"]["compact"].items():
            print(f"{group}: {metrics['success']}/{metrics['episodes']} success, "
                  f"{metrics['damage']} damage, {metrics['timeout']} timeout")
        if args.episodes == 1 and not args.known:
            print(f"Seed {args.seed}: {result['records'][0]['steps']} actions")
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
