"""Repeat attempt 005 from scratch with a predeclared tenfold update budget."""

import argparse
import importlib.util
from pathlib import Path

RUNNER = Path(__file__).resolve().parents[1] / "005_fixed_scene_fit/run_fixed_scene_fit.py"
spec = importlib.util.spec_from_file_location("fixed_scene_diagnostic", RUNNER)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite a previous diagnostic: {output}")
    source = {
        "driver_sha256": diagnostic.file_hash(Path(__file__)),
        "runner_sha256": diagnostic.file_hash(RUNNER),
        "updates_per_arm": 20000,
        "evaluation_updates": [0, 500, 1000, 2000, 5000, 10000, 20000],
        "note": "Fresh identical initialization; no optimizer or weights resumed from attempt 005.",
    }
    try:
        diagnostic.run(
            diagnostic.read_mapping(diagnostic.ROOT / "configs/environment_multiroute.yaml"),
            diagnostic.DQNConfig(**diagnostic.read_mapping(
                diagnostic.ROOT / "configs/dqn_multiroute_pilot.yaml",
            )),
            output, updates=source["updates_per_arm"], checkpoints=source["evaluation_updates"],
        )
    finally:
        if (output / "run_status.json").is_file():
            diagnostic.save_json(output / "driver_source.json", source)


if __name__ == "__main__":
    main()
