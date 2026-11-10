"""Reload final diagnostic checkpoints and verify their recorded fitting and rollout."""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from me5418_harvesting.agents.dqn_agent import DQNAgent
from me5418_harvesting.agents.dqn_learning import TransitionBatch

RUNNER = Path(__file__).with_name("run_fixed_scene_fit.py")
spec = importlib.util.spec_from_file_location("fixed_scene_diagnostic", RUNNER)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def verify(run_dir, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite verification: {output}")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    assert diagnostic.file_hash(RUNNER) == summary["protocol"]["script_sha256"]
    for name, digest in provenance["source_sha256"].items():
        assert diagnostic.file_hash(diagnostic.ROOT / name) == digest, name
    assert diagnostic.file_hash(run_dir / "demonstration.npz") == summary["dataset"]["sha256"]
    with np.load(run_dir / "demonstration.npz", allow_pickle=False) as dataset:
        batch = TransitionBatch(**{
            name: torch.from_numpy(dataset[name]) for name in TransitionBatch.__dataclass_fields__
        })
    torch.set_num_threads(1)
    results = {}
    for objective, arm in summary["arms"].items():
        checkpoint_name = f"update_{arm['optimizer_updates']}.pt"
        path = run_dir / objective / checkpoint_name
        assert diagnostic.file_hash(path) == arm["checkpoint_sha256"][checkpoint_name]
        agent, env, metadata = DQNAgent.load(path)
        try:
            assert metadata["objective"] == objective
            assert metadata["updates"] == arm["optimizer_updates"]
            assert diagnostic.parameter_hash(agent) == arm["final_parameter_sha256"]
            fit = diagnostic.fit_metrics(agent, batch, summary["protocol"]["margin"])
            rollout = diagnostic.fixed_scene_rollout(
                agent, env, summary["protocol"]["scene_seed"], batch,
            )
            assert fit == arm["final_fit"] == metadata["fit"], objective
            assert rollout == arm["final_rollout"], objective
            results[objective] = {
                "checkpoint_sha256": diagnostic.file_hash(path),
                "metadata_matches": True, "fit_matches_exactly": True,
                "rollout_matches_exactly": True, "evaluation_steps": rollout["steps"],
            }
        finally:
            env.close()
    record = {
        "sources_match": True, "dataset_hash_matches": True, "arms": results,
        "additional_verification_environment_steps": sum(
            row["evaluation_steps"] for row in results.values()
        ),
        "optimizer_updates": 0, "training_scene_only": True,
        "note": "Additional post-run verification; excludes reset feasibility probes.",
        "verifier_sha256": diagnostic.file_hash(Path(__file__)),
    }
    diagnostic.save_json(output, record)
    print(json.dumps(record, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.run_dir, args.output)


if __name__ == "__main__":
    main()
