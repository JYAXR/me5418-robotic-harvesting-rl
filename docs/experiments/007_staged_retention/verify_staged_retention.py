"""Verify staged experiment records and independently reload the four final policies."""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from me5418_harvesting.agents.dqn_agent import DQNAgent
from me5418_harvesting.agents.dqn_learning import TransitionBatch

RUNNER = Path(__file__).with_name("run_staged_retention.py")
SPEC = importlib.util.spec_from_file_location("staged_retention_diagnostic", RUNNER)
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


def require(condition, description):
    if not condition:
        raise ValueError(f"Verification failed: {description}")


def verify(run_dir, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite verification evidence: {output}")
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    summary = read(run_dir / "summary.json")
    provenance = read(run_dir / "provenance.json")
    require(read(run_dir / "run_status.json")["status"] == "completed", "run completion")
    require(diagnostic.fit.file_hash(RUNNER) == summary["protocol"]["script_sha256"], "runner hash")
    require(diagnostic.fit.file_hash(diagnostic.FIT_RUNNER)
            == summary["protocol"]["fit_runner_sha256"], "fitting helper hash")
    for name, digest in provenance["source_sha256"].items():
        require(diagnostic.fit.file_hash(diagnostic.ROOT / name) == digest, f"source: {name}")
    dataset = read(run_dir / "dataset.json")
    require(diagnostic.fit.file_hash(run_dir / "demonstration.npz") == dataset["sha256"],
            "encoded dataset hash")
    require(all(read(run_dir / "pretraining_reproduction.json").values()), "supervised reproduction")
    with np.load(run_dir / "demonstration.npz", allow_pickle=False) as data:
        batch = TransitionBatch(**{
            name: torch.from_numpy(data[name]) for name in TransitionBatch.__dataclass_fields__
        })
    torch.set_num_threads(1)
    phases = {"pretraining": summary["pretraining"], **summary["arms"]}
    boundaries = read(run_dir / "phase_boundaries.json")
    records = {}
    for name, phase in phases.items():
        directory = run_dir / name
        updates = [json.loads(line) for line in
                   (directory / "updates.jsonl").read_text(encoding="utf-8").splitlines()]
        count = phase["optimizer_updates"]
        require([row["update"] for row in updates] == list(range(1, count + 1)), f"{name} updates")
        syncs = sum(row["target_synchronized"] for row in updates)
        require(syncs == phase["target_synchronizations"], f"{name} synchronization log count")
        expected_syncs = count // read(run_dir / "settings.json")["dqn"]["target_update_interval"]
        if name in ("pretraining", "margin_only"):
            expected_syncs = 0
        require(syncs == expected_syncs, f"{name} target schedule")
        for checkpoint, digest in phase["checkpoint_sha256"].items():
            require(diagnostic.fit.file_hash(directory / checkpoint) == digest,
                    f"{name}/{checkpoint} hash")
        observations = read(directory / "observations.json")
        expected_checkpoints = summary["protocol"][
            "pretraining_checkpoints" if name == "pretraining" else "phase_checkpoints"
        ]
        require([row["update"] for row in observations] == expected_checkpoints,
                f"{name} observation schedule")
        if name != "pretraining":
            boundary = boundaries[name]
            require(boundary["target_equals_online"] and boundary["optimizer_state_empty"]
                    and boundary["optimizer_reset"] and boundary["phase_updates"] == 0,
                    f"{name} phase handoff")
            require(boundary["policy_parameter_sha256"]
                    == summary["pretraining"]["final_parameter_sha256"], f"{name} initial weights")
            require(observations[0]["fit"] == summary["pretraining"]["final_fit"]
                    and observations[0]["rollout"] == summary["pretraining"]["final_rollout"],
                    f"{name} phase-zero behavior")
        failed = [row["update"] for row in observations if not row["rollout"]["success"]]
        require((min(failed) if failed else None) == phase["first_sampled_failure_update"],
                f"{name} first sampled failure")
        checkpoint = directory / f"update_{count}.pt"
        agent, env, metadata = DQNAgent.load(checkpoint)
        try:
            require(metadata["objective"] == name and metadata["updates"] == count,
                    f"{name} checkpoint metadata")
            require(diagnostic.fit.parameter_hash(agent) == phase["final_parameter_sha256"],
                    f"{name} final weights")
            current_fit = diagnostic.fit.fit_metrics(agent, batch, summary["protocol"]["margin"])
            current_rollout = diagnostic.fit.fixed_scene_rollout(
                agent, env, summary["protocol"]["scene_seed"], batch,
            )
            require(current_fit == phase["final_fit"] == metadata["fit"], f"{name} reloaded fit")
            require(current_rollout == phase["final_rollout"], f"{name} reloaded rollout")
            records[name] = {
                "all_checkpoint_hashes_match": True, "checkpoints_verified": len(observations),
                "update_records_verified": count, "target_synchronizations": syncs,
                "final_fit_matches_exactly": True, "final_rollout_matches_exactly": True,
                "reload_environment_steps": current_rollout["steps"],
                "checkpoint_sha256": diagnostic.fit.file_hash(checkpoint),
            }
        finally:
            env.close()
    result = {
        "sources_match": True, "encoded_dataset_hash_matches": True,
        "phases": records, "optimizer_updates_performed": 0,
        "additional_verification_environment_steps": sum(
            record["reload_environment_steps"] for record in records.values()
        ),
        "training_scene_only": True,
        "note": "Additional post-run verification; reset feasibility probes are not counted.",
        "target_metric_limit": (
            "Inference checkpoints omit the training target and optimizer. Reload verifies action "
            "fits and rollouts, not intermediate value metrics against the historical target."
        ),
        "verifier_sha256": diagnostic.fit.file_hash(Path(__file__)),
    }
    diagnostic.fit.save_json(output, result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.run_dir, args.output)


if __name__ == "__main__":
    main()
