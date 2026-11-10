"""Verify the rate comparison and independently reload its two final policies."""

import argparse
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from me5418_harvesting.agents.dqn_agent import DQNAgent
from me5418_harvesting.agents.dqn_learning import TransitionBatch

RUNNER = Path(__file__).with_name("run_learning_rate_retention.py")
SPEC = importlib.util.spec_from_file_location("learning_rate_diagnostic", RUNNER)
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)
fit = diagnostic.staged.fit


def require(condition, description):
    if not condition:
        raise ValueError(f"Verification failed: {description}")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def parameter_vector(payload):
    """The saved MLP state contains only its ordered weights and biases."""
    return torch.cat([value.reshape(-1) for value in payload["online"].values()])


def verify(run_dir, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite verification evidence: {output}")
    require(read_json(run_dir / "run_status.json")["status"] == "completed", "run completion")
    summary = read_json(run_dir / "summary.json")
    settings = read_json(run_dir / "settings.json")
    protocol = summary["protocol"]
    require(protocol == settings["training"], "resolved protocol")
    require(fit.file_hash(RUNNER) == protocol["script_sha256"], "rate runner hash")
    require(fit.file_hash(diagnostic.STAGED_RUNNER) == protocol["staged_runner_sha256"],
            "staged runner hash")
    require(fit.file_hash(diagnostic.staged.FIT_RUNNER) == protocol["fit_runner_sha256"],
            "fitting helper hash")
    provenance = read_json(run_dir / "provenance.json")
    for name, digest in provenance["source_sha256"].items():
        require(fit.file_hash(diagnostic.ROOT / name) == digest, f"source: {name}")

    manifest = read_json(run_dir / "input_manifest.json")
    reference_dir = Path(protocol["reference_run"])
    require(str(reference_dir) == manifest["reference_run"], "reference location")
    for name, digest in manifest["reference_files_sha256"].items():
        require(fit.file_hash(reference_dir / name) == digest, f"reference input: {name}")
    require(read_json(reference_dir / "run_status.json")["status"] == "completed",
            "reference completion")
    reference = read_json(reference_dir / "summary.json")
    reference_settings = read_json(reference_dir / "settings.json")
    reference_sources = read_json(reference_dir / "provenance.json")["source_sha256"]
    require(reference_sources == provenance["source_sha256"], "unchanged production sources")
    require(settings["environment"] == reference_settings["environment"], "unchanged environment")
    require(settings["dqn"] == reference_settings["dqn"], "recorded historical DQN settings")
    require(summary["pretrained_policy"] == reference["pretraining"], "pretrained reference")
    historical = read_json(run_dir / "historical_td_margin.json")
    require(historical == summary["historical_td_margin"], "historical evidence copy")
    require(historical["summary"] == reference["arms"]["td_margin"], "historical summary")
    require(historical["observations"]
            == read_json(reference_dir / "td_margin/observations.json"), "historical observations")
    require(historical["historical_reference_only"]
            and not manifest["historical_baseline_reexecuted"], "historical baseline accounting")

    dataset = read_json(run_dir / "dataset.json")
    require(dataset == read_json(reference_dir / "dataset.json"), "dataset metadata")
    require(fit.file_hash(run_dir / "demonstration.npz") == dataset["sha256"]
            == protocol["reference_dataset_sha256"], "copied encoded data hash")
    checkpoint = reference_dir / "pretraining/update_20000.pt"
    require(fit.file_hash(checkpoint) == protocol["reference_checkpoint_sha256"]
            == reference["pretraining"]["checkpoint_sha256"][checkpoint.name],
            "frozen supervised checkpoint hash")
    pretrained_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    pretrained_vector = parameter_vector(pretrained_payload)
    with np.load(run_dir / "demonstration.npz", allow_pickle=False) as data:
        batch = TransitionBatch(**{
            name: torch.from_numpy(data[name]) for name in TransitionBatch.__dataclass_fields__
        })
    require(len(batch.actions) == dataset["transitions"], "full fixed dataset count")
    require(protocol["learning_rates"] == dict(diagnostic.RATES), "declared rate arms")
    require(set(summary["arms"]) == set(protocol["learning_rates"]), "completed rate arms")
    require(protocol["phase_updates"] == diagnostic.PHASE_UPDATES,
            "declared adaptation budget")
    require(protocol["pretraining_updates"] == protocol["pretraining_updates_reused"]
            == diagnostic.PRETRAIN_UPDATES and protocol["pretraining_updates_performed"] == 0,
            "inherited pretraining budget")
    torch.set_num_threads(1)
    records = {}
    for label, arm in summary["arms"].items():
        rate = protocol["learning_rates"][label]
        require(arm["directory"] == f"{label}/td_margin", f"{label} recorded directory")
        directory = run_dir / arm["directory"]
        arm_settings = read_json(run_dir / label / "settings.json")
        expected_dqn = {**settings["dqn"], "learning_rate": rate}
        require(arm["learning_rate"] == rate and arm_settings["dqn"] == expected_dqn,
                f"{label} learning rate is the only DQN override")
        require(arm_settings["environment"] == settings["environment"], f"{label} environment")
        boundary = read_json(run_dir / label / "phase_boundary.json")
        require(boundary["learning_rate"] == rate and boundary["optimizer_reset"]
                and boundary["optimizer_state_empty"] and boundary["target_synchronized"]
                and boundary["target_equals_online"] and boundary["phase_updates"] == 0
                and boundary["pretraining_updates"] == diagnostic.PRETRAIN_UPDATES,
                f"{label} explicit phase boundary")
        expected_hash = reference["pretraining"]["final_parameter_sha256"]
        require(boundary["policy_parameter_sha256"] == arm["initial_parameter_sha256"]
                == expected_hash, f"{label} initial weights")
        updates = [json.loads(line) for line in
                   (directory / "updates.jsonl").read_text(encoding="utf-8").splitlines()]
        count = diagnostic.PHASE_UPDATES
        require(arm["optimizer_updates"] == count
                and [row["update"] for row in updates] == list(range(1, count + 1)),
                f"{label} complete update sequence")
        interval = expected_dqn["target_update_interval"]
        synchronized = [row["update"] for row in updates if row["target_synchronized"]]
        require(synchronized == list(range(interval, count + 1, interval))
                and arm["target_synchronizations"] == len(synchronized) == 10,
                f"{label} target synchronization schedule")
        require(any(row["parameter_delta_norm"] > 0 for row in updates),
                f"{label} nonzero optimization steps")
        observations = read_json(directory / "observations.json")
        require([row["update"] for row in observations] == protocol["phase_checkpoints"],
                f"{label} observation schedule")
        require(observations[0]["fit"] == reference["pretraining"]["final_fit"]
                and observations[0]["rollout"] == reference["pretraining"]["final_rollout"],
                f"{label} preserved phase-zero behavior")
        require(observations[-1]["fit"] == arm["final_fit"]
                and observations[-1]["rollout"] == arm["final_rollout"], f"{label} final record")
        failures = [row["update"] for row in observations if not row["rollout"]["success"]]
        require(arm["first_sampled_failure_update"] == (min(failures) if failures else None)
                and arm["success_at_every_sampled_checkpoint"] == (not failures),
                f"{label} retention summary")
        require(arm["evaluation_environment_steps"] == sum(
            row["rollout"]["steps"] for row in observations
        ), f"{label} evaluation action count")
        require(set(arm["checkpoint_sha256"]) == {
            f"update_{step}.pt" for step in protocol["phase_checkpoints"]
        }, f"{label} all scheduled checkpoints")
        for filename, digest in arm["checkpoint_sha256"].items():
            path = directory / filename
            require(fit.file_hash(path) == digest, f"{label}/{filename} hash")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            require(payload["dqn_config"] == expected_dqn, f"{label}/{filename} actual rate")
            require(payload["environment_config"] == settings["environment"],
                    f"{label}/{filename} actual environment")
            if filename == "update_0.pt":
                require(torch.equal(parameter_vector(payload), pretrained_vector),
                        f"{label} phase-zero parameter values")
        final_checkpoint = directory / f"update_{count}.pt"
        agent, env, metadata = DQNAgent.load(final_checkpoint)
        try:
            require(json.loads(json.dumps(asdict(agent.learner.config))) == expected_dqn,
                    f"{label} reloaded actual configuration")
            require(metadata["objective"] == "td_margin" and metadata["updates"] == count
                    and metadata["pretraining_updates"] == diagnostic.PRETRAIN_UPDATES
                    and metadata["total_updates_on_branch"] == diagnostic.PRETRAIN_UPDATES + count,
                    f"{label} inherited and new update accounting")
            require(fit.parameter_hash(agent) == arm["final_parameter_sha256"] != expected_hash
                    and arm["weights_changed"], f"{label} changed final policy weights")
            final_vector = torch.nn.utils.parameters_to_vector(
                agent.learner.online.parameters(),
            ).detach()
            delta = torch.linalg.vector_norm(final_vector - pretrained_vector).item()
            require(np.isfinite(delta) and delta > 0, f"{label} nonzero final parameter change")
            actual_fit = fit.fit_metrics(agent, batch, protocol["margin"])
            actual_rollout = fit.fixed_scene_rollout(agent, env, protocol["scene_seed"], batch)
            require(actual_fit == arm["final_fit"] == metadata["fit"], f"{label} reloaded fitting")
            require(actual_rollout == arm["final_rollout"], f"{label} reloaded full rollout")
            records[label] = {
                "learning_rate": rate, "phase_zero_matches_reference": True,
                "checkpoint_hashes_verified": len(arm["checkpoint_sha256"]),
                "update_records_verified": count, "target_synchronizations": len(synchronized),
                "parameter_change_l2": delta, "final_fit_matches_exactly": True,
                "final_rollout_matches_exactly": True,
                "additional_evaluation_steps": actual_rollout["steps"],
                "final_checkpoint_sha256": fit.file_hash(final_checkpoint),
            }
        finally:
            env.close()
    require(summary["optimizer_updates_total"] == summary["td_optimizer_updates"]
            == sum(row["update_records_verified"] for row in records.values()) == 4000,
            "new optimization accounting")
    require(summary["pretraining_updates_performed"] == summary["demonstration_collection_steps"]
            == summary["online_training_steps"] == summary["held_out_scenes_accessed"] == 0,
            "no new pretraining, collection or held-out scenes")
    require(summary["evaluation_environment_steps"] == sum(
        arm["evaluation_environment_steps"] for arm in summary["arms"].values()
    ), "total evaluation action count")
    result = {
        "sources_match": True, "reference_inputs_match": True, "encoded_dataset_matches": True,
        "historical_baseline_not_reexecuted": True, "arms": records,
        "additional_verification_environment_steps": sum(
            row["additional_evaluation_steps"] for row in records.values()
        ),
        "optimizer_updates_performed": 0, "training_scene_only": True,
        "note": "Additional post-run verification; reset feasibility probes are excluded.",
        "target_metric_limit": (
            "Inference checkpoints omit training targets. Verify action fits and full rollouts, "
            "not historical value metrics against unsaved target-network weights."
        ),
        "verifier_sha256": fit.file_hash(Path(__file__)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    fit.save_json(output, result)
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.run_dir.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
