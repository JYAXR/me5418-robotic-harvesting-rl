"""Verify return-calibration evidence and reload each branch's two final policies."""

import argparse
import hashlib
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from me5418_harvesting.agents.dqn_agent import DQNAgent
from me5418_harvesting.agents.dqn_learning import TransitionBatch

RUNNER = Path(__file__).with_name("run_return_calibration.py")
SPEC = importlib.util.spec_from_file_location("return_calibration_diagnostic", RUNNER)
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, description):
    if not condition:
        raise ValueError(f"Verification failed: {description}")


def state_hash(state):
    digest = hashlib.sha256()
    for value in state.values():
        require(torch.isfinite(value).all().item(), "finite saved parameter values")
        digest.update(value.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def independent_targets(batch, record, gamma):
    """Recompute returns with NumPy float32 and a separate Python float64 loop."""
    rewards = batch.rewards.cpu().numpy()
    require(rewards.dtype == np.float32 and rewards.ndim == 1 and len(rewards) > 0,
            "recorded finite float32 reward vector")
    require(np.isfinite(rewards).all(), "finite rewards")
    require(batch.terminated[-1].item() and not batch.terminated[:-1].any().item()
            and not batch.truncated.any().item(), "complete final task termination")
    require(torch.equal(batch.next_observations[:-1], batch.observations[1:]),
            "continuous demonstration ordering")
    single = np.empty_like(rewards)
    double = [0.0] * len(rewards)
    following_single, following_double = np.float32(0.0), 0.0
    for index in range(len(rewards) - 1, -1, -1):
        following_single = rewards[index] + np.float32(gamma) * following_single
        following_double = float(rewards[index]) + float(gamma) * following_double
        single[index], double[index] = following_single, following_double
    require(record["gamma"] == gamma and record["transitions"] == len(rewards), "MC target layout")
    require(record["rewards"] == rewards.tolist(), "MC source rewards")
    require(record["returns"] == single.tolist(), "independent float32 MC returns")
    require(record["python_float64_returns"] == double, "independent float64 MC returns")
    require(record["complete_ordered_terminal_trajectory_verified"]
            and record["network_bootstrapping_used"] is False, "non-bootstrap MC target meaning")
    maximum_difference = float(np.max(np.abs(single.astype(np.float64) - np.asarray(double))))
    require(record["reference_tolerance"] == 1e-4 and maximum_difference <= 1e-4,
            "MC precision tolerance")
    require(record["maximum_float64_reference_difference"] == maximum_difference,
            "MC precision error record")
    following = np.concatenate((single[1:].astype(np.float64), np.zeros(1)))
    residual = single.astype(np.float64) - rewards.astype(np.float64) - float(gamma) * following
    require(record["maximum_recurrence_residual"] == float(np.max(np.abs(residual))),
            "MC recurrence residual")
    require(single[-1] == rewards[-1], "terminal reward appears once")
    require(record["minimum_return"] == float(single.min())
            and record["maximum_return"] == float(single.max())
            and record["initial_discounted_return"] == float(single[0]), "MC range records")
    require(record["undiscounted_episode_return_from_stored_rewards"]
            == sum(float(value) for value in rewards), "stored undiscounted episode return")
    return torch.from_numpy(single), maximum_difference


def verify(run_dir, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite verification evidence: {output}")
    summary = read_json(run_dir / "summary.json")
    settings = read_json(run_dir / "settings.json")
    protocol = summary["protocol"]
    require(read_json(run_dir / "run_status.json")["status"] == "completed", "run completion")
    require(settings["training"] == protocol, "resolved protocol consistency")
    require(protocol["calibration_updates"] == diagnostic.CALIBRATION_UPDATES
            and protocol["td_updates"] == diagnostic.TD_UPDATES
            and protocol["inherited_pretraining_updates"] == diagnostic.INHERITED_UPDATES,
            "declared update budgets")
    require(tuple(protocol["arms"]) == diagnostic.ARMS, "declared branches")
    script_paths = {
        "script_sha256": RUNNER,
        "reference_loader_sha256": diagnostic.REFERENCE_LOADER,
        "staged_runner_sha256": diagnostic.reference_module.STAGED_RUNNER,
        "fit_runner_sha256": diagnostic.staged.FIT_RUNNER,
    }
    for name, path in script_paths.items():
        require(diagnostic.staged.fit.file_hash(path) == protocol[name], f"helper hash: {name}")
    provenance = read_json(run_dir / "provenance.json")
    for name, digest in provenance["source_sha256"].items():
        require(diagnostic.staged.fit.file_hash(diagnostic.ROOT / name) == digest,
                f"launch source hash: {name}")
    manifest = read_json(run_dir / "input_manifest.json")
    reference_run = Path(manifest["reference_run"])
    require(str(reference_run) == protocol["reference_run"], "reference directory")
    for name, digest in manifest["reference_files_sha256"].items():
        require(diagnostic.staged.fit.file_hash(reference_run / name) == digest,
                f"reference input hash: {name}")
    require(manifest["optimizer_state_reused"] is False and manifest["weights_and_data_reused"]
            and manifest["new_collection_steps"] == 0, "explicit reuse accounting")
    reference_summary = read_json(reference_run / "summary.json")
    reference_settings = read_json(reference_run / "settings.json")
    reference_provenance = read_json(reference_run / "provenance.json")
    require(manifest["reference_source_files_verified"]
            == len(reference_provenance["source_sha256"]), "reference source count")
    require(settings["environment"] == reference_settings["environment"]
            and settings["dqn"] == reference_settings["dqn"], "unchanged environment and learner")
    require(protocol["learning_rate"] == settings["dqn"]["learning_rate"] == 3e-4,
            "fixed learning rate")
    require(summary["pretrained_policy"] == reference_summary["pretraining"],
            "inherited supervised policy record")
    initial_hash = reference_summary["pretraining"]["final_parameter_sha256"]
    dataset = read_json(run_dir / "dataset.json")
    require(dataset == read_json(reference_run / "dataset.json"), "frozen dataset metadata")
    require(diagnostic.staged.fit.file_hash(run_dir / "demonstration.npz") == dataset["sha256"]
            == protocol["reference_dataset_sha256"], "frozen dataset hash")
    require(diagnostic.staged.fit.file_hash(run_dir / "mc_targets.json")
            == manifest["mc_targets_sha256"], "MC target record hash")
    with np.load(run_dir / "demonstration.npz", allow_pickle=False) as values:
        batch = TransitionBatch(**{
            name: torch.from_numpy(values[name]) for name in TransitionBatch.__dataclass_fields__
        })
    returns, precision_difference = independent_targets(
        batch, read_json(run_dir / "mc_targets.json"), protocol["gamma"],
    )
    torch.set_num_threads(1)
    verified, new_updates, td_updates, evaluations = {}, 0, 0, 0
    for branch, arm in summary["arms"].items():
        require(read_json(run_dir / branch / "summary.json") == arm, f"{branch} summary copy")
        eligible = bool(arm["calibration"]["final_rollout"]["success"])
        require(arm["retention_eligible"] == eligible, f"{branch} retention eligibility")
        require(arm["retained_at_all_sampled_td_checkpoints"]
                == (eligible and arm["td"]["success_at_every_sampled_checkpoint"]),
                f"{branch} retention conclusion")
        expected_interpretation = "retention_of_successful_policy" if eligible else (
            "downstream_behavior_after_unsuccessful_calibration"
        )
        require(arm["td_interpretation"] == expected_interpretation, f"{branch} TD interpretation")
        verified[branch] = {}
        for phase in ("calibration", "td"):
            record = arm[phase]
            directory = run_dir / branch / phase
            count = protocol["calibration_updates" if phase == "calibration" else "td_updates"]
            prior = protocol["inherited_pretraining_updates"] + (
                0 if phase == "calibration" else protocol["calibration_updates"]
            )
            require(record["directory"] == f"{branch}/{phase}" and record["phase"] == phase,
                    f"{branch}/{phase} directory identity")
            require(record["optimizer_updates"] == count and record["prior_updates_on_branch"] == prior
                    and record["total_updates_on_branch"] == prior + count,
                    f"{branch}/{phase} update history")
            require(read_json(directory / "summary.json") == record, f"{branch}/{phase} summary copy")
            log = [json.loads(line) for line in
                   (directory / "updates.jsonl").read_text(encoding="utf-8").splitlines()]
            require([row["update"] for row in log] == list(range(1, count + 1)),
                    f"{branch}/{phase} complete update sequence")
            target_period = settings["dqn"]["target_update_interval"]
            flags = [phase == "td" and step % target_period == 0 for step in range(1, count + 1)]
            require([row["target_synchronized"] for row in log] == flags,
                    f"{branch}/{phase} complete target schedule")
            require(record["target_synchronizations"] == sum(flags), f"{branch}/{phase} target count")
            snapshots = read_json(directory / "observations.json")
            checkpoints = protocol["calibration_checkpoints" if phase == "calibration"
                                   else "td_checkpoints"]
            require([row["update"] for row in snapshots] == checkpoints,
                    f"{branch}/{phase} checkpoint schedule")
            require([row["total_updates_on_branch"] for row in snapshots]
                    == [prior + step for step in checkpoints], f"{branch}/{phase} snapshot counters")
            expected_start = (reference_summary["pretraining"] if phase == "calibration"
                              else arm["calibration"])
            require(snapshots[0]["fit"] == expected_start["final_fit"]
                    and snapshots[0]["rollout"] == expected_start["final_rollout"],
                    f"{branch}/{phase} exact phase-zero behavior")
            expected_hash = initial_hash if phase == "calibration" else (
                arm["calibration"]["final_parameter_sha256"]
            )
            boundary = read_json(run_dir / branch / f"{phase}_boundary.json")
            require(boundary["policy_parameter_sha256"] == record["initial_parameter_sha256"]
                    == expected_hash, f"{branch}/{phase} starting parameter hash")
            require(boundary["inherited_pretraining_updates"] == protocol["inherited_pretraining_updates"]
                    and boundary["completed_new_updates_before_phase"]
                    == (0 if phase == "calibration" else protocol["calibration_updates"])
                    and boundary["updates_on_branch_before_phase"] == prior,
                    f"{branch}/{phase} boundary history")
            helper = boundary["helper_record"]
            require(helper["pretraining_updates"] == 20000 and helper["phase_updates"] == 0
                    and helper["optimizer_reset"] and helper["target_synchronized"]
                    and boundary["target_equals_online"] and boundary["optimizer_state_empty"]
                    and boundary["learning_rate"] == protocol["learning_rate"],
                    f"{branch}/{phase} fresh optimizer and target boundary")
            failures = [row["update"] for row in snapshots if not row["rollout"]["success"]]
            require(record["first_sampled_failure_update"] == (min(failures) if failures else None)
                    and record["success_at_every_sampled_checkpoint"] == (not failures),
                    f"{branch}/{phase} retained failure classification")
            require(record["final_fit"] == snapshots[-1]["fit"]
                    and record["final_rollout"] == snapshots[-1]["rollout"]
                    and record["initial_return_fit"] == snapshots[0]["return_fit"]
                    and record["final_return_fit"] == snapshots[-1]["return_fit"],
                    f"{branch}/{phase} final and initial observations")
            require(record["evaluation_environment_steps"]
                    == sum(row["rollout"]["steps"] for row in snapshots),
                    f"{branch}/{phase} evaluation accounting")
            require(set(record["checkpoint_sha256"]) == {f"update_{step}.pt" for step in checkpoints},
                    f"{branch}/{phase} complete checkpoint inventory")
            for step in checkpoints:
                path = directory / f"update_{step}.pt"
                require(diagnostic.staged.fit.file_hash(path) == record["checkpoint_sha256"][path.name],
                        f"{branch}/{phase}/{path.name} file hash")
                payload = torch.load(path, map_location="cpu", weights_only=True)
                metadata = payload["metadata"]
                require(json.loads(json.dumps(payload["dqn_config"])) == settings["dqn"]
                        and payload["environment_config"] == settings["environment"],
                        f"{branch}/{phase}/{path.name} actual configuration")
                require(metadata["objective"] == f"{branch}/{phase}" and metadata["updates"] == step
                        and metadata["pretraining_updates"] == prior
                        and metadata["total_updates_on_branch"] == prior + step,
                        f"{branch}/{phase}/{path.name} checkpoint counters")
                if step == 0:
                    require(state_hash(payload["online"]) == expected_hash,
                            f"{branch}/{phase} phase-zero saved parameters")
            checkpoint = directory / f"update_{count}.pt"
            agent, env, metadata = DQNAgent.load(checkpoint)
            try:
                require(json.loads(json.dumps(asdict(agent.learner.config))) == settings["dqn"],
                        f"{branch}/{phase} loaded learner configuration")
                require(diagnostic.staged.fit.parameter_hash(agent) == record["final_parameter_sha256"],
                        f"{branch}/{phase} loaded final parameters")
                action_fit = diagnostic.staged.fit.fit_metrics(agent, batch, protocol["margin"])
                rollout = diagnostic.staged.fit.fixed_scene_rollout(
                    agent, env, protocol["scene_seed"], batch,
                )
                calibration_fit = diagnostic.return_fit(agent, batch, returns)
                require(action_fit == record["final_fit"] == metadata["fit"],
                        f"{branch}/{phase} reloaded action fitting")
                require(rollout == record["final_rollout"], f"{branch}/{phase} reloaded full rollout")
                require(calibration_fit == record["final_return_fit"],
                        f"{branch}/{phase} reloaded per-state return fitting")
                verified[branch][phase] = {
                    "checkpoints_verified": len(checkpoints), "update_records_verified": count,
                    "target_synchronizations": sum(flags), "checkpoint_metadata_verified": True,
                    "phase_zero_preserved": True, "final_parameter_hash_matches": True,
                    "final_fit_matches_exactly": True, "final_rollout_matches_exactly": True,
                    "final_return_fit_matches_exactly": True,
                    "reload_environment_steps": rollout["steps"],
                    "checkpoint_sha256": diagnostic.staged.fit.file_hash(checkpoint),
                }
            finally:
                env.close()
            new_updates += count
            td_updates += count if phase == "td" else 0
            evaluations += record["evaluation_environment_steps"]
    require(new_updates == summary["optimizer_updates_total"] == 44000
            and td_updates == summary["td_optimizer_updates"] == 4000,
            "new update accounting")
    require(evaluations == summary["evaluation_environment_steps"], "scheduled evaluation accounting")
    require(summary["inherited_pretraining_updates"] == 20000
            and summary["pretraining_reexecuted"] is False
            and summary["demonstration_collection_steps"] == 0
            and summary["online_training_steps"] == 0 and summary["held_out_scenes_accessed"] == 0,
            "inherited work and data-use accounting")
    result = {
        "source_files_verified": len(provenance["source_sha256"]),
        "reference_input_files_verified": len(manifest["reference_files_sha256"]),
        "runner_and_helper_hashes_verified": True, "dataset_hash_verified": True,
        "mc_targets_recomputed_independently": True,
        "mc_float32_targets_match_exactly": True,
        "mc_float64_maximum_difference": precision_difference,
        "arms": verified, "optimizer_updates_performed": 0,
        "additional_verification_environment_steps": sum(
            phase["reload_environment_steps"] for arm in verified.values() for phase in arm.values()
        ),
        "training_scene_only": True,
        "note": "Additional post-run verification; reset feasibility probes are not counted.",
        "target_metric_limit": (
            "Inference checkpoints omit historical target/optimizer state. This verifies fixed "
            "MC errors, action choices and rollouts, not historical target-based value_metrics."
        ),
        "verifier_sha256": diagnostic.staged.fit.file_hash(Path(__file__)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
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
