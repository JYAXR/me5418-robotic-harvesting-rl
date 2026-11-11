"""Audit Adam-state evidence and independently replay the first ten TD updates."""

import argparse
import hashlib
import importlib.util
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from me5418_harvesting.agents.dqn import DQNConfig
from me5418_harvesting.agents.dqn_agent import DQNAgent
from me5418_harvesting.agents.dqn_learning import TransitionBatch
from me5418_harvesting.agents.optimizer_transition import optimizer_state_summary
from me5418_harvesting.envs.harvesting_env import HarvestingEnv

ROOT = Path(__file__).resolve().parents[3]
RUNNER = Path(__file__).with_name("run_adam_state.py")
SPEC = importlib.util.spec_from_file_location("adam_state_diagnostic", RUNNER)
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)
MC_VERIFIER = ROOT / "docs/experiments/009_return_calibration/verify_return_calibration.py"
MC_SPEC = importlib.util.spec_from_file_location("adam_state_mc_check", MC_VERIFIER)
mc_check = importlib.util.module_from_spec(MC_SPEC)
MC_SPEC.loader.exec_module(mc_check)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_log(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, description):
    if not condition:
        raise ValueError(f"Verification failed: {description}")


def require_equal(actual, expected, description):
    """Compare saved tensor trees exactly, including optimizer moment tensors."""
    if isinstance(expected, torch.Tensor):
        require(isinstance(actual, torch.Tensor) and actual.dtype == expected.dtype
                and actual.shape == expected.shape and torch.equal(actual, expected), description)
    elif isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), description)
        for name in expected:
            require_equal(actual[name], expected[name], f"{description}/{name}")
    elif isinstance(expected, (tuple, list)):
        require(isinstance(actual, type(expected)) and len(actual) == len(expected), description)
        for index, (first, second) in enumerate(zip(actual, expected)):
            require_equal(first, second, f"{description}/{index}")
    else:
        require(actual == expected, description)


def state_hash(state):
    digest = hashlib.sha256()
    for value in state.values():
        require(torch.isfinite(value).all().item(), "finite saved parameter values")
        digest.update(value.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def independent_restore(payload):
    """Restore without calling the experiment's snapshot loader or boundary helper."""
    env = HarvestingEnv(payload["environment_config"])
    agent = DQNAgent(env, DQNConfig(**payload["dqn_config"]), seed=0)
    agent.learner.online.load_state_dict(deepcopy(payload["online"]))
    agent.learner.target.load_state_dict(deepcopy(payload["target"]))
    agent.learner.optimizer.load_state_dict(deepcopy(payload["optimizer"]))
    agent.learner.update_count = payload["update_count"]
    return agent, env


def independent_boundary(agent, preserve):
    """Apply the declared intervention independently of its production helper."""
    learner = agent.learner
    if not preserve:
        groups = [{**{name: deepcopy(value) for name, value in group.items() if name != "params"},
                   "params": list(group["params"])} for group in learner.optimizer.param_groups]
        learner.optimizer = torch.optim.Adam(groups, **deepcopy(learner.optimizer.defaults))
    learner.online.zero_grad(set_to_none=True)
    learner.target.load_state_dict(learner.online.state_dict())
    learner.target.requires_grad_(False)
    learner.target.eval()
    learner.target.zero_grad(set_to_none=True)
    learner.update_count = 0


def replay_early_updates(payload, branch, run_dir, batch, protocol, observations, expected_boundary):
    """Recompute ten TD updates; these are separately counted verification work."""
    agent, env = independent_restore(payload)
    try:
        independent_boundary(agent, preserve=branch == "preserve")
        require(optimizer_state_summary(agent.learner) == expected_boundary,
                f"{branch} independently restored boundary optimizer")
        log = read_log(run_dir / branch / "td/updates.jsonl")
        saved = {row["update"]: row for row in observations}
        for step in range(1, 11):
            actual = {"update": step, **asdict(agent.learner.update(
                batch, demonstration_weight=protocol["demonstration_weight"],
                demonstration_margin=protocol["margin"],
            ))}
            require(actual == log[step - 1], f"{branch} independent TD update {step} metrics")
            checkpoint = torch.load(run_dir / branch / f"td/update_{step}.pt",
                                    map_location="cpu", weights_only=True)
            require_equal(agent.learner.online.state_dict(), checkpoint["online"],
                          f"{branch} independent TD update {step} weights")
            require(optimizer_state_summary(agent.learner) == saved[step]["optimizer_state"],
                    f"{branch} independent TD update {step} Adam state")
        return {"optimizer_updates_performed": 10, "exact_update_metrics": True,
                "exact_online_tensors": True, "exact_optimizer_summaries": True}
    finally:
        env.close()


def verify_phase(run_dir, directory, record, settings, checkpoints, expected_start_hash,
                 objective, prior, is_td):
    """Check every saved policy, update counter, target event and reported outcome."""
    path = run_dir / directory
    require(read_json(path / "summary.json") == record, f"{directory} summary copy")
    logs = read_log(path / "updates.jsonl")
    count = record["optimizer_updates"]
    require([row["update"] for row in logs] == list(range(1, count + 1)),
            f"{directory} complete update sequence")
    target_period = settings["dqn"]["target_update_interval"]
    flags = [is_td and step % target_period == 0 for step in range(1, count + 1)]
    require([row["target_synchronized"] for row in logs] == flags,
            f"{directory} complete target schedule")
    require(record["target_synchronizations"] == sum(flags), f"{directory} target event count")
    rows = read_json(path / "observations.json")
    require([row["update"] for row in rows] == checkpoints, f"{directory} observation schedule")
    require(record["prior_updates_on_branch"] == prior
            and record["total_updates_on_branch"] == prior + count,
            f"{directory} update accounting")
    require([row["total_updates_on_branch"] for row in rows]
            == [prior + step for step in checkpoints], f"{directory} observation update history")
    require(record["initial_parameter_sha256"] == expected_start_hash,
            f"{directory} initial parameter hash")
    failures = [row["update"] for row in rows if not row["rollout"]["success"]]
    require(record["success_at_every_sampled_checkpoint"] == (not failures)
            and record["first_sampled_failure_update"] == (min(failures) if failures else None),
            f"{directory} complete failure classification")
    require(record["final_fit"] == rows[-1]["fit"]
            and record["final_rollout"] == rows[-1]["rollout"]
            and record["initial_return_fit"] == rows[0]["return_fit"]
            and record["final_return_fit"] == rows[-1]["return_fit"],
            f"{directory} final and initial metrics")
    require(record["evaluation_environment_steps"] == sum(row["rollout"]["steps"] for row in rows),
            f"{directory} evaluation accounting")
    expected_files = {f"update_{step}.pt" for step in checkpoints}
    if is_td:
        expected_files.add("training_state.pt")
    require(set(record["checkpoint_sha256"]) == expected_files,
            f"{directory} complete checkpoint inventory")
    for step in checkpoints:
        checkpoint = path / f"update_{step}.pt"
        require(file_hash(checkpoint) == record["checkpoint_sha256"][checkpoint.name],
                f"{directory}/{checkpoint.name} checkpoint file hash")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        metadata = payload["metadata"]
        require(json.loads(json.dumps(payload["dqn_config"])) == settings["dqn"]
                and payload["environment_config"] == settings["environment"],
                f"{directory}/{checkpoint.name} checkpoint configuration")
        require(metadata["objective"] == objective and metadata["updates"] == step
                and metadata["pretraining_updates"] == prior
                and metadata["total_updates_on_branch"] == prior + step
                and metadata["scene_seed"] == settings["training"]["scene_seed"]
                and metadata["training_scene_only"], f"{directory}/{checkpoint.name} metadata")
        require(metadata["fit"] == rows[checkpoints.index(step)]["fit"],
                f"{directory}/{checkpoint.name} saved fit metadata")
        if step == 0:
            require(state_hash(payload["online"]) == expected_start_hash,
                    f"{directory} saved phase-zero weights")
        if step == count:
            require(state_hash(payload["online"]) == record["final_parameter_sha256"],
                    f"{directory} saved final weights")
    return rows, logs


def reload_policies(run_dir, branch, batch, returns, protocol, observations):
    """Evaluate early failures and final behavior without updating loaded policies."""
    rows = {row["update"]: row for row in observations}
    verified = {}
    for step in (1, 10, 2000):
        checkpoint = run_dir / branch / f"td/update_{step}.pt"
        agent, env, metadata = DQNAgent.load(checkpoint)
        try:
            actual_fit = diagnostic.fit.fit_metrics(agent, batch, protocol["margin"])
            actual_return_fit = diagnostic.calibration.return_fit(agent, batch, returns)
            actual_rollout = diagnostic.fit.fixed_scene_rollout(
                agent, env, protocol["scene_seed"], batch,
            )
            require(actual_fit == rows[step]["fit"] == metadata["fit"],
                    f"{branch} update {step} reloaded action fit")
            require(actual_return_fit == rows[step]["return_fit"],
                    f"{branch} update {step} reloaded return fit")
            require(actual_rollout == rows[step]["rollout"],
                    f"{branch} update {step} reloaded full trajectory")
            verified[str(step)] = {
                "action_fit_matches_exactly": True, "return_fit_matches_exactly": True,
                "full_rollout_matches_exactly": True,
                "reload_environment_steps": actual_rollout["steps"],
                "success": actual_rollout["success"], "damage": actual_rollout["damage"],
                "checkpoint_sha256": file_hash(checkpoint),
            }
        finally:
            env.close()
    return verified


def verify(run_dir, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite verification evidence: {output}")
    torch.set_num_threads(1)
    summary = read_json(run_dir / "summary.json")
    settings = read_json(run_dir / "settings.json")
    protocol = summary["protocol"]
    require(read_json(run_dir / "run_status.json")["status"] == "completed", "completed run")
    require(protocol == settings["training"], "resolved protocol consistency")
    require(tuple(protocol["arms"]) == diagnostic.ARMS == ("reset", "preserve")
            and protocol["td_checkpoints"] == list(diagnostic.CHECKPOINTS)
            and protocol["td_updates"] == diagnostic.TD_UPDATES == 2000
            and protocol["calibration_updates"] == diagnostic.CALIBRATION_UPDATES == 20000
            and protocol["inherited_pretraining_updates"] == 20000,
            "declared treatment identities and budgets")
    for field, path in {
        "script_sha256": RUNNER, "calibration_runner_sha256": diagnostic.CALIBRATION_RUNNER,
        "reference_loader_sha256": diagnostic.calibration.REFERENCE_LOADER,
        "staged_runner_sha256": diagnostic.calibration.reference_module.STAGED_RUNNER,
        "fit_runner_sha256": diagnostic.staged.FIT_RUNNER,
    }.items():
        require(file_hash(path) == protocol[field], f"runner/helper hash: {field}")
    provenance = read_json(run_dir / "provenance.json")
    for name, expected in provenance["source_sha256"].items():
        require(file_hash(ROOT / name) == expected, f"launch source: {name}")
    manifest = read_json(run_dir / "input_manifest.json")
    reference = Path(manifest["reference_run"])
    pretraining = Path(manifest["pretraining_run"])
    require(str(reference) == protocol["reference_run"]
            and str(pretraining) == protocol["pretraining_run"], "reference directory identities")
    for prefix, directory in (("reference", reference), ("pretraining", pretraining)):
        for name, expected in manifest[f"{prefix}_files_sha256"].items():
            require(file_hash(directory / name) == expected, f"{prefix} input: {name}")
    require(manifest["inference_checkpoint_optimizer_state_available"] is False
            and manifest["calibration_optimizer_state_reconstructed"], "honest state reconstruction")
    require(settings["dqn"] == read_json(reference / "settings.json")["dqn"]
            and settings["environment"] == read_json(reference / "settings.json")["environment"],
            "unchanged learner and environment configuration")
    dataset = read_json(run_dir / "dataset.json")
    require(dataset == read_json(reference / "dataset.json")
            and file_hash(run_dir / "demonstration.npz") == dataset["sha256"], "frozen dataset")
    with np.load(run_dir / "demonstration.npz", allow_pickle=False) as values:
        batch = TransitionBatch(**{name: torch.from_numpy(values[name])
                                   for name in TransitionBatch.__dataclass_fields__})
    require(read_json(run_dir / "mc_targets.json") == read_json(reference / "mc_targets.json"),
            "same demonstrated returns")
    returns, precision_difference = mc_check.independent_targets(
        batch, read_json(run_dir / "mc_targets.json"), protocol["gamma"],
    )
    reference_summary = read_json(reference / "summary.json")
    original_calibration = reference_summary["arms"]["calibrated"]["calibration"]
    reproduced = summary["reproduced_calibration"]
    calibration_directory = "reproduction/calibrated/calibration"
    original_start = read_json(pretraining / "summary.json")["pretraining"]["final_parameter_sha256"]
    calibration_rows, calibration_log = verify_phase(
        run_dir, calibration_directory, reproduced, settings, protocol["calibration_checkpoints"],
        original_start, "calibrated/calibration", 20000, False,
    )
    checks = read_json(run_dir / "calibration_reproduction.json")
    require(checks == summary["calibration_reproduction"] and all(value is True for value in checks.values()),
            "all declared calibration reproduction checks")
    require(calibration_rows == read_json(reference / "calibrated/calibration/observations.json")
            and calibration_log == read_log(reference / "calibrated/calibration/updates.jsonl")
            and reproduced["final_parameter_sha256"] == original_calibration["final_parameter_sha256"],
            "independent complete calibration reproduction")
    info = read_json(run_dir / "training_snapshot.json")
    require(info == summary["training_snapshot"] and info["snapshot_saved_after_exact_reproduction"],
            "shared training snapshot description")
    require(file_hash(run_dir / info["file"]) == info["sha256"], "full snapshot file hash")
    payload = torch.load(run_dir / info["file"], map_location="cpu", weights_only=True)
    require(payload["format_version"] == 1 and payload["kind"] == "fixed_data_optimizer_boundary"
            and payload["update_count"] == info["calibration_updates"] == 20000,
            "full snapshot identity and calibration clock")
    require(json.loads(json.dumps(payload["dqn_config"])) == settings["dqn"]
            and payload["environment_config"] == settings["environment"], "snapshot configuration")
    require(state_hash(payload["online"]) == info["parameter_sha256"]
            == payload["parameter_sha256"] == reproduced["final_parameter_sha256"],
            "shared calibrated online weights")
    require(state_hash(payload["target"]) == original_start, "calibration's unchanged target weights")
    agent, env = independent_restore(payload)
    try:
        state = optimizer_state_summary(agent.learner)
        require(state == payload["optimizer_summary"] == info["optimizer_summary"],
                "full saved Adam moment and setting hashes")
        require(state["initialized_parameter_count"] == state["parameter_count"]
                and state["step_min"] == state["step_max"] == 20000
                and state["first_moment_l2"] > 0 and state["second_moment_l2"] > 0,
                "complete nonzero reconstructed Adam state with calibration step count")
        require_equal(agent.learner.optimizer.state_dict(), payload["optimizer"],
                      "exactly reconstructed saved optimizer tensors")
    finally:
        env.close()
    verified, evaluations = {}, reproduced["evaluation_environment_steps"]
    for branch in protocol["arms"]:
        record = summary["arms"][branch]
        rows, logs = verify_phase(run_dir, f"{branch}/td", record, settings,
                                  protocol["td_checkpoints"], info["parameter_sha256"],
                                  f"{branch}/td", 40000, True)
        boundary = read_json(run_dir / branch / "td_boundary.json")
        require(boundary == summary["boundaries"][branch], f"{branch} boundary copy")
        require(boundary["training_snapshot_sha256"] == info["sha256"]
                and boundary["policy_parameter_sha256"] == info["parameter_sha256"]
                and boundary["snapshot_optimizer_storage_disjoint"], f"{branch} shared snapshot identity")
        require(boundary["previous_phase_updates"] == boundary["reproduced_calibration_updates"] == 20000
                and boundary["phase_updates"] == 0 and boundary["target_synchronized"]
                and boundary["total_prior_updates_on_branch"] == 40000
                and boundary["optimizer_reset"] == (branch == "reset"), f"{branch} TD phase boundary")
        before, after = boundary["optimizer_before"], boundary["optimizer_after"]
        require(before == state and after == record["initial_optimizer_state"]
                == rows[0]["optimizer_state"], f"{branch} recorded optimizer boundary")
        require(after["settings_sha256"] == before["settings_sha256"], f"{branch} identical Adam options")
        if branch == "preserve":
            require(after == before, "preserved complete Adam state")
        else:
            require(after["initialized_parameter_count"] == 0 and after["steps"] == {}
                    and after["first_moment_l2"] == after["second_moment_l2"] == 0,
                    "reset Adam has no accumulated moments or bias-correction steps")
        for row in rows:
            step = row["update"]
            if step or branch == "preserve":
                expected_step = step + (20000 if branch == "preserve" else 0)
                adam = row["optimizer_state"]
                require(adam["step_min"] == adam["step_max"] == expected_step
                        and adam["initialized_parameter_count"] == adam["parameter_count"],
                        f"{branch} TD update {step} distinct Adam bias-correction clock")
            require(row["optimizer_state"]["settings_sha256"] == state["settings_sha256"],
                    f"{branch} TD update {step} unchanged Adam settings")
        require(record["final_optimizer_state"] == rows[-1]["optimizer_state"],
                f"{branch} final optimizer record")
        final_snapshot = run_dir / record["training_snapshot_file"]
        require(file_hash(final_snapshot) == record["training_snapshot_sha256"]
                == record["checkpoint_sha256"]["training_state.pt"],
                f"{branch} final full training snapshot hash")
        final_payload = torch.load(final_snapshot, map_location="cpu", weights_only=True)
        require(final_payload["format_version"] == 1
                and final_payload["kind"] == "fixed_data_optimizer_boundary"
                and final_payload["update_count"] == 2000
                and json.loads(json.dumps(final_payload["dqn_config"])) == settings["dqn"]
                and final_payload["environment_config"] == settings["environment"],
                f"{branch} final training snapshot configuration")
        require(state_hash(final_payload["online"]) == final_payload["parameter_sha256"]
                == record["final_parameter_sha256"], f"{branch} final full snapshot weights")
        require_equal(final_payload["target"], final_payload["online"],
                      f"{branch} final target synchronized at TD update 2000")
        final_agent, final_env = independent_restore(final_payload)
        try:
            require(optimizer_state_summary(final_agent.learner)
                    == final_payload["optimizer_summary"] == record["final_optimizer_state"],
                    f"{branch} full final Adam state restored exactly")
            require_equal(final_agent.learner.optimizer.state_dict(), final_payload["optimizer"],
                          f"{branch} exact full final Adam tensors")
        finally:
            final_env.close()
        require(rows[0]["fit"] == reproduced["final_fit"]
                and rows[0]["rollout"] == reproduced["final_rollout"]
                and rows[0]["return_fit"] == reproduced["final_return_fit"],
                f"{branch} unchanged starting behavior")
        failures = [row["update"] for row in rows if not row["rollout"]["success"]]
        require(record["retention_eligible"] and record["failed_sampled_updates"] == failures
                and record["retained_at_all_sampled_td_checkpoints"] == (not failures)
                and record["first_ten_updates_success_count"] == sum(
                    row["rollout"]["success"] for row in rows if 1 <= row["update"] <= 10),
                f"{branch} eligibility and early retention classification")
        if branch == "reset":
            reset_checks = read_json(run_dir / "reset_reproduction.json")
            require(reset_checks == summary["reset_reproduction"]
                    and all(value is True for key, value in reset_checks.items()
                            if key != "shared_checkpoint_updates"), "declared reset reproduction")
            old_rows = read_json(reference / "calibrated/td/observations.json")
            indexed = {row["update"]: row for row in rows}
            fields = ("fit", "rollout", "value_metrics", "return_fit", "total_updates_on_branch")
            require(logs == read_log(reference / "calibrated/td/updates.jsonl")
                    and all(all(indexed[row["update"]][field] == row[field] for field in fields)
                            for row in old_rows), "independent full reset-control reproduction")
        early = replay_early_updates(payload, branch, run_dir, batch, protocol, rows, after)
        reloaded = reload_policies(run_dir, branch, batch, returns, protocol, rows)
        verified[branch] = {
            "checkpoints_verified": len(rows), "update_records_verified": len(logs),
            "target_synchronizations": record["target_synchronizations"],
            "full_final_training_snapshot_verified": True,
            "independent_early_replay": early, "reloaded_policies": reloaded,
        }
        evaluations += record["evaluation_environment_steps"]
    accounting = read_json(run_dir / "accounting.json")
    require(all(summary[name] == value for name, value in accounting.items()), "accounting summary copy")
    require(summary["optimizer_updates_total"] == sum(accounting["update_records_per_log"].values()) == 24000
            and summary["td_optimizer_updates"] == 4000
            and summary["calibration_optimizer_updates_performed"] == 20000,
            "actual new experiment optimizer work")
    require(evaluations == summary["evaluation_environment_steps"], "actual scheduled evaluation actions")
    require(summary["pretraining_reexecuted"] is False and summary["inherited_pretraining_updates"] == 20000
            and summary["demonstration_collection_steps"] == summary["online_training_steps"]
            == summary["held_out_scenes_accessed"] == 0, "inherited work and zero new data access")
    result = {
        "source_files_verified": len(provenance["source_sha256"]),
        "reference_input_files_verified": len(manifest["reference_files_sha256"]),
        "pretraining_input_files_verified": len(manifest["pretraining_files_sha256"]),
        "runner_and_helper_hashes_verified": True, "dataset_hash_verified": True,
        "independent_mc_returns_verified": True, "mc_float64_maximum_difference": precision_difference,
        "calibration_reproduction_verified": True, "reset_reproduction_verified": True,
        "full_training_snapshot_verified": True, "calibration_checkpoints_verified": len(calibration_rows),
        "calibration_update_records_verified": len(calibration_log), "arms": verified,
        "additional_verification_optimizer_updates": 20,
        "additional_verification_environment_steps": sum(
            row["reload_environment_steps"] for arm in verified.values()
            for row in arm["reloaded_policies"].values()
        ),
        "training_scene_only": True,
        "note": "Independent post-run work; reset feasibility probes are excluded from action counts.",
        "target_metric_limit": "Inference-only reloads verify action fit, fixed MC errors and full trajectories; historical target losses are checked in the independent early replay.",
        "verifier_sha256": file_hash(Path(__file__)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.run_dir.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
