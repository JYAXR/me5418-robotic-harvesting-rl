"""Compare retained and reset Adam state after exactly reproduced return calibration.

This diagnostic uses one known scene and fixed demonstration data. It replays
calibration once to recover optimizer moments omitted by inference checkpoints.
"""

import argparse
import importlib.util
import json
import shutil
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import torch

from me5418_harvesting.agents.dqn import DQNConfig
from me5418_harvesting.agents.dqn_agent import DQNAgent
from me5418_harvesting.agents.optimizer_transition import (
    begin_td_with_optimizer,
    optimizer_state_summary,
)
from me5418_harvesting.agents.run_record import begin_run, finish_run
from me5418_harvesting.envs.harvesting_env import HarvestingEnv

ROOT = Path(__file__).resolve().parents[3]
CALIBRATION_RUNNER = ROOT / "docs/experiments/009_return_calibration/run_return_calibration.py"
SPEC = importlib.util.spec_from_file_location("adam_calibration_reference", CALIBRATION_RUNNER)
calibration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(calibration)
staged = calibration.staged
fit = staged.fit
ARMS = ("reset", "preserve")
CHECKPOINTS = (*range(11), 20, 50, 100, 200, 500, 1000, 2000)
TD_UPDATES = 2000
CALIBRATION_UPDATES = 20000
INHERITED_UPDATES = 20000


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_updates(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def load_reference(reference_run, pretraining_run):
    """Validate recorded sources and inputs before reproducing optimizer state."""
    inherited = calibration.reference_module.load_reference(pretraining_run)
    reference = read_json(reference_run / "summary.json")
    settings = read_json(reference_run / "settings.json")
    provenance = read_json(reference_run / "provenance.json")
    protocol = reference["protocol"]
    if (read_json(reference_run / "run_status.json")["status"] != "completed"
            or protocol != settings["training"]
            or protocol["calibration_updates"] != CALIBRATION_UPDATES
            or protocol["td_updates"] != TD_UPDATES
            or protocol["inherited_pretraining_updates"] != INHERITED_UPDATES
            or protocol["learning_rate"] != 3e-4 or protocol["gamma"] != 0.99
            or protocol["margin"] != 0.8 or protocol["demonstration_weight"] != 1.0
            or protocol["scene_seed"] != staged.SCENE_SEED
            or tuple(protocol["calibration_checkpoints"]) != calibration.CALIBRATION_CHECKPOINTS
            or tuple(protocol["td_checkpoints"]) != calibration.TD_CHECKPOINTS
            or settings["dqn"] != inherited[1]["dqn"]
            or settings["environment"] != inherited[1]["environment"]):
        raise ValueError("The reference does not match the completed attempt-009 protocol.")
    for name, expected in provenance["source_sha256"].items():
        if fit.file_hash(ROOT / name) != expected:
            raise ValueError(f"A calibration reference source has changed: {name}")
    helpers = {
        "script_sha256": CALIBRATION_RUNNER,
        "reference_loader_sha256": calibration.REFERENCE_LOADER,
        "staged_runner_sha256": calibration.reference_module.STAGED_RUNNER,
        "fit_runner_sha256": staged.FIT_RUNNER,
    }
    for field, path in helpers.items():
        if fit.file_hash(path) != protocol[field]:
            raise ValueError(f"A calibration reference helper has changed: {field}")
    manifest = read_json(reference_run / "input_manifest.json")
    for name, expected in manifest["reference_files_sha256"].items():
        if fit.file_hash(pretraining_run / name) != expected:
            raise ValueError(f"An inherited demonstration input differs: {name}")
    if (fit.file_hash(reference_run / "demonstration.npz") != inherited[3]["sha256"]
            or fit.file_hash(reference_run / "mc_targets.json") != manifest["mc_targets_sha256"]):
        raise ValueError("The reference demonstration or return targets have changed.")
    for phase in ("calibration", "td"):
        result = reference["arms"]["calibrated"][phase]
        for name, expected in result["checkpoint_sha256"].items():
            if fit.file_hash(reference_run / "calibrated" / phase / name) != expected:
                raise ValueError(f"A reference calibrated checkpoint differs: {phase}/{name}")
    if not reference["arms"]["calibrated"]["calibration"]["final_rollout"]["success"]:
        raise ValueError("The reference calibrated policy must be successful before TD.")
    return reference, settings, provenance, inherited


def training_snapshot(agent, environment):
    """Detach a full boundary state, including Adam moments and bias-correction steps."""
    return deepcopy({
        "format_version": 1, "kind": "fixed_data_optimizer_boundary",
        "environment_config": environment, "dqn_config": asdict(agent.learner.config),
        "online": agent.learner.online.state_dict(),
        "target": agent.learner.target.state_dict(),
        "optimizer": agent.learner.optimizer.state_dict(),
        "update_count": agent.learner.update_count,
        "agent_rng_state": agent.rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
        "optimizer_summary": optimizer_state_summary(agent.learner),
        "parameter_sha256": fit.parameter_hash(agent),
    })


def restore_training_snapshot(snapshot):
    """Build an independent policy and optimizer; no tensors alias the input snapshot."""
    if snapshot["format_version"] != 1 or snapshot["kind"] != "fixed_data_optimizer_boundary":
        raise ValueError("Unsupported optimizer-boundary snapshot.")
    payload = deepcopy(snapshot)
    env = HarvestingEnv(payload["environment_config"])
    try:
        agent = DQNAgent(env, DQNConfig(**payload["dqn_config"]), seed=0)
        agent.learner.online.load_state_dict(payload["online"])
        agent.learner.target.load_state_dict(payload["target"])
        agent.learner.optimizer.load_state_dict(payload["optimizer"])
        agent.learner.update_count = payload["update_count"]
        agent.rng.bit_generator.state = payload["agent_rng_state"]
        torch.set_rng_state(payload["torch_rng_state"])
        if (optimizer_state_summary(agent.learner) != snapshot["optimizer_summary"]
                or fit.parameter_hash(agent) != snapshot["parameter_sha256"]):
            raise RuntimeError("Restoring the training snapshot changed weights or Adam state.")
        for actual, source in zip(agent.learner.optimizer.state.values(),
                                  snapshot["optimizer"]["state"].values()):
            for key in actual:
                if isinstance(actual[key], torch.Tensor) and actual[key].data_ptr() == source[key].data_ptr():
                    raise RuntimeError("Restored Adam state aliases the shared snapshot.")
        return agent, env
    except BaseException:
        env.close()
        raise


def compare_reproduction(result, reference, directory, reference_directory, output):
    """Reject a changed calibration trajectory before either TD treatment starts."""
    current_rows = read_json(directory / "observations.json")
    old_rows = read_json(reference_directory / "observations.json")
    checks = {
        "all_update_records_equal": read_updates(directory / "updates.jsonl")
        == read_updates(reference_directory / "updates.jsonl"),
        "all_observations_equal": current_rows == old_rows,
        "final_parameters_equal": result["final_parameter_sha256"]
        == reference["final_parameter_sha256"],
        "final_fit_equal": result["final_fit"] == reference["final_fit"],
        "final_rollout_equal": result["final_rollout"] == reference["final_rollout"],
        "final_return_fit_equal": result["final_return_fit"] == reference["final_return_fit"],
        "update_count_equal": result["optimizer_updates"] == reference["optimizer_updates"],
    }
    fit.save_json(output, checks)
    if not all(checks.values()):
        raise RuntimeError("Calibration replay did not exactly reproduce attempt 009; TD was not started.")
    return checks


def compare_reset(result, directory, reference_directory, output):
    """Check the reset control against attempt 009 while retaining denser new observations."""
    previous = read_json(reference_directory / "summary.json")
    observations = {row["update"]: row for row in read_json(directory / "observations.json")}
    old_rows = read_json(reference_directory / "observations.json")
    keys = ("fit", "rollout", "value_metrics", "return_fit", "total_updates_on_branch")
    checks = {
        "all_update_records_equal": read_updates(directory / "updates.jsonl")
        == read_updates(reference_directory / "updates.jsonl"),
        "all_shared_observations_equal": all(
            all(observations[row["update"]][key] == row[key] for key in keys) for row in old_rows
        ),
        "final_parameters_equal": result["final_parameter_sha256"]
        == previous["final_parameter_sha256"],
        "target_synchronizations_equal": result["target_synchronizations"]
        == previous["target_synchronizations"],
        "shared_checkpoint_updates": [row["update"] for row in old_rows],
    }
    fit.save_json(output, checks)
    if not all(checks[key] for key in checks if key != "shared_checkpoint_updates"):
        raise RuntimeError("The reset control did not exactly reproduce the attempt-009 TD trajectory.")
    return checks


def start_td(agent, branch, snapshot, snapshot_hash, output):
    if branch not in ARMS:
        raise ValueError(f"Unknown optimizer treatment: {branch}")
    before = optimizer_state_summary(agent.learner)
    if before != snapshot["optimizer_summary"]:
        raise RuntimeError("The TD treatment did not receive the shared calibration optimizer state.")
    boundary = begin_td_with_optimizer(agent.learner, preserve_optimizer=branch == "preserve")
    if fit.parameter_hash(agent) != snapshot["parameter_sha256"]:
        raise RuntimeError("The TD transition changed the calibrated online parameters.")
    if (agent.learner.update_count != 0
            or optimizer_state_summary(agent.learner)["settings_sha256"] != before["settings_sha256"]):
        raise RuntimeError("The treatments must share optimizer settings and a zero TD phase clock.")
    record = {
        **boundary, "branch": branch, "training_snapshot_sha256": snapshot_hash,
        "policy_parameter_sha256": snapshot["parameter_sha256"],
        "inherited_pretraining_updates": INHERITED_UPDATES,
        "reproduced_calibration_updates": snapshot["update_count"],
        "total_prior_updates_on_branch": INHERITED_UPDATES + snapshot["update_count"],
        "snapshot_optimizer_storage_disjoint": True,
    }
    fit.save_json(output / branch / "td_boundary.json", record)
    return record


def execute_td(agent, env, environment, batch, returns, branch, output, expected_start,
               updates=TD_UPDATES, checkpoints=CHECKPOINTS):
    """Evaluate each of the first ten updates and every prespecified later checkpoint."""
    directory = output / branch / "td"
    directory.mkdir(parents=True, exist_ok=False)
    initial_hash = fit.parameter_hash(agent)
    initial_optimizer = optimizer_state_summary(agent.learner)
    observations, synchronizations = [], 0
    prior = INHERITED_UPDATES + expected_start["optimizer_updates"]
    with (directory / "updates.jsonl").open("x", encoding="utf-8") as log:
        for step in range(updates + 1):
            if step:
                metrics = asdict(agent.learner.update(
                    batch, demonstration_weight=1.0, demonstration_margin=0.8,
                ))
                synchronizations += int(metrics["target_synchronized"])
                log.write(json.dumps({"update": step, **metrics}, allow_nan=False) + "\n")
            if step in checkpoints:
                row = staged.observe(agent, env, environment, batch, f"{branch}/td", step, prior, directory)
                row["return_fit"] = calibration.return_fit(agent, batch, returns)
                row["optimizer_state"] = optimizer_state_summary(agent.learner)
                observations.append(row)
                fit.save_json(directory / "observations.json", observations)
                if step == 0 and (row["fit"] != expected_start["final_fit"]
                                  or row["rollout"] != expected_start["final_rollout"]
                                  or row["return_fit"] != expected_start["final_return_fit"]):
                    raise RuntimeError("The TD boundary changed the calibrated starting behavior.")
                log.flush()
            elif step % 100 == 0:
                log.flush()
    final_snapshot = directory / "training_state.pt"
    torch.save(training_snapshot(agent, environment), final_snapshot)
    failures = [row["update"] for row in observations if not row["rollout"]["success"]]
    result = {
        "directory": f"{branch}/td", "optimizer_treatment": branch,
        "initial_parameter_sha256": initial_hash,
        "final_parameter_sha256": fit.parameter_hash(agent),
        "optimizer_updates": agent.learner.update_count,
        "prior_updates_on_branch": prior,
        "total_updates_on_branch": prior + agent.learner.update_count,
        "target_synchronizations": synchronizations,
        "initial_optimizer_state": initial_optimizer,
        "final_optimizer_state": optimizer_state_summary(agent.learner),
        "training_snapshot_file": f"{branch}/td/training_state.pt",
        "training_snapshot_sha256": fit.file_hash(final_snapshot),
        "initial_return_fit": observations[0]["return_fit"],
        "final_return_fit": observations[-1]["return_fit"],
        "final_fit": observations[-1]["fit"], "final_rollout": observations[-1]["rollout"],
        "retention_eligible": bool(expected_start["final_rollout"]["success"]),
        "success_at_every_sampled_checkpoint": not failures,
        "retained_at_all_sampled_td_checkpoints": bool(expected_start["final_rollout"]["success"])
        and not failures,
        "first_sampled_failure_update": min(failures) if failures else None,
        "failed_sampled_updates": failures,
        "first_ten_updates_success_count": sum(
            bool(row["rollout"]["success"]) for row in observations if 1 <= row["update"] <= 10
        ),
        "evaluation_environment_steps": sum(row["rollout"]["steps"] for row in observations),
        "checkpoint_sha256": {
            path.name: fit.file_hash(path) for path in sorted(directory.glob("*.pt"))
        },
    }
    fit.save_json(directory / "summary.json", result)
    return result


def write_partial_accounting(output):
    """Retain actual performed work when a numerical or reproduction check fails."""
    counts = {path.relative_to(output).as_posix(): len(read_updates(path))
              for path in sorted(output.rglob("updates.jsonl"))}
    evaluations = sum(row["rollout"]["steps"] for path in output.rglob("observations.json")
                      for row in read_json(path))
    record = {
        "update_records_per_log": counts, "optimizer_updates_total": sum(counts.values()),
        "td_optimizer_updates": sum(count for name, count in counts.items() if "/td/" in name),
        "evaluation_environment_steps": evaluations,
        "inherited_pretraining_updates": INHERITED_UPDATES,
        "demonstration_collection_steps": 0, "online_training_steps": 0,
        "held_out_scenes_accessed": 0,
        "note": "Counts cover recorded completed updates and evaluations; reset probes are excluded.",
    }
    fit.save_json(output / "accounting.json", record)
    return record


def run(reference_run, pretraining_run, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an Adam-state diagnostic: {output}")
    reference, settings, provenance, inherited = load_reference(reference_run, pretraining_run)
    inherited_summary, _, _, dataset, batch, checkpoint = inherited
    returns, target_record = calibration.compute_targets(batch, settings["dqn"]["gamma"])
    if target_record != read_json(reference_run / "mc_targets.json"):
        raise ValueError("Recomputed return targets differ from attempt 009.")
    protocol = {
        "kind": "retain_versus_reset_adam_after_reproduced_return_calibration",
        "arms": ARMS, "td_checkpoints": CHECKPOINTS, "td_updates": TD_UPDATES,
        "calibration_updates": CALIBRATION_UPDATES,
        "calibration_checkpoints": calibration.CALIBRATION_CHECKPOINTS,
        "inherited_pretraining_updates": INHERITED_UPDATES,
        "scene_seed": staged.SCENE_SEED, "gamma": 0.99, "learning_rate": 3e-4,
        "margin": 0.8, "demonstration_weight": 1.0,
        "calibration_objective": "fixed_demonstration_return_huber_plus_margin",
        "td_objective": "one_step_huber_td_plus_margin",
        "only_treatment_difference": "Adam moments and per-parameter bias-correction step",
        "phase_boundary": "same_weights_settings_synced_target_zero_td_phase_clock",
        "reconstruct_calibration_once": True, "early_stopping": False,
        "reference_run": str(reference_run.resolve()),
        "pretraining_run": str(pretraining_run.resolve()),
        "reference_git_head": provenance["git_head"],
        "script_sha256": fit.file_hash(Path(__file__)),
        "calibration_runner_sha256": fit.file_hash(CALIBRATION_RUNNER),
        "reference_loader_sha256": fit.file_hash(calibration.REFERENCE_LOADER),
        "staged_runner_sha256": fit.file_hash(calibration.reference_module.STAGED_RUNNER),
        "fit_runner_sha256": fit.file_hash(staged.FIT_RUNNER),
        "online_training_steps": 0, "held_out_scenes_accessed": 0,
    }
    output.mkdir(parents=True, exist_ok=False)
    try:
        begin_run(output, settings["environment"], settings["dqn"], protocol)
        torch.set_num_threads(1)
        source_names = (
            "run_status.json", "summary.json", "settings.json", "provenance.json",
            "input_manifest.json", "dataset.json", "demonstration.npz", "mc_targets.json",
            "calibrated/calibration/updates.jsonl", "calibrated/calibration/observations.json",
            "calibrated/calibration/summary.json", "calibrated/calibration/update_20000.pt",
            "calibrated/td/updates.jsonl", "calibrated/td/observations.json",
            "calibrated/td/summary.json", "calibrated/td/update_2000.pt",
        )
        fit.save_json(output / "input_manifest.json", {
            "reference_run": str(reference_run.resolve()),
            "reference_files_sha256": {name: fit.file_hash(reference_run / name) for name in source_names},
            "pretraining_run": str(pretraining_run.resolve()),
            "pretraining_files_sha256": read_json(reference_run / "input_manifest.json")[
                "reference_files_sha256"],
            "reference_source_files_verified": len(provenance["source_sha256"]),
            "inference_checkpoint_optimizer_state_available": False,
            "calibration_optimizer_state_reconstructed": True,
        })
        shutil.copy2(reference_run / "demonstration.npz", output / "demonstration.npz")
        fit.save_json(output / "dataset.json", dataset)
        fit.save_json(output / "mc_targets.json", target_record)
        torch.manual_seed(inherited_summary["protocol"]["initialization_seed"])
        agent, env, metadata = DQNAgent.load(checkpoint)
        try:
            expected = inherited_summary["pretraining"]
            if (metadata["objective"] != "pretraining" or metadata["updates"] != INHERITED_UPDATES
                    or fit.parameter_hash(agent) != expected["final_parameter_sha256"]):
                raise ValueError("The calibration replay did not start from the expected supervised weights.")
            agent.learner.update_count = INHERITED_UPDATES
            replay_root = output / "reproduction"
            (replay_root / "calibrated").mkdir(parents=True)
            calibration.start_phase(agent, "calibrated", "calibration", replay_root,
                                    expected["final_parameter_sha256"], 0)
            reproduced = calibration.execute_phase(
                agent, env, settings["environment"], batch, returns, "calibrated", "calibration",
                CALIBRATION_UPDATES, calibration.CALIBRATION_CHECKPOINTS, replay_root,
                INHERITED_UPDATES, expected,
            )
            reproduced_checks = compare_reproduction(
                reproduced, reference["arms"]["calibrated"]["calibration"],
                replay_root / "calibrated/calibration", reference_run / "calibrated/calibration",
                output / "calibration_reproduction.json",
            )
            snapshot = training_snapshot(agent, settings["environment"])
            snapshot_path = output / "calibrated_training_state.pt"
            torch.save(snapshot, snapshot_path)
            snapshot_hash = fit.file_hash(snapshot_path)
            fit.save_json(output / "training_snapshot.json", {
                "file": snapshot_path.name, "sha256": snapshot_hash,
                "parameter_sha256": snapshot["parameter_sha256"],
                "optimizer_summary": snapshot["optimizer_summary"],
                "calibration_updates": snapshot["update_count"],
                "kind": snapshot["kind"], "snapshot_saved_after_exact_reproduction": True,
            })
        finally:
            env.close()
        arms, boundaries = {}, {}
        for branch in ARMS:
            (output / branch).mkdir()
            # Reload the same serialized state independently for each treatment.
            if fit.file_hash(snapshot_path) != snapshot_hash:
                raise RuntimeError("The shared training snapshot changed before a treatment.")
            payload = torch.load(snapshot_path, map_location="cpu", weights_only=True)
            agent, env = restore_training_snapshot(payload)
            try:
                boundaries[branch] = start_td(agent, branch, payload, snapshot_hash, output)
                arms[branch] = execute_td(
                    agent, env, settings["environment"], batch, returns, branch, output, reproduced,
                )
                if branch == "reset":
                    reset_checks = compare_reset(
                        arms[branch], output / branch / "td", reference_run / "calibrated/td",
                        output / "reset_reproduction.json",
                    )
            finally:
                env.close()
        summary = {
            "protocol": protocol, "calibration_reproduction": reproduced_checks,
            "reset_reproduction": reset_checks, "reproduced_calibration": reproduced,
            "training_snapshot": read_json(output / "training_snapshot.json"),
            "arms": arms, "boundaries": boundaries, **write_partial_accounting(output),
            "calibration_optimizer_updates_performed": CALIBRATION_UPDATES,
            "pretraining_reexecuted": False,
            "limitations": [
                "One known training scene and one inherited initialization; generalization is untested.",
                "The treatment changes all Adam state, not individual moments or the step counter separately.",
                "All first-ten updates are evaluated; later retention is checked only at sampled checkpoints.",
                "Fixed demonstration data are used without online exploration or held-out evaluation.",
                "Remaining-budget inputs may support memorization of the fixed demonstrated route.",
            ],
        }
        fit.save_json(output / "summary.json", summary)
    except BaseException as exc:
        if (output / "run_status.json").exists():
            write_partial_accounting(output)
            finish_run(output, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", exc)
        raise
    else:
        finish_run(output, "completed")
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path,
                        default=ROOT / "results/training/return_calibration_20261111")
    parser.add_argument("--pretraining-run", type=Path,
                        default=ROOT / "results/training/staged_retention_20261111")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.reference_run.resolve(), args.pretraining_run.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
