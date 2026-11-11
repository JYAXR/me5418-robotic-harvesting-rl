"""Continue the exact attempt-013 treatment to 10,000 absolute TD updates."""

import argparse
import hashlib
import importlib.util
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch

from me5418_harvesting.agents.run_record import begin_run, finish_run

ROOT = Path(__file__).resolve().parents[3]
PREVIOUS_RUNNER = ROOT / "docs/experiments/013_td_learning_rate/run_learning_rate_control.py"
SPEC = importlib.util.spec_from_file_location("extension014_previous", PREVIOUS_RUNNER)
previous = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(previous)
control = previous.control
adam, fit, multi = control.adam, control.fit, control.multi
read_json, read_updates, exact_equal = control.read_json, control.read_updates, control.exact_equal

START_TD_UPDATE = 2000
FINAL_TD_UPDATE = 10000
EXTENSION_UPDATES = FINAL_TD_UPDATE - START_TD_UPDATE
PRIOR_PHASE_UPDATES = 40000
INHERITED_UPDATES = PRIOR_PHASE_UPDATES + START_TD_UPDATE
LEARNING_RATE = 1e-4
DEMONSTRATION_WEIGHT = 1.0
MARGIN = 0.8
TARGET_INTERVAL = 200
CHECKPOINTS = (2000, *range(2001, 2011), 2200, 2201, *range(2500, 10001, 500))
SENTINEL_SEED = 10001
SENTINEL_STEPS = (6, 18, 22, 24)


def target_source_updates(checkpoints=CHECKPOINTS, interval=TARGET_INTERVAL):
    """Save missing target generations without introducing physical evaluations."""
    return tuple(sorted({step - step % interval for step in checkpoints} - set(checkpoints)))


TARGET_SOURCE_UPDATES = target_source_updates()


def verify_archive(reference, archive):
    """Bind all local attempt-013 evidence to its committed archive manifest."""
    manifest_path = archive / "archive_manifest.json"
    relative = manifest_path.relative_to(ROOT).as_posix()
    committed = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)
    archived = read_json(manifest_path)
    if json.loads(committed) != archived or archived["status"] != "completed":
        raise ValueError("The completed attempt-013 archive differs from committed evidence.")
    checked = {}
    for name, record in archived["files"].items():
        control.require_hash(archive / name, record["sha256"], "archived attempt-013 evidence")
        if not name.endswith("update_samples.jsonl"):
            control.require_hash(reference / name, record["sha256"], "attempt-013 reference evidence")
            checked[name] = record["sha256"]
    for name, record in archived["update_log_coverage"].items():
        control.require_hash(reference / name, record["raw_sha256"], "attempt-013 raw updates")
        if [row["update"] for row in read_updates(reference / name)] != list(range(1, 2001)):
            raise ValueError("Attempt-013 update records do not cover the declared absolute clock.")
        if record["raw_records"] != 2000:
            raise ValueError("Attempt-013 archived raw update count is inconsistent.")
        checked[name] = record["raw_sha256"]
    return {
        "archive_manifest": relative,
        "archive_manifest_sha256": fit.file_hash(manifest_path),
        "committed_archive_manifest_sha256": hashlib.sha256(committed).hexdigest(),
        "committed_archive_verified": True,
        "archive_comparison": "Exact JSON equality; Git checkout line endings may differ.",
        "reference_files_sha256": checked,
    }


def verify_previous_source(settings, provenance):
    """Verify the recorded launch files and their committed source at attempt 013."""
    declaration = settings["training"]
    previous.validate_fixed_protocol(declaration, declaration["base_runner_sha256"])
    identity = previous.verify_base_runner()
    if any(declaration[name] != value for name, value in identity.items()):
        raise ValueError("Attempt-013 immutable implementation identity has changed.")
    if declaration["attempt"] != 13:
        raise ValueError("The resume source must be attempt 013.")
    sources = provenance["source_sha256"]
    if len(sources) != 52:
        raise ValueError("Expected the 52 recorded production source files.")
    helpers = declaration["helper_sha256"]
    launch = {
        declaration["entrypoint_script"]: declaration["entrypoint_sha256"],
        declaration["base_runner_script"]: declaration["base_runner_sha256"],
    }
    if declaration["entrypoint_script"] != PREVIOUS_RUNNER.relative_to(ROOT).as_posix():
        raise ValueError("Unexpected attempt-013 entrypoint.")
    for name, expected in {**sources, **helpers, **launch}.items():
        control.require_hash(ROOT / name, expected, "attempt-013 source or helper")
    committed_launch = {}
    for name in launch:
        blob = subprocess.check_output(["git", "show", f"{provenance['git_head']}:{name}"], cwd=ROOT)
        if blob.replace(b"\r\n", b"\n") != (ROOT / name).read_bytes().replace(b"\r\n", b"\n"):
            raise ValueError(f"The attempt-013 launch source differs from its commit: {name}")
        committed_launch[name] = hashlib.sha256(blob).hexdigest()
    return {
        "reference_source_sha256": sources, "helper_sha256": helpers,
        "launch_script_sha256": launch, "launch_git_head": provenance["git_head"],
        "committed_launch_blob_sha256": committed_launch,
    }


def validate_resume_snapshot(snapshot, settings, phase):
    """Check the stored TD and Adam clocks without adjusting either of them."""
    expected_config = {**settings["dqn"], "learning_rate": LEARNING_RATE}
    optimizer = snapshot["optimizer_summary"]
    expected = (
        snapshot["update_count"] == START_TD_UPDATE,
        snapshot["dqn_config"] == expected_config,
        snapshot["environment_config"] == settings["environment"],
        optimizer["step_min"] == 22000, optimizer["step_max"] == 22000,
        optimizer["parameter_count"] == optimizer["initialized_parameter_count"],
        phase["optimizer_updates"] == START_TD_UPDATE,
        phase["total_updates_on_branch"] == INHERITED_UPDATES,
        phase["learning_rate"] == LEARNING_RATE,
        phase["demonstration_weight"] == DEMONSTRATION_WEIGHT,
        phase["final_parameter_sha256"] == snapshot["parameter_sha256"],
        phase["final_observation"]["optimizer_state"] == optimizer,
        snapshot["dqn_config"]["target_update_interval"] == TARGET_INTERVAL,
        all(group["lr"] == LEARNING_RATE for group in snapshot["optimizer"]["param_groups"]),
        all(row["rollout"]["success"] for row in phase["final_observation"]["scenes"]),
    )
    if not all(expected):
        raise ValueError("The snapshot is not the exact successful attempt-013 treatment at TD 2000.")


def load_inputs(reference, archive):
    """Check the 013 checkpoint and the original 011 data/return provenance chain."""
    archived = verify_archive(reference, archive)
    settings = read_json(reference / "settings.json")
    if read_json(reference / "run_status.json")["status"] != "completed":
        raise ValueError("The reference run must have completed.")
    source = verify_previous_source(settings, read_json(reference / "provenance.json"))
    inputs = read_json(reference / "input_manifest.json")
    inherited = control.load_reference(Path(inputs["reference_run"]), ROOT / inputs["archive_manifest"].rsplit("/", 1)[0])
    for name in ("reference_files_sha256", "reference_source_sha256", "config_sha256", "helper_sha256"):
        if inputs[name] != inherited["manifest"][name]:
            raise ValueError(f"The attempt-011 inheritance chain differs at {name}.")
    dataset = read_json(reference / "dataset.json")
    targets = read_json(reference / "mc_targets.json")
    if (dataset != inherited["dataset"] or targets["returns"] != inherited["returns"].tolist()
            or targets != read_json(Path(inputs["reference_run"]) / "mc_targets.json")):
        raise ValueError("The frozen demonstrations or actual discounted returns have changed.")
    phase = read_json(reference / "treatment/td/summary.json")
    if read_json(reference / "summary.json")["arms"]["treatment"] != phase:
        raise ValueError("Attempt-013 treatment summaries disagree.")
    for name, expected in phase["checkpoint_sha256"].items():
        control.require_hash(reference / "treatment/td" / name, expected, "attempt-013 treatment checkpoint")
    snapshot_path = reference / phase["training_snapshot_file"]
    if phase["training_snapshot_file"] != "treatment/td/training_state.pt":
        raise ValueError("Expected the final attempt-013 treatment training snapshot.")
    control.require_hash(snapshot_path, phase["training_snapshot_sha256"], "resumed training snapshot")
    snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=True)
    validate_resume_snapshot(snapshot, settings, phase)
    observations = read_json(reference / "treatment/td/observations.json")
    if ([row["update"] for row in observations] != list(previous.CHECKPOINTS)
            or observations[-1] != phase["final_observation"]):
        raise ValueError("The attempt-013 physical observations are incomplete or inconsistent.")
    manifest = {
        "reference_run": str(reference), "attempt_013": {**archived, **source},
        "attempt_011": inherited["manifest"],
        "resumed_snapshot_file": phase["training_snapshot_file"],
        "resumed_snapshot_sha256": phase["training_snapshot_sha256"],
        "inherited_td_updates": START_TD_UPDATE, "inherited_optimizer_updates": INHERITED_UPDATES,
        "same_frozen_117_transitions": True, "episode_returns_recomputed_exactly": True,
    }
    return {
        **inherited, "settings": settings, "snapshot": snapshot, "dataset": dataset,
        "targets": targets, "previous_phase": phase, "manifest": manifest,
    }


def restore_continuation(snapshot):
    """Restore all stored state; do not invoke a phase-boundary reset or target sync."""
    agent, env = adam.restore_training_snapshot(snapshot)
    if not exact_equal(adam.training_snapshot(agent, snapshot["environment_config"]), snapshot):
        env.close()
        raise RuntimeError("Restoration changed the complete continuation snapshot.")
    return agent, env


def sentinel_record(agent, batch, episodes, update, start=START_TD_UPDATE,
                    reference_steps=SENTINEL_STEPS):
    """Probe fixed demonstrated inputs with batch size one, without an environment step."""
    episode = next(row for row in episodes if row["seed"] == SENTINEL_SEED)
    states = []
    # Restore every module's flag, including any deliberately mixed train/eval modes.
    modes = [(module, module.training) for module in agent.learner.online.modules()]
    try:
        agent.learner.online.eval()
        with torch.no_grad():
            for step in reference_steps:
                index = episode["start"] + step - 1
                if not episode["start"] <= index < episode["stop"]:
                    raise ValueError("A sentinel reference step lies outside its fixed episode.")
                q = agent.learner.online(batch.observations[index:index + 1])[0]
                if q.shape != (8,) or not bool(torch.isfinite(q).all()):
                    raise FloatingPointError("Sentinel Q values must contain eight finite action scores.")
                action = int(batch.actions[index])
                other = torch.cat((q[:action], q[action + 1:]))
                states.append({
                    "reference_step": step, "global_transition_index": index,
                    "demonstrated_action": action, "greedy_action": int(q.argmax()),
                    "q_values": q.tolist(),
                    "signed_demonstration_gap": float(q[action]) - float(other.max()),
                })
    finally:
        for module, mode in modes:
            module.training = mode
    return {"update": update, "extension_update": update - start,
            "scene_seed": SENTINEL_SEED, "states": states}


def retention_metrics(observations, expected_start, start):
    result = multi.retention_metrics(observations, expected_start)
    for seed, record in result["per_scene"].items():
        record.pop("first_ten_updates_success_count")
        record.pop("first_ten_updates_observation_count")
        early = [row for row in observations if start < row["update"] <= start + 10]
        record["first_ten_extension_updates_success_count"] = sum(
            next(scene for scene in row["scenes"] if scene["seed"] == int(seed))["rollout"]["success"]
            for row in early)
        record["first_ten_extension_updates_observation_count"] = len(early)
    return result


def execute_extension(agent, env, environment, batch, returns, episodes, output, expected_start,
                      start=START_TD_UPDATE, final=FINAL_TD_UPDATE, checkpoints=CHECKPOINTS,
                      reference_steps=SENTINEL_STEPS):
    """Continue a fixed batch while preserving absolute target and optimizer clocks."""
    if (agent.learner.update_count != start or not start < final
            or tuple(sorted(set(checkpoints))) != tuple(checkpoints)
            or checkpoints[0] != start or checkpoints[-1] != final):
        raise ValueError("The continuation clock or declared evaluation schedule is inconsistent.")
    directory = output / "td"
    directory.mkdir(parents=True, exist_ok=False)
    target_sources = target_source_updates(checkpoints, agent.learner.config.target_update_interval)
    if any(step < start for step in target_sources):
        # The resume snapshot already contains a target from before the continuation.
        target_sources = tuple(step for step in target_sources if step >= start)
    initial_hash = fit.parameter_hash(agent)
    observations, synchronizations = [], 0
    with (directory / "updates.jsonl").open("x", encoding="utf-8") as updates, (
        directory / "sentinel_q.jsonl"
    ).open("x", encoding="utf-8") as sentinels:
        for step in range(start, final + 1):
            if step > start:
                metrics = asdict(agent.learner.update(
                    batch, demonstration_weight=DEMONSTRATION_WEIGHT, demonstration_margin=MARGIN))
                if agent.learner.update_count != step:
                    raise RuntimeError("The absolute TD counter changed unexpectedly.")
                synchronizations += int(metrics["target_synchronized"])
                updates.write(json.dumps({"update": step, "extension_update": step - start, **metrics},
                                         allow_nan=False) + "\n")
            probe = sentinel_record(agent, batch, episodes, step, start, reference_steps)
            sentinels.write(json.dumps(probe, allow_nan=False) + "\n")
            if step in checkpoints:
                row = multi.observe(agent, env, environment, batch, returns, episodes,
                                    "extension/td", step, PRIOR_PHASE_UPDATES, directory)
                observations.append(row)
                fit.save_json(directory / "observations.json", observations)
                torch.save(adam.training_snapshot(agent, environment),
                           directory / f"training_state_{step}.pt")
                if step == start:
                    equal = row == expected_start
                    fit.save_json(output / "initial_equivalence.json", {
                        "passed": equal, "exact_previous_final_observation": equal,
                        "absolute_td_update": start, "new_updates_before_gate": 0,
                    })
                    if not equal:
                        raise RuntimeError("Restored behavior differs from the previous final observation.")
                updates.flush()
                sentinels.flush()
            elif step in target_sources:
                torch.save(adam.training_snapshot(agent, environment),
                           directory / f"target_source_state_{step}.pt")
            if step % 100 == 0:
                updates.flush()
                sentinels.flush()
    snapshot_path = directory / "training_state.pt"
    torch.save(adam.training_snapshot(agent, environment), snapshot_path)
    result = {
        "directory": "td", "initial_parameter_sha256": initial_hash,
        "final_parameter_sha256": fit.parameter_hash(agent),
        "optimizer_updates": final - start, "initial_td_update": start, "final_td_update": final,
        "prior_updates_on_branch": PRIOR_PHASE_UPDATES,
        "inherited_optimizer_updates": PRIOR_PHASE_UPDATES + start,
        "total_updates_on_branch": PRIOR_PHASE_UPDATES + final,
        "target_synchronizations": synchronizations, "target_source_updates": list(target_sources),
        "demonstration_weight": DEMONSTRATION_WEIGHT, "learning_rate": agent.learner.config.learning_rate,
        "initial_observation": observations[0], "final_observation": observations[-1],
        "retention": retention_metrics(observations, expected_start, start),
        "evaluation_environment_steps": sum(row["aggregate"]["evaluation_environment_steps"]
                                            for row in observations),
        "sentinel_records": final - start + 1, "sentinel_environment_steps": 0,
        "training_snapshot_file": "td/training_state.pt",
        "training_snapshot_sha256": fit.file_hash(snapshot_path),
        "checkpoint_sha256": {path.name: fit.file_hash(path) for path in sorted(directory.glob("*.pt"))},
    }
    fit.save_json(directory / "summary.json", result)
    return result


def write_accounting(output):
    update_path = output / "td/updates.jsonl"
    observation_path = output / "td/observations.json"
    sentinel_path = output / "td/sentinel_q.jsonl"
    updates = read_updates(update_path) if update_path.exists() else []
    observations = read_json(observation_path) if observation_path.exists() else []
    sentinel_count = sum(1 for _ in sentinel_path.open(encoding="utf-8")) if sentinel_path.exists() else 0
    result = {
        "optimizer_updates_total": len(updates), "td_optimizer_updates": len(updates),
        "inherited_optimizer_updates": INHERITED_UPDATES,
        "inherited_supervised_updates": 20000, "inherited_calibration_updates": 20000,
        "inherited_td_updates": START_TD_UPDATE,
        "total_updates_on_branch": INHERITED_UPDATES + len(updates),
        "absolute_td_update": START_TD_UPDATE + len(updates),
        "evaluation_environment_steps": sum(row["aggregate"]["evaluation_environment_steps"]
                                            for row in observations),
        "scheduled_evaluation_episodes": sum(len(row["scenes"]) for row in observations),
        "sentinel_records": sentinel_count, "sentinel_forward_passes": sentinel_count * len(SENTINEL_STEPS),
        "sentinel_environment_steps": 0, "demonstration_collection_steps": 0,
        "online_training_steps": 0, "held_out_scenes_accessed": 0,
        "note": "Counts completed updates and completed physical checkpoint observations. Inherited updates are separate; independent verification is recorded separately.",
    }
    fit.save_json(output / "accounting.json", result)
    return result


def run(output, reference, archive):
    output, reference, archive = output.resolve(), reference.resolve(), archive.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite the continuation experiment: {output}")
    inherited = load_inputs(reference, archive)
    snapshot = inherited["snapshot"]
    protocol = {
        "attempt": 14, "kind": "exact_low_rate_td_continuation", "scene_seeds": list(multi.SEEDS),
        "initial_td_update": START_TD_UPDATE, "final_td_update": FINAL_TD_UPDATE,
        "extension_updates": EXTENSION_UPDATES, "td_checkpoints": list(CHECKPOINTS),
        "target_source_updates": list(TARGET_SOURCE_UPDATES),
        "inherited_optimizer_updates": INHERITED_UPDATES,
        "learning_rate": LEARNING_RATE, "demonstration_weight": DEMONSTRATION_WEIGHT, "margin": MARGIN,
        "target_update_interval": TARGET_INTERVAL, "effective_batch_size": 117,
        "full_batch_weighting": "equal_per_transition", "early_stopping": False,
        "resume_source": "attempt_013_treatment_final_training_state",
        "preserve_adam_moments_and_parameter_steps": True,
        "preserve_absolute_td_counter": True, "synchronize_target_at_resume": False,
        "initial_gate": "exact_full_training_state_and_previous_final_observation",
        "sentinel_seed": SENTINEL_SEED, "sentinel_reference_steps": list(SENTINEL_STEPS),
        "sentinel_batch_size": 1, "sentinel_updates": "every absolute update from 2000 through 10000 inclusive",
        "sentinel_gap": "Q(demonstrated action) - max Q(other actions)",
        "sentinel_timing": "after the optimizer update and any scheduled target synchronization",
        "sentinel_is_physical_evaluation": False,
        "online_training_steps": 0, "held_out_scenes_accessed": 0,
        "script_sha256": fit.file_hash(Path(__file__)),
        "helper_sha256": {**inherited["manifest"]["attempt_013"]["helper_sha256"],
                          **inherited["manifest"]["attempt_013"]["launch_script_sha256"]},
    }
    output.mkdir(parents=True, exist_ok=False)
    env = None
    try:
        begin_run(output, snapshot["environment_config"], snapshot["dqn_config"], protocol)
        fit.save_json(output / "input_manifest.json", inherited["manifest"])
        fit.save_json(output / "dataset.json", inherited["dataset"])
        fit.save_json(output / "mc_targets.json", inherited["targets"])
        torch.set_num_threads(1)
        agent, env = restore_continuation(snapshot)
        fit.save_json(output / "restore_checks.json", {
            "exact_full_snapshot": True, "target_synchronized_at_restore": False,
            "optimizer_reset_at_restore": False, "td_counter_reset_at_restore": False,
            "absolute_td_update": agent.learner.update_count,
            "optimizer_state": snapshot["optimizer_summary"],
        })
        phase = execute_extension(
            agent, env, snapshot["environment_config"], inherited["batch"], inherited["returns"],
            inherited["episodes"], output, inherited["previous_phase"]["final_observation"],
        )
        summary = {
            "protocol": protocol, "dataset": inherited["dataset"], "extension": phase,
            "accounting": write_accounting(output),
            "limitations": [
                "Three known training scenes and one inherited initialization; no held-out generalization test.",
                "Frozen privileged demonstrations, with no new experience or exploration.",
                "Physical success is observed only at predeclared checkpoints.",
                "Sentinel action changes describe fixed demonstrated states, not actual rollout failures.",
            ],
        }
        fit.save_json(output / "summary.json", summary)
    except BaseException as exc:
        if (output / "run_status.json").exists():
            write_accounting(output)
            finish_run(output, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", exc)
        raise
    else:
        finish_run(output, "completed")
        return summary
    finally:
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.output_dir, args.reference_run, args.archive_dir)


if __name__ == "__main__":
    main()
