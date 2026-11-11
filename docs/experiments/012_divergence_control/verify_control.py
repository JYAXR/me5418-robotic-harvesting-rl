"""Independently verify the diagnosed single-factor TD comparison."""

import argparse
import importlib.util
import json
import subprocess
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from me5418_harvesting.agents.dqn_learning import TransitionBatch
from me5418_harvesting.agents.optimizer_transition import optimizer_state_summary

ROOT = Path(__file__).resolve().parents[3]
RUNNER = Path(__file__).with_name("run_control.py")
MULTISCENE_VERIFIER = ROOT / "docs/experiments/011_multiscene_retention/verify_multiscene_retention.py"


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


multi = import_file("divergence_independent_multiscene", MULTISCENE_VERIFIER)
audit = multi.audit
read_json, read_log = audit.read_json, audit.read_log
require, require_equal, file_hash = audit.require, audit.require_equal, audit.file_hash


def expected_settings(settings, factor, value, treatment):
    """Allow exactly one declared scalar difference in learner configuration."""
    result = deepcopy(settings)
    require(factor in ("learning_rate", "demonstration_weight"), "supported single factor")
    require(isinstance(value, (float, int)) and not isinstance(value, bool)
            and np.isfinite(value) and value > 0, "finite positive treatment value")
    if factor == "learning_rate" and treatment:
        result["dqn"]["learning_rate"] = value
    return result


def independent_intervention(agent, factor, value, treatment):
    """Preserve Adam tensors while applying the scalar without runner helpers."""
    audit.independent_boundary(agent, preserve=True)
    if treatment and factor == "learning_rate":
        learner = agent.learner
        learner.config = replace(learner.config, learning_rate=value)
        learner.optimizer.defaults["lr"] = value
        for group in learner.optimizer.param_groups:
            group["lr"] = value
    return value if treatment and factor == "demonstration_weight" else 1.0


def verify_initial_state(payload, calibrated, settings, factor, value, treatment):
    """Check all shared state, allowing only the requested learning-rate option."""
    require(payload["update_count"] == 0, "TD clock starts at zero")
    require_equal(payload["online"], calibrated["online"], "unchanged calibrated policy")
    require_equal(payload["target"], calibrated["online"], "initial target synchronized to policy")
    expected = expected_settings(settings, factor, value, treatment)
    require(json.loads(json.dumps(payload["dqn_config"])) == expected["dqn"]
            and payload["environment_config"] == expected["environment"],
            "only declared learner scalar changes")
    optimizer = deepcopy(calibrated["optimizer"])
    if treatment and factor == "learning_rate":
        for group in optimizer["param_groups"]:
            group["lr"] = value
    require_equal(payload["optimizer"], optimizer, "preserved exact Adam moments and internal steps")
    require_equal(payload["agent_rng_state"], calibrated["agent_rng_state"], "preserved exploration RNG")
    require_equal(payload["torch_rng_state"], calibrated["torch_rng_state"], "preserved torch RNG")


def replay_early_updates(calibrated, branch, run_dir, batch, factor, value, rows):
    """Recompute ten updates and compare full parameter and Adam tensor trees."""
    agent, env = audit.independent_restore(calibrated)
    try:
        weight = independent_intervention(agent, factor, value, branch == "treatment")
        saved = {row["update"]: row for row in rows}
        require(optimizer_state_summary(agent.learner) == saved[0]["optimizer_state"],
                f"{branch} independent initial Adam state")
        logs = read_log(run_dir / branch / "td/updates.jsonl")
        for step in range(1, 11):
            stats = agent.learner.update(batch, demonstration_weight=weight,
                                         demonstration_margin=0.8)
            require({"update": step, **asdict(stats)} == logs[step - 1],
                    f"{branch}/{step} independent TD metrics")
            payload = torch.load(run_dir / branch / f"td/training_state_{step}.pt",
                                 map_location="cpu", weights_only=True)
            require_equal(agent.learner.online.state_dict(), payload["online"],
                          f"{branch}/{step} exact online tensors")
            require_equal(agent.learner.target.state_dict(), payload["target"],
                          f"{branch}/{step} exact target tensors")
            require_equal(agent.learner.optimizer.state_dict(), payload["optimizer"],
                          f"{branch}/{step} exact Adam tensors and settings")
            require(optimizer_state_summary(agent.learner) == payload["optimizer_summary"]
                    == saved[step]["optimizer_state"], f"{branch}/{step} exact Adam summary")
        return {"optimizer_updates_performed": 10, "exact_update_metrics": True,
                "exact_online_tensors": True, "exact_target_tensors": True,
                "exact_optimizer_tensors_and_settings": True}
    finally:
        env.close()


def verify_phase(run_dir, branch, record, settings, episodes, checkpoints, calibrated,
                 preceding, factor, value):
    directory = f"{branch}/td"
    path = run_dir / directory
    require(record["directory"] == directory and read_json(path / "summary.json") == record,
            f"{branch} summary identity")
    logs, rows = read_log(path / "updates.jsonl"), read_json(path / "observations.json")
    require([row["update"] for row in logs] == list(range(1, 2001)),
            f"{branch} all 2000 update records")
    require([row["update"] for row in rows] == checkpoints, f"{branch} all 18 observations")
    require(record["optimizer_updates"] == 2000 and record["prior_updates_on_branch"] == 40000
            and record["total_updates_on_branch"] == 42000, f"{branch} actual and inherited updates")
    require(record["initial_parameter_sha256"] == calibrated["parameter_sha256"],
            f"{branch} common calibrated policy")
    flags = [step % 200 == 0 for step in range(1, 2001)]
    require([row["target_synchronized"] for row in logs] == flags
            and record["target_synchronizations"] == 10, f"{branch} target schedule")
    require(rows[0] == record["initial_observation"] and rows[-1] == record["final_observation"],
            f"{branch} endpoint records")
    for field in ("fit", "return_fit", "scenes", "aggregate"):
        require(rows[0][field] == preceding[field], f"{branch} unchanged initial {field}")
    expected = expected_settings(settings, factor, value, branch == "treatment")
    expected_names = {f"update_{step}.pt" for step in checkpoints}
    expected_names |= {f"training_state_{step}.pt" for step in checkpoints} | {"training_state.pt"}
    require(set(record["checkpoint_sha256"]) == expected_names, f"{branch} all saved checkpoints")
    for name, digest in record["checkpoint_sha256"].items():
        require(file_hash(path / name) == digest, f"{directory}/{name} file hash")
    for row in rows:
        step = row["update"]
        multi.verify_aggregate(row, episodes)
        require(row["total_updates_on_branch"] == 40000 + step, f"{branch}/{step} update history")
        inference = torch.load(path / f"update_{step}.pt", map_location="cpu", weights_only=True)
        require(inference["environment_config"] == expected["environment"]
                and json.loads(json.dumps(inference["dqn_config"])) == expected["dqn"],
                f"{branch}/{step} inference configuration")
        metadata = inference["metadata"]
        require(metadata["phase"] == "multiscene_retention_diagnostic"
                and metadata["objective"] == directory and metadata["updates"] == step
                and metadata["prior_updates_on_branch"] == 40000
                and metadata["total_updates_on_branch"] == 40000 + step
                and metadata["training_scene_only"]
                and metadata["scene_seeds"] == [10000, 10001, 10002]
                and metadata["fit"] == row["fit"] and metadata["aggregate"] == row["aggregate"],
                f"{branch}/{step} inference metadata")
        state_hash = audit.state_hash(inference["online"])
        saved_full = torch.load(path / f"training_state_{step}.pt", map_location="cpu", weights_only=True)
        if step < 200:
            target_hash = calibrated["parameter_sha256"]
        elif step % 200 == 0:
            target_hash = state_hash
        else:
            # Update 500 uses the unsaved update-400 target; only its saved integrity is checked.
            target_hash = audit.state_hash(saved_full["target"])
        payload = multi.verify_snapshot(path / f"training_state_{step}.pt", state_hash,
                                         expected, step, target_hash, row["optimizer_state"])
        require_equal(payload["online"], inference["online"], f"{branch}/{step} inference/full weights")
        require(row["optimizer_state"]["step_min"] == row["optimizer_state"]["step_max"] == 20000 + step,
                f"{branch}/{step} inherited Adam step clock")
        if step == 0:
            verify_initial_state(payload, calibrated, settings, factor, value, branch == "treatment")
        if step == 2000:
            require(state_hash == record["final_parameter_sha256"], f"{branch} final parameters")
            final = torch.load(path / "training_state.pt", map_location="cpu", weights_only=True)
            require_equal(final, payload, f"{branch} final full snapshot alias")
    multi.verify_retention(record["retention"], rows, preceding)
    require(record["evaluation_environment_steps"] == sum(
        row["aggregate"]["evaluation_environment_steps"] for row in rows),
        f"{branch} actual evaluation actions")
    return rows, logs


def verify_inputs(run_dir, protocol):
    """Check archived hashes independently of the control runner's loader."""
    manifest = read_json(run_dir / "input_manifest.json")
    reference = Path(manifest["reference_run"])
    archive_file = ROOT / manifest["archive_manifest"]
    committed = subprocess.check_output(
        ["git", "show", f"HEAD:{archive_file.relative_to(ROOT).as_posix()}"], cwd=ROOT,
    )
    archive = read_json(archive_file)
    require(json.loads(committed) == archive
            and audit.hashlib.sha256(committed).hexdigest() == manifest["committed_archive_manifest_sha256"]
            and file_hash(archive_file) == manifest["archive_manifest_sha256"],
            "committed attempt-011 archive identity, allowing Git line-ending conversion")
    for name, record in archive["files"].items():
        require(file_hash(archive_file.parent / name) == record["sha256"], f"archive file {name}")
        if not name.endswith("update_samples.jsonl"):
            require(file_hash(reference / name) == record["sha256"], f"archived raw input {name}")
    for name, record in archive["update_log_coverage"].items():
        require(file_hash(reference / name) == record["raw_sha256"]
                and len(read_log(reference / name)) == record["raw_records"], f"archived log {name}")
    for name, digest in manifest["reference_files_sha256"].items():
        require(file_hash(reference / name) == digest, f"reference input {name}")
    require(manifest["committed_archive_verified"] and manifest["inherited_checkpoint_used"]
            and manifest["inherited_optimizer_updates"] == 40000
            and manifest["episode_returns_recomputed_exactly"], "declared inherited input semantics")
    for group in ("reference_source_sha256", "config_sha256", "helper_sha256"):
        for name, digest in manifest[group].items():
            require(file_hash(ROOT / name) == digest, f"{group}: {name}")
    provenance = read_json(run_dir / "provenance.json")
    for name, digest in provenance["source_sha256"].items():
        require(file_hash(ROOT / name) == digest, f"launch source: {name}")
    require(provenance["source_sha256"] == manifest["reference_source_sha256"]
            and len(provenance["source_sha256"]) == 52, "unchanged 52 production source files")
    require(protocol["helper_sha256"] == manifest["helper_sha256"]
            and file_hash(RUNNER) == protocol["script_sha256"], "runner and helper provenance")
    require(read_json(reference / "run_status.json")["status"] == "completed", "completed reference")
    return manifest, reference, provenance


def verify_boundary(record, initial, calibrated_observation, factor, value, treatment):
    transition, intervention = record["transition"], record["intervention"]
    before = calibrated_observation["optimizer_state"]
    multi.verify_boundary(transition, before, before, transition["policy_parameter_sha256"], 40000, True)
    baseline = {"learning_rate": 0.0003, "demonstration_weight": 1.0}
    expected = {**baseline, factor: value} if treatment else baseline
    require(intervention["factor"] == factor and intervention["value"] == expected[factor]
            and intervention["applied"] == treatment
            and intervention["settings_before"] == baseline
            and intervention["settings_after"] == expected
            and intervention["optimizer_before"] == before
            and intervention["optimizer_after"] == initial["optimizer_state"]
            and intervention["weights_unchanged"] and intervention["adam_state_unchanged"],
            "one scalar intervention after preserved-Adam boundary")


def verify(run_dir, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite single-factor verification: {output}")
    torch.set_num_threads(1)
    summary, settings = read_json(run_dir / "summary.json"), read_json(run_dir / "settings.json")
    protocol = summary["protocol"]
    require(read_json(run_dir / "run_status.json")["status"] == "completed", "completed control")
    require(protocol == settings["training"] and protocol["arms"] == ["baseline", "treatment"]
            and protocol["scene_seeds"] == [10000, 10001, 10002], "declared arms and training scenes")
    checkpoints = [*range(11), 20, 50, 100, 200, 500, 1000, 2000]
    require(protocol["td_checkpoints"] == checkpoints and protocol["td_updates"] == 2000
            and protocol["inherited_optimizer_updates"] == 40000 and not protocol["early_stopping"],
            "fixed update budgets and observation schedule")
    factor, value = protocol["factor"], protocol["treatment_value"]
    expected_settings(settings, factor, value, True)
    baseline = {"learning_rate": 0.0003, "demonstration_weight": 1.0}
    require(protocol["baseline_settings"] == baseline and protocol["baseline_value"] == baseline[factor]
            and value != baseline[factor] and protocol["treatment_settings"] == {**baseline, factor: value}
            and protocol["margin"] == 0.8 and protocol["preserve_adam_moments_and_parameter_steps"],
            "one explicit changed scalar and preserved Adam")
    require(protocol["full_batch_weighting"] == "equal_per_transition"
            and protocol["effective_batch_size"] == 117
            and protocol["online_training_steps"] == protocol["held_out_scenes_accessed"] == 0,
            "fixed known-scene full-batch design")
    manifest, reference, provenance = verify_inputs(run_dir, protocol)
    reference_settings = read_json(reference / "settings.json")
    require(settings["environment"] == reference_settings["environment"]
            and settings["dqn"] == reference_settings["dqn"], "unchanged base environment and learner")
    require(settings["dqn"]["target_update_interval"] == 200 and settings["dqn"]["gamma"] == 0.99
            and settings["dqn"]["gradient_clip_norm"] == 10
            and settings["dqn"]["hidden_sizes"] == [256, 256], "unchanged model and other hyperparameters")
    dataset, targets = read_json(run_dir / "dataset.json"), read_json(run_dir / "mc_targets.json")
    require(dataset == summary["dataset"] == read_json(reference / "dataset.json")
            and targets == read_json(reference / "mc_targets.json")
            and file_hash(reference / "demonstration.npz") == dataset["sha256"], "unchanged frozen dataset")
    with np.load(reference / "demonstration.npz", allow_pickle=False) as saved:
        batch = TransitionBatch(**{name: torch.from_numpy(saved[name].copy())
                                   for name in TransitionBatch.__dataclass_fields__})
    episodes = dataset["episodes"]
    require([entry["seed"] for entry in episodes] == [10000, 10001, 10002]
            and [(entry["start"], entry["stop"]) for entry in episodes] == [(0, 35), (35, 76), (76, 117)],
            "ordered complete three-scene dataset")
    returns, return_checks, precision = multi.independent_returns(batch, episodes, 0.99, targets["returns"])
    require(targets["rewards"] == batch.rewards.tolist()
            and targets["network_bootstrapping_used"] is False and targets["episode_boundaries_respected"],
            "actual rewards and separate finite-horizon returns")
    reference_summary = read_json(reference / "summary.json")
    calibration = reference_summary["calibration"]
    snapshot_path = reference / manifest["calibrated_snapshot_file"]
    require(file_hash(snapshot_path) == manifest["calibrated_snapshot_sha256"]
            == calibration["training_snapshot_sha256"], "shared calibrated snapshot hash")
    preceding = calibration["final_observation"]
    require(all(scene["rollout"]["success"] for scene in preceding["scenes"]),
            "all three scenes eligible for retention")
    calibrated = multi.verify_snapshot(
        snapshot_path, calibration["final_parameter_sha256"], settings, 20000,
        reference_summary["pretraining"]["final_parameter_sha256"], preceding["optimizer_state"],
    )
    arms, reloads, action_count = {}, [], 0
    for branch in ("baseline", "treatment"):
        record = summary["arms"][branch]
        rows, logs = verify_phase(run_dir, branch, record, settings, episodes, checkpoints,
                                  calibrated, preceding, factor, value)
        current = {**baseline, factor: value} if branch == "treatment" else baseline
        require(record["learning_rate"] == current["learning_rate"]
                and record["demonstration_weight"] == current["demonstration_weight"],
                f"{branch} actual declared objective settings")
        boundary = read_json(run_dir / branch / "td_boundary.json")
        require(boundary == summary["td_boundaries"][branch]
                and read_json(run_dir / branch / "transition.json") == boundary["transition"],
                f"{branch} complete boundary records")
        verify_boundary(boundary, rows[0], preceding, factor, value, branch == "treatment")
        require(boundary["transition"]["policy_parameter_sha256"] == calibrated["parameter_sha256"],
                f"{branch} calibrated boundary policy")
        if branch == "baseline":
            require(rows == read_json(reference / "preserve/td/observations.json")
                    and logs == read_log(reference / "preserve/td/updates.jsonl"),
                    "all baseline observations and updates reproduce attempt 011 exactly")
            original = torch.load(reference / "preserve/td/training_state.pt", map_location="cpu", weights_only=True)
            repeated = torch.load(run_dir / "baseline/td/training_state.pt", map_location="cpu", weights_only=True)
            require_equal(repeated, original, "exact complete baseline final state")
        early = replay_early_updates(calibrated, branch, run_dir, batch, factor, value, rows)
        for step in (0, 1, 2000):
            row = next(item for item in rows if item["update"] == step)
            reloads.append(multi.reload_policy(run_dir, f"{branch}/td", step, row, batch, returns, episodes))
        arms[branch] = {"checkpoints_verified": len(rows), "update_records_verified": len(logs),
                        "full_training_snapshots_verified": len(rows) + 1,
                        "independent_early_replay": early, "retention_verified": record["retention"]}
        action_count += record["evaluation_environment_steps"]
    equivalence = read_json(run_dir / "baseline_equivalence.json")
    require(equivalence == summary["baseline_equivalence"] == {
        "passed": True, "exact_update_records": 2000, "exact_observations": 18,
        "exact_final_training_state": True, "treatment_started": False,
    }, "completed exact baseline gate before treatment")
    accounting = read_json(run_dir / "accounting.json")
    require(all(summary[name] == item for name, item in accounting.items()), "accounting copy")
    actual_logs = {path.relative_to(run_dir).as_posix(): len(read_log(path))
                   for path in run_dir.rglob("updates.jsonl")}
    require(accounting["update_records_per_log"] == actual_logs
            and accounting["optimizer_updates_total"] == accounting["td_optimizer_updates"]
            == sum(actual_logs.values()) == 4000, "all actual new TD updates counted")
    require(accounting["inherited_optimizer_updates"] == 40000
            and accounting["inherited_supervised_updates"] == accounting["inherited_calibration_updates"] == 20000
            and accounting["scheduled_evaluation_episodes"] == 108
            and accounting["evaluation_environment_steps"] == action_count
            and accounting["demonstration_collection_steps"] == accounting["online_training_steps"]
            == accounting["held_out_scenes_accessed"] == 0, "separate inherited, evaluation and new-data accounting")
    result = {
        "source_files_verified": len(provenance["source_sha256"]),
        "reference_input_files_verified": len(manifest["reference_files_sha256"]),
        "committed_archive_verified": True, "runner_and_helper_hashes_verified": True,
        "dataset_hash_verified": True, "independent_episode_returns": return_checks,
        "mc_float64_maximum_difference": precision, "exact_baseline_reproduction_verified": True,
        "single_scalar_boundary_verified": True, "shared_calibrated_full_snapshot_verified": True,
        "arms": arms, "reloaded_policies": reloads,
        "additional_verification_optimizer_updates": 20,
        "additional_verification_episodes": 18,
        "additional_verification_environment_steps": sum(
            scene["reload_environment_steps"] for item in reloads for scene in item["scenes"]),
        "training_scene_only": True, "verifier_sha256": file_hash(Path(__file__)),
        "multiscene_verifier_sha256": file_hash(MULTISCENE_VERIFIER),
        "independent_adam_verifier_sha256": file_hash(multi.INDEPENDENT_HELPER),
        "target_verification_limit": "The first ten updates are independently replayed; update-500 target integrity is checked, but its unsaved update-400 source is not numerically replayed.",
        "note": "Verification updates and actions are additional work; reset feasibility probes are excluded.",
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
