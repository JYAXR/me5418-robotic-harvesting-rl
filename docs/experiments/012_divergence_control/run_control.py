"""Replay attempt 011 and test one explicitly selected scalar after Adam restoration."""

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from me5418_harvesting.agents.dqn_learning import TransitionBatch
from me5418_harvesting.agents.multiscene_demonstrations import (
    build_multiscene_batch,
    slice_transition_batch,
)
from me5418_harvesting.agents.optimizer_transition import optimizer_state_summary
from me5418_harvesting.agents.run_record import begin_run, finish_run

ROOT = Path(__file__).resolve().parents[3]
REFERENCE_RUNNER = ROOT / "docs/experiments/011_multiscene_retention/run_multiscene_retention.py"
SPEC = importlib.util.spec_from_file_location("control_multiscene_reference", REFERENCE_RUNNER)
multi = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(multi)
adam, fit = multi.adam, multi.fit
SEEDS = multi.SEEDS
ARMS = ("baseline", "treatment")
TD_UPDATES = 2000
CHECKPOINTS = multi.TD_CHECKPOINTS
INHERITED_UPDATES = 40000
BASELINE = {"learning_rate": 3e-4, "demonstration_weight": 1.0}
MARGIN = 0.8


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_updates(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def exact_equal(left, right):
    """Compare complete snapshots without relying on zip serialization timestamps."""
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.dtype == right.dtype
                and left.shape == right.shape and torch.equal(left, right))
    if isinstance(left, dict):
        return (isinstance(right, dict) and left.keys() == right.keys()
                and all(exact_equal(value, right[key]) for key, value in left.items()))
    if isinstance(left, (list, tuple)):
        return (isinstance(right, (list, tuple)) and len(left) == len(right)
                and all(exact_equal(a, b) for a, b in zip(left, right)))
    return left == right


def require_hash(path, expected, description):
    if fit.file_hash(path) != expected:
        raise ValueError(f"Changed {description}: {path}")


def load_reference(reference, archive):
    """Anchor data, optimizer state, source files and comparison logs in committed evidence."""
    archive_path = archive / "archive_manifest.json"
    relative = archive_path.relative_to(ROOT).as_posix()
    committed = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)
    archived = read_json(archive_path)
    if json.loads(committed) != archived:
        raise ValueError("The attempt-011 archive manifest differs from its committed version.")
    checked = {}
    for name, record in archived["files"].items():
        require_hash(archive / name, record["sha256"], "archived evidence")
        if not name.endswith("update_samples.jsonl"):
            require_hash(reference / name, record["sha256"], "reference evidence")
            checked[name] = record["sha256"]
    for name, record in archived["update_log_coverage"].items():
        require_hash(reference / name, record["raw_sha256"], "reference update log")
        if len(read_updates(reference / name)) != record["raw_records"]:
            raise ValueError(f"The recorded update count differs: {name}")
        checked[name] = record["raw_sha256"]
    settings = read_json(reference / "settings.json")
    provenance = read_json(reference / "provenance.json")
    source_hashes = provenance["source_sha256"]
    if len(source_hashes) != 52:
        raise ValueError("Expected exactly the 52 source files recorded by attempt 011.")
    for name, expected in source_hashes.items():
        require_hash(ROOT / name, expected, "reference source")
    inputs = read_json(reference / "input_manifest.json")
    for mapping in (inputs["config_sha256"], inputs["helper_sha256"]):
        for name, expected in mapping.items():
            require_hash(ROOT / name, expected, "reference helper or configuration")
    require_hash(REFERENCE_RUNNER, settings["training"]["script_sha256"], "reference runner")
    protocol = settings["training"]
    if (read_json(reference / "run_status.json")["status"] != "completed"
            or protocol["scene_seeds"] != list(SEEDS)
            or protocol["pretraining_updates"] != 20000
            or protocol["calibration_updates"] != 20000
            or protocol["td_updates"] != TD_UPDATES
            or tuple(protocol["td_checkpoints"]) != CHECKPOINTS
            or protocol["learning_rate"] != BASELINE["learning_rate"]
            or protocol["demonstration_weight"] != BASELINE["demonstration_weight"]
            or protocol["margin"] != MARGIN):
        raise ValueError("The reference is not the completed, declared attempt-011 experiment.")
    dataset = read_json(reference / "dataset.json")
    targets = read_json(reference / "mc_targets.json")
    require_hash(reference / "demonstration.npz", dataset["sha256"], "frozen demonstration")
    checked["demonstration.npz"] = dataset["sha256"]
    with np.load(reference / "demonstration.npz", allow_pickle=False) as data:
        batch = TransitionBatch(**{name: torch.from_numpy(data[name].copy())
                                   for name in TransitionBatch.__dataclass_fields__})
    episodes = dataset["episodes"]
    if (len(batch.actions) != 117 or dataset["seeds"] != list(SEEDS)
            or [(row["start"], row["stop"]) for row in episodes] != [(0, 35), (35, 76), (76, 117)]):
        raise ValueError("Expected the same 117 transitions and three episode boundaries.")
    _, returns, rebuilt = build_multiscene_batch([
        slice_transition_batch(batch, row["start"], row["stop"]) for row in episodes
    ], SEEDS, settings["dqn"]["gamma"])
    if (returns.tolist() != targets["returns"] or batch.rewards.tolist() != targets["rewards"]
            or any(any(row[key] != original[key] for key in row)
                   for row, original in zip(rebuilt, episodes))):
        raise ValueError("The actual discounted returns or episode boundaries have changed.")
    calibrated = read_json(reference / "calibration/summary.json")
    baseline = read_json(reference / "preserve/td/summary.json")
    for phase, summary in (("calibration", calibrated), ("preserve/td", baseline)):
        for name, expected in summary["checkpoint_sha256"].items():
            require_hash(reference / phase / name, expected, "reference checkpoint")
            checked[f"{phase}/{name}"] = expected
    snapshot = torch.load(reference / calibrated["training_snapshot_file"],
                          map_location="cpu", weights_only=True)
    optimizer = snapshot["optimizer_summary"]
    if (snapshot["update_count"] != 20000 or optimizer["step_min"] != 20000
            or optimizer["step_max"] != 20000
            or optimizer["initialized_parameter_count"] != optimizer["parameter_count"]
            or snapshot["environment_config"] != settings["environment"]
            or snapshot["dqn_config"] != settings["dqn"]
            or not all(row["rollout"]["success"] for row in calibrated["final_observation"]["scenes"])):
        raise ValueError("All three scenes and all Adam parameter states must be acquired at calibration.")
    reference_observations = read_json(reference / "preserve/td/observations.json")
    reference_updates = read_updates(reference / "preserve/td/updates.jsonl")
    if ([row["update"] for row in reference_observations] != list(CHECKPOINTS)
            or [row["update"] for row in reference_updates] != list(range(1, TD_UPDATES + 1))):
        raise ValueError("Reference logs do not contain every scheduled observation and update.")
    manifest = {
        "reference_run": str(reference), "archive_manifest": relative,
        "archive_manifest_sha256": fit.file_hash(archive_path), "committed_archive_verified": True,
        "committed_archive_manifest_sha256": hashlib.sha256(committed).hexdigest(),
        "archive_comparison": "Exact JSON equality; Git checkout line endings may differ.",
        "reference_files_sha256": checked, "reference_source_sha256": source_hashes,
        "config_sha256": inputs["config_sha256"],
        "helper_sha256": {**inputs["helper_sha256"],
                          REFERENCE_RUNNER.relative_to(ROOT).as_posix(): fit.file_hash(REFERENCE_RUNNER)},
        "inherited_checkpoint_used": True, "inherited_optimizer_updates": INHERITED_UPDATES,
        "calibrated_snapshot_file": calibrated["training_snapshot_file"],
        "calibrated_snapshot_sha256": calibrated["training_snapshot_sha256"],
        "episode_returns_recomputed_exactly": True,
    }
    return {"settings": settings, "dataset": dataset, "batch": batch, "returns": returns,
            "episodes": episodes, "calibrated": calibrated, "snapshot": snapshot,
            "reference_updates": reference_updates, "reference_observations": reference_observations,
            "baseline_summary": baseline, "manifest": manifest}


def validate_factor(factor, value):
    if factor not in BASELINE:
        raise ValueError("The single factor must be learning_rate or demonstration_weight.")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("The treatment value must be a finite scalar.")
    if value <= 0 or value == BASELINE[factor]:
        raise ValueError("The treatment must be positive and different from its baseline.")


def apply_factor(agent, factor, value):
    """Change exactly one scalar after restore, preserving every Adam tensor and model weight."""
    validate_factor(factor, value)
    before = adam.training_snapshot(agent, {})
    settings_before = {"learning_rate": agent.learner.config.learning_rate,
                       "demonstration_weight": BASELINE["demonstration_weight"]}
    if settings_before != BASELINE or agent.learner.update_count != 0:
        raise ValueError("Apply the treatment only at the restored baseline TD boundary.")
    if factor == "learning_rate":
        agent.learner.config = replace(agent.learner.config, learning_rate=float(value))
        agent.learner.optimizer.defaults["lr"] = float(value)
        for group in agent.learner.optimizer.param_groups:
            group["lr"] = float(value)
    after = adam.training_snapshot(agent, {})
    if (not exact_equal(before["optimizer"]["state"], after["optimizer"]["state"])
            or any(not exact_equal(before[key], after[key])
                   for key in ("online", "target", "agent_rng_state", "torch_rng_state", "update_count"))):
        raise RuntimeError("The scalar intervention changed model weights, optimizer state or RNG.")
    expected_config = {**before["dqn_config"]}
    expected_optimizer = deepcopy(before["optimizer"])
    if factor == "learning_rate":
        expected_config["learning_rate"] = float(value)
        for group in expected_optimizer["param_groups"]:
            group["lr"] = float(value)
    if (after["dqn_config"] != expected_config
            or not exact_equal(after["optimizer"], expected_optimizer)):
        raise RuntimeError("The intervention changed another optimizer or learner setting.")
    return {"factor": factor, "value": value, "applied": True,
            "settings_before": settings_before, "settings_after": {**settings_before, factor: value},
            "optimizer_before": before["optimizer_summary"],
            "optimizer_after": after["optimizer_summary"],
            "weights_unchanged": True, "adam_state_unchanged": True}


def compare_record(actual, expected, output, kind, index):
    """Record a failed exact-reproduction gate before stopping the treatment comparison."""
    if actual != expected:
        record = {"passed": False, "kind": kind, "index": index,
                  "actual": actual, "expected": expected, "treatment_started": False}
        fit.save_json(output / "baseline_equivalence.json", record)
        raise RuntimeError(f"Baseline reproduction differs at {kind} {index}; treatment is blocked.")


def execute_td(agent, env, environment, batch, returns, episodes, arm, output,
               expected_start, demonstration_weight, updates=TD_UPDATES,
               checkpoints=CHECKPOINTS, reference_updates=None, reference_observations=None):
    directory = output / arm / "td"
    directory.mkdir(parents=True, exist_ok=False)
    initial_hash = fit.parameter_hash(agent)
    observations, synchronizations = [], 0
    reference_rows = {row["update"]: row for row in reference_observations or []}
    with (directory / "updates.jsonl").open("x", encoding="utf-8") as log:
        for step in range(updates + 1):
            if step:
                metrics = asdict(agent.learner.update(batch, demonstration_weight=demonstration_weight,
                                                      demonstration_margin=MARGIN))
                synchronizations += int(metrics["target_synchronized"])
                record = {"update": step, **metrics}
                log.write(json.dumps(record, allow_nan=False) + "\n")
                if reference_updates is not None:
                    log.flush()
                    compare_record(record, reference_updates[step - 1], output, "update", step)
            if step in checkpoints:
                row = multi.observe(agent, env, environment, batch, returns, episodes,
                                    f"{arm}/td", step, INHERITED_UPDATES, directory)
                observations.append(row)
                fit.save_json(directory / "observations.json", observations)
                torch.save(adam.training_snapshot(agent, environment),
                           directory / f"training_state_{step}.pt")
                if step == 0 and any(row[key] != expected_start[key]
                                     for key in ("fit", "return_fit", "scenes", "aggregate")):
                    raise RuntimeError("The TD boundary changed acquired per-scene behavior.")
                if reference_observations is not None:
                    compare_record(row, reference_rows[step], output, "observation", step)
                log.flush()
            elif step % 100 == 0:
                log.flush()
    snapshot_path = directory / "training_state.pt"
    torch.save(adam.training_snapshot(agent, environment), snapshot_path)
    result = {
        "directory": f"{arm}/td", "initial_parameter_sha256": initial_hash,
        "final_parameter_sha256": fit.parameter_hash(agent),
        "optimizer_updates": agent.learner.update_count,
        "prior_updates_on_branch": INHERITED_UPDATES,
        "total_updates_on_branch": INHERITED_UPDATES + agent.learner.update_count,
        "target_synchronizations": synchronizations,
        "demonstration_weight": demonstration_weight, "learning_rate": agent.learner.config.learning_rate,
        "initial_observation": observations[0], "final_observation": observations[-1],
        "retention": multi.retention_metrics(observations, expected_start),
        "evaluation_environment_steps": sum(row["aggregate"]["evaluation_environment_steps"]
                                            for row in observations),
        "training_snapshot_file": f"{arm}/td/training_state.pt",
        "training_snapshot_sha256": fit.file_hash(snapshot_path),
        "checkpoint_sha256": {path.name: fit.file_hash(path) for path in sorted(directory.glob("*.pt"))},
    }
    fit.save_json(directory / "summary.json", result)
    return result


def write_accounting(output):
    counts = {path.relative_to(output).as_posix(): len(read_updates(path))
              for path in output.rglob("updates.jsonl")}
    rows = [row for path in output.rglob("observations.json") for row in read_json(path)]
    record = {
        "update_records_per_log": counts, "optimizer_updates_total": sum(counts.values()),
        "td_optimizer_updates": sum(counts.values()), "inherited_optimizer_updates": INHERITED_UPDATES,
        "inherited_supervised_updates": 20000, "inherited_calibration_updates": 20000,
        "evaluation_environment_steps": sum(row["aggregate"]["evaluation_environment_steps"] for row in rows),
        "scheduled_evaluation_episodes": sum(len(row["scenes"]) for row in rows),
        "demonstration_collection_steps": 0, "online_training_steps": 0, "held_out_scenes_accessed": 0,
        "note": "Inherited updates are counted once and separately from new paired TD updates.",
    }
    fit.save_json(output / "accounting.json", record)
    return record


def run(output, reference, archive, factor, value):
    validate_factor(factor, value)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite a controlled experiment: {output}")
    inherited = load_reference(reference, archive)
    settings = inherited["settings"]
    protocol = {
        "kind": "diagnosis_informed_single_factor_td_control", "scene_seeds": list(SEEDS),
        "arms": list(ARMS), "factor": factor, "baseline_value": BASELINE[factor],
        "treatment_value": value, "baseline_settings": BASELINE,
        "treatment_settings": {**BASELINE, factor: value}, "margin": MARGIN,
        "td_updates": TD_UPDATES, "td_checkpoints": list(CHECKPOINTS),
        "inherited_optimizer_updates": INHERITED_UPDATES,
        "shared_start": "attempt_011_calibration_training_state",
        "preserve_adam_moments_and_parameter_steps": True,
        "td_boundary": "synchronized_targets_zero_phase_clock_before_scalar_intervention",
        "full_batch_weighting": "equal_per_transition", "effective_batch_size": 117,
        "early_stopping": False, "baseline_gate": "exact_all_updates_observations_and_final_state",
        "retention_eligibility": "All three scenes succeeded at the shared calibration boundary.",
        "script_sha256": fit.file_hash(Path(__file__)),
        "helper_sha256": inherited["manifest"]["helper_sha256"],
        "online_training_steps": 0, "held_out_scenes_accessed": 0,
    }
    output.mkdir(parents=True, exist_ok=False)
    env = None
    try:
        begin_run(output, settings["environment"], settings["dqn"], protocol)
        fit.save_json(output / "input_manifest.json", inherited["manifest"])
        fit.save_json(output / "dataset.json", inherited["dataset"])
        fit.save_json(output / "mc_targets.json", read_json(reference / "mc_targets.json"))
        torch.set_num_threads(1)
        arms, boundaries = {}, {}
        for arm in ARMS:
            (output / arm).mkdir()
            require_hash(reference / inherited["manifest"]["calibrated_snapshot_file"],
                         inherited["manifest"]["calibrated_snapshot_sha256"], "shared calibrated snapshot")
            agent, env = adam.restore_training_snapshot(inherited["snapshot"])
            transition = multi.start_phase(agent, True, inherited["snapshot"]["parameter_sha256"],
                                           INHERITED_UPDATES, output / arm / "transition.json")
            before = optimizer_state_summary(agent.learner)
            intervention = apply_factor(agent, factor, value) if arm == "treatment" else {
                "factor": factor, "value": BASELINE[factor], "applied": False,
                "settings_before": BASELINE, "settings_after": BASELINE,
                "optimizer_before": before, "optimizer_after": before,
                "weights_unchanged": True, "adam_state_unchanged": True,
            }
            boundaries[arm] = {"transition": transition, "intervention": intervention}
            fit.save_json(output / arm / "td_boundary.json", boundaries[arm])
            arms[arm] = execute_td(
                agent, env, settings["environment"], inherited["batch"], inherited["returns"],
                inherited["episodes"], arm, output, inherited["calibrated"]["final_observation"],
                intervention["settings_after"]["demonstration_weight"],
                reference_updates=inherited["reference_updates"] if arm == "baseline" else None,
                reference_observations=inherited["reference_observations"] if arm == "baseline" else None,
            )
            if arm == "baseline":
                previous = torch.load(reference / "preserve/td/training_state.pt",
                                      map_location="cpu", weights_only=True)
                same = exact_equal(adam.training_snapshot(agent, settings["environment"]), previous)
                equivalence = {"passed": same, "exact_update_records": TD_UPDATES,
                               "exact_observations": len(CHECKPOINTS), "exact_final_training_state": same,
                               "treatment_started": False}
                fit.save_json(output / "baseline_equivalence.json", equivalence)
                if not same:
                    raise RuntimeError("Baseline final full snapshot differs; treatment is blocked.")
            env.close()
            env = None
        summary = {
            "protocol": protocol, "dataset": inherited["dataset"], "arms": arms,
            "td_boundaries": boundaries, "baseline_equivalence": read_json(output / "baseline_equivalence.json"),
            **write_accounting(output),
            "limitations": [
                "Three known training scenes and one inherited initialization; no generalization claim.",
                "Fixed privileged demonstrations; no online collection or exploration.",
                "Sampled retention does not establish success at every unsampled update.",
                "The factor was selected from earlier failures; this is a diagnostic, not a held-out test.",
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
    parser.add_argument("--factor", choices=tuple(BASELINE), required=True)
    parser.add_argument("--value", type=float, required=True)
    args = parser.parse_args()
    run(args.output_dir.resolve(), args.reference_run.resolve(), args.archive_dir.resolve(),
        args.factor, args.value)


if __name__ == "__main__":
    main()
