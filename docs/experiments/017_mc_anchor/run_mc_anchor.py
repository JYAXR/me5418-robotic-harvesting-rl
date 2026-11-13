"""Test persistent demonstration-return anchoring against a verified historical control."""

import argparse
import hashlib
import importlib.util
import json
import subprocess
from math import isfinite
from numbers import Real
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from me5418_harvesting.agents.dqn_learning import demonstration_margin_loss, td_loss
from me5418_harvesting.agents.run_record import begin_run, finish_run

ROOT = Path(__file__).resolve().parents[3]
PREVIOUS = ROOT / "docs/experiments/016_local_supervision/run_local_supervision.py"
SPEC = importlib.util.spec_from_file_location("anchor017_local", PREVIOUS)
local = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(local)
extension, adam, fit, multi = local.extension, local.adam, local.fit, local.multi
read_json, read_updates = local.read_json, local.read_updates
require, require_equal, exact_equal = local.require, local.require_equal, local.exact_equal
START, END, MARGIN, NORMALIZATION, ANCHOR_WEIGHT = 2000, 10000, 0.8, 117, 1.0
CHECKPOINTS = local.CHECKPOINTS
TARGET_SOURCE_UPDATES = local.TARGET_SOURCE_UPDATES
SENTINEL_STEPS = local.SENTINEL_STEPS


def validate_anchor(batch, returns, anchor_weight):
    """Reject malformed fixed targets before modifying modes, gradients or optimizer state."""
    if (isinstance(anchor_weight, bool) or not isinstance(anchor_weight, Real)
            or not isfinite(anchor_weight) or anchor_weight < 0):
        raise ValueError("The return anchor weight must be finite and nonnegative.")
    if (not isinstance(returns, torch.Tensor) or returns.ndim != 1 or returns.numel() == 0
            or not returns.is_floating_point() or returns.shape != batch.rewards.shape
            or returns.device != batch.observations.device or returns.device != batch.rewards.device
            or returns.dtype != batch.observations.dtype or not torch.isfinite(returns).all()):
        raise ValueError("Fixed returns must be a finite floating vector matching the batch layout.")
    if (batch.observations.ndim != 2 or len(batch.observations) != len(returns)
            or batch.actions.shape != returns.shape or batch.actions.dtype != torch.long
            or batch.actions.device != returns.device or not torch.isfinite(batch.observations).all()):
        raise ValueError("Current observations and demonstration actions must match the fixed returns.")
    return returns.detach()


def anchored_update(learner, batch, returns, anchor_weight=ANCHOR_WEIGHT):
    """Add one mean return loss to the unchanged one-step TD and action-margin losses."""
    fixed_returns = validate_anchor(batch, returns, anchor_weight)
    learner.online.train()
    learner.optimizer.zero_grad(set_to_none=True)
    before = nn.utils.parameters_to_vector(learner.online.parameters()).detach().clone()
    td_before, chosen_before, targets = td_loss(learner.online, learner.target, batch, learner.config.gamma)
    q_values = learner.online(batch.observations)
    margin_before = demonstration_margin_loss(q_values, batch.actions, MARGIN)
    return_before = F.smooth_l1_loss(chosen_before, fixed_returns)
    accuracy = float((q_values.detach().argmax(dim=1) == batch.actions).float().mean())
    loss = td_before + 1.0 * margin_before
    # Omitting an inactive term preserves the original updater's exact graph.
    if anchor_weight > 0:
        loss = loss + anchor_weight * return_before
    difference = targets.detach().double() - fixed_returns.double()
    target_diagnostics = {
        "td_target_return_rmse": float(difference.square().mean().sqrt()),
        "td_target_return_mean_difference": float(difference.mean()),
        "td_target_return_max_abs_difference": float(difference.abs().max()),
    }
    if not torch.isfinite(loss):
        raise FloatingPointError("The return-anchored TD loss is not finite.")
    loss.backward()
    gradient_norm = nn.utils.clip_grad_norm_(
        learner.online.parameters(), learner.config.gradient_clip_norm, error_if_nonfinite=True,
    )
    learner.optimizer.step()
    with torch.no_grad():
        q_after = learner.online(batch.observations)
        chosen_after = q_after.gather(1, batch.actions[:, None]).squeeze(1)
        td_after = F.smooth_l1_loss(chosen_after, targets)
        margin_after = demonstration_margin_loss(q_after, batch.actions, MARGIN)
        return_after = F.smooth_l1_loss(chosen_after, fixed_returns)
        loss_after = td_after + 1.0 * margin_after
        if anchor_weight > 0:
            loss_after = loss_after + anchor_weight * return_after
        after = nn.utils.parameters_to_vector(learner.online.parameters())
        delta = torch.linalg.vector_norm(after - before)
        if not torch.isfinite(after).all() or not torch.isfinite(loss_after):
            raise FloatingPointError("The return-anchored update produced non-finite values.")
    learner.update_count += 1
    synchronized = learner.update_count % learner.config.target_update_interval == 0
    if synchronized:
        learner.target.load_state_dict(learner.online.state_dict())
    return {
        "loss_before": loss.item(), "loss_after": loss_after.item(),
        "gradient_norm": gradient_norm.item(), "parameter_delta_norm": delta.item(),
        "target_synchronized": synchronized, "td_loss_before": td_before.item(),
        "td_loss_after": td_after.item(), "demonstration_loss_before": margin_before.item(),
        "demonstration_loss_after": margin_after.item(), "demonstration_action_accuracy": accuracy,
        "return_loss_before": return_before.item(), "return_loss_after": return_after.item(),
        "return_anchor_weight": float(anchor_weight), **target_diagnostics,
    }


def verify_archive(reference, archive):
    """Check every archived byte and bind complete local logs to the committed manifest."""
    manifest_path = archive / "archive_manifest.json"
    relative = manifest_path.relative_to(ROOT).as_posix()
    committed = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)
    manifest = read_json(manifest_path)
    require(committed == manifest_path.read_bytes() and manifest["status"] == "completed"
            and Path(manifest["source_run"]) == reference, "committed completed attempt-016 archive")
    files, raw = {}, {}
    for name, record in manifest["files"].items():
        path = archive / name
        require(fit.file_hash(path) == record["sha256"] and path.stat().st_size == record["bytes"],
                f"unchanged archived attempt-016 evidence: {name}")
        files[name] = record["sha256"]
        if not name.endswith("_samples.jsonl"):
            require(fit.file_hash(reference / name) == record["sha256"], f"unchanged raw attempt-016 evidence: {name}")
            raw[name] = record["sha256"]
    require(len(files) == 30, "all 30 archived evidence files plus their committed manifest")
    for name, coverage in manifest["raw_log_coverage"].items():
        path = reference / name
        require(fit.file_hash(path) == coverage["raw_sha256"], f"unchanged complete attempt-016 log: {name}")
        records = read_updates(path)
        first = START if name.endswith("sentinel_q.jsonl") else START + 1
        require([row["update"] for row in records] == list(range(first, END + 1))
                and len(records) == coverage["raw_records"], f"complete historical log clock: {name}")
        raw[name] = coverage["raw_sha256"]
    require(len(manifest["raw_log_coverage"]) == 6, "six complete source logs")
    return {"archive_manifest": relative, "archive_manifest_sha256": fit.file_hash(manifest_path),
            "committed_archive_manifest_sha256": hashlib.sha256(committed).hexdigest(),
            "committed_archive_verified": True, "archived_files_sha256": files,
            "reference_files_sha256": raw}


def load_inputs(reference, archive):
    """Load one complete TD-2000 start and a verified, explicitly historical control."""
    reference, archive = reference.resolve(), archive.resolve()
    archived = verify_archive(reference, archive)
    settings = read_json(reference / "settings.json")
    old_manifest = read_json(reference / "input_manifest.json")
    old_summary = read_json(reference / "summary.json")
    require(read_json(reference / "run_status.json")["status"] == "completed"
            and settings["training"]["attempt"] == 16
            and settings["training"]["baseline_selected_sample_weight"] == 1.0
            and settings["training"]["demonstration_weight"] == 1.0
            and read_json(reference / "reload_checks.json")["passed"], "verified complete attempt-016 source")
    require(fit.file_hash(PREVIOUS) == settings["training"]["script_sha256"], "unchanged attempt-016 runner")
    for name, digest in settings["training"]["helper_sha256"].items():
        require(fit.file_hash(ROOT / name) == digest, f"unchanged source helper: {name}")
    reference014 = Path(old_manifest["reference_run"])
    archive014 = (ROOT / old_manifest["attempt_014"]["archive_manifest"]).parent
    inherited = local.load_inputs(reference014, archive014)
    require(inherited["manifest"] == old_manifest, "unchanged inheritance through the original demonstrations")
    baseline = read_json(reference / "baseline/td/summary.json")
    rows = read_json(reference / "baseline/td/observations.json")
    expected_files = {f"update_{step}.pt" for step in CHECKPOINTS}
    expected_files |= {f"training_state_{step}.pt" for step in CHECKPOINTS} | {"training_state.pt"}
    expected_files |= {f"target_source_state_{step}.pt" for step in TARGET_SOURCE_UPDATES}
    for arm in ("baseline", "treatment"):
        phase = read_json(reference / arm / "td/summary.json")
        require(phase == old_summary["arms"][arm] and set(phase["checkpoint_sha256"]) == expected_files,
                "all 67 historical checkpoint files per arm")
        for name, digest in phase["checkpoint_sha256"].items():
            path = reference / arm / "td" / name
            require(fit.file_hash(path) == digest, f"unchanged historical model: {arm}/{name}")
            if arm == "baseline":
                require_equal(torch.load(path, map_location="cpu", weights_only=True),
                              torch.load(reference014 / "td" / name, map_location="cpu", weights_only=True),
                              f"historical control exactly reproduces attempt 014: {name}")
    require(rows == inherited["reference_rows"] and [row["update"] for row in rows] == list(CHECKPOINTS)
            and read_updates(reference / "baseline/td/updates.jsonl") == read_updates(reference014 / "td/updates.jsonl"),
            "historical control observations and all updates exactly match attempt 014")
    gate = {"passed": True, "exact_update_records": 8000, "exact_observations": 29,
            "exact_checkpoint_payloads": 67, "exact_inherited_sentinel_rows": 8001,
            "treatment_started_before_gate": False}
    require(read_json(reference / "baseline_equivalence.json") == old_summary["baseline_equivalence"] == gate,
            "the recorded original baseline reproduction gate")
    snapshot_path = reference / "baseline/td/training_state_2000.pt"
    snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=True)
    require_equal(snapshot, inherited["snapshot"], "the same full TD-2000 continuation state")
    require(rows[0] == inherited["previous_phase"]["final_observation"], "unchanged successful starting behavior")
    require(read_json(reference / "dataset.json") == inherited["dataset"]
            and read_json(reference / "mc_targets.json") == inherited["targets"], "same data and episodewise actual returns")
    helpers = {**old_manifest["helper_sha256"], PREVIOUS.relative_to(ROOT).as_posix(): fit.file_hash(PREVIOUS)}
    baseline_reference = {
        "reference_run": str(reference), "directory": "baseline/td", "reused_not_rerun": True,
        "return_anchor_weight": 0.0, "demonstration_weight": 1.0, "all_sample_weights": 1.0,
        "optimizer_updates": 8000, "scheduled_evaluation_episodes": 87,
        "evaluation_environment_steps": baseline["evaluation_environment_steps"],
        "new_optimizer_updates": 0, "new_evaluation_episodes": 0, "new_environment_steps": 0,
        "exact_attempt_014_reproduction": gate,
        "observations_sha256": fit.file_hash(reference / "baseline/td/observations.json"),
        "updates_sha256": fit.file_hash(reference / "baseline/td/updates.jsonl"),
        "sentinel_sha256": fit.file_hash(reference / "baseline/td/sentinel_q.jsonl"),
        "summary_sha256": fit.file_hash(reference / "baseline/td/summary.json"),
        "checkpoint_sha256": baseline["checkpoint_sha256"],
    }
    require(baseline_reference["evaluation_environment_steps"] == 3711, "historical control action accounting")
    manifest = {
        "reference_run": str(reference), "attempt_016": archived, "inherited_manifest": old_manifest,
        "reference_014_run": str(reference014),
        "reference_013_run": old_manifest["attempt_013"]["reference_run"],
        "original_data_run": old_manifest["original_data_run"], "dataset_sha256": old_manifest["dataset_sha256"],
        "resumed_snapshot_file": "baseline/td/training_state_2000.pt",
        "resumed_snapshot_sha256": fit.file_hash(snapshot_path),
        "inherited_optimizer_updates": 42000, "inherited_td_updates": START,
        "helper_sha256": helpers, "production_source_sha256": old_manifest["production_source_sha256"],
    }
    return {**inherited, "snapshot": snapshot, "manifest": manifest, "baseline_reference": baseline_reference,
            "reference_rows": rows, "reference_phase": baseline}


def make_protocol(manifest):
    """Resolve the fixed experiment before any optimizer or environment action."""
    return {
        "attempt": 17, "kind": "persistent_demonstration_return_anchor",
        "scene_seeds": [10000, 10001, 10002], "initial_td_update": START, "final_td_update": END,
        "new_treatment_optimizer_updates": END - START, "td_checkpoints": list(CHECKPOINTS),
        "target_source_updates": list(TARGET_SOURCE_UPDATES), "inherited_shared_optimizer_updates": 42000,
        "learning_rate": 0.0001, "gamma": 0.99, "gradient_clip_norm": 10.0,
        "demonstration_weight": 1.0, "all_sample_weights": 1.0, "margin": MARGIN,
        "return_anchor_weight": ANCHOR_WEIGHT, "baseline_return_anchor_weight": 0.0,
        "target_update_interval": 200, "effective_batch_size": NORMALIZATION,
        "normalization_denominator": NORMALIZATION,
        "objective": "mean TD Huber + mean demonstration hinge + mean fixed-return Huber",
        "return_targets": "Original episodewise actual discounted demonstration returns; detached and unchanged.",
        "total_loss_divided_by_three": False, "td_targets_mixed_with_returns": False,
        "historical_control_reused_not_rerun": True, "historical_control_arm": "attempt_016/baseline",
        "preserve_adam_moments_and_parameter_steps": True, "preserve_absolute_td_counter": True,
        "synchronize_target_at_resume": False, "early_stopping": False,
        "sentinel_seed": 10001, "sentinel_reference_steps": list(SENTINEL_STEPS),
        "sentinel_batch_size": 1, "sentinel_updates": "every absolute update from 2000 through 10000 inclusive",
        "sentinel_gap": "Q(demonstrated action) - max Q(other actions)",
        "sentinel_is_physical_evaluation": False, "online_training_steps": 0, "held_out_scenes_accessed": 0,
        "update_accuracy_semantics": "Demonstration action accuracy is measured before the update.",
        "update_target_semantics": "Before/after TD losses use the same frozen pre-update Bellman targets.",
        "script_sha256": fit.file_hash(Path(__file__)), "helper_sha256": manifest["helper_sha256"],
    }


def execute_treatment(agent, env, environment, batch, returns, episodes, output, expected_start,
                      start=START, final=END, checkpoints=CHECKPOINTS, reference_steps=SENTINEL_STEPS):
    """Continue the original absolute clock with one added fixed-return objective."""
    require(agent.learner.update_count == start and tuple(sorted(set(checkpoints))) == tuple(checkpoints)
            and checkpoints[0] == start and checkpoints[-1] == final and start < final,
            "declared continuation clock and evaluation schedule")
    directory = output / "treatment/td"
    directory.mkdir(parents=True, exist_ok=False)
    target_sources = tuple(step for step in extension.target_source_updates(
        checkpoints, agent.learner.config.target_update_interval) if step >= start)
    initial_hash, rows, syncs = fit.parameter_hash(agent), [], 0
    print(f"Starting treatment: TD {start} to {final}, return anchor weight {ANCHOR_WEIGHT}.", flush=True)
    with (directory / "updates.jsonl").open("x", encoding="utf-8") as updates, (
        directory / "sentinel_q.jsonl"
    ).open("x", encoding="utf-8") as sentinels:
        for step in range(start, final + 1):
            if step > start:
                metrics = anchored_update(agent.learner, batch, returns)
                require(agent.learner.update_count == step, "preserved absolute TD clock")
                syncs += int(metrics["target_synchronized"])
                updates.write(json.dumps({"update": step, "extension_update": step - start, **metrics},
                                         allow_nan=False) + "\n")
            probe = extension.sentinel_record(agent, batch, episodes, step, start, reference_steps)
            sentinels.write(json.dumps(probe, allow_nan=False) + "\n")
            if step in checkpoints:
                row = multi.observe(agent, env, environment, batch, returns, episodes,
                                    "extension/td", step, 40000, directory)
                rows.append(row)
                fit.save_json(directory / "observations.json", rows)
                torch.save(adam.training_snapshot(agent, environment), directory / f"training_state_{step}.pt")
                if step == start:
                    passed = row == expected_start
                    fit.save_json(output / "treatment/initial_equivalence.json", {
                        "passed": passed, "exact_previous_final_observation": passed,
                        "absolute_td_update": start, "new_updates_before_gate": 0,
                    })
                    require(passed, "the treatment starts from exact source behavior before any new update")
                updates.flush()
                sentinels.flush()
            elif step in target_sources:
                torch.save(adam.training_snapshot(agent, environment), directory / f"target_source_state_{step}.pt")
            if step % 100 == 0:
                updates.flush()
                sentinels.flush()
    snapshot_path = directory / "training_state.pt"
    torch.save(adam.training_snapshot(agent, environment), snapshot_path)
    result = {
        "directory": "treatment/td", "initial_parameter_sha256": initial_hash,
        "final_parameter_sha256": fit.parameter_hash(agent), "optimizer_updates": final - start,
        "initial_td_update": start, "final_td_update": final, "prior_updates_on_branch": 40000,
        "inherited_optimizer_updates": 40000 + start, "total_updates_on_branch": 40000 + final,
        "target_synchronizations": syncs, "target_source_updates": list(target_sources),
        "demonstration_weight": 1.0, "all_sample_weights": 1.0, "return_anchor_weight": ANCHOR_WEIGHT,
        "normalization_denominator": len(batch.actions), "learning_rate": agent.learner.config.learning_rate,
        "initial_observation": rows[0], "final_observation": rows[-1],
        "retention": extension.retention_metrics(rows, expected_start, start),
        "evaluation_environment_steps": sum(row["aggregate"]["evaluation_environment_steps"] for row in rows),
        "sentinel_records": final - start + 1, "sentinel_environment_steps": 0,
        "training_snapshot_file": "treatment/td/training_state.pt", "training_snapshot_sha256": fit.file_hash(snapshot_path),
        "checkpoint_sha256": {path.name: fit.file_hash(path) for path in sorted(directory.glob("*.pt"))},
    }
    fit.save_json(directory / "summary.json", result)
    return result


def write_accounting(output, baseline_reference):
    """Keep completed new work separate from reused training and evaluation evidence."""
    directory = output / "treatment/td"
    updates = read_updates(directory / "updates.jsonl") if (directory / "updates.jsonl").exists() else []
    rows = read_json(directory / "observations.json") if (directory / "observations.json").exists() else []
    sentinels = sum(1 for _ in (directory / "sentinel_q.jsonl").open(encoding="utf-8")) if (
        directory / "sentinel_q.jsonl").exists() else 0
    result = {
        "optimizer_updates_total": len(updates), "inherited_shared_optimizer_updates": 42000,
        "absolute_td_update": START + len(updates), "total_updates_on_treatment_branch": 42000 + len(updates),
        "scheduled_evaluation_episodes": sum(len(row["scenes"]) for row in rows),
        "evaluation_environment_steps": sum(row["aggregate"]["evaluation_environment_steps"] for row in rows),
        "sentinel_records": sentinels, "sentinel_forward_passes": sentinels * len(SENTINEL_STEPS),
        "sentinel_environment_steps": 0, "online_training_steps": 0, "demonstration_collection_steps": 0,
        "held_out_scenes_accessed": 0,
        "reused_historical_control": {key: baseline_reference[key] for key in (
            "optimizer_updates", "scheduled_evaluation_episodes", "evaluation_environment_steps",
            "new_optimizer_updates", "new_evaluation_episodes", "new_environment_steps")},
        "note": "Only completed new treatment work is counted; historical control and inherited training are reused. Independent verification is separate.",
    }
    fit.save_json(output / "accounting.json", result)
    return result


def run(reference, archive, output):
    reference, archive, output = reference.resolve(), archive.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite return-anchor evidence: {output}")
    torch.set_num_threads(1)
    inherited = load_inputs(reference, archive)
    snapshot, baseline_reference = inherited["snapshot"], inherited["baseline_reference"]
    protocol = make_protocol(inherited["manifest"])
    output.mkdir(parents=True, exist_ok=False)
    env = None
    try:
        begin_run(output, snapshot["environment_config"], snapshot["dqn_config"], protocol)
        for name, value in (("input_manifest", inherited["manifest"]), ("dataset", inherited["dataset"]),
                            ("mc_targets", inherited["targets"]), ("baseline_reference", baseline_reference)):
            fit.save_json(output / f"{name}.json", value)
        agent, env = extension.restore_continuation(snapshot)
        (output / "treatment").mkdir()
        fit.save_json(output / "treatment/restore_checks.json", {
            "exact_full_snapshot": True, "target_synchronized_at_restore": False,
            "optimizer_reset_at_restore": False, "td_counter_reset_at_restore": False,
            "absolute_td_update": START, "optimizer_state": snapshot["optimizer_summary"],
        })
        phase = execute_treatment(agent, env, snapshot["environment_config"], inherited["batch"],
                                  inherited["returns"], inherited["episodes"], output, inherited["reference_rows"][0])
        env.close()
        env = None
        checked = load_inputs(reference, archive)
        require(checked["manifest"] == inherited["manifest"] and checked["baseline_reference"] == baseline_reference,
                "all immutable source inputs and historical control remain unchanged")
        summary = {"protocol": protocol, "dataset": inherited["dataset"], "treatment": phase,
                   "historical_control": baseline_reference, "accounting": write_accounting(output, baseline_reference),
                   "limitations": [
                       "One inherited initialization and three known demonstration scenes; no held-out evaluation.",
                       "The control is previously verified historical evidence and was not rerun in this main experiment.",
                       "Actual demonstrated returns describe the demonstration continuation, not guaranteed optimal values.",
                       "Reference-state Q rankings do not establish physical outcomes at unsampled updates.",
                       "All new physical evaluations use the autonomous greedy policy without forced actions.",
                   ]}
        fit.save_json(output / "summary.json", summary)
    except BaseException as error:
        write_accounting(output, baseline_reference)
        if (output / "run_status.json").exists():
            finish_run(output, "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error)
        raise
    else:
        finish_run(output, "completed")
        return summary
    finally:
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.reference_run, args.archive_dir, args.output_dir)


if __name__ == "__main__":
    main()
