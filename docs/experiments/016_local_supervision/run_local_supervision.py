"""Compare one doubled demonstration coefficient with an exact continuation control."""

import argparse
import hashlib
import importlib.util
import json
import subprocess
from dataclasses import asdict
from math import isfinite
from numbers import Real
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from me5418_harvesting.agents.dqn_learning import UpdateStats, td_loss
from me5418_harvesting.agents.run_record import begin_run, finish_run

ROOT = Path(__file__).resolve().parents[3]
EXTENSION = ROOT / "docs/experiments/014_td_extension/run_td_extension.py"
RESCUE = ROOT / "docs/experiments/015_single_action_rescue/run_single_action_rescue.py"
DIAGNOSIS_ARCHIVE = ROOT / "docs/experiments/015_single_action_rescue/evidence"


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


extension = import_file("local016_extension", EXTENSION)
rescue = import_file("local016_rescue", RESCUE)
adam, fit, multi = extension.adam, extension.fit, extension.multi
read_json, read_updates, exact_equal = extension.read_json, extension.read_updates, extension.exact_equal
require, require_equal = rescue.require, rescue.require_equal
START, END, ROW, SCENE, STEP, ACTION = 2000, 10000, 55, 10001, 21, 0
CHECKPOINTS = extension.CHECKPOINTS
TARGET_SOURCE_UPDATES = extension.TARGET_SOURCE_UPDATES
SENTINEL_STEPS = (6, 18, 20, 21, 22, 23, 24)
ARM_WEIGHTS = {"baseline": 1.0, "treatment": 2.0}
MARGIN, NORMALIZATION = 0.8, 117


def margin_terms(q_values, actions, margin=MARGIN):
    """Return the original per-row hinge before any averaging or weighting."""
    if (q_values.ndim != 2 or len(q_values) == 0 or actions.ndim != 1
            or len(actions) != len(q_values)):
        raise ValueError("Expected nonempty Q rows and one demonstration action per row.")
    offsets = torch.full_like(q_values, margin)
    offsets.scatter_(1, actions[:, None], 0.0)
    chosen = q_values.gather(1, actions[:, None]).squeeze(1)
    return (q_values + offsets).max(dim=1).values - chosen


def weighted_margin_loss(q_values, actions, selected_row, selected_weight):
    """Change one coefficient; retain the original batch-size denominator."""
    if (isinstance(selected_row, bool) or not isinstance(selected_row, int)
            or not 0 <= selected_row < len(actions)):
        raise ValueError("The selected demonstration row must be a valid integer index.")
    if (isinstance(selected_weight, bool) or not isinstance(selected_weight, Real)
            or not isfinite(selected_weight) or selected_weight <= 0):
        raise ValueError("The selected supervision coefficient must be finite and positive.")
    terms = margin_terms(q_values, actions)
    weights = torch.ones_like(terms)
    weights[selected_row] = selected_weight
    return (terms * weights).mean()


def weighted_update(learner, batch, selected_row=ROW, selected_weight=2.0):
    """Mirror the frozen DQN update, changing only one supervised hinge coefficient."""
    # Validate before mutating training flags, gradients, or optimizer state.
    if (isinstance(selected_row, bool) or not isinstance(selected_row, int)
            or not 0 <= selected_row < len(batch.actions)):
        raise ValueError("The selected demonstration row must be a valid integer index.")
    if (isinstance(selected_weight, bool) or not isinstance(selected_weight, Real)
            or not isfinite(selected_weight) or selected_weight <= 0):
        raise ValueError("The selected supervision coefficient must be finite and positive.")
    learner.online.train()
    learner.optimizer.zero_grad(set_to_none=True)
    parameters_before = nn.utils.parameters_to_vector(learner.online.parameters()).detach().clone()
    td_before, _, targets = td_loss(learner.online, learner.target, batch, learner.config.gamma)
    q_values = learner.online(batch.observations)
    demonstration_before = weighted_margin_loss(q_values, batch.actions, selected_row, selected_weight)
    accuracy = float((q_values.detach().argmax(dim=1) == batch.actions).float().mean())
    loss = td_before + demonstration_before
    if not torch.isfinite(loss):
        raise FloatingPointError("The locally weighted DQN loss is not finite.")
    loss.backward()
    gradient_norm = nn.utils.clip_grad_norm_(
        learner.online.parameters(), learner.config.gradient_clip_norm, error_if_nonfinite=True,
    )
    learner.optimizer.step()
    with torch.no_grad():
        q_values_after = learner.online(batch.observations)
        chosen_q = q_values_after.gather(1, batch.actions[:, None]).squeeze(1)
        td_after = F.smooth_l1_loss(chosen_q, targets)
        demonstration_after = weighted_margin_loss(q_values_after, batch.actions, selected_row, selected_weight)
        loss_after = td_after + demonstration_after
        parameters_after = nn.utils.parameters_to_vector(learner.online.parameters())
        delta = torch.linalg.vector_norm(parameters_after - parameters_before)
        if not torch.isfinite(parameters_after).all() or not torch.isfinite(loss_after):
            raise FloatingPointError("The locally weighted update produced non-finite values.")
    learner.update_count += 1
    synchronized = learner.update_count % learner.config.target_update_interval == 0
    if synchronized:
        learner.target.load_state_dict(learner.online.state_dict())
    return UpdateStats(
        loss.item(), loss_after.item(), gradient_norm.item(), delta.item(), synchronized,
        td_before.item(), td_after.item(), demonstration_before.item(), demonstration_after.item(), accuracy,
    )


def supervision_probe(agent, batch, selected_row=ROW, selected_weight=1.0):
    """Measure raw and weighted hinge terms without changing persistent training state."""
    modes = [(module, module.training) for module in agent.learner.online.modules()]
    try:
        agent.learner.online.eval()
        with torch.no_grad():
            values = agent.learner.online(batch.observations)
            terms = margin_terms(values, batch.actions)
            weighted = weighted_margin_loss(values, batch.actions, selected_row, selected_weight)
            return {"unweighted_margin": float(terms.mean()), "weighted_margin": float(weighted),
                    "selected_hinge": float(terms[selected_row])}
    finally:
        for module, mode in modes:
            module.training = mode


def verify_diagnosis(reference):
    """Bind the local-weight motivation to the already committed four-episode diagnosis."""
    archive = DIAGNOSIS_ARCHIVE
    manifest_path = archive / "archive_manifest.json"
    relative = manifest_path.relative_to(ROOT).as_posix()
    committed = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)
    manifest = read_json(manifest_path)
    require(json.loads(committed) == manifest and manifest["status"] == "completed",
            "committed completed attempt-015 diagnosis")
    raw = Path(manifest["source_run"])
    files = {}
    for name, record in manifest["files"].items():
        rescue.diagnostic.require_hash(archive / name, record["sha256"])
        rescue.diagnostic.require_hash(raw / name, record["sha256"])
        files[name] = record["sha256"]
    settings = read_json(raw / "settings.json")
    summary = read_json(raw / "summary.json")
    require(read_json(raw / "verification.json")["passed"]
            and summary["tested_checkpoint_count"] == summary["rescued_checkpoint_count"] == 2
            and summary["td_updates"] == [2009, 10000]
            and Path(read_json(raw / "input_manifest.json")["reference_run"]) == reference,
            "the two step-21 rescues diagnose this exact reference extension")
    require(settings["training"]["intervention_step"] == STEP
            and settings["training"]["intervention_action"] == ACTION
            and fit.file_hash(RESCUE) == settings["training"]["script_sha256"],
            "the unchanged fixed-step diagnosis runner")
    fork_hashes = set()
    for update in (2009, 10000):
        for mode in ("greedy", "counterfactual"):
            trace = read_json(raw / f"update_{update}_{mode}.json")
            fork = trace["records"][STEP - 1]
            require(fork["step"] == STEP and fork["policy_action"] == 4,
                    "the diagnosed step-21 decision state")
            fork_hashes.add(fork["encoded_observation_sha256"])
    require(len(fork_hashes) == 1, "all four diagnostic forks share the same encoded input")
    return {"archive_manifest": relative, "archive_manifest_sha256": fit.file_hash(manifest_path),
            "committed_archive_manifest_sha256": hashlib.sha256(committed).hexdigest(),
            "reference_run": str(raw), "reference_files_sha256": files,
            "committed_archive_verified": True,
            "selected_observation_sha256": next(iter(fork_hashes))}


def selected_identity(batch, episode, expected_hash):
    """Bind the selected row to the actual shared input from the rescue experiment."""
    require(episode["seed"] == SCENE and episode["start"] + STEP - 1 == ROW
            and episode["start"] == 35 and episode["stop"] == 76
            and len(batch.actions) == NORMALIZATION and int(batch.actions[ROW]) == ACTION,
            "the sole reweighted transition is scene 10001 step 21 at global row 55")
    digest = rescue.diagnostic.array_hash(batch.observations[ROW].cpu().numpy())
    require(digest == expected_hash, "the selected encoded state exactly matches all attempt-015 forks")
    return {"scene_seed": SCENE, "reference_step": STEP,
            "global_transition_index": ROW, "demonstrated_action": ACTION,
            "encoded_observation_sha256": digest}


def load_inputs(reference, archive):
    """Reuse exact TD-2000 state and frozen data, retaining both diagnostic source anchors."""
    source = rescue.load_inputs(reference, archive)
    old_manifest = read_json(reference / "input_manifest.json")
    previous_run = Path(old_manifest["reference_run"])
    previous_archive = ROOT / old_manifest["attempt_013"]["archive_manifest"]
    inherited = extension.load_inputs(previous_run, previous_archive.parent)
    require(inherited["manifest"] == old_manifest, "the exact original attempt-014 resume chain")
    diagnosis = verify_diagnosis(reference)
    episode = next(row for row in inherited["episodes"] if row["seed"] == SCENE)
    batch = inherited["batch"]
    selected = selected_identity(batch, episode, diagnosis["selected_observation_sha256"])
    reference_rows = read_json(reference / "td/observations.json")
    require(reference_rows[0] == inherited["previous_phase"]["final_observation"]
            and [row["update"] for row in reference_rows] == list(CHECKPOINTS),
            "the reference begins at the exact successful inherited TD-2000 state")
    old_resume = torch.load(reference / "td/training_state_2000.pt", map_location="cpu", weights_only=True)
    require_equal(old_resume, inherited["snapshot"], "unchanged full TD-2000 source snapshot")
    helpers = {**source["manifest"]["helper_sha256"],
               EXTENSION.relative_to(ROOT).as_posix(): fit.file_hash(EXTENSION),
               RESCUE.relative_to(ROOT).as_posix(): fit.file_hash(RESCUE)}
    manifest = {
        "reference_run": str(reference), "attempt_014": source["manifest"],
        "attempt_013": inherited["manifest"], "attempt_015": diagnosis,
        "resumed_snapshot_file": "td/training_state_2000.pt",
        "resumed_snapshot_sha256": fit.file_hash(reference / "td/training_state_2000.pt"),
        "original_data_run": source["manifest"]["original_data_run"],
        "dataset_sha256": source["manifest"]["dataset_sha256"],
        "inherited_optimizer_updates": 42000, "inherited_td_updates": START,
        "selected_transition": selected,
        "helper_sha256": helpers, "production_source_sha256": source["manifest"]["source_production_sha256"],
    }
    return {**inherited, "manifest": manifest, "reference_rows": reference_rows,
            "reference_phase": read_json(reference / "td/summary.json")}


def execute_arm(agent, env, environment, batch, returns, episodes, output, arm, expected_start,
                start=START, final=END, checkpoints=CHECKPOINTS, selected_row=ROW,
                reference_steps=SENTINEL_STEPS):
    """Run an unchanged absolute TD clock with either original or locally weighted supervision."""
    require(arm in ARM_WEIGHTS and agent.learner.update_count == start
            and tuple(sorted(set(checkpoints))) == tuple(checkpoints)
            and checkpoints[0] == start and checkpoints[-1] == final and start < final,
            "declared arm, continuation clock and evaluation schedule")
    selected_weight = ARM_WEIGHTS[arm]
    directory = output / arm / "td"
    directory.mkdir(parents=True, exist_ok=False)
    target_sources = tuple(step for step in extension.target_source_updates(
        checkpoints, agent.learner.config.target_update_interval) if step >= start)
    initial_hash, rows, syncs = fit.parameter_hash(agent), [], 0
    print(f"Starting {arm}: TD {start} to {final}, selected supervision weight {selected_weight}.", flush=True)
    with (directory / "updates.jsonl").open("x", encoding="utf-8") as updates, (
        directory / "supervision.jsonl"
    ).open("x", encoding="utf-8") as supervision, (
        directory / "sentinel_q.jsonl"
    ).open("x", encoding="utf-8") as sentinels:
        for step in range(start, final + 1):
            if step > start:
                before = supervision_probe(agent, batch, selected_row, selected_weight)
                stats = (agent.learner.update(batch, demonstration_weight=1.0, demonstration_margin=MARGIN)
                         if arm == "baseline" else weighted_update(agent.learner, batch, selected_row, selected_weight))
                after = supervision_probe(agent, batch, selected_row, selected_weight)
                require(agent.learner.update_count == step, "preserved absolute TD clock")
                metrics = asdict(stats)
                syncs += int(stats.target_synchronized)
                updates.write(json.dumps({"update": step, "extension_update": step - start, **metrics},
                                         allow_nan=False) + "\n")
                record = {"update": step, "extension_update": step - start,
                          "selected_sample_weight": selected_weight, "normalization_denominator": len(batch.actions)}
                record.update({f"{name}_{when}": value for when, values in (("before", before), ("after", after))
                               for name, value in values.items()})
                supervision.write(json.dumps(record, allow_nan=False) + "\n")
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
                    fit.save_json(output / arm / "initial_equivalence.json", {
                        "passed": passed, "exact_previous_final_observation": passed,
                        "absolute_td_update": start, "new_updates_before_gate": 0,
                    })
                    require(passed, "each arm starts from exact source behavior before any new update")
                updates.flush()
                supervision.flush()
                sentinels.flush()
            elif step in target_sources:
                torch.save(adam.training_snapshot(agent, environment), directory / f"target_source_state_{step}.pt")
            if step % 100 == 0:
                updates.flush()
                supervision.flush()
                sentinels.flush()
    snapshot_path = directory / "training_state.pt"
    torch.save(adam.training_snapshot(agent, environment), snapshot_path)
    result = {
        "directory": f"{arm}/td", "initial_parameter_sha256": initial_hash,
        "final_parameter_sha256": fit.parameter_hash(agent), "optimizer_updates": final - start,
        "initial_td_update": start, "final_td_update": final, "prior_updates_on_branch": 40000,
        "inherited_optimizer_updates": 40000 + start, "total_updates_on_branch": 40000 + final,
        "target_synchronizations": syncs, "target_source_updates": list(target_sources),
        "demonstration_weight": 1.0, "selected_sample_weight": selected_weight,
        "selected_transition_index": selected_row, "normalization_denominator": len(batch.actions),
        "learning_rate": agent.learner.config.learning_rate,
        "initial_observation": rows[0], "final_observation": rows[-1],
        "retention": extension.retention_metrics(rows, expected_start, start),
        "evaluation_environment_steps": sum(row["aggregate"]["evaluation_environment_steps"] for row in rows),
        "sentinel_records": final - start + 1, "sentinel_environment_steps": 0,
        "supervision_records": final - start,
        "training_snapshot_file": f"{arm}/td/training_state.pt",
        "training_snapshot_sha256": fit.file_hash(snapshot_path),
        "checkpoint_sha256": {path.name: fit.file_hash(path) for path in sorted(directory.glob("*.pt"))},
    }
    fit.save_json(directory / "summary.json", result)
    return result


def compare_baseline(output, reference):
    """Gate treatment on exact reproduction of all historical baseline training and behavior."""
    directory, source = output / "baseline/td", reference / "td"
    observations = read_json(directory / "observations.json")
    require(observations == read_json(source / "observations.json"), "all 29 baseline observations reproduce attempt 014")
    records = read_updates(directory / "updates.jsonl")
    require(records == read_updates(source / "updates.jsonl"), "all 8000 baseline update logs reproduce attempt 014")
    phase = read_json(directory / "summary.json")
    old = read_json(source / "summary.json")
    require(set(phase["checkpoint_sha256"]) == set(old["checkpoint_sha256"]), "the same 67 baseline checkpoint files")
    for name in phase["checkpoint_sha256"]:
        current = torch.load(directory / name, map_location="cpu", weights_only=True)
        previous = torch.load(source / name, map_location="cpu", weights_only=True)
        require_equal(current, previous, f"exact baseline checkpoint payload: {name}")
    baseline_probes, previous_probes = read_updates(directory / "sentinel_q.jsonl"), read_updates(source / "sentinel_q.jsonl")
    for current, previous in zip(baseline_probes, previous_probes, strict=True):
        common = {**current, "states": [state for state in current["states"]
                                       if state["reference_step"] in extension.SENTINEL_STEPS]}
        require(common == previous, "all inherited four-state baseline Q probes reproduce attempt 014")
    result = {"passed": True, "exact_update_records": len(records), "exact_observations": len(observations),
              "exact_checkpoint_payloads": len(phase["checkpoint_sha256"]),
              "exact_inherited_sentinel_rows": len(baseline_probes), "treatment_started_before_gate": False}
    fit.save_json(output / "baseline_equivalence.json", result)
    return result


def write_accounting(output):
    arms = {}
    for arm in ARM_WEIGHTS:
        directory = output / arm / "td"
        updates = read_updates(directory / "updates.jsonl") if (directory / "updates.jsonl").exists() else []
        rows = read_json(directory / "observations.json") if (directory / "observations.json").exists() else []
        sentinels = sum(1 for _ in (directory / "sentinel_q.jsonl").open(encoding="utf-8")) if (
            directory / "sentinel_q.jsonl").exists() else 0
        arms[arm] = {"optimizer_updates": len(updates), "inherited_optimizer_updates": 42000,
                     "absolute_td_update": START + len(updates), "total_updates_on_branch": 42000 + len(updates),
                     "scheduled_evaluation_episodes": sum(len(row["scenes"]) for row in rows),
                     "evaluation_environment_steps": sum(row["aggregate"]["evaluation_environment_steps"] for row in rows),
                     "sentinel_records": sentinels, "sentinel_forward_passes": sentinels * len(SENTINEL_STEPS)}
    result = {"arms": arms, "optimizer_updates_total": sum(row["optimizer_updates"] for row in arms.values()),
              "inherited_shared_optimizer_updates": 42000,
              "scheduled_evaluation_episodes": sum(row["scheduled_evaluation_episodes"] for row in arms.values()),
              "evaluation_environment_steps": sum(row["evaluation_environment_steps"] for row in arms.values()),
              "sentinel_records": sum(row["sentinel_records"] for row in arms.values()),
              "sentinel_forward_passes": sum(row["sentinel_forward_passes"] for row in arms.values()),
              "sentinel_environment_steps": 0, "online_training_steps": 0,
              "demonstration_collection_steps": 0, "held_out_scenes_accessed": 0,
              "note": "Completed updates and completed physical checkpoint evaluations only; shared inherited training is counted once, and independent verification is separate."}
    fit.save_json(output / "accounting.json", result)
    return result


def run(reference, archive, output):
    reference, archive, output = reference.resolve(), archive.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite local-supervision evidence: {output}")
    inherited = load_inputs(reference, archive)
    snapshot = inherited["snapshot"]
    protocol = {
        "attempt": 16, "kind": "single_transition_supervision_weight_control",
        "scene_seeds": [10000, 10001, 10002], "initial_td_update": START, "final_td_update": END,
        "extension_updates_per_arm": END - START, "td_checkpoints": list(CHECKPOINTS),
        "target_source_updates": list(TARGET_SOURCE_UPDATES), "inherited_shared_optimizer_updates": 42000,
        "learning_rate": 0.0001, "demonstration_weight": 1.0, "margin": MARGIN,
        "target_update_interval": 200, "effective_batch_size": NORMALIZATION,
        "selected_scene_seed": SCENE, "selected_reference_step": STEP,
        "selected_transition_index": ROW, "selected_demonstrated_action": ACTION,
        "baseline_selected_sample_weight": 1.0, "treatment_selected_sample_weight": 2.0,
        "other_sample_weights": 1.0, "normalization_denominator": NORMALIZATION,
        "normalization_rule": "sum of weighted per-row hinges divided by unchanged batch size 117",
        "td_sample_weights_changed": False, "demonstrations_duplicated": False,
        "preserve_adam_moments_and_parameter_steps": True, "preserve_absolute_td_counter": True,
        "synchronize_target_at_resume": False, "early_stopping": False,
        "baseline_must_exactly_reproduce_attempt_014_before_treatment": True,
        "sentinel_seed": SCENE, "sentinel_reference_steps": list(SENTINEL_STEPS),
        "sentinel_batch_size": 1, "sentinel_updates": "every absolute update from 2000 through 10000 inclusive",
        "sentinel_gap": "Q(demonstrated action) - max Q(other actions)",
        "sentinel_is_physical_evaluation": False, "online_training_steps": 0, "held_out_scenes_accessed": 0,
        "update_demonstration_loss_semantics": "Weighted mean hinge with fixed denominator; unweighted and selected-row terms are in supervision.jsonl.",
        "script_sha256": fit.file_hash(Path(__file__)), "helper_sha256": inherited["manifest"]["helper_sha256"],
    }
    output.mkdir(parents=True, exist_ok=False)
    env = None
    try:
        begin_run(output, snapshot["environment_config"], snapshot["dqn_config"], protocol)
        fit.save_json(output / "input_manifest.json", inherited["manifest"])
        fit.save_json(output / "dataset.json", inherited["dataset"])
        fit.save_json(output / "mc_targets.json", inherited["targets"])
        fit.save_json(output / "supervision_weights.json", {
            "selected_transition": inherited["manifest"]["selected_transition"],
            "normalization_denominator": NORMALIZATION,
            "arms": {arm: [weight if index == ROW else 1.0 for index in range(NORMALIZATION)]
                     for arm, weight in ARM_WEIGHTS.items()},
            "td_weights_changed": False,
        })
        torch.set_num_threads(1)
        arms, equivalence = {}, None
        for arm in ARM_WEIGHTS:
            agent, env = extension.restore_continuation(snapshot)
            (output / arm).mkdir()
            fit.save_json(output / arm / "restore_checks.json", {
                "exact_full_snapshot": True, "target_synchronized_at_restore": False,
                "optimizer_reset_at_restore": False, "td_counter_reset_at_restore": False,
                "absolute_td_update": START, "optimizer_state": snapshot["optimizer_summary"],
            })
            arms[arm] = execute_arm(agent, env, snapshot["environment_config"], inherited["batch"],
                                     inherited["returns"], inherited["episodes"], output, arm,
                                     inherited["previous_phase"]["final_observation"])
            env.close()
            env = None
            write_accounting(output)
            if arm == "baseline":
                equivalence = compare_baseline(output, reference)
        require(load_inputs(reference, archive)["manifest"] == inherited["manifest"],
                "all immutable source inputs remain unchanged after both arms")
        summary = {"protocol": protocol, "dataset": inherited["dataset"], "arms": arms,
                   "baseline_equivalence": equivalence, "accounting": write_accounting(output),
                   "limitations": [
                       "One inherited initialization and three known demonstration scenes; no held-out evaluation.",
                       "Only the supervised coefficient of one selected demonstrated state differs.",
                       "Reference-state Q rankings do not establish physical outcomes at unsampled updates.",
                       "All physical evaluations use the autonomous greedy policy without forced actions.",
                   ]}
        fit.save_json(output / "summary.json", summary)
    except BaseException as error:
        write_accounting(output)
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
