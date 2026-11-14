"""Continue a verified baseline with only its TD objective coefficient reduced."""

import argparse
import hashlib
import importlib.util
import json
from copy import deepcopy
from math import isfinite
from numbers import Real
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from me5418_harvesting.agents.dqn_learning import demonstration_margin_loss, td_loss
from me5418_harvesting.agents.run_record import begin_run, finish_run

ROOT = Path(__file__).resolve().parents[3]
PREVIOUS = ROOT / "docs/experiments/017_mc_anchor/run_mc_anchor.py"
SPEC = importlib.util.spec_from_file_location("weight019_anchor", PREVIOUS)
anchor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(anchor)
extension, adam, fit, multi = anchor.extension, anchor.adam, anchor.fit, anchor.multi
read_json, read_updates = anchor.read_json, anchor.read_updates
require, require_equal, exact_equal = anchor.require, anchor.require_equal, anchor.exact_equal
START, END, MARGIN, NORMALIZATION, TD_WEIGHT = 2000, 10000, 0.8, 117, 0.5
CHECKPOINTS = anchor.CHECKPOINTS
TARGET_SOURCE_UPDATES = anchor.TARGET_SOURCE_UPDATES
SENTINEL_STEPS = anchor.SENTINEL_STEPS


def weighted_td_update(learner, batch, td_weight=TD_WEIGHT):
    """Scale the mean TD loss only; retain the original margin, clipping and Adam semantics."""
    if (isinstance(td_weight, bool) or not isinstance(td_weight, Real)
            or not isfinite(td_weight) or td_weight not in (0.5, 1.0)):
        raise ValueError("TD weight must be the declared treatment 0.5 or control 1.0.")
    learner.online.train()
    learner.optimizer.zero_grad(set_to_none=True)
    before = nn.utils.parameters_to_vector(learner.online.parameters()).detach().clone()
    td_before, _, targets = td_loss(learner.online, learner.target, batch, learner.config.gamma)
    values = learner.online(batch.observations)
    margin_before = demonstration_margin_loss(values, batch.actions, MARGIN)
    accuracy = float((values.detach().argmax(dim=1) == batch.actions).float().mean())
    # Keep the original graph exactly at coefficient one; no MC term is constructed.
    scaled_td = td_before if td_weight == 1.0 else td_weight * td_before
    loss = scaled_td + 1.0 * margin_before
    if not torch.isfinite(loss):
        raise FloatingPointError("The weighted TD objective is not finite.")
    loss.backward()
    gradient_norm = nn.utils.clip_grad_norm_(
        learner.online.parameters(), learner.config.gradient_clip_norm, error_if_nonfinite=True)
    learner.optimizer.step()
    with torch.no_grad():
        after_values = learner.online(batch.observations)
        after_chosen = after_values.gather(1, batch.actions[:, None]).squeeze(1)
        td_after = F.smooth_l1_loss(after_chosen, targets)
        margin_after = demonstration_margin_loss(after_values, batch.actions, MARGIN)
        scaled_after = td_after if td_weight == 1.0 else td_weight * td_after
        loss_after = scaled_after + 1.0 * margin_after
        after = nn.utils.parameters_to_vector(learner.online.parameters())
        delta = torch.linalg.vector_norm(after - before)
        if not torch.isfinite(after).all() or not torch.isfinite(loss_after):
            raise FloatingPointError("The weighted TD update produced non-finite values.")
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
        "td_weight": float(td_weight),
    }


def load_inputs(reference, archive):
    """Reuse the previously verified no-MC control, its ordered data and complete TD-2000 state."""
    inherited = anchor.load_inputs(reference, archive)
    manifest = deepcopy(inherited["manifest"])
    manifest["helper_sha256"][PREVIOUS.relative_to(ROOT).as_posix()] = fit.file_hash(PREVIOUS)
    baseline = {**inherited["baseline_reference"], "td_weight": 1.0,
                "certification_accounting": "Historical policy-action counters exclude reset-time copied-environment certification; no new baseline simulator work is performed."}
    return {**inherited, "manifest": manifest, "baseline_reference": baseline}


def make_protocol(manifest):
    """Declare the sole objective change before any new update or physical evaluation."""
    return {
        "attempt": 19, "kind": "td_objective_weight_control",
        "scene_seeds": [10000, 10001, 10002], "initial_td_update": START, "final_td_update": END,
        "new_treatment_optimizer_updates": END - START, "td_checkpoints": list(CHECKPOINTS),
        "target_source_updates": list(TARGET_SOURCE_UPDATES), "inherited_shared_optimizer_updates": 42000,
        "learning_rate": 0.0001, "gamma": 0.99, "gradient_clip_norm": 10.0,
        "td_weight": TD_WEIGHT, "baseline_td_weight": 1.0,
        "demonstration_weight": 1.0, "all_sample_weights": 1.0, "margin": MARGIN,
        "return_anchor_weight": 0.0, "baseline_return_anchor_weight": 0.0,
        "target_update_interval": 200, "effective_batch_size": NORMALIZATION,
        "normalization_denominator": NORMALIZATION,
        "objective": "0.5 * mean TD Huber + 1.0 * mean demonstration hinge",
        "total_loss_rescaled": False, "mc_loss_constructed": False,
        "returns_used_only_for_evaluation": True, "td_targets_mixed_with_returns": False,
        "historical_control_reused_not_rerun": True, "historical_control_arm": "attempt_016/baseline",
        "preserve_adam_moments_and_parameter_steps": True, "preserve_absolute_td_counter": True,
        "synchronize_target_at_resume": False, "early_stopping": False,
        "sentinel_seed": 10001, "sentinel_reference_steps": list(SENTINEL_STEPS),
        "sentinel_batch_size": 1, "sentinel_updates": "every absolute update from 2000 through 10000 inclusive",
        "sentinel_gap": "Q(demonstrated action) - max Q(other actions)",
        "sentinel_is_physical_evaluation": False, "online_training_steps": 0, "held_out_scenes_accessed": 0,
        "reset_certification_preserved": True,
        "reset_certification_accounting": "An instance-scoped reset observer records returned certification attempts and copied-environment steps without changing reset arguments, results or physics.",
        "update_accuracy_semantics": "Demonstration action accuracy is measured before the update.",
        "update_target_semantics": "Before/after TD losses use the same frozen pre-update Bellman targets.",
        "script_sha256": fit.file_hash(Path(__file__)), "helper_sha256": manifest["helper_sha256"],
    }


class ResetCertificationRecorder:
    """Capture already-executed certification from reset results without adding simulator work."""

    def __init__(self, env):
        self.env = env
        self.records = []
        self.incomplete_resets = []

    def observe(self, callback, update):
        original = self.env.reset
        had_instance_reset = "reset" in self.env.__dict__
        previous_attribute = self.env.__dict__.get("reset")

        def reset(*args, **kwargs):
            try:
                result = original(*args, **kwargs)
            except BaseException:
                self.incomplete_resets.append({
                    "update": update, "seed": kwargs.get("seed"),
                    "certification_steps_unknown": True,
                    "reason": "Reset failed before returning its certification accounting.",
                })
                raise
            _, info = result
            checks = deepcopy(info.get("reference_route_checks", []))
            require(not self.env.verify_reference_route or len(checks) > 0,
                    "enabled certification returns all completed candidate checks")
            require(all(isinstance(row["steps"], int) and not isinstance(row["steps"], bool)
                        and row["steps"] >= 0 for row in checks), "valid copied-environment step accounting")
            self.records.append({
                "update": update, "seed": info.get("seed", kwargs.get("seed")),
                "certification_enabled": bool(self.env.verify_reference_route),
                "certification_attempts": len(checks),
                "certification_environment_steps": sum(row["steps"] for row in checks),
                "reference_route_checks": checks,
            })
            return result

        self.env.reset = reset
        try:
            return callback()
        finally:
            if had_instance_reset:
                self.env.reset = previous_attribute
            else:
                del self.env.reset

    def save(self, output):
        record = {
            "records": self.records, "completed_reset_count": len(self.records),
            "certification_attempts": sum(row["certification_attempts"] for row in self.records),
            "certification_environment_steps": sum(row["certification_environment_steps"] for row in self.records),
            "incomplete_resets": self.incomplete_resets,
            "accounting_complete": not self.incomplete_resets,
            "semantics": "Only returned reset records are summed. Certification steps occur on simulator copies; they are separate from policy evaluation actions. No extra certification is executed for recording.",
        }
        fit.save_json(output / "reset_certification.json", record)
        return record


def initial_q_equivalence(agent, batch, snapshot):
    """Measure all ordered initial Q vectors against the unchanged source model state."""
    source = deepcopy(agent.learner.online)
    source.load_state_dict(snapshot["online"])
    source.eval()
    modes = [(module, module.training) for module in agent.learner.online.modules()]
    try:
        agent.learner.online.eval()
        with torch.no_grad():
            expected = source(batch.observations)
            actual = agent.learner.online(batch.observations)
    finally:
        for module, mode in modes:
            module.training = mode
    require_equal(actual, expected, "all initial restored Q vectors equal the source model")
    digest = hashlib.sha256(actual.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    return {"passed": True, "rows": len(batch.actions), "shape": list(actual.shape),
            "source_and_restored_q_sha256": digest, "q_values": actual.tolist(),
            "maximum_absolute_difference": float((actual - expected).abs().max()),
            "new_optimizer_updates": 0, "environment_actions": 0}


def execute_treatment(agent, env, environment, batch, returns, episodes, output, expected_start,
                      start=START, final=END, checkpoints=CHECKPOINTS, reference_steps=SENTINEL_STEPS):
    """Continue unchanged data and absolute clocks with the predeclared TD coefficient."""
    require(agent.learner.update_count == start and tuple(sorted(set(checkpoints))) == tuple(checkpoints)
            and checkpoints[0] == start and checkpoints[-1] == final and start < final,
            "declared continuation clock and evaluation schedule")
    directory = output / "treatment/td"
    directory.mkdir(parents=True, exist_ok=False)
    target_sources = tuple(step for step in extension.target_source_updates(
        checkpoints, agent.learner.config.target_update_interval) if step >= start)
    initial_hash, rows, syncs = fit.parameter_hash(agent), [], 0
    certification = ResetCertificationRecorder(env)
    print(f"Starting treatment: TD {start} to {final}, TD weight {TD_WEIGHT}, no MC objective.", flush=True)
    try:
        with (directory / "updates.jsonl").open("x", encoding="utf-8") as updates, (
            directory / "sentinel_q.jsonl"
        ).open("x", encoding="utf-8") as sentinels:
            for step in range(start, final + 1):
                if step > start:
                    metrics = weighted_td_update(agent.learner, batch)
                    require(agent.learner.update_count == step, "preserved absolute TD clock")
                    syncs += int(metrics["target_synchronized"])
                    updates.write(json.dumps({"update": step, "extension_update": step - start, **metrics},
                                             allow_nan=False) + "\n")
                probe = extension.sentinel_record(agent, batch, episodes, step, start, reference_steps)
                sentinels.write(json.dumps(probe, allow_nan=False) + "\n")
                if step in checkpoints:
                    initial_reset_count = len(certification.records)
                    row = certification.observe(lambda step=step: multi.observe(
                        agent, env, environment, batch, returns, episodes,
                        "extension/td", step, 40000, directory), step)
                    require(len(certification.records) - initial_reset_count == len(episodes),
                            "one recorded environment reset per physical scene evaluation")
                    rows.append(row)
                    fit.save_json(directory / "observations.json", rows)
                    certification.save(output)
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
    finally:
        certification.save(output)
    snapshot_path = directory / "training_state.pt"
    torch.save(adam.training_snapshot(agent, environment), snapshot_path)
    result = {
        "directory": "treatment/td", "initial_parameter_sha256": initial_hash,
        "final_parameter_sha256": fit.parameter_hash(agent), "optimizer_updates": final - start,
        "initial_td_update": start, "final_td_update": final, "prior_updates_on_branch": 40000,
        "inherited_optimizer_updates": 40000 + start, "total_updates_on_branch": 40000 + final,
        "target_synchronizations": syncs, "target_source_updates": list(target_sources),
        "td_weight": TD_WEIGHT, "demonstration_weight": 1.0, "all_sample_weights": 1.0,
        "return_anchor_weight": 0.0, "normalization_denominator": len(batch.actions),
        "learning_rate": agent.learner.config.learning_rate,
        "initial_observation": rows[0], "final_observation": rows[-1],
        "retention": extension.retention_metrics(rows, expected_start, start),
        "evaluation_environment_steps": sum(row["aggregate"]["evaluation_environment_steps"] for row in rows),
        "reset_certification": certification.save(output),
        "sentinel_records": final - start + 1, "sentinel_environment_steps": 0,
        "training_snapshot_file": "treatment/td/training_state.pt", "training_snapshot_sha256": fit.file_hash(snapshot_path),
        "checkpoint_sha256": {path.name: fit.file_hash(path) for path in sorted(directory.glob("*.pt"))},
    }
    fit.save_json(directory / "summary.json", result)
    return result


def write_accounting(output, baseline_reference):
    """Separate policy actions, copied-simulator certification and reused historical work."""
    directory = output / "treatment/td"
    updates = read_updates(directory / "updates.jsonl") if (directory / "updates.jsonl").exists() else []
    rows = read_json(directory / "observations.json") if (directory / "observations.json").exists() else []
    sentinels = read_updates(directory / "sentinel_q.jsonl") if (directory / "sentinel_q.jsonl").exists() else []
    certification = read_json(output / "reset_certification.json") if (output / "reset_certification.json").exists() else {
        "completed_reset_count": 0, "certification_attempts": 0, "certification_environment_steps": 0,
        "incomplete_resets": [], "accounting_complete": True}
    initial = rows[0]["update"] if rows else START
    policy_steps = sum(row["aggregate"]["evaluation_environment_steps"] for row in rows)
    result = {
        "optimizer_updates_total": len(updates), "inherited_shared_optimizer_updates": 40000 + initial,
        "absolute_td_update": initial + len(updates), "total_updates_on_treatment_branch": 40000 + initial + len(updates),
        "scheduled_evaluation_episodes": sum(len(row["scenes"]) for row in rows),
        "evaluation_environment_steps": policy_steps,
        "environment_resets": certification["completed_reset_count"],
        "reset_certification_attempts": certification["certification_attempts"],
        "reset_certification_environment_steps": certification["certification_environment_steps"],
        "total_simulator_environment_steps": policy_steps + certification["certification_environment_steps"],
        "reset_accounting_complete": certification["accounting_complete"],
        "incomplete_resets": certification["incomplete_resets"],
        "sentinel_records": len(sentinels), "sentinel_forward_passes": sum(len(row["states"]) for row in sentinels),
        "sentinel_environment_steps": 0, "online_training_steps": 0, "demonstration_collection_steps": 0,
        "held_out_scenes_accessed": 0,
        "reused_historical_control": {key: baseline_reference[key] for key in (
            "optimizer_updates", "scheduled_evaluation_episodes", "evaluation_environment_steps",
            "new_optimizer_updates", "new_evaluation_episodes", "new_environment_steps")},
        "note": "Completed new treatment work only. Policy actions and reset-time certification on simulator copies are separate; inherited and historical control work are reused. Independent verification is separate.",
    }
    fit.save_json(output / "accounting.json", result)
    return result


def run(reference, archive, output):
    reference, archive, output = reference.resolve(), archive.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite TD-weight evidence: {output}")
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
        fit.save_json(output / "initial_q_equivalence.json", initial_q_equivalence(agent, inherited["batch"], snapshot))
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
                       "The verified historical control was not rerun in this main experiment.",
                       "The objective coefficient changes relative gradient contributions; it is not a learning-rate change.",
                       "Reference-state Q rankings do not establish physical outcomes at unsampled updates.",
                       "All new physical evaluations use the autonomous greedy policy without forced actions.",
                       "Actual discounted returns are evaluation targets only; no MC objective is constructed.",
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
