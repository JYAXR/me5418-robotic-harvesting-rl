"""Compare fixed-return calibration with matched supervision before TD adaptation.

Both branches reuse one verified supervised policy and one complete training
trajectory. Discounted returns describe that demonstration, not an optimal Q*.
"""

import argparse
import importlib.util
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from me5418_harvesting.agents.dqn_agent import DQNAgent
from me5418_harvesting.agents.return_calibration import (
    calibration_update,
    discounted_demonstration_returns,
)
from me5418_harvesting.agents.run_record import begin_run, finish_run
from me5418_harvesting.agents.staged_learning import begin_td_phase, supervised_update

ROOT = Path(__file__).resolve().parents[3]
REFERENCE_LOADER = ROOT / "docs/experiments/008_td_learning_rate/run_learning_rate_retention.py"
SPEC = importlib.util.spec_from_file_location("return_calibration_reference", REFERENCE_LOADER)
reference_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reference_module)
staged = reference_module.staged

ARMS = ("continued_margin", "calibrated")
INHERITED_UPDATES = 20000
CALIBRATION_UPDATES = 20000
TD_UPDATES = 2000
CALIBRATION_CHECKPOINTS = (0, 1, 10, 100, 1000, 5000, 10000, 20000)
TD_CHECKPOINTS = (0, 1, 10, 50, 100, 200, 500, 1000, 2000)
MARGIN = 0.8
DEMONSTRATION_WEIGHT = 1.0
LEARNING_RATE = 3e-4


def return_fit(agent, batch, returns):
    """Measure each demonstrated action score against its fixed observed return."""
    agent.learner.online.eval()
    with torch.no_grad():
        chosen = agent.learner.online(batch.observations).gather(
            1, batch.actions[:, None],
        ).squeeze(1)
        if not torch.isfinite(chosen).all():
            raise FloatingPointError("The calibrated policy produced non-finite action values.")
        errors = chosen.double() - returns.double()
        return {
            "rmse": torch.sqrt(errors.square().mean()).item(),
            "mae": errors.abs().mean().item(), "max_abs_error": errors.abs().max().item(),
            "bias": errors.mean().item(), "chosen_values": chosen.tolist(),
            "mc_targets": returns.tolist(),
        }


def compute_targets(batch, gamma):
    """Cross-check tensor targets against a separate Python float64 recurrence."""
    returns = discounted_demonstration_returns(batch, gamma)
    rewards = [float(value) for value in batch.rewards.tolist()]
    independent = [0.0] * len(rewards)
    following = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        following = rewards[index] + float(gamma) * following
        independent[index] = following
    recorded = np.asarray(returns.tolist(), dtype=np.float64)
    reference = np.asarray(independent, dtype=np.float64)
    difference = float(np.max(np.abs(recorded - reference)))
    tolerance = 1e-4
    if difference > tolerance:
        raise RuntimeError("Discounted return targets do not match the independent recurrence.")
    next_returns = np.concatenate((recorded[1:], np.zeros(1, dtype=np.float64)))
    residual = recorded - np.asarray(rewards, dtype=np.float64) - float(gamma) * next_returns
    if float(recorded[-1]) != rewards[-1]:
        raise RuntimeError("The final observed reward must appear exactly once in its target.")
    return returns, {
        "gamma": gamma, "rewards": rewards, "returns": returns.tolist(),
        "python_float64_returns": independent, "transitions": len(rewards),
        "minimum_return": float(recorded.min()), "maximum_return": float(recorded.max()),
        "initial_discounted_return": float(recorded[0]),
        "undiscounted_episode_return_from_stored_rewards": sum(rewards),
        "maximum_recurrence_residual": float(np.max(np.abs(residual))),
        "maximum_float64_reference_difference": difference,
        "reference_tolerance": tolerance, "source_rewards_dtype": str(batch.rewards.dtype),
        "complete_ordered_terminal_trajectory_verified": True,
        "network_bootstrapping_used": False,
        "note": "Targets describe the recorded demonstration continuation, not an optimal Q*.",
    }


def apply_update(agent, batch, returns, branch, phase):
    if branch not in ARMS:
        raise ValueError(f"Unknown calibration branch: {branch}")
    if phase == "td":
        return asdict(agent.learner.update(
            batch, demonstration_weight=DEMONSTRATION_WEIGHT, demonstration_margin=MARGIN,
        ))
    if phase != "calibration":
        raise ValueError(f"Unknown experiment phase: {phase}")
    if branch == "continued_margin":
        return supervised_update(agent.learner, batch, objective="margin_only", margin=MARGIN)
    if branch == "calibrated":
        return calibration_update(
            agent.learner, batch, returns, demonstration_weight=DEMONSTRATION_WEIGHT,
            demonstration_margin=MARGIN,
        )
    raise ValueError(f"Unknown calibration branch: {branch}")


def execute_phase(agent, env, environment, batch, returns, branch, phase, updates,
                  checkpoints, output, prior_updates, expected_start):
    """Keep all scheduled failures, including failures before the first TD update."""
    directory = output / branch / phase
    directory.mkdir(parents=True, exist_ok=False)
    initial_hash = staged.fit.parameter_hash(agent)
    observations, target_synchronizations = [], 0
    with (directory / "updates.jsonl").open("x", encoding="utf-8") as log:
        for step in range(updates + 1):
            if step:
                metrics = apply_update(agent, batch, returns, branch, phase)
                target_synchronizations += int(metrics["target_synchronized"])
                log.write(json.dumps({"update": step, **metrics}, allow_nan=False) + "\n")
            if step in checkpoints:
                observation = staged.observe(
                    agent, env, environment, batch, f"{branch}/{phase}", step,
                    prior_updates, directory,
                )
                observation["return_fit"] = return_fit(agent, batch, returns)
                observations.append(observation)
                staged.fit.save_json(directory / "observations.json", observations)
                if (step == 0 and (observation["fit"] != expected_start["final_fit"]
                                   or observation["rollout"] != expected_start["final_rollout"])):
                    raise RuntimeError("The phase boundary changed the expected starting behavior.")
                log.flush()
            elif step % 100 == 0:
                log.flush()
    failures = [row["update"] for row in observations if not row["rollout"]["success"]]
    result = {
        "directory": f"{branch}/{phase}", "phase": phase,
        "initial_parameter_sha256": initial_hash,
        "final_parameter_sha256": staged.fit.parameter_hash(agent),
        "optimizer_updates": agent.learner.update_count,
        "prior_updates_on_branch": prior_updates,
        "total_updates_on_branch": prior_updates + agent.learner.update_count,
        "target_synchronizations": target_synchronizations,
        "initial_return_fit": observations[0]["return_fit"],
        "final_return_fit": observations[-1]["return_fit"],
        "final_fit": observations[-1]["fit"], "final_rollout": observations[-1]["rollout"],
        "success_at_every_sampled_checkpoint": not failures,
        "first_sampled_failure_update": min(failures) if failures else None,
        "evaluation_environment_steps": sum(row["rollout"]["steps"] for row in observations),
        "checkpoint_sha256": {
            path.name: staged.fit.file_hash(path) for path in sorted(directory.glob("*.pt"))
        },
    }
    staged.fit.save_json(directory / "summary.json", result)
    return result


def start_phase(agent, branch, phase, output, expected_hash, completed_new_updates):
    """Reset optimizer state while explicitly separating phase and inherited counts."""
    helper_record = begin_td_phase(agent.learner)
    if staged.fit.parameter_hash(agent) != expected_hash:
        raise RuntimeError("The fresh phase boundary changed the online parameters.")
    target_matches = all(torch.equal(online, target) for online, target in zip(
        agent.learner.online.parameters(), agent.learner.target.parameters(),
    ))
    if (not target_matches or agent.learner.optimizer.state or agent.learner.update_count
            or agent.learner.optimizer.param_groups[0]["lr"] != LEARNING_RATE):
        raise RuntimeError("Expected fresh Adam, synchronized targets and a zero phase clock.")
    record = {
        "helper_record": helper_record,
        "helper_count_meaning": "The helper pretraining_updates field is the prior phase only.",
        "inherited_pretraining_updates": INHERITED_UPDATES,
        "completed_new_updates_before_phase": completed_new_updates,
        "updates_on_branch_before_phase": INHERITED_UPDATES + completed_new_updates,
        "policy_parameter_sha256": expected_hash, "target_equals_online": True,
        "optimizer_state_empty": True, "learning_rate": LEARNING_RATE,
    }
    staged.fit.save_json(output / branch / f"{phase}_boundary.json", record)


def run(reference_run, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing calibration experiment: {output}")
    reference, settings, provenance, dataset, batch, checkpoint = reference_module.load_reference(
        reference_run,
    )
    if settings["dqn"]["learning_rate"] != LEARNING_RATE:
        raise ValueError("The reference learning rate differs from the declared matched comparison.")
    gamma = settings["dqn"]["gamma"]
    returns, target_record = compute_targets(batch, gamma)
    protocol = {
        "kind": "fixed_return_calibration_before_td_with_matched_supervision_control",
        "arms": ARMS, "scene_seed": staged.SCENE_SEED, "gamma": gamma,
        "learning_rate": LEARNING_RATE, "margin": MARGIN,
        "demonstration_weight": DEMONSTRATION_WEIGHT,
        "inherited_pretraining_updates": INHERITED_UPDATES,
        "calibration_updates": CALIBRATION_UPDATES, "td_updates": TD_UPDATES,
        "calibration_checkpoints": CALIBRATION_CHECKPOINTS, "td_checkpoints": TD_CHECKPOINTS,
        "calibration_objectives": {
            "continued_margin": "supervised_margin_only",
            "calibrated": "fixed_demonstration_return_huber_plus_margin",
        },
        "td_objective": "one_step_huber_td_plus_margin",
        "phase_boundary": "preserved_weights_fresh_adam_synced_target_zero_phase_clock",
        "td_executed_when_calibration_fails": True,
        "retention_eligibility": "Successful greedy rollout at the end of the calibration phase.",
        "batch_mode": "same_complete_ordered_demonstration",
        "early_stopping": False, "online_training_steps": 0, "held_out_scenes_accessed": 0,
        "reference_run": str(reference_run.resolve()), "reference_git_head": provenance["git_head"],
        "reference_checkpoint_sha256": staged.fit.file_hash(checkpoint),
        "reference_dataset_sha256": dataset["sha256"],
        "script_sha256": staged.fit.file_hash(Path(__file__)),
        "reference_loader_sha256": staged.fit.file_hash(REFERENCE_LOADER),
        "staged_runner_sha256": staged.fit.file_hash(reference_module.STAGED_RUNNER),
        "fit_runner_sha256": staged.fit.file_hash(staged.FIT_RUNNER),
    }
    output.mkdir(parents=True, exist_ok=False)
    try:
        begin_run(output, settings["environment"], settings["dqn"], protocol)
        torch.set_num_threads(1)
        reference_files = (
            "run_status.json", "summary.json", "settings.json", "provenance.json",
            "pretraining_reproduction.json", "dataset.json", "demonstration.npz",
            "pretraining/update_20000.pt", "pretraining/updates.jsonl",
        )
        staged.fit.save_json(output / "mc_targets.json", target_record)
        staged.fit.save_json(output / "input_manifest.json", {
            "reference_run": str(reference_run.resolve()),
            "reference_git_head": provenance["git_head"],
            "reference_files_sha256": {
                name: staged.fit.file_hash(reference_run / name) for name in reference_files
            },
            "reference_source_files_verified": len(provenance["source_sha256"]),
            "mc_targets_sha256": staged.fit.file_hash(output / "mc_targets.json"),
            "optimizer_state_reused": False, "weights_and_data_reused": True,
            "new_collection_steps": 0, "inherited_pretraining_updates": INHERITED_UPDATES,
        })
        shutil.copy2(reference_run / "demonstration.npz", output / "demonstration.npz")
        if staged.fit.file_hash(output / "demonstration.npz") != dataset["sha256"]:
            raise RuntimeError("The frozen demonstration copy differs from the reference.")
        staged.fit.save_json(output / "dataset.json", dataset)
        arms = {}
        for branch in ARMS:
            (output / branch).mkdir()
            torch.manual_seed(reference["protocol"]["initialization_seed"])
            agent, env, metadata = DQNAgent.load(checkpoint)
            try:
                if (metadata["objective"] != "pretraining" or metadata["updates"] != INHERITED_UPDATES
                        or metadata["fit"] != reference["pretraining"]["final_fit"]):
                    raise ValueError("The checkpoint is not the expected acquired supervised policy.")
                expected_hash = reference["pretraining"]["final_parameter_sha256"]
                if staged.fit.parameter_hash(agent) != expected_hash:
                    raise ValueError("The inherited pretrained parameter hash differs from the reference.")
                loaded_settings = json.loads(json.dumps(asdict(agent.learner.config)))
                if loaded_settings != settings["dqn"]:
                    raise ValueError("The loaded model configuration differs from the fixed comparison.")
                agent.learner.update_count = INHERITED_UPDATES
                start_phase(agent, branch, "calibration", output, expected_hash, 0)
                calibration = execute_phase(
                    agent, env, settings["environment"], batch, returns, branch, "calibration",
                    CALIBRATION_UPDATES, CALIBRATION_CHECKPOINTS, output, INHERITED_UPDATES,
                    reference["pretraining"],
                )
                eligible = bool(calibration["final_rollout"]["success"])
                start_phase(
                    agent, branch, "td", output, calibration["final_parameter_sha256"],
                    CALIBRATION_UPDATES,
                )
                td = execute_phase(
                    agent, env, settings["environment"], batch, returns, branch, "td", TD_UPDATES,
                    TD_CHECKPOINTS, output, INHERITED_UPDATES + CALIBRATION_UPDATES, calibration,
                )
                arms[branch] = {
                    "calibration": calibration, "td": td, "retention_eligible": eligible,
                    "retained_at_all_sampled_td_checkpoints": (
                        eligible and td["success_at_every_sampled_checkpoint"]
                    ),
                    "td_interpretation": "retention_of_successful_policy" if eligible
                    else "downstream_behavior_after_unsuccessful_calibration",
                }
                staged.fit.save_json(output / branch / "summary.json", arms[branch])
            finally:
                env.close()
        summary = {
            "protocol": protocol, "arms": arms, "pretrained_policy": reference["pretraining"],
            "optimizer_updates_total": sum(
                arm[phase]["optimizer_updates"] for arm in arms.values()
                for phase in ("calibration", "td")
            ),
            "td_optimizer_updates": sum(arm["td"]["optimizer_updates"] for arm in arms.values()),
            "inherited_pretraining_updates": INHERITED_UPDATES,
            "pretraining_reexecuted": False, "demonstration_collection_steps": 0,
            "evaluation_environment_steps": sum(
                arm[phase]["evaluation_environment_steps"] for arm in arms.values()
                for phase in ("calibration", "td")
            ),
            "online_training_steps": 0, "held_out_scenes_accessed": 0,
            "accounting_note": "New evaluation actions exclude reset-time feasibility probes.",
            "limitations": [
                "Returns describe one demonstrated continuation, not a known optimal Q function.",
                "One fixed training scene and one inherited initialization; generalization untested.",
                "Both TD phases run even if calibration fails; only successful starts test retention.",
                "This tests a calibration objective, not scale mismatch as an isolated sole cause.",
                "TD uses frozen demonstration data, with no online collection or exploration.",
                "Inherited pretraining is counted once in the reference, not as new work here.",
                "All sampled failures are retained; no early stopping or checkpoint selection is used.",
            ],
        }
        staged.fit.save_json(output / "summary.json", summary)
    except BaseException as exc:
        if (output / "run_status.json").exists():
            finish_run(output, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", exc)
        raise
    else:
        finish_run(output, "completed")
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path,
                        default=ROOT / "results/training/staged_retention_20261111")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.reference_run.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
