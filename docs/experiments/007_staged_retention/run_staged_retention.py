"""Reproduce supervised fitting, then test retention during fixed-data TD updates.

All policies use one known training scene. This isolates loss changes from new
data and exploration; it is not online training or a generalization experiment.
"""

import argparse
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from me5418_harvesting.agents.demonstrations import seed_replay_from_reference
from me5418_harvesting.agents.dqn import DQNConfig
from me5418_harvesting.agents.dqn_agent import DQNAgent
from me5418_harvesting.agents.dqn_learning import td_loss
from me5418_harvesting.agents.replay_buffer import ReplayBuffer
from me5418_harvesting.agents.run_record import begin_run, finish_run
from me5418_harvesting.agents.staged_learning import begin_td_phase, supervised_update
from me5418_harvesting.agents.training import read_mapping
from me5418_harvesting.envs.harvesting_env import HarvestingEnv

ROOT = Path(__file__).resolve().parents[3]
FIT_RUNNER = ROOT / "docs/experiments/005_fixed_scene_fit/run_fixed_scene_fit.py"
SPEC = importlib.util.spec_from_file_location("fixed_scene_diagnostic", FIT_RUNNER)
fit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fit)

ARMS = ("margin_only", "td_only", "td_margin")
PRETRAIN_UPDATES = 20000
PHASE_UPDATES = 2000
PRETRAIN_CHECKPOINTS = (0, 5000, 10000, 20000)
PHASE_CHECKPOINTS = (0, 1, 10, 50, 100, 200, 500, 1000, 2000)
SCENE_SEED = 10000
INIT_SEED = 0
MARGIN = 0.8


def value_metrics(agent, batch):
    """Describe current scores and targets; supervised scores are not calibrated returns."""
    with torch.no_grad():
        loss, chosen, targets = td_loss(
            agent.learner.online, agent.learner.target, batch, agent.learner.config.gamma,
        )
        return {
            "current_target_td_loss": loss.item(),
            "chosen_score_min": chosen.min().item(), "chosen_score_mean": chosen.mean().item(),
            "chosen_score_max": chosen.max().item(), "target_min": targets.min().item(),
            "target_mean": targets.mean().item(), "target_max": targets.max().item(),
        }


def observe(agent, env, environment, batch, objective, phase_update, pretraining_updates, directory):
    weights_before = fit.parameter_hash(agent)
    rng_before = json.dumps(agent.rng.bit_generator.state)
    result = {
        "update": phase_update, "total_updates_on_branch": pretraining_updates + phase_update,
        "fit": fit.fit_metrics(agent, batch, MARGIN),
        "value_metrics": value_metrics(agent, batch),
        "rollout": fit.fixed_scene_rollout(agent, env, SCENE_SEED, batch),
    }
    if weights_before != fit.parameter_hash(agent):
        raise RuntimeError("Evaluation changed the policy parameters.")
    if rng_before != json.dumps(agent.rng.bit_generator.state):
        raise RuntimeError("Evaluation consumed the policy exploration RNG.")
    agent.save(directory / f"update_{phase_update}.pt", environment, {
        "phase": "staged_retention_diagnostic", "objective": objective,
        "updates": phase_update, "pretraining_updates": pretraining_updates,
        "total_updates_on_branch": pretraining_updates + phase_update,
        "scene_seed": SCENE_SEED, "training_scene_only": True, "fit": result["fit"],
    })
    print(f"{objective}: update={phase_update}, fit={result['fit']['correct']}/"
          f"{len(batch.actions)}, success={result['rollout']['success']}, "
          f"damage={result['rollout']['damage']}, steps={result['rollout']['steps']}", flush=True)
    return result


def update_agent(agent, batch, objective):
    if objective in ("pretraining", "margin_only"):
        return supervised_update(agent.learner, batch, objective="margin_only", margin=MARGIN)
    if objective == "td_only":
        return asdict(agent.learner.update(batch))
    if objective == "td_margin":
        return asdict(agent.learner.update(
            batch, demonstration_weight=1.0, demonstration_margin=MARGIN,
        ))
    raise ValueError(f"Unknown stage objective: {objective}")


def execute_phase(agent, env, environment, batch, objective, updates, checkpoints, output,
                  pretraining_updates, expected_start=None):
    directory = output / objective
    directory.mkdir(exist_ok=False)
    initial_hash = fit.parameter_hash(agent)
    observations, target_syncs = [], 0
    with (directory / "updates.jsonl").open("x", encoding="utf-8") as log:
        for step in range(updates + 1):
            if step:
                metrics = update_agent(agent, batch, objective)
                target_syncs += int(metrics["target_synchronized"])
                log.write(json.dumps({"update": step, **metrics}, allow_nan=False) + "\n")
            if step in checkpoints:
                observations.append(observe(
                    agent, env, environment, batch, objective, step, pretraining_updates, directory,
                ))
                fit.save_json(directory / "observations.json", observations)
                if (step == 0 and expected_start is not None
                        and (observations[0]["fit"] != expected_start["final_fit"]
                             or observations[0]["rollout"] != expected_start["final_rollout"])):
                    raise RuntimeError("The phase-zero policy changed the supervised behavior.")
                log.flush()
            elif step % 100 == 0:
                log.flush()
    failed = [row["update"] for row in observations if not row["rollout"]["success"]]
    result = {
        "initial_parameter_sha256": initial_hash, "final_parameter_sha256": fit.parameter_hash(agent),
        "optimizer_updates": agent.learner.update_count, "target_synchronizations": target_syncs,
        "final_fit": observations[-1]["fit"], "final_rollout": observations[-1]["rollout"],
        "success_at_every_sampled_checkpoint": not failed,
        "first_sampled_failure_update": min(failed) if failed else None,
        "evaluation_environment_steps": sum(row["rollout"]["steps"] for row in observations),
        "checkpoint_sha256": {
            path.name: fit.file_hash(path) for path in sorted(directory.glob("*.pt"))
        },
    }
    fit.save_json(directory / "summary.json", result)
    return result, observations


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    environment = read_mapping(ROOT / "configs/environment_multiroute.yaml")
    dqn = DQNConfig(**read_mapping(ROOT / "configs/dqn_multiroute_pilot.yaml"))
    reference = ROOT / "docs/experiments/006_fixed_scene_budget/evidence"
    reference_summary = json.loads((reference / "summary.json").read_text(encoding="utf-8"))
    protocol = {
        "kind": "supervised_pretraining_then_offline_td_retention", "scene_seed": SCENE_SEED,
        "initialization_seed": INIT_SEED, "pretraining_objective": "margin_only",
        "pretraining_updates": PRETRAIN_UPDATES, "pretraining_checkpoints": PRETRAIN_CHECKPOINTS,
        "arms": ARMS, "phase_updates": PHASE_UPDATES, "phase_checkpoints": PHASE_CHECKPOINTS,
        "margin": MARGIN, "td_margin_weight": 1.0, "batch_mode": "full_single_demonstration",
        "phase_boundary": "identical_weights_fresh_adam_synced_target_zero_phase_clock",
        "online_training_steps": 0, "held_out_scenes_accessed": 0, "early_stopping": False,
        "script_sha256": fit.file_hash(Path(__file__)),
        "fit_runner_sha256": fit.file_hash(FIT_RUNNER),
        "reference_summary_sha256": fit.file_hash(reference / "summary.json"),
        "reference_demonstrations_sha256": fit.file_hash(reference / "demonstrations.json"),
    }
    env = None
    try:
        begin_run(output, environment, asdict(dqn), protocol)
        torch.set_num_threads(1)
        torch.manual_seed(INIT_SEED)
        env = HarvestingEnv(environment)
        encoder_agent = DQNAgent(env, dqn, INIT_SEED)
        replay = ReplayBuffer(dqn.replay_capacity)
        collection = seed_replay_from_reference(
            environment, encoder_agent.encoder, replay, 1, SCENE_SEED,
            progress=lambda record: fit.save_json(output / "demonstrations.json", record),
        )
        expected_collection = json.loads(
            (reference / "demonstrations.json").read_text(encoding="utf-8"),
        )
        if collection != expected_collection or collection["successful_episodes"] != 1:
            raise RuntimeError("The demonstration does not exactly reproduce attempt 006.")
        batch = fit.complete_batch(replay)
        if (not bool(batch.terminated[-1]) or bool(batch.terminated[:-1].any())
                or bool(batch.truncated.any()) or not torch.equal(
                    batch.next_observations[:-1], batch.observations[1:],
                )):
            raise ValueError("Expected one continuous successful terminal demonstration.")
        np.savez_compressed(output / "demonstration.npz", **{
            name: getattr(batch, name).numpy() for name in batch.__dataclass_fields__
        })
        if fit.file_hash(output / "demonstration.npz") != reference_summary["dataset"]["sha256"]:
            raise RuntimeError("The encoded data differ from attempt 006.")
        fit.save_json(output / "dataset.json", reference_summary["dataset"])
        torch.manual_seed(INIT_SEED)
        pretrained = DQNAgent(env, dqn, INIT_SEED)
        pretraining, _ = execute_phase(
            pretrained, env, environment, batch, "pretraining", PRETRAIN_UPDATES,
            PRETRAIN_CHECKPOINTS, output, 0,
        )
        expected = reference_summary["arms"]["margin_only"]
        reproduction = {
            "dataset_matches": True, "collection_matches": True,
            "initial_parameters_match": (
                pretraining["initial_parameter_sha256"] == expected["initial_parameter_sha256"]
            ),
            "final_parameters_match": (
                pretraining["final_parameter_sha256"] == expected["final_parameter_sha256"]
            ),
            "fit_matches": pretraining["final_fit"] == expected["final_fit"],
            "rollout_matches": pretraining["final_rollout"] == expected["final_rollout"],
            "pretraining_success": pretraining["final_rollout"]["success"],
        }
        fit.save_json(output / "pretraining_reproduction.json", reproduction)
        if not all(reproduction.values()):
            raise RuntimeError("Supervised reproduction failed; do not begin TD retention testing.")
        arms, boundaries = {}, {}
        for objective in ARMS:
            agent = DQNAgent(env, dqn, INIT_SEED)
            agent.learner.online.load_state_dict(pretrained.learner.online.state_dict())
            agent.learner.optimizer.load_state_dict(pretrained.learner.optimizer.state_dict())
            agent.learner.update_count = PRETRAIN_UPDATES
            boundary = begin_td_phase(agent.learner)
            if fit.parameter_hash(agent) != pretraining["final_parameter_sha256"]:
                raise RuntimeError("The phase boundary changed pretrained policy weights.")
            target_matches = all(torch.equal(online, target) for online, target in zip(
                agent.learner.online.parameters(), agent.learner.target.parameters(),
            ))
            if not target_matches or agent.learner.optimizer.state or agent.learner.update_count:
                raise RuntimeError("The phase must start with synced targets and a fresh optimizer.")
            boundaries[objective] = {
                **boundary, "policy_parameter_sha256": fit.parameter_hash(agent),
                "target_equals_online": True, "optimizer_state_empty": True,
            }
            fit.save_json(output / "phase_boundaries.json", boundaries)
            arm, _ = execute_phase(
                agent, env, environment, batch, objective, PHASE_UPDATES, PHASE_CHECKPOINTS, output,
                PRETRAIN_UPDATES, expected_start=pretraining,
            )
            arms[objective] = arm
        summary = {
            "protocol": protocol, "pretraining": pretraining, "arms": arms,
            "pretraining_reproduction": reproduction,
            "optimizer_updates_total": PRETRAIN_UPDATES + sum(
                arm["optimizer_updates"] for arm in arms.values()
            ),
            "td_optimizer_updates": sum(arms[name]["optimizer_updates"]
                                        for name in ("td_only", "td_margin")),
            "demonstration_collection_steps": collection["collection_environment_steps"],
            "evaluation_environment_steps": pretraining["evaluation_environment_steps"] + sum(
                arm["evaluation_environment_steps"] for arm in arms.values()
            ),
            "online_training_steps": 0, "held_out_scenes_accessed": 0,
            "accounting_note": "Counts exclude planner search and reset feasibility probes.",
            "limitations": [
                "One known training scene and one initialization; no generalization claim.",
                "TD uses fixed demonstrations; no online exploration or newly collected replay.",
                "A shared fresh Adam reset defines this handoff; optimizer retention is untested.",
                "Supervised scores are not calibrated discounted returns at the phase boundary.",
                "First failures refer to sampled checkpoints, not an exact onset time.",
                "Remaining-budget input can support memorization of the fixed route.",
            ],
        }
        fit.save_json(output / "summary.json", summary)
    except BaseException as exc:
        if (output / "run_status.json").exists():
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
    args = parser.parse_args()
    run(args.output_dir.resolve())


if __name__ == "__main__":
    main()
