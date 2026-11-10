"""Compare three fitting objectives on one fixed, training-only reference route.

This is a memorization diagnostic, not an online RL or generalization experiment.
Supervised-only scores are used for action selection, not interpreted as returns.
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from me5418_harvesting.agents.demonstrations import seed_replay_from_reference
from me5418_harvesting.agents.dqn import DQNConfig
from me5418_harvesting.agents.dqn_agent import DQNAgent
from me5418_harvesting.agents.dqn_learning import TransitionBatch, demonstration_margin_loss
from me5418_harvesting.agents.replay_buffer import ReplayBuffer
from me5418_harvesting.agents.run_record import begin_run, finish_run
from me5418_harvesting.agents.training import episode_metrics, read_mapping
from me5418_harvesting.envs.harvesting_env import HarvestingEnv

ROOT = Path(__file__).resolve().parents[3]
OBJECTIVES = ("td_margin", "margin_only", "cross_entropy")
ACTION_NAMES = (
    "north_small", "south_small", "east_small", "west_small",
    "north_large", "south_large", "east_large", "west_large",
)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parameter_hash(agent):
    values = torch.nn.utils.parameters_to_vector(agent.learner.online.parameters())
    return hashlib.sha256(values.detach().cpu().numpy().tobytes()).hexdigest()


def complete_batch(replay):
    """Preserve trajectory order; never sample or drop its final transition."""
    if not replay.records:
        raise ValueError("A complete successful reference trajectory is required.")
    columns = list(zip(*replay.records))
    return TransitionBatch(
        torch.from_numpy(np.stack(columns[0])),
        torch.tensor(columns[1], dtype=torch.long),
        torch.tensor(columns[2], dtype=torch.float32),
        torch.from_numpy(np.stack(columns[3])),
        torch.tensor(columns[4], dtype=torch.bool),
        torch.tensor(columns[5], dtype=torch.bool),
    )


def fit_metrics(agent, batch, margin):
    """Measure the entire frozen training set at the current weights."""
    agent.learner.online.eval()
    with torch.no_grad():
        scores = agent.learner.online(batch.observations)
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Non-finite policy scores in the fitting diagnostic.")
        predictions = scores.argmax(dim=1)
        correct = int((predictions == batch.actions).sum())
        confusion = torch.bincount(
            batch.actions * agent.action_count + predictions,
            minlength=agent.action_count ** 2,
        ).reshape(agent.action_count, agent.action_count)
        return {
            "transitions": len(batch.actions), "correct": correct,
            "action_agreement": correct / len(batch.actions),
            "cross_entropy": F.cross_entropy(scores, batch.actions).item(),
            "margin_loss": demonstration_margin_loss(scores, batch.actions, margin).item(),
            "prediction_counts": torch.bincount(
                predictions, minlength=agent.action_count,
            ).tolist(),
            "confusion_true_rows_predicted_columns": confusion.tolist(),
        }


def fitting_update(agent, batch, objective, margin):
    """Use the production TD update or a deliberately isolated supervised control."""
    if objective not in OBJECTIVES:
        raise ValueError(f"Unknown fitting objective: {objective}")
    if objective == "td_margin":
        return asdict(agent.learner.update(
            batch, demonstration_weight=1.0, demonstration_margin=margin,
        ))
    learner = agent.learner
    learner.online.train()
    learner.optimizer.zero_grad(set_to_none=True)
    before = torch.nn.utils.parameters_to_vector(learner.online.parameters()).detach().clone()
    scores = learner.online(batch.observations)
    loss = (F.cross_entropy(scores, batch.actions) if objective == "cross_entropy"
            else demonstration_margin_loss(scores, batch.actions, margin))
    if not torch.isfinite(loss):
        raise FloatingPointError("The supervised fitting loss is not finite.")
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        learner.online.parameters(), learner.config.gradient_clip_norm, error_if_nonfinite=True,
    )
    learner.optimizer.step()
    after = torch.nn.utils.parameters_to_vector(learner.online.parameters()).detach()
    if not torch.isfinite(after).all():
        raise FloatingPointError("The supervised update produced non-finite parameters.")
    learner.update_count += 1
    return {
        "loss_before": loss.item(), "gradient_norm": gradient_norm.item(),
        "parameter_delta_norm": torch.linalg.vector_norm(after - before).item(),
        "target_synchronized": False,
    }


def fixed_scene_rollout(agent, env, seed, batch):
    """Execute only greedy neural actions, recording divergence from the reference."""
    observation, _ = env.reset(seed=seed)
    if not np.array_equal(agent.encoder(observation), batch.observations[0].numpy()):
        raise ValueError("The rollout reset does not match the fixed demonstration scene.")
    total_reward, stationary, blocked = 0.0, 0, 0
    actions, path = [], [observation[:2].tolist()]
    first_divergence = None
    for index in range(env.max_steps):
        action = agent.act(observation, epsilon=0.0)
        if first_divergence is None:
            if index >= len(batch.actions) or action != int(batch.actions[index]):
                first_divergence = {
                    "step": index + 1, "predicted_action": action,
                    "demonstrated_action": int(batch.actions[index])
                    if index < len(batch.actions) else None,
                }
            elif not np.array_equal(agent.encoder(observation), batch.observations[index].numpy()):
                raise RuntimeError("Matching actions did not reproduce the demonstration states.")
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        stationary += int(np.linalg.norm(info["executed_motion"]) <= 1e-6)
        blocked += int(info["collision_blocked"])
        actions.append(action)
        path.append(observation[:2].tolist())
        if terminated or truncated:
            break
    else:
        raise RuntimeError("The fixed task did not terminate within its declared budget.")
    return {
        "training_scene_only": True, "seed": seed,
        **episode_metrics(total_reward, len(actions), info, stationary, blocked),
        "first_reference_action_divergence": first_divergence,
        "actions": actions, "robot_path": path,
    }


def run(environment, dqn, output, *, scene_seed=10000, init_seed=0, updates=2000,
        checkpoints=(0, 500, 1000, 2000), margin=0.8):
    if updates <= 0 or scene_seed < 0 or init_seed < 0:
        raise ValueError("Require positive updates and nonnegative seeds.")
    if tuple(sorted(set(checkpoints))) != tuple(checkpoints):
        raise ValueError("Checkpoints must be sorted and unique.")
    if not checkpoints or checkpoints[0] != 0 or checkpoints[-1] != updates:
        raise ValueError("Checkpoints must include zero and the final update.")
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "kind": "fixed_training_scene_memorization_diagnostic", "objectives": OBJECTIVES,
        "scene_seed": scene_seed, "initialization_seed": init_seed,
        "updates_per_arm": updates, "evaluation_updates": checkpoints,
        "margin": margin, "td_margin_weight": 1.0,
        "batch_mode": "entire_single_trajectory_in_order", "online_training_steps": 0,
        "held_out_scenes_accessed": 0, "early_stopping": False,
        "script_sha256": file_hash(Path(__file__)),
    }
    env = None
    try:
        begin_run(output, environment, asdict(dqn), protocol)
        torch.set_num_threads(1)
        torch.manual_seed(init_seed)
        env = HarvestingEnv(environment)
        encoder_agent = DQNAgent(env, dqn, init_seed)
        replay = ReplayBuffer(dqn.replay_capacity)
        collection = seed_replay_from_reference(
            environment, encoder_agent.encoder, replay, 1, scene_seed,
            progress=lambda record: save_json(output / "demonstrations.json", record),
        )
        if collection["successful_episodes"] != 1:
            raise RuntimeError("The fixed scene did not yield one successful demonstration.")
        batch = complete_batch(replay)
        if not bool(batch.terminated[-1]) or bool(batch.terminated[:-1].any()):
            raise ValueError("The single trajectory must end with exactly one task termination.")
        if bool(batch.truncated.any()) or not torch.equal(
            batch.next_observations[:-1], batch.observations[1:],
        ):
            raise ValueError("Expected a continuous demonstration without external truncations.")
        dataset_path = output / "demonstration.npz"
        np.savez_compressed(dataset_path, **{
            name: getattr(batch, name).numpy() for name in batch.__dataclass_fields__
        })
        labels = torch.bincount(batch.actions, minlength=env.action_space.n)
        inputs = batch.observations.numpy()
        unique_rows, inverse, counts = np.unique(
            inputs, axis=0, return_inverse=True, return_counts=True,
        )
        conflicting = sum(
            len(np.unique(batch.actions.numpy()[inverse == index])) > 1
            for index in range(len(unique_rows)) if counts[index] > 1
        )
        dataset = {
            "sha256": file_hash(dataset_path), "transitions": len(replay),
            "effective_batch_size": len(replay), "configured_batch_size_unused": dqn.batch_size,
            "observation_size": env.observation_size, "action_order": ACTION_NAMES,
            "label_counts": labels.tolist(),
            "majority_label_agreement": int(labels.max()) / len(replay),
            "duplicate_input_groups_with_conflicting_actions": int(conflicting),
            "varying_input_dimensions": int((np.ptp(inputs, axis=0) > 0).sum()),
            "budget_input_retained": True,
            "note": "The budget feature and fixed geometry can support route-order memorization.",
        }
        save_json(output / "dataset.json", dataset)
        arms = {}
        initial_hash = None
        for objective in OBJECTIVES:
            arm_dir = output / objective
            arm_dir.mkdir()
            torch.manual_seed(init_seed)
            agent = DQNAgent(env, dqn, init_seed)
            current_hash = parameter_hash(agent)
            if initial_hash is None:
                initial_hash = current_hash
            elif current_hash != initial_hash:
                raise RuntimeError("The fitting arms did not start with identical weights.")
            rng_before = json.dumps(agent.rng.bit_generator.state)
            observations = []
            with (arm_dir / "updates.jsonl").open("w", encoding="utf-8") as log:
                for step in range(updates + 1):
                    if step > 0:
                        record = fitting_update(agent, batch, objective, margin)
                        log.write(json.dumps({"update": step, **record}, allow_nan=False) + "\n")
                    if step in checkpoints:
                        fit = fit_metrics(agent, batch, margin)
                        before_rollout = parameter_hash(agent)
                        rollout = fixed_scene_rollout(agent, env, scene_seed, batch)
                        if parameter_hash(agent) != before_rollout:
                            raise RuntimeError("Greedy evaluation changed model weights.")
                        record = {"update": step, "fit": fit, "rollout": rollout}
                        observations.append(record)
                        save_json(arm_dir / "observations.json", observations)
                        agent.save(arm_dir / f"update_{step}.pt", environment, {
                            "phase": "fixed_scene_diagnostic", "objective": objective,
                            "updates": step, "scene_seed": scene_seed, "training_scene_only": True,
                            "scores_are_return_estimates": objective == "td_margin",
                            "fit": fit,
                        })
                        log.flush()
                        print(f"{objective}: update={step}, agreement={fit['correct']}/"
                              f"{len(replay)}, success={rollout['success']}, "
                              f"steps={rollout['steps']}", flush=True)
                    elif step % 100 == 0:
                        log.flush()
            if json.dumps(agent.rng.bit_generator.state) != rng_before:
                raise RuntimeError("The diagnostic consumed the exploration RNG.")
            arms[objective] = {
                "initial_parameter_sha256": current_hash,
                "final_parameter_sha256": parameter_hash(agent),
                "optimizer_updates": agent.learner.update_count,
                "target_synchronizations": updates // dqn.target_update_interval
                if objective == "td_margin" else 0,
                "final_fit": observations[-1]["fit"],
                "final_rollout": observations[-1]["rollout"],
                "evaluation_environment_steps": sum(row["rollout"]["steps"] for row in observations),
                "checkpoint_sha256": {
                    path.name: file_hash(path) for path in sorted(arm_dir.glob("*.pt"))
                },
            }
            save_json(arm_dir / "summary.json", arms[objective])
        summary = {
            "protocol": protocol, "dataset": dataset, "arms": arms,
            "optimizer_updates_total": sum(arm["optimizer_updates"] for arm in arms.values()),
            "demonstration_collection_steps": collection["collection_environment_steps"],
            "evaluation_environment_steps": sum(
                arm["evaluation_environment_steps"] for arm in arms.values()
            ),
            "online_training_steps": 0, "held_out_scenes_accessed": 0,
            "accounting_note": "Environment steps exclude A* search and reset feasibility probes.",
            "limitations": [
                "One scene and one initialization cannot establish generalization or RL success.",
                "The single-route budget input permits memorization of action order.",
                "CE changes the supervised loss as well as removing TD; margin-only isolates TD.",
                "The full-batch protocol differs from attempt 004 and is not a performance comparison.",
                "Action labels are not necessarily unique successful choices.",
            ],
        }
        save_json(output / "summary.json", summary)
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
    run(
        read_mapping(ROOT / "configs/environment_multiroute.yaml"),
        DQNConfig(**read_mapping(ROOT / "configs/dqn_multiroute_pilot.yaml")),
        args.output_dir.resolve(),
    )


if __name__ == "__main__":
    main()
