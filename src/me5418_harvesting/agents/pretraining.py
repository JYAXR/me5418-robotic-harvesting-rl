"""Bounded DQN updates on collected demonstrations before online interaction."""

import json
from pathlib import Path

import numpy as np

from me5418_harvesting.agents.dqn_agent import DQNAgent
from me5418_harvesting.agents.dqn_learning import validate_demonstration_loss
from me5418_harvesting.agents.replay_buffer import ReplayBuffer


def pretrain_on_replay(
    agent: DQNAgent, replay: ReplayBuffer, updates: int, seed: int, output: Path,
    *, demonstration_weight: float = 0.0, demonstration_margin: float = 0.8,
) -> dict:
    """Use an independent sampler without advancing online actions or exploration.

    The existing TD loss, optimizer, gradient clipping and target-update schedule
    are retained. Optimizer and target states continue into online training.
    An optional supervised margin applies only to these demonstration batches.
    This limited objective is not an implementation of the full DQfD algorithm.
    """
    if type(updates) is not int or updates <= 0:
        raise ValueError("Pretraining updates must be a positive integer.")
    if type(seed) is not int or seed < 0:
        raise ValueError("Pretraining seed must be a nonnegative integer.")
    validate_demonstration_loss(demonstration_weight, demonstration_margin)
    batch_size = agent.learner.config.batch_size
    if len(replay) < batch_size:
        raise ValueError("Demonstration replay must contain at least one full pretraining batch.")
    summary_path = output / "pretraining.json"
    metrics_path = output / "pretraining_metrics.jsonl"
    if summary_path.exists() or metrics_path.exists():
        raise FileExistsError("Refusing to overwrite an existing pretraining record.")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    summary = {
        "method": ("demonstration_dqn_td_and_margin_updates" if demonstration_weight > 0
                   else "demonstration_only_dqn_td_updates"),
        "demonstration_weight": demonstration_weight,
        "demonstration_margin": demonstration_margin,
        "status": "running", "requested_updates": updates, "completed_updates": 0,
        "input_transitions": len(replay), "batch_size": batch_size,
        "sampler_seed_sequence": [seed, 1], "environment_steps": 0,
        "first_loss": None, "last_loss": None, "target_synchronizations": 0,
        "initial_total_updates": agent.learner.update_count,
    }

    def save_summary():
        summary["total_updates"] = agent.learner.update_count
        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")

    save_summary()
    try:
        with metrics_path.open("x", encoding="utf-8") as log:
            for index in range(1, updates + 1):
                batch = replay.sample(batch_size, rng)
                if demonstration_weight > 0:
                    stats = agent.learner.update(
                        batch, demonstration_weight=demonstration_weight,
                        demonstration_margin=demonstration_margin,
                    )
                else:
                    stats = agent.learner.update(batch)
                summary["completed_updates"] = index
                if summary["first_loss"] is None:
                    summary["first_loss"] = stats.loss_before
                summary["last_loss"] = stats.loss_before
                summary["target_synchronizations"] += int(stats.target_synchronized)
                record = {
                    "pretraining_update": index, "total_updates": agent.learner.update_count,
                    "loss_before": stats.loss_before, "loss_after": stats.loss_after,
                    "gradient_norm": stats.gradient_norm,
                    "parameter_delta_norm": stats.parameter_delta_norm,
                    "target_synchronized": stats.target_synchronized,
                }
                if demonstration_weight > 0:
                    record.update({
                        "td_loss_before": stats.td_loss_before,
                        "td_loss_after": stats.td_loss_after,
                        "demonstration_loss_before": stats.demonstration_loss_before,
                        "demonstration_loss_after": stats.demonstration_loss_after,
                        "demonstration_action_accuracy": stats.demonstration_action_accuracy,
                    })
                log.write(json.dumps(record, allow_nan=False) + "\n")
                if index % 100 == 0 or index == updates:
                    log.flush()
                    save_summary()
                    print(json.dumps({"pretraining_progress": summary}, allow_nan=False), flush=True)
        summary["status"] = "completed"
    except BaseException as exc:
        summary.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error_type=type(exc).__name__, error_message=str(exc),
        )
        raise
    finally:
        save_summary()
    return summary
