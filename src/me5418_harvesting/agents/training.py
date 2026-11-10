"""Episode orchestration, replay updates, metrics, and independent evaluation."""

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
import yaml

from me5418_harvesting.agents.demonstrations import seed_replay_from_reference
from me5418_harvesting.agents.dqn import DQNConfig
from me5418_harvesting.agents.dqn_agent import DQNAgent
from me5418_harvesting.agents.dqn_learning import validate_demonstration_loss
from me5418_harvesting.agents.pretraining import pretrain_on_replay
from me5418_harvesting.agents.replay_buffer import ReplayBuffer
from me5418_harvesting.agents.run_record import begin_run, finish_run
from me5418_harvesting.envs.harvesting_env import HarvestingEnv


@dataclass(frozen=True)
class TrainingConfig:
    episodes: int = 200
    learning_starts: int = 5000
    train_frequency: int = 4
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 100000
    evaluation_interval: int = 20
    evaluation_episodes: int = 5
    seed: int = 0
    validation_seed: int | None = None
    evaluate_before_training: bool = False
    demonstration_episodes: int = 0
    demonstration_seed: int = 10000
    demonstration_pretrain_updates: int = 0
    demonstration_pretrain_weight: float = 0.0
    demonstration_pretrain_margin: float = 0.8

    def __post_init__(self):
        for name in (
            "episodes", "train_frequency", "epsilon_decay_steps",
            "evaluation_interval", "evaluation_episodes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        for name in (
            "learning_starts", "seed", "demonstration_episodes", "demonstration_seed",
            "demonstration_pretrain_updates",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")
        if self.demonstration_pretrain_updates and not self.demonstration_episodes:
            raise ValueError("Demonstration pretraining requires demonstration episodes.")
        validate_demonstration_loss(
            self.demonstration_pretrain_weight, self.demonstration_pretrain_margin,
        )
        if self.demonstration_pretrain_weight > 0 and not self.demonstration_pretrain_updates:
            raise ValueError("Supervised demonstration loss requires pretraining updates.")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("Require 0 <= epsilon_end <= epsilon_start <= 1.")
        if type(self.evaluate_before_training) is not bool:
            raise TypeError("evaluate_before_training must be a boolean.")
        if self.validation_seed is not None:
            if type(self.validation_seed) is not int or self.validation_seed < 0:
                raise ValueError("validation_seed must be a nonnegative integer or null.")
            if (
                self.validation_seed < self.seed + self.episodes
                and self.seed < self.validation_seed + self.evaluation_episodes
            ):
                raise ValueError("Training and validation seed ranges must not overlap.")
        if self.demonstration_episodes:
            validation_seed = (
                self.seed + self.episodes if self.validation_seed is None else self.validation_seed
            )
            for start, count, name in (
                (self.seed, self.episodes, "online training"),
                (validation_seed, self.evaluation_episodes, "validation"),
            ):
                if (self.demonstration_seed < start + count
                        and start < self.demonstration_seed + self.demonstration_episodes):
                    raise ValueError(f"Demonstration and {name} seed ranges must not overlap.")

    def epsilon(self, steps: int) -> float:
        fraction = min(max(steps, 0) / self.epsilon_decay_steps, 1.0)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)


def read_mapping(path: Path) -> dict:
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise TypeError(f"Expected a YAML mapping: {path}")
    return values


def load_settings(dqn_path: Path, training_path: Path):
    """Read existing DQN settings; explicit training settings override schedules."""
    dqn_values = read_mapping(dqn_path)
    names = {field.name for field in fields(DQNConfig)}
    schedules = {"learning_starts", "epsilon_start", "epsilon_end", "epsilon_decay_steps"}
    unknown = dqn_values.keys() - names - schedules
    if unknown:
        raise ValueError(f"Unknown DQN settings: {sorted(unknown)}")
    network = {key: value for key, value in dqn_values.items() if key in names}
    if "hidden_sizes" in network:
        network["hidden_sizes"] = tuple(network["hidden_sizes"])
    training = {key: value for key, value in dqn_values.items() if key in schedules}
    training.update(read_mapping(training_path))
    return DQNConfig(**network), TrainingConfig(**training)


def episode_metrics(
    reward: float, steps: int, info: dict, stationary_steps: int = 0, blocked_steps: int = 0,
) -> dict:
    return {
        "return": reward, "steps": steps,
        "success": bool(info["success"]), "damage": bool(info["damage"]),
        "budget_exhausted": bool(info["budget_exhausted"]),
        "travel_distance": float(info["travel_distance"]),
        "peak_deformation": float(info["peak_normalized_deformation"]),
        "distance_to_target": float(info["distance_to_target"]),
        "stationary_steps": stationary_steps,
        "blocked_steps": blocked_steps,
    }


def evaluate(agent: DQNAgent, env: HarvestingEnv, episodes: int, seed: int) -> dict:
    """Run greedy episodes without replay writes, exploration, or weight updates."""
    if episodes <= 0 or seed < 0:
        raise ValueError("Evaluation episodes must be positive and seed nonnegative.")
    records = []
    for index in range(episodes):
        observation, _ = env.reset(seed=seed + index)
        total_reward, steps = 0.0, 0
        stationary_steps, blocked_steps = 0, 0
        while True:
            action = agent.act(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            stationary_steps += int(np.linalg.norm(info["executed_motion"]) <= 1e-6)
            blocked_steps += int(info["collision_blocked"])
            if terminated or truncated:
                break
        records.append({
            "seed": seed + index,
            **episode_metrics(total_reward, steps, info, stationary_steps, blocked_steps),
        })
    successful_steps = [row["steps"] for row in records if row["success"]]
    return {
        "episodes": episodes, "seed": seed,
        "mean_return": float(np.mean([row["return"] for row in records])),
        "success_rate": float(np.mean([row["success"] for row in records])),
        "damage_rate": float(np.mean([row["damage"] for row in records])),
        "budget_exhaustion_rate": float(np.mean([row["budget_exhausted"] for row in records])),
        "mean_success_steps": float(np.mean(successful_steps)) if successful_steps else None,
        "mean_travel_distance": float(np.mean([row["travel_distance"] for row in records])),
        "mean_peak_deformation": float(np.mean([row["peak_deformation"] for row in records])),
        "mean_distance_to_target": float(np.mean([row["distance_to_target"] for row in records])),
        "mean_stationary_fraction": float(np.mean([
            row["stationary_steps"] / row["steps"] for row in records
        ])),
        "mean_blocked_fraction": float(np.mean([
            row["blocked_steps"] / row["steps"] for row in records
        ])),
        "records": records,
    }


def train(environment_config: dict, dqn: DQNConfig, config: TrainingConfig, output: Path):
    """Record provenance and completion or failure for every started training run."""
    if dqn.replay_capacity < dqn.batch_size:
        raise ValueError("Replay capacity must be at least the batch size.")
    output.mkdir(parents=True, exist_ok=False)
    try:
        begin_run(output, environment_config, asdict(dqn), asdict(config))
        summary = _train(environment_config, dqn, config, output)
    except BaseException as exc:
        finish_run(output, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", exc)
        raise
    finish_run(output, "completed")
    return summary


def _train(environment_config: dict, dqn: DQNConfig, config: TrainingConfig, output: Path):
    """Follow the lecture loop, preserving replay and global counters across episodes."""
    torch.manual_seed(config.seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(config.seed)
    env = HarvestingEnv(environment_config)
    evaluation_env = HarvestingEnv(environment_config)
    agent = DQNAgent(env, dqn, config.seed)
    replay = ReplayBuffer(dqn.replay_capacity)
    # Fixed validation scenes are disjoint from this run's training scenes.
    validation_seed = (
        config.seed + config.episodes if config.validation_seed is None else config.validation_seed
    )
    settings = {
        "environment": environment_config, "dqn": asdict(dqn),
        "training": asdict(config), "validation_seed": validation_seed,
        "encoder_version": agent.encoder.VERSION,
    }
    (output / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    global_steps, best_score = 0, (-1.0, -float("inf"))
    last_loss = None
    initial_validation = None
    demonstrations = None
    pretraining = None
    pretraining_validation = None
    pretraining_updates = 0
    try:
        if config.evaluate_before_training or config.demonstration_pretrain_updates:
            initial_validation = evaluate(
                agent, evaluation_env, config.evaluation_episodes, validation_seed,
            )
            (output / "initial_validation.json").write_text(
                json.dumps(initial_validation, indent=2, allow_nan=False), encoding="utf-8",
            )
            print(json.dumps({"initial_validation": initial_validation}, allow_nan=False), flush=True)
        if config.demonstration_episodes:
            def save_demonstrations(snapshot):
                (output / "demonstrations.json").write_text(
                    json.dumps(snapshot, indent=2, allow_nan=False), encoding="utf-8",
                )

            demonstrations = seed_replay_from_reference(
                environment_config, agent.encoder, replay,
                config.demonstration_episodes, config.demonstration_seed,
                progress=save_demonstrations,
            )
            print(json.dumps({"demonstrations": demonstrations}, allow_nan=False), flush=True)
            if demonstrations["accepted_transitions"] == 0:
                raise RuntimeError("No successful demonstration transitions were collected.")
        if config.demonstration_pretrain_updates:
            pretraining = pretrain_on_replay(
                agent, replay, config.demonstration_pretrain_updates, config.seed, output,
                demonstration_weight=config.demonstration_pretrain_weight,
                demonstration_margin=config.demonstration_pretrain_margin,
            )
            pretraining_updates = pretraining["completed_updates"]
            pretraining_validation = evaluate(
                agent, evaluation_env, config.evaluation_episodes, validation_seed,
            )
            (output / "pretraining_validation.json").write_text(
                json.dumps(pretraining_validation, indent=2, allow_nan=False), encoding="utf-8",
            )
            print(json.dumps({"pretraining_validation": pretraining_validation},
                             allow_nan=False), flush=True)
            # Keep a separate snapshot; best.pt still compares online checkpoints only.
            agent.save(output / "pretrained.pt", environment_config, {
                "phase": "pretraining", "episode": 0, "environment_steps": 0,
                "updates": agent.learner.update_count, "pretraining_updates": pretraining_updates,
                "online_updates": 0, "validation": pretraining_validation,
            })
        with (output / "metrics.jsonl").open("w", encoding="utf-8") as log:
            for episode in range(1, config.episodes + 1):
                observation, _ = env.reset(seed=config.seed + episode - 1)
                total_reward, steps, episode_updates = 0.0, 0, 0
                stationary_steps, blocked_steps = 0, 0
                while True:
                    epsilon = config.epsilon(global_steps)
                    action = agent.act(observation, epsilon)
                    next_observation, reward, terminated, truncated, info = env.step(action)
                    # Store the final next state before reset, including both end flags.
                    replay.add(
                        agent.encoder(observation), action, reward,
                        agent.encoder(next_observation), terminated, truncated,
                    )
                    global_steps += 1
                    steps += 1
                    total_reward += reward
                    stationary_steps += int(np.linalg.norm(info["executed_motion"]) <= 1e-6)
                    blocked_steps += int(info["collision_blocked"])
                    if (
                        global_steps >= config.learning_starts
                        and len(replay) >= dqn.batch_size
                        and global_steps % config.train_frequency == 0
                    ):
                        stats = agent.learner.update(replay.sample(dqn.batch_size, rng))
                        last_loss = stats.loss_before
                        episode_updates += 1
                    observation = next_observation
                    if terminated or truncated:
                        break
                record = {
                    "episode": episode, "environment_steps": global_steps,
                    "updates": agent.learner.update_count, "episode_updates": episode_updates,
                    "pretraining_updates": pretraining_updates,
                    "online_updates": agent.learner.update_count - pretraining_updates,
                    "epsilon": epsilon, "replay_size": len(replay), "last_loss": last_loss,
                    **episode_metrics(total_reward, steps, info, stationary_steps, blocked_steps),
                }
                if episode % config.evaluation_interval == 0 or episode == config.episodes:
                    validation = evaluate(
                        agent, evaluation_env, config.evaluation_episodes, validation_seed,
                    )
                    record["validation"] = validation
                    score = (validation["success_rate"], validation["mean_return"])
                    if score > best_score:
                        best_score = score
                        agent.save(output / "best.pt", environment_config, record)
                    agent.save(output / "latest.pt", environment_config, record)
                line = json.dumps(record, allow_nan=False)
                log.write(line + "\n")
                log.flush()
                print(line, flush=True)
        summary = {
            "episodes": config.episodes, "environment_steps": global_steps,
            "updates": agent.learner.update_count, "replay_size": len(replay),
            "pretraining_updates": pretraining_updates,
            "online_updates": agent.learner.update_count - pretraining_updates,
            "last_loss": last_loss, "validation": validation,
            "checkpoint": str(output / "latest.pt"),
        }
        if initial_validation is not None:
            summary["initial_validation"] = initial_validation
        if pretraining is not None:
            summary["pretraining"] = pretraining
            summary["pretraining_validation"] = pretraining_validation
        if demonstrations is not None:
            summary["demonstrations"] = demonstrations
            summary["total_collection_steps"] = (
                global_steps + demonstrations["collection_environment_steps"]
            )
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8",
        )
        return summary
    finally:
        env.close()
        evaluation_env.close()
