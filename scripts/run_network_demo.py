"""Show a real-environment Q-network forward pass, backward pass, and Adam update."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import yaml

try:
    import torch
except ImportError as exc:
    raise SystemExit('Install the network dependencies: python -m pip install -e ".[network]"') from exc

from me5418_harvesting import HarvestingEnv
from me5418_harvesting.agents.dqn import DQNConfig
from me5418_harvesting.agents.dqn_learning import DQNUpdater, collect_transitions, td_loss

ROOT = Path(__file__).resolve().parents[1]


def load_dqn_config(path: Path) -> DQNConfig:
    """Read architecture/update settings from the full DQN configuration."""
    settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        raise TypeError("The DQN configuration must be a YAML mapping.")
    names = {field.name for field in fields(DQNConfig)}
    # Warmup and epsilon schedules describe full training, not this short demo.
    training_only = {"learning_starts", "epsilon_start", "epsilon_end", "epsilon_decay_steps"}
    unknown = settings.keys() - names - training_only
    if unknown:
        raise ValueError(f"Unknown DQN configuration settings: {sorted(unknown)}")
    values = {name: settings[name] for name in names if name in settings}
    if "hidden_sizes" in values:
        values["hidden_sizes"] = tuple(values["hidden_sizes"])
    return DQNConfig(**values)


def demonstrate_network_action(env: HarvestingEnv, learner: DQNUpdater, seed: int) -> dict:
    """Execute one action selected by the updated network in the real environment."""
    observation, _ = env.reset(seed=seed)
    position_before = env.scene.end_effector.copy()
    learner.online.eval()
    with torch.no_grad():
        q_values = learner.online(torch.from_numpy(observation).unsqueeze(0))[0]
    action = int(q_values.argmax().item())
    next_observation, reward, terminated, truncated, info = env.step(action)
    return {
        "q_values": q_values.tolist(), "action": action, "reward": reward,
        "terminated": terminated, "truncated": truncated,
        "position_before": position_before.tolist(),
        "position_after": env.scene.end_effector.tolist(),
        "executed_motion": info["executed_motion"].tolist(),
        "next_observation_shape": list(next_observation.shape),
    }


def run_demo(
    environment_config: dict, config: DQNConfig, seed: int, transitions: int, updates: int
) -> dict:
    """Return JSON-serializable evidence of actual inference and learning operations."""
    if updates <= 0:
        raise ValueError("The number of updates must be positive.")
    if transitions < config.batch_size:
        raise ValueError("Collect at least batch_size transitions.")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    env = HarvestingEnv(environment_config)
    try:
        experience = collect_transitions(env, transitions, seed)
        input_dim = int(env.observation_space.shape[0])
        action_count = int(env.action_space.n)
        learner = DQNUpdater(input_dim, action_count, config)
        rng = np.random.default_rng(seed)
        batch = experience.sample(config.batch_size, rng)
        with torch.no_grad():
            q_values = learner.online(batch.observations)
            initial_loss, _, targets = td_loss(
                learner.online, learner.target, batch, config.gamma
            )
        forward = {
            "input_shape": list(batch.observations.shape),
            "output_shape": list(q_values.shape),
            "sample_input": batch.observations[0].tolist(),
            "sample_q_values": q_values[0].tolist(),
            "greedy_action": int(q_values[0].argmax().item()),
            "sample_collected_action": int(batch.actions[0].item()),
            "sample_reward": float(batch.rewards[0].item()),
            "sample_terminated": bool(batch.terminated[0].item()),
            "sample_truncated": bool(batch.truncated[0].item()),
            "sample_td_target": float(targets[0].item()),
            "initial_huber_loss": float(initial_loss.item()),
        }
        history = []
        for index in range(updates):
            if index:
                batch = experience.sample(config.batch_size, rng)
            stats = learner.update(batch)
            if stats.gradient_norm <= 0 or stats.parameter_delta_norm <= 0:
                raise RuntimeError("The demo did not produce a nonzero gradient and parameter update.")
            history.append({"update": index + 1, **asdict(stats)})
        return {
            "purpose": "Forward/backward/update demonstration; not a trained harvesting policy.",
            "device": "cpu",
            "seed": seed,
            "torch_version": torch.__version__,
            "architecture": [input_dim, *config.hidden_sizes, action_count],
            "parameter_count": sum(parameter.numel() for parameter in learner.online.parameters()),
            "transitions_collected": transitions,
            "terminated_transitions": int(experience.terminated.sum().item()),
            "truncated_transitions": int(experience.truncated.sum().item()),
            "forward": forward,
            "updates": history,
            "target_has_gradients": any(p.grad is not None for p in learner.target.parameters()),
            "network_controlled_step": demonstrate_network_action(env, learner, seed),
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/environment.yaml")
    parser.add_argument("--dqn-config", type=Path, default=ROOT / "configs/dqn.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--transitions", type=int, default=128)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--batch-size", type=int, help="Override the configured minibatch size.")
    args = parser.parse_args()
    try:
        config = load_dqn_config(args.dqn_config)
        if args.batch_size is not None:
            config = DQNConfig(**{**asdict(config), "batch_size": args.batch_size})
        environment_config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        if not isinstance(environment_config, dict):
            raise TypeError("The environment configuration must be a YAML mapping.")
        result = run_demo(environment_config, config, args.seed, args.transitions, args.updates)
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    # Console output only: no checkpoints or pre-generated result files are bundled.
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
