"""Train the harvesting DQN or run a short end-to-end training check."""

import argparse
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import yaml

from me5418_harvesting.agents.training import load_settings, read_mapping, train

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/environment.yaml")
    parser.add_argument("--dqn-config", type=Path, default=ROOT / "configs/dqn.yaml")
    parser.add_argument("--training-config", type=Path, default=ROOT / "configs/training.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke", action="store_true", help="Use short learning schedules.")
    args = parser.parse_args()
    try:
        dqn, config = load_settings(args.dqn_config, args.training_config)
        if args.smoke:
            dqn = replace(dqn, batch_size=16, replay_capacity=512, target_update_interval=10)
            config = replace(
                config, episodes=3, learning_starts=16, epsilon_decay_steps=200,
                evaluation_interval=3, evaluation_episodes=1,
            )
        if args.episodes is not None:
            config = replace(config, episodes=args.episodes)
        if args.seed is not None:
            config = replace(config, seed=args.seed)
        output = args.output_dir or (
            ROOT / "results" / "training" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )
        result = train(read_mapping(args.config), dqn, config, output)
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    print(f"Training finished: {result['updates']} updates; checkpoint: {result['checkpoint']}")
    if result["updates"] == 0:
        print("No updates occurred. Increase training length or reduce warmup/batch size.")


if __name__ == "__main__":
    main()
