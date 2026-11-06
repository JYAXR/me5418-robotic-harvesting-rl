"""Load a saved DQN policy and evaluate it on new seeded scenes."""

import argparse
import json
from pathlib import Path

import torch

from me5418_harvesting.agents.dqn_agent import DQNAgent
from me5418_harvesting.agents.training import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000000)
    args = parser.parse_args()
    try:
        torch.set_num_threads(1)
        agent, env, _ = DQNAgent.load(args.checkpoint)
        try:
            result = evaluate(agent, env, args.episodes, args.seed)
        finally:
            env.close()
    except (OSError, TypeError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
