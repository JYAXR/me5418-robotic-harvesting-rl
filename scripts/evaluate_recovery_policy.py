"""Evaluate an experimental geometry residual policy without action overrides."""

import argparse
import json
from pathlib import Path

import torch

from me5418_harvesting.agents.recovery_policy import RecoveryAgent
from me5418_harvesting.agents.training import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(1)
    agent, env, _ = RecoveryAgent.load(args.checkpoint)
    try:
        result = evaluate(agent, env, args.episodes, args.seed)
    finally:
        env.close()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
