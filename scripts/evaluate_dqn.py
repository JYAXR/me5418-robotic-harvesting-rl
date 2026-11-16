"""Load a saved DQN policy and evaluate it on new seeded scenes."""

import argparse
import json
from pathlib import Path

import torch

from me5418_harvesting.agents.dqn_agent import DQNAgent
from me5418_harvesting.agents.training import evaluate
from me5418_harvesting.envs.scene_cache import CertifiedSceneCache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000000)
    parser.add_argument("--scene-cache-capacity", type=int, default=0,
                        help="Opt-in process-local certified scene cache; zero disables it.")
    parser.add_argument("--timings", type=Path, help="Write separate evaluation timing telemetry.")
    args = parser.parse_args()
    try:
        torch.set_num_threads(1)
        agent, env, _ = DQNAgent.load(args.checkpoint)
        if args.scene_cache_capacity < 0:
            raise ValueError("Scene cache capacity must be nonnegative.")
        if args.scene_cache_capacity:
            env.scene_cache = CertifiedSceneCache(args.scene_cache_capacity)
        env.profile_reset = args.timings is not None
        timings = [] if args.timings is not None else None
        try:
            result = evaluate(agent, env, args.episodes, args.seed, timings=timings)
            if timings is not None:
                args.timings.write_text(json.dumps(timings, indent=2), encoding="utf-8")
        finally:
            env.close()
    except (OSError, TypeError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
