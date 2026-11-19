"""Collect training-only A* demonstrations through the harvesting dynamics."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import yaml

from me5418_harvesting import HarvestingEnv
from me5418_harvesting.agents.dqn_agent import ObservationEncoder
from me5418_harvesting.planners.collision_free import NoCollisionFreeRoute, collision_free_actions

ROOT = Path(__file__).resolve().parents[1]


def collect_scene(config: dict, seed: int) -> tuple[int, list[tuple], str]:
    """Return the first certified route, using its certification rollout as data."""
    if float(config.get("execution_noise_fraction", 0.0)) != 0.0:
        raise ValueError("Demonstration collection requires deterministic execution.")

    env = HarvestingEnv(config)
    attempts = env.reference_max_attempts if env.verify_reference_route else 1
    env.verify_reference_route = False
    encoder = ObservationEncoder(env)
    reason = "no collision-free route"
    try:
        for attempt in range(attempts):
            # Subsequent resets retain the seeded generator, as certification does.
            # Deterministic route execution does not consume any scene-generator randomness.
            try:
                observation, _ = env.reset(seed=seed if attempt == 0 else None)
            except ValueError as error:
                return seed, [], f"scene generation failed: {error}"
            try:
                # The current shape equals the resting shape immediately after reset.
                actions = collision_free_actions(env)
            except NoCollisionFreeRoute:
                reason = "no collision-free route"
                continue

            records = []
            encoded = encoder(observation)
            for action in actions:
                next_observation, reward, terminated, truncated, info = env.step(action)
                next_encoded = encoder(next_observation)
                records.append((encoded, action, reward, next_encoded, terminated, truncated, seed))
                encoded = next_encoded
                if terminated or truncated:
                    if info["success"] and not info["damage"]:
                        return seed, records, "success"
                    reason = "damage" if info["damage"] else "budget exhausted"
                    break
            else:
                reason = "route exhausted before success"
        return seed, [], reason
    finally:
        env.close()


def save_dataset(output: Path, records: list[tuple]) -> None:
    observations, actions, rewards, next_observations, terminated, truncated, seeds = zip(*records)
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(
            temporary,
            observations=np.stack(observations),
            actions=np.asarray(actions, dtype=np.int64),
            rewards=np.asarray(rewards, dtype=np.float32),
            next_observations=np.stack(next_observations),
            terminated=np.asarray(terminated, dtype=np.bool_),
            truncated=np.asarray(truncated, dtype=np.bool_),
            scene_seeds=np.asarray(seeds, dtype=np.int64),
        )
        if output.exists():
            raise FileExistsError(f"Output already exists: {output}")
        temporary.rename(output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/environment_multiroute.yaml")
    parser.add_argument("--seed", type=int, required=True, help="First training-only scene seed.")
    parser.add_argument("--count", type=int, required=True, help="Number of scene seeds to attempt.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.seed < 0 or args.count < 1 or args.workers < 1:
        parser.error("Use a nonnegative seed and positive count and worker count.")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        parser.error("The environment configuration must be a YAML mapping.")
    if float(config.get("execution_noise_fraction", 0.0)) != 0.0:
        parser.error("Demonstration collection requires deterministic execution.")

    collect = partial(collect_scene, config)
    seeds = range(args.seed, args.seed + args.count)
    records = []
    successes = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, (seed, transitions, reason) in enumerate(pool.map(collect, seeds), start=1):
            if transitions:
                successes += 1
                records.extend(transitions)
            else:
                print(f"Skipped scene {seed}: {reason}", flush=True)
            if index % 16 == 0 or index == args.count:
                print(f"Attempted {index}/{args.count}, retained {successes} scenes "
                      f"and {len(records)} transitions", flush=True)
    if not records:
        parser.error("No successful demonstrations were collected.")
    save_dataset(args.output, records)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
