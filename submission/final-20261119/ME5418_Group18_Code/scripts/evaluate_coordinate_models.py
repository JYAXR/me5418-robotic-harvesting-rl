"""Evaluate coordinate DQN checkpoints on paired layouts without changing dynamics."""

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from functools import partial
from pathlib import Path

import torch

from me5418_harvesting.agents.training import evaluate
from me5418_harvesting.envs.scene_cache import CertifiedSceneCache

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "research_results/20261118_generalization"
KNOWN_DATASET = ROOT / "docs/experiments/030_expanded_stages/evidence/dataset.json"


def resolve_models(arguments: list[str]) -> dict:
    """Resolve named archival models or explicit NAME=PATH checkpoints."""
    archive = json.loads((ARCHIVE / "models.json").read_text(encoding="utf-8"))
    models = {}
    for argument in arguments:
        name, separator, filename = argument.partition("=")
        if not name or name in models:
            raise ValueError(f"Model labels must be nonempty and unique: {name!r}")
        if separator and filename:
            path = Path(filename).resolve()
        elif not separator and name in archive:
            path = ARCHIVE / archive[name]["file"]
        else:
            raise ValueError(f"Unknown archive model or missing checkpoint path: {argument!r}")
        with path.open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        if not separator and digest != archive[name]["sha256"]:
            raise ValueError(f"Archived checkpoint hash mismatch for {name!r}")
        models[name] = {"path": str(path), "sha256": digest}
    return models


def paired_environment_config(models: dict) -> dict:
    """Require every checkpoint to carry the same scene and dynamics configuration."""
    reference = None
    for name, entry in models.items():
        payload = torch.load(entry["path"], map_location="cpu", weights_only=True)
        entry["format_version"] = payload["format_version"]
        config = payload["environment_config"]
        if reference is None:
            reference = config
        elif config != reference:
            raise ValueError(f"Environment configuration differs for model {name!r}")
    return reference


def evaluation_groups(
    seed: int | None, episodes: int | None, dataset: Path | None
) -> dict[str, list[int]]:
    """Keep consecutively seeded trials distinct from the known-scene regression set."""
    groups = {}
    if seed is not None or episodes is not None:
        if seed is None or seed < 0 or episodes is None or episodes <= 0:
            raise ValueError("Supply a nonnegative --seed and positive --episodes together.")
        groups["range"] = list(range(seed, seed + episodes))
    if dataset is not None:
        known = json.loads(dataset.read_text(encoding="utf-8"))["seeds"]
        if (
            not isinstance(known, list)
            or not known
            or any(type(item) is not int or item < 0 for item in known)
            or len(set(known)) != len(known)
        ):
            raise ValueError("Known dataset seeds must be a nonempty list of unique nonnegative ints.")
        if set(known).intersection(groups.get("range", ())):
            raise ValueError("Consecutive evaluation seeds overlap with the known-scene dataset.")
        groups["known"] = known
    if not groups:
        raise ValueError("Supply --seed and --episodes, or --known-dataset.")
    return groups


def evaluate_batch(trials: list[tuple[str, int]], *, models: dict) -> list[dict]:
    """Evaluate seeds in matched order with one process-local shared reset cache."""
    sys.path.insert(0, str(ARCHIVE))
    from coordinate_policy import CoordinateAgent
    from stall_aware_policy import StallAwarePolicy

    formats = {entry["format_version"] for entry in models.values()}
    agent_types = {CoordinateAgent.FORMAT: CoordinateAgent}
    if "geometric-coordinate-dqn-v1" in formats:
        from geometric_coordinate_policy import GeometricCoordinateAgent

        agent_types[GeometricCoordinateAgent.FORMAT] = GeometricCoordinateAgent
    if formats.intersection({"guided-coordinate-dqn-v1", "compact-guided-dqn-v1"}):
        from guided_coordinate_policy import CompactGuidedAgent, GuidedCoordinateAgent

        agent_types[GuidedCoordinateAgent.FORMAT] = GuidedCoordinateAgent
        agent_types[CompactGuidedAgent.FORMAT] = CompactGuidedAgent

    torch.set_num_threads(1)
    cache = CertifiedSceneCache(capacity=1)
    records = []
    with ExitStack() as resources, torch.inference_mode():
        policies = []
        for name, entry in models.items():
            agent_type = agent_types.get(entry["format_version"])
            if agent_type is None:
                raise ValueError(f"Unsupported checkpoint format for {name!r}")
            agent, env, _ = agent_type.load(entry["path"])
            resources.callback(env.close)
            env.scene_cache = cache
            if entry.get("policy") == "stall_aware":
                agent = StallAwarePolicy(agent)
            policies.append((name, agent, env))
        for group, seed in trials:
            for name, agent, env in policies:
                if isinstance(agent, StallAwarePolicy):
                    agent.reset()
                row = evaluate(agent, env, episodes=1, seed=seed)["records"][0]
                timeout = row["budget_exhausted"] and not row["success"] and not row["damage"]
                records.append({
                    "model": name, "group": group, **row,
                    "timeout": timeout, "contact_steps": env.contact_steps,
                    "action_overrides": getattr(agent, "action_overrides", 0),
                })
    return records


def evaluate_models(models: dict, groups: dict[str, list[int]], workers: int) -> dict:
    """Compute summaries over a complete list of paired rollout rows."""
    if workers < 1:
        raise ValueError("--workers must be positive.")
    config = paired_environment_config(models)
    trials = [(group, seed) for group, seeds in groups.items() for seed in seeds]
    if workers == 1:
        records = evaluate_batch(trials, models=models)
    else:
        batch_size = (len(trials) + workers - 1) // workers
        batches = [trials[start:start + batch_size] for start in range(0, len(trials), batch_size)]
        with ProcessPoolExecutor(max_workers=len(batches)) as pool:
            worker = partial(evaluate_batch, models=models)
            futures = {pool.submit(worker, batch): index for index, batch in enumerate(batches)}
            chunks = [None] * len(batches)
            completed = 0
            for future in as_completed(futures):
                chunk = future.result()
                chunks[futures[future]] = chunk
                completed += len(chunk) // len(models)
                print(f"Evaluated {completed}/{len(trials)} paired scenes", file=sys.stderr,
                      flush=True)
            records = [row for chunk in chunks for row in chunk]
    summary = {}
    for model in models:
        summary[model] = {}
        for group in groups:
            outcomes = [row for row in records if row["model"] == model and row["group"] == group]
            summary[model][group] = {
                "episodes": len(outcomes),
                **{name: sum(row[name] for row in outcomes)
                   for name in ("success", "damage", "timeout")},
            }
    return {
        "environment_config": config, "groups": groups, "models": models,
        "summary": summary, "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", default=[], metavar="NAME[=PATH]",
                        help="Archive model alias or NAME=PATH; repeat to compare models.")
    parser.add_argument("--stall-aware", action="append", default=[], metavar="NAME[=PATH]",
                        help="Apply the optional observation-history action guard for this model.")
    parser.add_argument("--seed", type=int, help="First seed in the consecutive trial group.")
    parser.add_argument("--episodes", type=int, help="Number of consecutive trials per model.")
    parser.add_argument("--known-dataset", type=Path, nargs="?", const=KNOWN_DATASET,
                        help="Add known seeds from a dataset JSON. Defaults to the original 33.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, help="Write evaluation JSON here instead of stdout.")
    args = parser.parse_args()
    try:
        if args.output is not None and args.output.exists():
            raise FileExistsError(f"Output already exists: {args.output}")
        if not args.model and not args.stall_aware:
            raise ValueError("Supply at least one --model or --stall-aware.")
        stall_aware = {value.partition("=")[0] for value in args.stall_aware}
        models = resolve_models(args.model + args.stall_aware)
        for name, entry in models.items():
            entry["policy"] = "stall_aware" if name in stall_aware else "greedy"
        groups = evaluation_groups(args.seed, args.episodes, args.known_dataset)
        result = evaluate_models(models, groups, args.workers)
        payload = json.dumps(result, indent=2, allow_nan=False)
        if args.output is None:
            print(payload)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as file:
                file.write(payload + "\n")
    except (OSError, TypeError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
