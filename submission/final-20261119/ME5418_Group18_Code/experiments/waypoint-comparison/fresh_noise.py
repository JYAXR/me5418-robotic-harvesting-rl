"""Paired clean and noisy rollouts on six frozen, previously unused layouts."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from compare_waypoints import (
    ARCHIVE, MODEL, MODEL_SHA256, CompactGuidedAgent, guided_rollout, make_physics, torch,
)

SEEDS = tuple(range(5420100, 5420106))
NOISE = {"clean": 0.0, "noise10": 0.10}
METHODS = {"dqn": None, "h5": 0.0, "h5_waypoint": 10.0}
OUTCOMES = ("success", "damage", "timeout")
NOISE_SEED_PREFIX = 913714
DEPTH, BEAM = 5, 32


def encode_json(data):
    return json.dumps(data, indent=2, allow_nan=False) + "\n"


def noise_seed(seed, replicate=0):
    # Preserve the exact seed sequence of the first experiment.
    return [NOISE_SEED_PREFIX, seed] + ([replicate] if replicate else [])


def environment(initial, seed, condition, replicate=0):
    env = deepcopy(initial)
    env.noise_fraction = NOISE[condition]
    env.np_random = np.random.default_rng(noise_seed(seed, replicate))
    return env


def actual_step(env, action, step):
    obs, reward, done, trunc, info = env.step(action)
    return obs, info, {
        "step": step, "action": int(action), "xy": env.scene.end_effector.tolist(),
        "reward": reward, "travel": env.total_travel, "peak": env.peak_deformation,
        "contact_steps": env.contact_steps, "terminal": done or trunc,
    }


def summarize_episode(env, trace, info):
    total_reward = 0.0
    for row in trace:
        total_reward += row["reward"]
    return {
        "steps": len(trace), "success": bool(info["success"]), "damage": bool(info["damage"]),
        "timeout": bool(info["budget_exhausted"] and not info["success"] and not info["damage"]),
        "travel": env.total_travel, "peak": env.peak_deformation, "reward": total_reward,
        "contacts": env.contact_steps,
    }


def run_dqn(env, obs, agent):
    trace = []
    for step in range(1, env.max_steps + 1):
        obs, info, record = actual_step(env, agent.act(obs, epsilon=0), step)
        trace.append(record)
        if record["terminal"]:
            break
    return dict(summarize_episode(env, trace, info), trace=trace)


def replay(initial, seed, condition, row, replicate=0):
    env = environment(initial, seed, condition, replicate)
    reference = row["trace"]
    for step, expected in enumerate(reference, 1):
        _, info, actual = actual_step(env, expected["action"], step)
        assert all(expected[k] == value for k, value in actual.items()), (seed, condition, step)
        assert actual["terminal"] == (step == len(reference))
    assert all(row[k] == v for k, v in summarize_episode(env, reference, info).items())
    env.close()


def run_layout(job, replicate=0):
    seed, folder, conditions, methods = job
    torch.set_num_threads(1)
    agent, initial, _ = CompactGuidedAgent.load(MODEL)
    obs, info = initial.reset(seed=seed)
    assert initial.noise_fraction == 0 and initial.max_steps == 200
    assert info["reference_route"]["success"]
    assert replicate == 0 or conditions == ("noise10",)
    digest = hashlib.sha256(obs.tobytes()).hexdigest()
    physical = make_physics(initial, memoize_branches=True) if set(methods) != {"dqn"} else None
    with torch.inference_mode():
        for method in methods:
            for condition in conditions:
                path = folder / f"{seed}-{condition}-{method}.json"
                if path.is_file():
                    saved = json.loads(path.read_text(encoding="utf-8"))
                    assert saved["initial_observation_sha256"] == digest
                    assert saved["noise_seed"] == noise_seed(seed, replicate)
                    continue
                env = environment(initial, seed, condition, replicate)

                label = f"{seed} {condition} {method}"
                if replicate:
                    label += f" repeat={replicate}"

                def progress(step):
                    print(f"{label}: {step} steps", flush=True)

                weight = METHODS[method]
                if weight is None:
                    row = run_dqn(env, obs, agent)
                else:
                    row = guided_rollout(env, agent.encoder, obs, DEPTH, BEAM, weight, physical, progress)
                env.close()
                replay(initial, seed, condition, row, replicate)
                row.update(seed=seed, condition=condition, method=method,
                           initial_observation_sha256=digest, noise_fraction=NOISE[condition],
                           waypoint_weight=weight, noise_seed=noise_seed(seed, replicate))
                if replicate:
                    row["noise_replicate"] = replicate
                temporary = path.with_suffix(f".{os.getpid()}.tmp")
                temporary.write_text(encode_json(row), encoding="utf-8")
                temporary.replace(path)
                print(f"{label}: {row['steps']} steps, "
                      f"{next(k for k in OUTCOMES if row[k])} (replayed)", flush=True)
    initial.close()


def write_csv(path, records):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def summarize(folder):
    cases, summary = [], []
    for seed in SEEDS:
        hashes = set()
        for condition, fraction in NOISE.items():
            for method, weight in METHODS.items():
                row = json.loads((folder / f"{seed}-{condition}-{method}.json").read_text())
                assert (row["seed"], row["condition"], row["method"]) == (seed, condition, method)
                assert row["noise_fraction"] == fraction and row["waypoint_weight"] == weight
                assert row["noise_seed"] == [NOISE_SEED_PREFIX, seed]
                assert sum(row[k] for k in OUTCOMES) == 1 and row["trace"][-1]["terminal"]
                hashes.add(row["initial_observation_sha256"])
                cases.append({k: row[k] for k in (
                    "seed", "condition", "method", *OUTCOMES, "steps", "travel", "peak", "reward",
                    "contacts", "initial_observation_sha256",
                )})
        assert len(hashes) == 1
    for condition in NOISE:
        for method in METHODS:
            group = [r for r in cases if (r["condition"], r["method"]) == (condition, method)]
            counts = {k: sum(r[k] for r in group) for k in OUTCOMES}
            row = dict(condition=condition, method=method, layouts=len(group), **counts)
            print(row)
            summary.append(row)
    write_csv(folder / "episodes.csv", cases)
    write_csv(folder / "summary.csv", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--conditions", nargs="+", choices=NOISE, default=list(NOISE))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    groups = (args.seeds, args.conditions, args.methods)
    if args.workers < 1 or any(len(set(values)) != len(values) for values in groups):
        parser.error("Use positive workers and unique seeds, methods and conditions.")
    assert hashlib.sha256(MODEL.read_bytes()).hexdigest() == MODEL_SHA256
    archived = json.loads((ARCHIVE / "data/dqn_133.json").read_text())
    assert set(SEEDS).isdisjoint(seed for seeds in archived["groups"].values() for seed in seeds)
    args.out.mkdir(parents=True, exist_ok=True)
    protocol = args.out / "protocol.json"
    description = dict(seeds=list(SEEDS), noise=NOISE, noise_seed_prefix=NOISE_SEED_PREFIX,
                       noise_replicates=1, methods=METHODS, planner_depth=DEPTH, planner_beam=BEAM,
                       planner_prediction_noise=0.0, max_steps=200, model_sha256=MODEL_SHA256,
                       layout="Certify at zero noise and clone the same initial state for all runs.",
                       pairing="NumPy default_rng([prefix, layout_seed]); common draws by step.")
    if protocol.exists():
        assert json.loads(protocol.read_text()) == description
    else:
        protocol.write_text(encode_json(description), encoding="utf-8")
    if not args.summarize_only:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(args.seeds))) as pool:
            list(pool.map(run_layout, [(s, args.out, args.conditions, args.methods) for s in args.seeds]))
    if args.summarize_only or all(
        set(selected) == set(expected) for selected, expected in zip(groups, (SEEDS, NOISE, METHODS))
    ):
        summarize(args.out)


if __name__ == "__main__":
    main()
