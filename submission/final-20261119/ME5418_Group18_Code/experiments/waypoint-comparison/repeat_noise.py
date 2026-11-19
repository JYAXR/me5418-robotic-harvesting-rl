"""Repeat the frozen 10% execution-noise comparison on the same six layouts."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from itertools import combinations
import json
from pathlib import Path
from statistics import mean

import numpy as np

from fresh_noise import (
    BEAM, DEPTH, METHODS, MODEL, MODEL_SHA256, NOISE, OUTCOMES, SEEDS, encode_json,
    noise_seed, run_layout, write_csv,
)

HERE = Path(__file__).resolve().parent
REPEATS = (0, 1, 2)
RUN_ORDER = ("dqn", "h5_waypoint", "h5")
METRICS = ("steps", "travel", "peak", "reward", "contacts")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def baseline_protocol(baseline):
    info = read_json(baseline / "protocol.json")
    expected = dict(seeds=list(SEEDS), noise=NOISE, noise_replicates=1, methods=METHODS,
                    planner_depth=DEPTH, planner_beam=BEAM, planner_prediction_noise=0.0,
                    max_steps=200, model_sha256=MODEL_SHA256)
    assert all(info[k] == v for k, v in expected.items())
    assert info["noise_seed_prefix"] == noise_seed(SEEDS[0])[0]
    return {
        f"{seed}-{condition}-{method}.json": digest(baseline / f"{seed}-{condition}-{method}.json")
        for seed in SEEDS for condition in NOISE for method in METHODS
    }


def freeze(folder, baseline):
    stream_hashes = {
        f"{seed}-r{rep}": hashlib.sha256(
            np.random.default_rng(noise_seed(seed, rep)).uniform(-1, 1, size=(200, 2)).tobytes()
        ).hexdigest()
        for seed in SEEDS for rep in REPEATS
    }
    assert len(set(stream_hashes.values())) == len(stream_hashes)
    design = dict(
        layouts=list(SEEDS), noise_fraction=NOISE["noise10"], repeats=list(REPEATS),
        max_steps=200, model_sha256=MODEL_SHA256, planner_depth=DEPTH, planner_beam=BEAM,
        planner_prediction_noise=0.0, methods=METHODS,
        noise_seed_rule="[913714, layout] for r0; append positive replicate number for r1/r2",
        pairing="Common standardized noise draws by step. Amplitude scales with action size.",
        primary_metric="Success counts by layout and method; average layout-level success rates.",
        standardized_stream_sha256=stream_hashes,
        baseline_trial_sha256=baseline_protocol(baseline),
    )
    protocol = folder / "protocol.json"
    if protocol.exists():
        assert read_json(protocol) == design
    else:
        protocol.write_text(encode_json(design), encoding="utf-8")


def load_episode(folder, seed, condition, method, replicate):
    path = folder / f"{seed}-{condition}-{method}.json"
    row = read_json(path)
    assert (row["seed"], row["condition"], row["method"]) == (seed, condition, method), path
    assert row.get("noise_replicate", 0) == replicate, path
    assert row["noise_seed"] == noise_seed(seed, replicate), path
    assert row["noise_fraction"] == NOISE[condition] and row["waypoint_weight"] == METHODS[method], path
    trace = row["trace"]
    assert 0 < len(trace) == row["steps"] <= 200, path
    assert sum(row[k] for k in OUTCOMES) == 1 and (not row["timeout"] or row["steps"] == 200), path
    assert not any(x["terminal"] for x in trace[:-1]) and trace[-1]["terminal"], path
    assert row["damage"] == (row["peak"] > 1), path
    return row


def summarize(folder, baseline):
    cases = []
    for seed in SEEDS:
        hashes = set()
        for condition, reps in (("clean", (0,)), ("noise10", REPEATS)):
            for rep in reps:
                source = baseline if rep == 0 else folder / f"r{rep}"
                for method in METHODS:
                    row = load_episode(source, seed, condition, method, rep)
                    hashes.add(row["initial_observation_sha256"])
                    cases.append(dict(replicate=rep, **{k: row[k] for k in (
                        "seed", "condition", "method", "noise_fraction", *OUTCOMES, *METRICS,
                        "initial_observation_sha256",
                    )}))
        assert len(hashes) == 1, seed

    counts = []
    by_layout = []
    by_replicate = []
    for condition in NOISE:
        for method in METHODS:
            group = [r for r in cases if (r["condition"], r["method"]) == (condition, method)]
            per_layout = []
            for seed in SEEDS:
                local = [r for r in group if r["seed"] == seed]
                local_row = dict(condition=condition, method=method, seed=seed, trials=len(local),
                                 **{k: sum(r[k] for r in local) for k in OUTCOMES})
                per_layout.append(local_row)
                by_layout.append(local_row)
            row = dict(condition=condition, method=method, layouts=len(per_layout), trials=len(group),
                       **{k: sum(r[k] for r in group) for k in OUTCOMES},
                       mean_layout_success=mean(r["success"] / r["trials"] for r in per_layout),
                       layouts_all_success=sum(r["success"] == r["trials"] for r in per_layout))
            counts.append(row)
            print(row, flush=True)
            for rep in sorted({r["replicate"] for r in group}):
                subset = [r for r in group if r["replicate"] == rep]
                by_replicate.append(dict(condition=condition, method=method, replicate=rep,
                                         layouts=len(subset),
                                         **{k: sum(r[k] for r in subset) for k in OUTCOMES}))

    pairs = []
    noise = {(r["seed"], r["replicate"], r["method"]): r
             for r in cases if r["condition"] == "noise10"}
    for a, b in combinations(METHODS, 2):
        switches = [(noise[s, r, a]["success"], noise[s, r, b]["success"])
                    for s in SEEDS for r in REPEATS]
        a_only = sum(x and not y for x, y in switches)
        b_only = sum(y and not x for x, y in switches)
        pairs.append(dict(method_a=a, method_b=b, layouts=len(SEEDS), paired_trials=len(switches),
                          a_only_success=a_only, b_only_success=b_only,
                          both_success=sum(x and y for x, y in switches),
                          both_fail=sum(not x and not y for x, y in switches),
                          success_delta_b_minus_a_pp=100 * (b_only - a_only) / len(switches)))
    write_csv(folder / "episodes.csv", cases)
    write_csv(folder / "summary.csv", counts)
    write_csv(folder / "by_layout.csv", by_layout)
    write_csv(folder / "by_replicate.csv", by_replicate)
    write_csv(folder / "paired_noise.csv", pairs)
    print(f"Checked {len(cases)} episodes over {len(SEEDS)} layouts.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=HERE / "runs/fresh-noise-v1")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(RUN_ORDER))
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if not __debug__ or args.workers < 1 or len(set(args.methods)) != len(args.methods):
        parser.error("Leave assertions enabled. Use positive workers and unique methods.")
    assert digest(MODEL) == MODEL_SHA256
    folder = args.out.resolve()
    baseline = args.baseline.resolve()
    if folder == baseline:
        parser.error("Use a separate output directory for the repeated trials.")
    folder.mkdir(parents=True, exist_ok=True)
    freeze(folder, baseline)
    if not args.summarize_only:
        jobs = []
        for rep in REPEATS[1:]:
            output = folder / f"r{rep}"
            output.mkdir(exist_ok=True)
            jobs.extend(((seed, output, ("noise10",), tuple(args.methods)), rep) for seed in SEEDS)
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
            futures = [pool.submit(run_layout, job, rep) for job, rep in jobs]
            for future in as_completed(futures):
                future.result()
    if args.summarize_only or set(args.methods) == set(METHODS):
        summarize(folder, baseline)


if __name__ == "__main__":
    main()
