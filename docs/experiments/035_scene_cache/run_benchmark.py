"""Measure cold/warm reset caching with exact historical and paired trajectory audits."""
import importlib.util
import json
import platform
import shutil
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import torch

from me5418_harvesting.agents.dqn_agent import DQNAgent
from me5418_harvesting.agents.training import evaluate
from me5418_harvesting.envs.scene_cache import CertifiedSceneCache

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).parent
OUTPUT = ROOT / "results/evaluation/scene_cache_035_20261116"
MODEL = ROOT / "models/supervision_034_targeted.pt"
MODEL_SHA = "202fd8d4f6929da61e991962c6f667dd80a9cbcc330b1b744434230f140f8b8b"
HISTORY = ROOT / "docs/experiments/034_scene10002_supervision/evidence"
SEEDS = [10000, 10001, 10002]
spec = importlib.util.spec_from_file_location("cache035helpers", ROOT / "docs/experiments/028_frozen_generalization/run_frozen_generalization.py")
frozen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)


def save(name, value):
    frozen.save(OUTPUT / name, value)


def require_equal(a, b, label):
    if not frozen.equal(a, b):
        raise AssertionError(label)


def episode(seed, cache, pass_index, arm):
    started = perf_counter()
    agent, env, _ = DQNAgent.load(MODEL)
    load_seconds = perf_counter() - started
    env.scene_cache, env.profile_reset = cache, True
    for network in (agent.learner.online, agent.learner.target):
        network.eval()
        network.requires_grad_(False)
    before = frozen.state(agent)
    original_reset, original_act, original_step = env.reset, agent.act, env.step
    pending, reset_record, rows, timings = {}, {}, [], []
    audit_seconds = 0.0

    def reset(**kwargs):
        nonlocal audit_seconds
        obs, info = original_reset(**kwargs)
        tick = perf_counter()
        reset_record.update(frozen.plain({"observation": obs.copy(), "info": deepcopy(info),
                            "rng": deepcopy(env.np_random.bit_generator.state),
                            "rng_seed": env.np_random_seed, "geometry": frozen.geometry(env)}))
        audit_seconds += perf_counter() - tick
        return obs, info

    def capture_values(module, inputs, output):
        nonlocal audit_seconds
        tick = perf_counter()
        values = output.detach()[0]
        sorted_q = values.sort(descending=True).values
        pending.update(q_values=values.tolist(), q_top_gap=float(sorted_q[0] - sorted_q[1]))
        audit_seconds += perf_counter() - tick

    def act(obs, epsilon=0.0):
        nonlocal audit_seconds
        tick = perf_counter()
        pending.clear()
        pending.update(step=len(rows) + 1, position=env.scene.end_effector.copy(),
                       encoded_observation_sha256=frozen.encoded_hash(agent.encoder(obs)),
                       raw_observation=obs.copy(), rng_before=deepcopy(env.np_random.bit_generator.state))
        audit_seconds += perf_counter() - tick
        return original_act(obs, epsilon=epsilon)

    def step(action):
        nonlocal audit_seconds
        next_obs, reward, terminated, truncated, info = original_step(action)
        tick = perf_counter()
        row = frozen.plain({**pending, "action": action, "action_name": frozen.ACTION_NAMES[action],
            "next_position": env.scene.end_effector.copy(),
            "next_encoded_observation_sha256": frozen.encoded_hash(agent.encoder(next_obs)),
            "raw_next_observation": next_obs.copy(), "rng_after": deepcopy(env.np_random.bit_generator.state),
            "reward": reward, "terminated": terminated, "truncated": truncated, "info": info})
        frozen.require(action == max(range(8), key=lambda a: row["q_values"][a]), "Non-greedy action.")
        rows.append(row)
        audit_seconds += perf_counter() - tick
        return next_obs, reward, terminated, truncated, info

    env.reset, agent.act, env.step = reset, act, step
    hook = agent.learner.online.register_forward_hook(capture_values)
    try:
        result = evaluate(agent, env, 1, seed, timings=timings)["records"][0]
        profile = deepcopy(env.last_reset_profile)
        require_equal(before, frozen.state(agent), "Frozen learner or RNG changed.")
        row = {"arm": arm, "pass": pass_index, "seed": seed, "metrics": result,
               "timing": timings[0], "model_load_seconds": load_seconds,
               "audit_instrumentation_seconds": audit_seconds, "reset_profile": profile,
               "reset_record": reset_record, "frozen_state_unchanged": True}
        tick = perf_counter()
        name = f"{arm}_{pass_index}_{seed}.jsonl"
        with (OUTPUT / name).open("x", encoding="utf-8", newline="\n") as handle:
            for item in rows:
                handle.write(json.dumps(item, separators=(",", ":"), allow_nan=False) + "\n")
        row.update(trace=name, trace_sha256=frozen.digest(OUTPUT / name),
                   trace_export_seconds=perf_counter() - tick)
        print(f"{arm} pass {pass_index} seed {seed}: {profile['cache_status']}, "
              f"{timings[0]['episode_seconds']:.3f}s, success={result['success']}.", flush=True)
        return row, rows
    finally:
        hook.remove()
        env.close()


def run():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    save("status.json", {"status": "running", "started_at_utc": started})
    try:
        torch.set_num_threads(1)
        frozen.require(frozen.digest(MODEL) == MODEL_SHA, "Wrong model.")
        paths = list((ROOT / "src").rglob("*.py")) + [Path(__file__), MODEL,
            HERE.parent / "035_scene_cache.md", ROOT / "tests/test_scene_cache.py"]
        paths += list(HISTORY.glob("targeted_2000_*.jsonl")) + [HISTORY / "evaluations.json"]
        hashes = {p.relative_to(ROOT).as_posix(): frozen.digest(p) for p in paths}
        save("provenance.json", {"source_hashes": hashes, "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(), "python": platform.python_version(),
            "torch": torch.__version__, "threads": 1, "seeds": SEEDS, "passes": 3,
            "cache_capacity": 3, "optimizer_updates": 0, "action_overrides": 0})
        historical = {seed: [json.loads(line) for line in
            (HISTORY / f"targeted_2000_{seed}.jsonl").read_text().splitlines()] for seed in SEEDS}
        old = json.loads((HISTORY / "evaluations.json").read_text())
        cache, records, pair_checks = CertifiedSceneCache(3), [], []
        for iteration in range(3):
            for seed in SEEDS:
                pair = {}
                order = ("cached", "uncached") if iteration == 1 else ("uncached", "cached")
                for arm in order:
                    record, rows = episode(seed, cache if arm == "cached" else None, iteration, arm)
                    require_equal(len(rows), len(historical[seed]), "Historical episode length")
                    for actual, expected in zip(rows, historical[seed], strict=True):
                        require_equal({k: actual[k] for k in expected}, expected, "Historical transition")
                    expected = next(r for r in old if r["arm"] == "targeted" and r["update"] == 2000 and r["seed"] == seed)
                    require_equal(record["metrics"], expected["metrics"], "Historical metrics")
                    require_equal(record["reset_record"]["info"], expected["reset_info"], "Historical reset info")
                    pair[arm] = (record, rows)
                    records.append(record)
                    save("episodes.json", records)
                require_equal(pair["uncached"][1], pair["cached"][1], "Paired raw observations, Q, RNG and transitions")
                require_equal(pair["uncached"][0]["reset_record"], pair["cached"][0]["reset_record"], "Paired reset state")
                pair_checks.append({"pass": iteration, "seed": seed, "all_transitions_exact": True})
                save("pair_verification.json", pair_checks)
        uncached = [r for r in records if r["arm"] == "uncached"]
        cached = [r for r in records if r["arm"] == "cached"]
        total = lambda rows, key: sum(r["timing"][key] for r in rows)
        warm = [r for r in cached if r["pass"] > 0]
        matched = [r for r in uncached if r["pass"] > 0]
        frozen.require(all(r["reset_profile"]["cache_status"] == "hit" and
                           r["reset_profile"]["certification_actions"] == 0 for r in warm), "Warm cache reran certification.")
        save("summary.json", {"all_pairs_exact": True, "all_historical_traces_exact": True,
            "optimizer_updates": 0, "episodes": len(records), "action_overrides": 0,
            "episode_actions": sum(r["metrics"]["steps"] for r in records),
            "actual_certification_actions": sum(r["reset_profile"]["certification_actions"] for r in records),
            "actual_certification_calls": sum(r["reset_profile"]["certification_calls"] for r in records),
            "cold_cache_episode_seconds": total(cached[:3], "episode_seconds"),
            "cold_uncached_episode_seconds": total(uncached[:3], "episode_seconds"),
            "uncached_total_seconds": total(uncached, "episode_seconds"),
            "cached_total_seconds_including_population": total(cached, "episode_seconds"),
            "whole_workload_speedup": total(uncached, "episode_seconds") / total(cached, "episode_seconds"),
            "warm_episode_speedup": total(matched, "episode_seconds") / total(warm, "episode_seconds"),
            "warm_reset_speedup": total(matched, "reset_seconds") / total(warm, "reset_seconds"),
            "segments": {arm: {k: total(rows, k) for k in ("episode_seconds", "reset_seconds", "policy_seconds", "physics_seconds")}
                         for arm, rows in (("uncached", uncached), ("cached", cached), ("warm_cached", warm), ("matched_uncached", matched))},
            "scope": "Single-process frozen evaluation; not an end-to-end training speedup."})
        for name, sha in hashes.items():
            frozen.require(frozen.digest(ROOT / name) == sha, "Input changed during benchmark.")
        save("status.json", {"status": "completed", "started_at_utc": started,
                             "finished_at_utc": datetime.now(timezone.utc).isoformat()})
    except BaseException as exc:
        save("status.json", {"status": "failed", "started_at_utc": started, "error": str(exc)})
        raise
    archive = HERE / "evidence"
    archive.mkdir()
    files = {}
    for path in OUTPUT.iterdir():
        dest = archive / path.name
        shutil.copy2(path, dest)
        frozen.require(path.read_bytes() == dest.read_bytes(), "Archive bytes changed.")
        files[path.name] = {"sha256": frozen.digest(path), "bytes": path.stat().st_size}
    frozen.save(archive / "archive_manifest.json", {"source_run": OUTPUT.relative_to(ROOT).as_posix(), "files": files})


if __name__ == "__main__":
    run()
