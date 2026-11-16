"""Test exactly one reference-action correction on each of three frozen scenes."""

import gzip
import hashlib
import importlib.util
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from me5418_harvesting.agents.training import evaluate

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUTPUT = HERE / "evidence"
HISTORY = HERE.parent / "001_training_coverage/evidence"
DATA = ROOT / "docs/experiments/029_training_coverage/evidence"
MODEL = ROOT / "models/supervision_034_targeted.pt"
MODEL_SHA = "202fd8d4f6929da61e991962c6f667dd80a9cbcc330b1b744434230f140f8b8b"
SOURCE = ROOT / "docs/experiments/034_scene10002_supervision/checkpoints/targeted_state_12000.pt"
SOURCE_SHA = "fe8e29438165cd4d99c6e143f819a37a2febc5e704a47efb9a895bb1074c610f"
CASES = {290007: (8, 0, 430), 290012: (19, 2, 653), 290021: (12, 0, 1052)}
REPLACEMENT = 6
HELPER = ROOT / "docs/experiments/031_scene10002_rescue/run_rescue.py"
spec = importlib.util.spec_from_file_location("own_route_rescue_helpers", HELPER)
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
frozen, equal, require = legacy.frozen, legacy.equal, legacy.require


def save(path, value):
    frozen.save(path, value)


def configure(seed, directory):
    """Use the parameterized primary rollout and pair auditor, not hardcoded replay."""
    legacy.SEED = seed
    legacy.FORK_STEP = CASES[seed][0]
    legacy.REPLACEMENT = REPLACEMENT
    legacy.OUTPUT = directory


def independent_replay(source, seed, corrected, expected, expected_result, directory):
    """Verify a fresh normal-evaluator rollout with just the predeclared override."""
    agent, env = legacy.load_frozen(source)
    original_act, original_step = agent.act, env.step
    seen, checks, pending = [], [], {}
    fork_step = CASES[seed][0]

    def observed_step(action):
        require(len(seen) < len(expected), "Replay exceeded expected trajectory.")
        row = expected[len(seen)]
        require(action == row["action"], "Replay action mismatch.")
        obs, reward, terminated, truncated, info = original_step(action)
        actual = {**pending, "next_position": env.scene.end_effector.tolist(),
                  "next_encoded_observation_sha256": frozen.encoded_hash(agent.encoder(obs)),
                  "reward": reward, "terminated": terminated, "truncated": truncated,
                  "info": frozen.plain(info)}
        equal(actual, row, "Independent full transition")
        seen.append(actual)
        return obs, reward, terminated, truncated, info

    def audited_act(obs, epsilon=0.0):
        require(epsilon == 0.0, "Replay exploration is forbidden.")
        step = len(seen) + 1
        if step == 1:
            equal(frozen.plain(frozen.geometry(env)), expected_result["initial_geometry"],
                  "Independent reset geometry")
            # Install only after reset, so certification is not intercepted.
            env.step = observed_step
        encoded = agent.encoder(obs)
        with torch.no_grad():
            q = agent.learner.online(torch.from_numpy(encoded).unsqueeze(0))[0]
        policy = original_act(obs, epsilon=0.0)
        forced = corrected and step == fork_step
        action = REPLACEMENT if forced else policy
        sorted_q = q.sort(descending=True).values
        pending.clear()
        pending.update({"step": step, "position": env.scene.end_effector.tolist(),
            "encoded_observation_sha256": frozen.encoded_hash(encoded),
            "q_values": q.tolist(), "q_top_gap": float(sorted_q[0] - sorted_q[1]),
            "policy_action": policy, "action": action, "action_name": frozen.ACTION_NAMES[action],
            "forced_action": forced, "scripted_reference": False})
        return action

    agent.act = audited_act
    try:
        with frozen.certification_checks(checks):
            metrics = evaluate(agent, env, 1, seed)["records"][0]
        require(len(seen) == len(expected), "Replay ended early.")
        equal(metrics, expected_result["metrics"], "Independent metrics")
        equal(checks, expected_result["certification"], "Independent certification")
        legacy.finish_state(agent, source)
        result = {"seed": seed, "mode": "corrected" if corrected else "baseline",
                  "metrics": metrics, "certification": checks, "full_trace_matches": True,
                  "frozen_full_state_verified": True,
                  "forced_actions": sum(row["forced_action"] for row in seen)}
        save(directory / f"verification_{result['mode']}.json", result)
        print(f"Verified seed {seed} {result['mode']}: success={metrics['success']}, steps={metrics['steps']}", flush=True)
        return result
    finally:
        env.close()


def inputs():
    require(frozen.digest(MODEL) == MODEL_SHA, "Inference model hash mismatch.")
    require(frozen.digest(SOURCE) == SOURCE_SHA, "Full snapshot hash mismatch.")
    source = torch.load(SOURCE, weights_only=True)
    inference = torch.load(MODEL, weights_only=True)
    for key in ("online", "environment_config", "dqn_config"):
        equal(source[key], inference[key], "Snapshot matches inference model")
    require(source["update_count"] == 12000, "Unexpected TD clock.")
    require(source["optimizer_summary"]["step_min"] == source["optimizer_summary"]["step_max"] == 32000,
            "Unexpected Adam clock.")
    metadata = legacy.read(DATA / "dataset.json")
    require(frozen.digest(DATA / "expanded_demonstrations.npz") == metadata["npz_sha256"], "Dataset changed.")
    history_records = legacy.read(HISTORY / "episodes.json")
    references, histories, metrics, forks = {}, {}, {}, {}
    with np.load(DATA / "expanded_demonstrations.npz") as arrays:
        for seed, (step, policy, dataset_row) in CASES.items():
            bounds = next(e for e in metadata["episodes"] if e["seed"] == seed)
            require(bounds["start"] + step - 1 == dataset_row, "Dataset index changed.")
            references[seed] = {k: arrays[k][bounds["start"]:bounds["stop"]].copy() for k in arrays}
            record = next(r for r in history_records if r["arm"] == "targeted" and r["seed"] == seed)
            path = HISTORY / record["trace"]
            require(frozen.digest(path) == record["trace_compressed_sha256"], "Historical trace changed.")
            raw = gzip.decompress(path.read_bytes())
            require(hashlib.sha256(raw).hexdigest() == record["trace_sha256"], "Historical raw trace changed.")
            histories[seed] = [json.loads(line) for line in raw.decode().splitlines()]
            forks[seed] = legacy.first_fork(histories[seed], references[seed])
            require((forks[seed]["step"], forks[seed]["policy_action"], forks[seed]["reference_action"])
                    == (step, policy, REPLACEMENT), "First disagreement changed.")
            metrics[seed] = record["metrics"]
    return source, references, histories, metrics, forks


def run():
    OUTPUT.mkdir(exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    clock = perf_counter()
    save(OUTPUT / "status.json", {"status": "running", "started_at_utc": started})
    results, replays, pair_checks = [], [], []
    try:
        torch.set_num_threads(1)
        source, references, histories, metrics, forks = inputs()
        paths = list((ROOT / "src").rglob("*.py")) + list((ROOT / "docs/experiments").rglob("*.py"))
        paths += [MODEL, SOURCE, HERE / "PROTOCOL.md", DATA / "dataset.json",
                  DATA / "expanded_demonstrations.npz", HISTORY / "episodes.json"]
        paths += [HISTORY / f"targeted_0_{seed}.jsonl.gz" for seed in CASES]
        hashes = {p.relative_to(ROOT).as_posix(): frozen.digest(p) for p in paths}
        save(OUTPUT / "provenance.json", {"input_hashes": hashes,
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
            "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "torch_threads": 1, "environment": source["environment_config"], "dqn": source["dqn_config"],
            "td_clock": source["update_count"], "adam": source["optimizer_summary"],
            "forks": forks, "optimizer_updates": 0, "new_training_samples": 0})
        for seed in CASES:
            directory = OUTPUT / str(seed)
            directory.mkdir()
            configure(seed, directory)
            traces, outcomes = {}, {}
            for mode in ("baseline", "corrected"):
                traces[mode], outcomes[mode] = legacy.main_rollout(source, references[seed], mode)
                results.append({"seed": seed, **outcomes[mode]})
                save(OUTPUT / "episodes.json", results)
                if mode == "baseline":
                    equal(outcomes[mode]["metrics"], metrics[seed], "Historical baseline metrics")
                    require(len(traces[mode]) == len(histories[seed]), "Historical trajectory length")
                    for actual, expected in zip(traces[mode], histories[seed], strict=True):
                        for key in actual.keys() & expected.keys():
                            equal(actual[key], expected[key], f"Historical transition {key}")
            equal(outcomes["baseline"]["initial_geometry"], outcomes["corrected"]["initial_geometry"], "Paired geometry")
            equal(outcomes["baseline"]["reset_info"], outcomes["corrected"]["reset_info"], "Paired reset info")
            pair_checks.append({"seed": seed, **legacy.audit_pair(traces["baseline"], traces["corrected"]),
                                "historical_baseline_exact": True})
            save(OUTPUT / "pair_checks.json", pair_checks)
            for mode in ("baseline", "corrected"):
                replays.append(independent_replay(source, seed, mode == "corrected", traces[mode], outcomes[mode], directory))
                save(OUTPUT / "replays.json", replays)
        for name, expected in hashes.items():
            require(frozen.digest(ROOT / name) == expected, "Source or model changed.")
        all_runs = results + replays
        save(OUTPUT / "summary.json", {"primary_episodes": len(results), "verification_episodes": len(replays),
            "outcomes": [{"seed": r["seed"], "mode": r["mode"], **r["metrics"]} for r in results],
            "primary_forced_actions": sum(r["forced_actions"] for r in results),
            "verification_forced_actions": sum(r["forced_actions"] for r in replays),
            "episode_actions": sum(r["metrics"]["steps"] for r in all_runs),
            "certification_actions": sum(c["steps"] for r in all_runs for c in r["certification"]),
            "certification_calls": sum(len(r["certification"]) for r in all_runs),
            "optimizer_updates": 0, "new_training_samples": 0, "all_checks_passed": True,
            "elapsed_seconds": perf_counter() - clock,
            "scope": "One-action intervention on three training scenes; not learned or generalized recovery."})
        save(OUTPUT / "status.json", {"status": "completed", "started_at_utc": started,
                                     "finished_at_utc": datetime.now(timezone.utc).isoformat()})
        save(OUTPUT / "manifest.json", {p.relative_to(OUTPUT).as_posix(): frozen.digest(p)
                                        for p in sorted(OUTPUT.rglob("*")) if p.is_file()})
    except BaseException as exc:
        save(OUTPUT / "status.json", {"status": "failed", "started_at_utc": started,
                                     "completed_primary_episodes": len(results), "error": repr(exc)})
        raise


if __name__ == "__main__":
    run()
