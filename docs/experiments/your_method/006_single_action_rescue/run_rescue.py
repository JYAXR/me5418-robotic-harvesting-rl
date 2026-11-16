"""Test four single-action interventions on the frozen attempt-004 endpoint."""

import importlib.util
import json
import platform
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import torch

from me5418_harvesting.agents.training import evaluate

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUTPUT = HERE / "evidence"
HISTORY = HERE.parent / "005_frozen_divergence/evidence"
CASES = {290012: (19, 6), 290021: (27, 0), 290017: (5, 5), 290022: (9, 0)}
spec = importlib.util.spec_from_file_location("own006prior", HERE.parent / "005_frozen_divergence/run_diagnosis.py")
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
frozen, equal, require = prior.frozen, prior.equal, prior.require
read, digest = prior.read, prior.digest


def selected_action(seed, step, policy, corrected):
    fork, replacement = CASES[seed]
    forced = corrected and step == fork
    return (replacement if forced else policy), forced


def strip_intervention(row):
    return {k: v for k, v in row.items() if k not in ("policy_action", "forced_action")}


def audit_pair(seed, baseline, corrected):
    """Reject any additional override or unequal pre-intervention state."""
    step, replacement = CASES[seed]
    require(len(baseline) >= step and len(corrected) >= step, "Intervention was not reached.")
    equal(baseline[:step - 1], corrected[:step - 1], "Exact common trajectory prefix")
    for key in ("position", "encoded_observation_sha256", "raw_observation", "rng_before",
                "q_values", "q_top_gap", "policy_action"):
        equal(baseline[step - 1][key], corrected[step - 1][key], "Identical intervention input and policy")
    for treatment, rows in ((False, baseline), (True, corrected)):
        expected_forced = [step] if treatment else []
        forced_steps = []
        for index, row in enumerate(rows, 1):
            require(row["step"] == index, "Trace ordering changed.")
            require(row["policy_action"] == max(range(8), key=lambda a: row["q_values"][a]),
                    "Policy was not greedy.")
            action, forced = selected_action(seed, index, row["policy_action"], treatment)
            require(row["action"] == action and row["forced_action"] == forced, "Unexpected action replacement.")
            if forced:
                forced_steps.append(index)
        equal(forced_steps, expected_forced, "Exact intervention count")
    row = baseline[step - 1]
    require(row["action"] != replacement, "Intervention does not change the selected action.")
    return {"seed": seed, "step": step, "replacement_action": replacement,
            "replacement_name": frozen.ACTION_NAMES[replacement],
            "policy_action": row["action"], "policy_name": row["action_name"],
            "policy_q": row["q_values"][row["action"]], "replacement_q": row["q_values"][replacement],
            "policy_minus_replacement_q": row["q_values"][row["action"]] - row["q_values"][replacement],
            "identical_prefix_steps": step - 1, "identical_fork_input_and_q": True,
            "baseline_overrides": 0, "corrected_overrides": 1, "all_other_actions_autonomous": True}


def episode(seed, corrected):
    """Trace a normal evaluation with exactly one optional action replacement."""
    arm = "corrected" if corrected else "baseline"
    pass_index = 0
    cache = None
    started = perf_counter()
    agent, env, _ = prior.FrozenSnapshotLoader.load(prior.MODEL)
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
        require(epsilon == 0.0, "Exploration is forbidden.")
        policy = original_act(obs, epsilon=epsilon)
        action, forced = selected_action(seed, len(rows) + 1, policy, corrected)
        pending.update(policy_action=policy, forced_action=forced)
        return action

    def step(action):
        nonlocal audit_seconds
        next_obs, reward, terminated, truncated, info = original_step(action)
        tick = perf_counter()
        row = frozen.plain({**pending, "action": action, "action_name": frozen.ACTION_NAMES[action],
            "next_position": env.scene.end_effector.copy(),
            "next_encoded_observation_sha256": frozen.encoded_hash(agent.encoder(next_obs)),
            "raw_next_observation": next_obs.copy(), "rng_after": deepcopy(env.np_random.bit_generator.state),
            "reward": reward, "terminated": terminated, "truncated": truncated, "info": info})
        require(row["policy_action"] == max(range(8), key=lambda a: row["q_values"][a]), "Non-greedy policy.")
        expected_action, forced = selected_action(seed, row["step"], row["policy_action"], corrected)
        require(action == expected_action and row["forced_action"] == forced, "Unexpected intervention.")
        rows.append(row)
        audit_seconds += perf_counter() - tick
        return next_obs, reward, terminated, truncated, info

    env.reset, agent.act, env.step = reset, act, step
    hook = agent.learner.online.register_forward_hook(capture_values)
    try:
        result = evaluate(agent, env, 1, seed, timings=timings)["records"][0]
        profile = deepcopy(env.last_reset_profile)
        equal(before, frozen.state(agent), "Frozen learner or RNG changed.")
        row = {"arm": arm, "pass": pass_index, "seed": seed, "metrics": result,
               "timing": timings[0], "model_load_seconds": load_seconds,
               "audit_instrumentation_seconds": audit_seconds, "reset_profile": profile,
               "reset_record": reset_record, "frozen_state_unchanged": True, "full_snapshot_unchanged": True,
               "forced_actions": sum(r["forced_action"] for r in rows)}
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
    OUTPUT.mkdir(exist_ok=False)
    started, tick = datetime.now(timezone.utc).isoformat(), perf_counter()
    frozen.save(OUTPUT / "status.json", {"status": "running", "started_at_utc": started})
    records, checks = [], []
    try:
        torch.set_num_threads(1)
        require(digest(prior.MODEL) == prior.MODEL_SHA, "Wrong inference model.")
        require(digest(prior.SNAPSHOT) == prior.SNAPSHOT_SHA, "Wrong full snapshot.")
        historical = {r["seed"]: r for r in read(HISTORY / "episodes.json")}
        for name, sha in read(HISTORY / "manifest.json").items():
            require(digest(HISTORY / name) == sha, "Historical evidence changed.")
        for name, sha in read(HISTORY / "provenance.json")["input_hashes"].items():
            require(digest(ROOT / name) == sha, "Historical input changed.")
        paths = list((ROOT / "src").rglob("*.py")) + list((ROOT / "docs/experiments").rglob("*.py"))
        paths += [prior.MODEL, prior.SNAPSHOT, HERE / "PROTOCOL.md"]
        paths += [p for p in HISTORY.iterdir() if p.is_file()]
        hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}
        source = torch.load(prior.SNAPSHOT, weights_only=True)
        require(source["update_count"] == 14000, "Unexpected TD clock.")
        require(source["optimizer_summary"]["step_min"] == source["optimizer_summary"]["step_max"] == 34000,
                "Unexpected Adam clock.")
        payload = torch.load(prior.MODEL, weights_only=True)
        for key in ("online", "environment_config", "dqn_config"):
            equal(payload[key], source[key], "Inference and resume endpoint match")
        frozen.save(OUTPUT / "provenance.json", {"input_hashes": hashes,
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
            "python": platform.python_version(), "torch": torch.__version__, "torch_threads": 1,
            "cases": CASES, "environment": source["environment_config"], "dqn": source["dqn_config"],
            "td_clock": source["update_count"], "adam": source["optimizer_summary"],
            "optimizer_updates": 0, "new_training_samples": 0})
        for seed in CASES:
            pair = {}
            for corrected in (False, True):
                record, rows = episode(seed, corrected)
                records.append(record)
                frozen.save(OUTPUT / "episodes.json", records)
                pair[corrected] = record, rows
                equal(record["reset_record"], historical[seed]["reset_record"], "Historical seeded reset")
                if not corrected:
                    old = [json.loads(line) for line in (HISTORY / historical[seed]["trace"]).read_text().splitlines()]
                    equal([strip_intervention(r) for r in rows], old, "Historical complete baseline trace")
                    equal(record["metrics"], historical[seed]["metrics"], "Historical baseline metrics")
            check = audit_pair(seed, pair[False][1], pair[True][1])
            check.update(historical_baseline_exact=True,
                         baseline_metrics=pair[False][0]["metrics"], corrected_metrics=pair[True][0]["metrics"])
            checks.append(check)
            frozen.save(OUTPUT / "pair_checks.json", checks)
        for name, sha in hashes.items():
            require(digest(ROOT / name) == sha, "Pinned input changed.")
        frozen.save(OUTPUT / "summary.json", {"episodes": len(records), "pairs": len(checks),
            "baseline_successes": sum(c["baseline_metrics"]["success"] for c in checks),
            "corrected_successes": sum(c["corrected_metrics"]["success"] for c in checks),
            "episode_actions": sum(r["metrics"]["steps"] for r in records),
            "certification_actions": sum(r["reset_profile"]["certification_actions"] for r in records),
            "certification_calls": sum(r["reset_profile"]["certification_calls"] for r in records),
            "action_overrides": sum(r["forced_actions"] for r in records), "optimizer_updates": 0,
            "full_snapshot_unchanged": True, "all_historical_baselines_exact": True,
            "elapsed_seconds": perf_counter() - tick,
            "scope": "Assisted single-action rescue of four development scenes, not learned recovery or generalization."})
        frozen.save(OUTPUT / "status.json", {"status": "completed", "started_at_utc": started,
                                            "finished_at_utc": datetime.now(timezone.utc).isoformat()})
        frozen.save(OUTPUT / "manifest.json", {p.name: digest(p) for p in sorted(OUTPUT.iterdir()) if p.is_file()})
    except BaseException as exc:
        frozen.save(OUTPUT / "status.json", {"status": "failed", "started_at_utc": started,
            "completed_episodes": len(records), "error": repr(exc)})
        raise


if __name__ == "__main__":
    run()
