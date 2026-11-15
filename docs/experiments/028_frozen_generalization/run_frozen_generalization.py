"""Evaluate one immutable policy on a predeclared unseen-scene set, without learning."""

import hashlib
import json
import math
import platform
import subprocess
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from me5418_harvesting.agents.dqn_agent import DQNAgent
from me5418_harvesting.agents.training import episode_metrics, evaluate
from me5418_harvesting.evaluation import scene_feasibility

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "results/evaluation/frozen_generalization_028_20261115"
MODEL = ROOT / "models/step6_repair_027.pt"
MODEL_SHA = "06733fccb5422d10ad7246e5368bbd5ed9cc3367f0b413847544ec26a96b0c88"
TRAIN_SEEDS = [10000, 10001, 10002]
TEST_SEEDS = list(range(280000, 280100))
ACTION_NAMES = ["north_small", "south_small", "east_small", "west_small",
                "north_large", "south_large", "east_large", "west_large"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encoded_hash(array):
    return hashlib.sha256(np.asarray(array, dtype=np.float32).tobytes()).hexdigest()


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def save(path, value):
    path.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def state(agent):
    return deepcopy({"online": agent.learner.online.state_dict(),
        "target": agent.learner.target.state_dict(),
        "optimizer": agent.learner.optimizer.state_dict(),
        "updates": agent.learner.update_count, "rng": agent.rng.bit_generator.state,
        "torch_rng": torch.get_rng_state()})


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(v, b[k]) for k, v in a.items())
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def load_frozen():
    require(digest(MODEL) == MODEL_SHA, "Frozen model SHA-256 mismatch.")
    agent, env, metadata = DQNAgent.load(MODEL)
    require(metadata["scene_seeds"] == TRAIN_SEEDS, "Unexpected training seed lineage.")
    for net in (agent.learner.online, agent.learner.target):
        net.eval()
        net.requires_grad_(False)
    def forbidden(*args, **kwargs):
        raise RuntimeError("Optimizer steps are forbidden during frozen evaluation.")
    agent.learner.optimizer.step = forbidden
    return agent, env


@contextmanager
def certification_checks(records):
    """Observe certification calls, including rejected reset candidates, unchanged."""
    original = scene_feasibility.verify_reference_route
    def recorded(env):
        result = original(env)
        records.append(deepcopy(result))
        return result
    scene_feasibility.verify_reference_route = recorded
    try:
        yield
    finally:
        scene_feasibility.verify_reference_route = original


def wilson(successes, count):
    if count == 0:
        return None
    z = 1.959963984540054
    rate = successes / count
    center = (rate + z * z / (2 * count)) / (1 + z * z / count)
    radius = z * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count**2)) / (1 + z * z / count)
    return [max(0.0, center - radius), min(1.0, center + radius)]


def failure_signature(metrics, trailing_stationary):
    if metrics["success"]:
        return "success"
    if metrics["damage"]:
        return "damage"
    if metrics["budget_exhausted"]:
        if trailing_stationary >= 20:
            return ("budget_blocked_stall" if metrics["blocked_steps"] >= metrics["steps"] / 2
                    else "budget_stationary_stall")
        return "budget_while_moving"
    return "external_truncation"


def geometry(env):
    return {"start": env.scene.end_effector.copy(), "target": env.scene.target.copy(),
        "layout": env.scene.layout, "passage_kinds": env.scene.passage_kinds,
        "branches": [{"rest_nodes": b.rest_nodes.copy(), "stiffness": b.stiffness,
                       "safe_limit": b.safe_limit} for b in env.scene.branches]}


def rollout(agent, env, seed, path):
    before = state(agent)
    checks = []
    try:
        with certification_checks(checks):
            observation, reset_info = env.reset(seed=seed)
    except ValueError as exc:
        require(equal(before, state(agent)), "State changed during failed reset.")
        return {"seed": seed, "status": "reset_error", "error": str(exc),
                "category": "reset_error", "certification": checks, "policy_actions": 0}
    require(checks == reset_info["reference_route_checks"], "Certification accounting mismatch.")
    require(checks[-1]["success"], "Accepted scene has no witnessed route.")
    initial = plain(geometry(env))
    initial_hash = encoded_hash(agent.encoder(observation))
    total, stationary, blocked, trailing, longest = 0.0, 0, 0, 0, 0
    first = {"blocked": None, "stationary": None, "contact": None}
    counts = Counter()
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for step in range(1, env.max_steps + 1):
            encoded = agent.encoder(observation)
            with torch.no_grad():
                values = agent.learner.online(torch.from_numpy(encoded).unsqueeze(0))[0]
            action = agent.act(observation)
            require(action == int(values.argmax()), "Policy action is not greedy.")
            sorted_q = values.sort(descending=True).values
            position = env.scene.end_effector.copy()
            next_obs, reward, terminated, truncated, info = env.step(action)
            stopped = bool(np.linalg.norm(info["executed_motion"]) <= 1e-6)
            is_blocked = bool(info["collision_blocked"])
            stationary += int(stopped)
            blocked += int(is_blocked)
            trailing = trailing + 1 if stopped else 0
            longest = max(longest, trailing)
            for key, event in (("blocked", is_blocked), ("stationary", stopped),
                               ("contact", info["contact_steps"] > 0)):
                if event and first[key] is None:
                    first[key] = step
            total += reward
            counts[ACTION_NAMES[action]] += 1
            row = {"step": step, "position": position, "next_position": env.scene.end_effector.copy(),
                "encoded_observation_sha256": encoded_hash(encoded),
                "next_encoded_observation_sha256": encoded_hash(agent.encoder(next_obs)),
                "action": action, "action_name": ACTION_NAMES[action], "q_values": values.tolist(),
                "q_top_gap": float(sorted_q[0] - sorted_q[1]), "reward": reward,
                "terminated": terminated, "truncated": truncated, "info": info}
            handle.write(json.dumps(plain(row), separators=(",", ":"), allow_nan=False) + "\n")
            observation = next_obs
            if terminated or truncated:
                break
        else:
            raise AssertionError("The unchanged task budget did not terminate the episode.")
    metrics = {"seed": seed, **episode_metrics(total, step, info, stationary, blocked)}
    require(equal(before, state(agent)), "Frozen agent state changed during rollout.")
    return {"seed": seed, "status": "evaluated", "category": failure_signature(metrics, trailing),
        "metrics": metrics, "initial_geometry": initial, "initial_encoded_sha256": initial_hash,
        "reset_info": reset_info, "certification": checks, "first_events": first,
        "trailing_stationary_steps": trailing, "longest_stationary_run": longest,
        "action_counts": dict(counts), "terminal_position": env.scene.end_effector.copy(),
        "terminal_contact_steps": info["contact_steps"], "policy_actions": step,
        "trace": path.name, "trace_sha256": digest(path), "frozen_state_verified": True}


def summary(rows):
    valid = [r for r in rows if r["status"] == "evaluated"]
    successes = sum(r["metrics"]["success"] for r in valid)
    return {"requested_scenes": len(rows), "valid_scenes": len(valid),
        "successes": successes, "success_rate_requested": successes / len(rows) if rows else None,
        "success_rate_valid": successes / len(valid) if valid else None,
        "wilson95_requested": wilson(successes, len(rows)),
        "wilson95_valid": wilson(successes, len(valid)),
        "categories": dict(Counter(r["category"] for r in rows)),
        "mean_return": float(np.mean([r["metrics"]["return"] for r in valid])) if valid else None,
        "mean_steps": float(np.mean([r["metrics"]["steps"] for r in valid])) if valid else None,
        "mean_success_steps": float(np.mean([r["metrics"]["steps"] for r in valid if r["metrics"]["success"]])) if successes else None,
        "policy_actions": sum(r["policy_actions"] for r in rows),
        "certification_actions": sum(c["steps"] for r in rows for c in r["certification"]),
        "certification_attempts": sum(len(r["certification"]) for r in rows)}


def run():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    save(OUTPUT / "status.json", {"status": "running", "started_at_utc": started})
    torch.set_num_threads(1)
    sources = list((ROOT / "src/me5418_harvesting").rglob("*.py")) + [Path(__file__),
        ROOT / "docs/experiments/028_frozen_generalization.md",
        ROOT / "tests/test_frozen_generalization.py", MODEL,
        ROOT / "docs/experiments/011_multiscene_retention/evidence/dataset.json",
        ROOT / "models/step6_repair_027.json"]
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in sources}
    dataset = json.loads(sources[-2].read_text())
    require(dataset["seeds"] == TRAIN_SEEDS and not set(TRAIN_SEEDS) & set(TEST_SEEDS), "Scene split overlap.")
    training_hashes = {r["initial_observation_sha256"] for r in dataset["episodes"]}
    config = torch.load(MODEL, weights_only=True)["environment_config"]
    save(OUTPUT / "provenance.json", {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "source_hashes": hashes, "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: version(name) for name in ("torch", "numpy", "gymnasium")},
        "training_seeds": TRAIN_SEEDS, "test_seeds": TEST_SEEDS, "environment": config,
        "training_initial_hashes": sorted(training_hashes), "optimizer_updates": 0})
    rows, controls = [], []
    seen = set()
    agent, env = load_frozen()
    try:
        for seed in TRAIN_SEEDS:
            row = rollout(agent, env, seed, OUTPUT / f"control_before_{seed}.jsonl")
            require(row["category"] == "success", "Known-scene control regressed before testing.")
            controls.append({"phase": "before", **row})
        save(OUTPUT / "controls.json", controls)
        for seed in TEST_SEEDS:
            row = rollout(agent, env, seed, OUTPUT / f"scene_{seed}.jsonl")
            rows.append(row)
            save(OUTPUT / "episodes.json", rows)
            if row["status"] == "evaluated":
                h = row["initial_encoded_sha256"]
                require(h not in training_hashes and h not in seen, "Duplicate training/test scene input.")
                seen.add(h)
            save(OUTPUT / "progress.json", summary(rows))
            print(json.dumps({"seed": seed, "category": row["category"], "completed": len(rows),
                "successes": sum(r["category"] == "success" for r in rows)}), flush=True)
        for seed in TRAIN_SEEDS:
            row = rollout(agent, env, seed, OUTPUT / f"control_after_{seed}.jsonl")
            original = next(r for r in controls if r["seed"] == seed)
            require(row["metrics"] == original["metrics"] and row["trace_sha256"] == original["trace_sha256"], "Control changed after evaluation.")
            controls.append({"phase": "after", **row})
        save(OUTPUT / "controls.json", controls)
    finally:
        env.close()
    valid = [r for r in rows if r["status"] == "evaluated"]
    selected = {r["seed"] for r in valid[:5]}
    for category in {r["category"] for r in valid}:
        selected.add(min(r["seed"] for r in valid if r["category"] == category))
    verification = []
    for seed in sorted(selected):
        agent, env = load_frozen()
        before = state(agent)
        checks = []
        try:
            with certification_checks(checks):
                actual = evaluate(agent, env, 1, seed)
            original = next(r for r in rows if r["seed"] == seed)
            require(actual["records"][0] == original["metrics"], "Normal evaluator metrics differ.")
            require(checks == original["certification"], "Reload certification differs.")
            require(equal(before, state(agent)), "Frozen state changed during reload.")
            verification.append({"seed": seed, "metrics": actual["records"][0], "certification": checks,
                "policy_actions": actual["records"][0]["steps"], "passed": True})
            save(OUTPUT / "verification.json", {"passed": False, "records": verification})
            print(f"Verified normal-evaluator reload for seed {seed}.", flush=True)
        finally:
            env.close()
    save(OUTPUT / "verification.json", {"passed": True, "records": verification, "optimizer_updates": 0})
    main_summary = summary(rows)
    save(OUTPUT / "summary.json", main_summary)
    phases = {"main": {"resets": len(rows), **{k: main_summary[k] for k in ("policy_actions", "certification_actions", "certification_attempts")}},
        "controls": {"resets": len(controls), **{k: summary(controls)[k] for k in ("policy_actions", "certification_actions", "certification_attempts")}},
        "verification": {"resets": len(verification), "policy_actions": sum(r["policy_actions"] for r in verification),
            "certification_actions": sum(c["steps"] for r in verification for c in r["certification"]),
            "certification_attempts": sum(len(r["certification"]) for r in verification)}}
    save(OUTPUT / "accounting.json", {"phases": phases, "optimizer_updates": 0,
        "total_resets": sum(p["resets"] for p in phases.values()),
        "total_policy_actions": sum(p["policy_actions"] for p in phases.values()),
        "total_certification_actions": sum(p["certification_actions"] for p in phases.values()),
        "scripted_actions": 0, "training_data_added": 0})
    for name, sha in hashes.items():
        require(digest(ROOT / name) == sha, f"Source changed: {name}")
    save(OUTPUT / "status.json", {"status": "completed", "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(), "model_sha256": digest(MODEL),
        "all_sources_unchanged": True})
    print(json.dumps(main_summary), flush=True)


if __name__ == "__main__":
    require(not OUTPUT.exists(), "Refusing to overwrite a previous evaluation run.")
    try:
        run()
    except BaseException as exc:
        if OUTPUT.exists():
            save(OUTPUT / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
            save(OUTPUT / "status.json", {"status": "failed", "finished_at_utc": datetime.now(timezone.utc).isoformat()})
        raise
