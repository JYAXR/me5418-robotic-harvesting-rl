"""Reconstruct unchanged late models and measure complete autonomous episodes."""
import hashlib
import importlib.util
import json
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "results/training/td_weight_20261114"
OUTPUT = ROOT / "results/training/late_rollouts_20261114"
WINDOW = tuple(range(9946, 9967))
RELOAD = (9955, 9956, 9957)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


v = module("late_verifier", ROOT / "docs/experiments/019_td_weight/verify_td_weight.py")
r = module("late_runner", ROOT / "docs/experiments/019_td_weight/run_td_weight.py")


def save(name, value):
    path = OUTPUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def snapshot(agent, env_config):
    return deepcopy(r.adam.training_snapshot(agent, env_config))


def trace_rollout(agent, env, seed, part):
    """Observe actual decisions without overriding the original greedy policy."""
    original = agent.act
    decisions = []
    prefix = True

    def traced(observation, epsilon=0.0):
        nonlocal prefix
        action = original(observation, epsilon=epsilon)
        encoded = agent.encoder(observation)
        index = len(decisions)
        matches = index < len(part.actions) and np.array_equal(encoded, part.observations[index].numpy())
        with torch.no_grad():
            values = agent.learner.online(torch.from_numpy(encoded).unsqueeze(0))[0].tolist()
        v.require(action == int(np.argmax(values)), "observer preserves greedy action")
        demonstrated = int(part.actions[index]) if index < len(part.actions) else None
        decisions.append({"step": index + 1, "action": action, "demonstrated_action": demonstrated,
                          "matches_reference_state": bool(matches), "reference_prefix_before_action": prefix,
                          "encoded_observation_sha256": hashlib.sha256(encoded.tobytes()).hexdigest(),
                          "q_values": values})
        prefix = prefix and matches and action == demonstrated
        return action

    agent.act = traced
    try:
        rollout = v.multi.diagnostic.fit.fixed_scene_rollout(agent, env, seed, part)
    finally:
        del agent.act
    return {"seed": seed, "rollout": rollout, "decisions": decisions}


def execute():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    save("status.json", {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()})
    torch.set_num_threads(1)
    settings, summary = v.read_json(SOURCE / "settings.json"), v.read_json(SOURCE / "summary.json")
    v.verify_protocol(settings["training"], settings)
    manifest, _, _, reference013, original = v.verify_inputs(SOURCE, settings["training"])
    batch, _, episodes, _ = v.prior.load_verified_batch(SOURCE, reference013, original, summary)
    phase = SOURCE / "treatment/td"
    model_map = summary["treatment"]["checkpoint_sha256"]
    for name in ("training_state_9500.pt", "training_state_10000.pt"):
        v.require(v.file_hash(phase / name) == model_map[name], "immutable boundary model")
    archive = v.read_json(ROOT / "docs/experiments/019_td_weight/evidence/archive_manifest.json")
    for name, info in archive["raw_log_coverage"].items():
        v.require(v.file_hash(SOURCE / name) == info["raw_sha256"], "immutable original raw log")
    source = torch.load(phase / "training_state_9500.pt", weights_only=True)
    endpoint = torch.load(phase / "training_state_10000.pt", weights_only=True)
    updates = {row["update"]: row for row in v.read_log(phase / "updates.jsonl")}
    sentinels = {row["update"]: row for row in v.read_log(phase / "sentinel_q.jsonl")}
    save("provenance.json", {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
                            "script_sha256": v.file_hash(Path(__file__)), "source_manifest": manifest,
                            "source_snapshot_sha256": model_map["training_state_9500.pt"],
                            "endpoint_sha256": model_map["training_state_10000.pt"],
                            "window": WINDOW, "source_protocol": settings["training"],
                            "source_run": str(SOURCE)})
    models = {}
    replay_records = []
    for independent in (False, True):
        agent, env = v.prior.independent_restore(source)
        try:
            for step in range(9501, 10001):
                stats = v.independent_update(agent.learner, batch, 0.5) if independent else r.weighted_td_update(agent.learner, batch)
                v.require({"update": step, "extension_update": step - 2000, **stats} == updates[step], "exact historical update")
                v.prior.verify_sentinel_network(agent, sentinels[step], batch)
                if step in WINDOW:
                    current = snapshot(agent, settings["environment"])
                    if independent:
                        v.require_equal(current, models[step], "independent reconstructed full state")
                    else:
                        models[step] = current
                        path = OUTPUT / "models" / f"training_state_{step}.pt"
                        path.parent.mkdir(exist_ok=True)
                        torch.save(current, path)
            v.require_equal(snapshot(agent, settings["environment"]), endpoint, "full TD-10000 endpoint matches")
            replay_records.append({"independent": independent, "updates": 500, "all_logs_and_q_match": True,
                                   "full_endpoint_matches": True, "window_full_states_checked": len(WINDOW)})
            print(f"Replay verified: independent={independent}, 500 updates, exact full endpoint.", flush=True)
        finally:
            env.close()
    save("reconstruction.json", replay_records)
    save("models.json", {path.name: v.file_hash(path) for path in sorted((OUTPUT / "models").glob("*.pt"))})
    observations, certification, reloads = [], [], []
    for verify in (False, True):
        for step in (RELOAD if verify else WINDOW):
            payload = torch.load(OUTPUT / "models" / f"training_state_{step}.pt", weights_only=True)
            agent, env = v.prior.independent_restore(payload)
            records, incomplete, scenes = [], [], []
            try:
                with v.capture_resets(env, step, records, incomplete):
                    for entry in episodes:
                        part = v.multi.episode_batch(batch, entry)
                        if verify:
                            result = v.multi.diagnostic.fit.fixed_scene_rollout(agent, env, entry["seed"], part)
                            expected = next(row for row in observations if row["update"] == step)
                            expected = next(row for row in expected["scenes"] if row["seed"] == entry["seed"])
                            v.require(result == expected["rollout"], "independent complete trajectory reload")
                            scenes.append({"seed": entry["seed"], "rollout": result})
                        else:
                            scenes.append(trace_rollout(agent, env, entry["seed"], part))
                v.require(not incomplete, "complete reset accounting")
                v.require_equal(snapshot(agent, settings["environment"]), payload, "evaluation leaves full training state unchanged")
                row = {"update": step, "scenes": scenes}
                if verify:
                    v.require(records == [record for record in certification if record["update"] == step], "exact reset certification reload")
                    reloads.append({**row, "reset_certification": records})
                    save("reloads.json", reloads)
                else:
                    observations.append(row)
                    certification.extend(records)
                    save("observations.json", observations)
                    save("reset_certification.json", certification)
                print(f"{'Reload' if verify else 'Evaluate'} TD {step}: " + ", ".join(f"{s['seed']}:{s['rollout']['success']}/{s['rollout']['steps']}" for s in scenes), flush=True)
            finally:
                env.close()
    policy = sum(s["rollout"]["steps"] for row in observations for s in row["scenes"])
    copied = sum(row["certification_environment_steps"] for row in certification)
    reload_policy = sum(s["rollout"]["steps"] for row in reloads for s in row["scenes"])
    reload_copied = sum(r["certification_environment_steps"] for row in reloads for r in row["reset_certification"])
    save("accounting.json", {"reconstruction_updates": 1000, "new_training_remedy_updates": 0,
                             "main_episodes": 63, "main_policy_actions": policy, "main_certification_actions": copied,
                             "reload_episodes": 9, "reload_policy_actions": reload_policy, "reload_certification_actions": reload_copied,
                             "total_simulator_actions": policy + copied + reload_policy + reload_copied,
                             "held_out_scenes": 0, "forced_actions": 0})
    save("verification.json", {"passed": True, "exact_replays": replay_records,
                               "exact_reloaded_episodes": 9, "all_evaluations_preserve_full_state": True})
    save("status.json", {"status": "completed", "finished_at_utc": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    try:
        execute()
    except BaseException as error:
        if OUTPUT.exists():
            save("failure.json", {"error_type": type(error).__name__, "message": str(error)})
        raise
