"""Replay frozen failures with exactly one south_small correction at the first reference divergence."""
import importlib.util
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "results/training/recovery_supervision_20261115"
OUTPUT = ROOT / "results/training/step6_rescue_20261115"
ARCHIVE = Path(__file__).parent / "evidence"
UPDATES = (2004, 2009)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


r = module("step6_training", ROOT / "docs/experiments/022_half_td_local_supervision/run_local_supervision.py")
diagnostic = module("step6_diagnostic", ROOT / "docs/experiments/012_divergence_control/diagnose_divergence.py")
v = r.v
read, require, equal = v.read_json, v.require, v.require_equal


def save(name, value):
    (OUTPUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def corrected_action(policy, step, intervene):
    return 1 if intervene and step == 6 else policy


def main_rollout(agent, env, part, original, intervene):
    """Execute a manual loop and retain every transition, without modifying any later policy decision."""
    obs, _ = env.reset(seed=10001)
    require(np.array_equal(agent.encoder(obs), part.observations[0].numpy()), "unchanged reset observation")
    records, actions, path = [], [], [obs[:2].tolist()]
    total, blocked, stationary = 0.0, 0, 0
    for index in range(env.max_steps):
        policy = agent.act(obs, epsilon=0.0)
        encoded = agent.encoder(obs)
        with torch.no_grad():
            q = agent.learner.online(torch.from_numpy(encoded).unsqueeze(0))[0].tolist()
        state_hash = diagnostic.array_hash(encoded)
        if index <= 5 or not intervene:
            require(policy == original["rollout"]["actions"][index]
                    and obs[:2].tolist() == original["rollout"]["robot_path"][index], "exact historical action/path prefix")
        if index == 5:
            require(policy == 2 and int(part.actions[index]) == 1
                    and np.array_equal(encoded, part.observations[index].numpy()), "visited reference fork with east_small error")
            require(q == original["fork_probe"]["q_values"], "exact source reference Q values")
        forced = intervene and index == 5
        action = corrected_action(policy, index + 1, intervene)
        following, reward, terminated, truncated, info = env.step(action)
        stopped = bool(np.linalg.norm(info["executed_motion"]) <= 1e-6)
        records.append({"step": index + 1, "position": obs[:2].tolist(), "next_position": following[:2].tolist(),
            "encoded_observation_sha256": state_hash, "q_values": q, "policy_action": policy,
            "executed_action": action, "forced_action": forced, "reward": float(reward),
            "terminated": bool(terminated), "truncated": bool(truncated), "stationary": stopped,
            "collision_blocked": bool(info["collision_blocked"]), "executed_motion": np.asarray(info["executed_motion"]).tolist()})
        actions.append(action)
        path.append(following[:2].tolist())
        total += float(reward)
        blocked += int(info["collision_blocked"])
        stationary += int(stopped)
        obs = following
        if terminated or truncated:
            break
    else:
        raise RuntimeError("The episode exceeded the unchanged budget.")
    result = {"training_scene_only": True, "seed": 10001,
        **v.multi.diagnostic.fit.episode_metrics(total, len(actions), info, stationary, blocked),
        "first_reference_action_divergence": diagnostic.first_divergence(actions, part.actions.numpy()),
        "actions": actions, "robot_path": path}
    if not intervene:
        require(result == original["rollout"], "exact full original failure")
    require(sum(row["forced_action"] for row in records) == int(intervene), "exact single override count")
    return {"rollout": result, "records": records, "forced_actions": int(intervene)}


def paired_fork(control, corrected):
    left, right = control["records"], corrected["records"]
    require(left[:5] == right[:5] and len(left) >= 6 and len(right) >= 6, "identical first five full transitions")
    a, b = left[5], right[5]
    require(all(a[k] == b[k] for k in ("step", "position", "encoded_observation_sha256", "q_values", "policy_action")), "identical fork observation and Q values")
    require(a["step"] == 6 and a["policy_action"] == 2 and a["executed_action"] == 2 and b["executed_action"] == 1,
            "only the prescribed step-6 action is replaced")
    for trace, intervene in ((control, False), (corrected, True)):
        require(sum(row["forced_action"] for row in trace["records"]) == int(intervene), "one forced action only")
        for row in trace["records"]:
            forced = intervene and row["step"] == 6
            require(row["forced_action"] == forced and row["executed_action"] == corrected_action(row["policy_action"], row["step"], intervene),
                    "all other decisions remain autonomous")
    return {"step": 6, "identical_prefix": True, "identical_fork_input_and_q": True,
            "policy_action": 2, "corrected_action": 1, "q_south_small_minus_east_small": a["q_values"][1] - a["q_values"][2]}


def independent_rollout(agent, env, part, expected, intervene):
    """Use the original loop and independently check all decisions and transitions."""
    original_act, original_step = agent.act, env.step
    decisions, transitions = [], []
    def observed_step(action):
        following, reward, terminated, truncated, info = original_step(action)
        record = {**decisions[-1], "step": len(decisions), "next_position": following[:2].tolist(),
            "reward": float(reward), "terminated": bool(terminated), "truncated": bool(truncated),
            "stationary": bool(np.linalg.norm(info["executed_motion"]) <= 1e-6),
            "collision_blocked": bool(info["collision_blocked"]), "executed_motion": np.asarray(info["executed_motion"]).tolist()}
        require(record == expected["records"][len(transitions)], "independent exact full transition")
        transitions.append(record)
        return following, reward, terminated, truncated, info
    def adapted(obs, epsilon=0.0):
        # Install only after reset has finished its copied-environment certification.
        if not decisions:
            env.step = observed_step
        policy = original_act(obs, epsilon=epsilon)
        index = len(decisions)
        encoded = agent.encoder(obs)
        with torch.no_grad():
            q = agent.learner.online(torch.from_numpy(encoded).unsqueeze(0))[0].tolist()
        action = 1 if intervene and index == 5 else policy
        row = {"position": obs[:2].tolist(), "encoded_observation_sha256": diagnostic.array_hash(encoded),
               "q_values": q, "policy_action": policy, "executed_action": action, "forced_action": intervene and index == 5}
        require(index < len(expected["records"]) and all(expected["records"][index][k] == value for k, value in row.items()), "independent exact decision")
        decisions.append(row)
        return action
    agent.act = adapted
    try:
        result = v.multi.diagnostic.fit.fixed_scene_rollout(agent, env, 10001, part)
    finally:
        del agent.act
        if "step" in env.__dict__:
            del env.step
    require(result == expected["rollout"] and transitions == expected["records"], "independent complete trajectory and transitions")
    return result


def run():
    OUTPUT.mkdir(exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    save("status.json", {"status": "running", "started_at_utc": started})
    torch.set_num_threads(1)
    archive = ROOT / "docs/experiments/025_recovery_supervision/evidence"
    manifest = read(archive / "archive_manifest.json")
    committed = subprocess.check_output(["git", "show", "HEAD:" + (archive / "archive_manifest.json").relative_to(ROOT).as_posix()], cwd=ROOT)
    require(json.loads(committed) == manifest, "committed source archive")
    bound = {str(path): v.file_hash(path) for path in (Path(__file__), ROOT / "tests/test_step6_rescue.py", Path(__file__).parent.parent / "026_step6_rescue.md")}
    for name, item in manifest["files"].items():
        path = archive / name
        require(v.file_hash(path) == item["sha256"], "immutable source archive")
        bound[str(path)] = item["sha256"]
        if name != "updates_samples.jsonl":
            path = SOURCE / name
            require(v.file_hash(path) == item["sha256"], "immutable local source evidence")
            bound[str(path)] = item["sha256"]
    for name, digest in read(SOURCE / "provenance.json")["source_hashes"].items():
        require(v.file_hash(Path(name)) == digest, "unchanged source dependency")
        bound[name] = digest
    path = SOURCE / "dense.json"
    digest = manifest["raw_log_coverage"]["dense.json"]["sha256"]
    require(v.file_hash(path) == digest, "exact full source Q log")
    bound[str(path)] = digest
    require(read(SOURCE / "status.json")["status"] == "completed" and read(SOURCE / "verification.json")["passed"], "verified source experiment")
    _, _, batch, _, episodes, _, _ = r.inputs()
    part = v.multi.episode_batch(batch, next(e for e in episodes if e["seed"] == 10001))
    scenes = {row["update"]: next(s for s in row["scenes"] if s["seed"] == 10001) for row in read(SOURCE / "observations.json") if row["update"] in UPDATES}
    probes = {row["update"]: next(p for p in row["reference"]["states"] if p["reference_step"] == 6)
              for row in read(SOURCE / "dense.json") if row["update"] in UPDATES}
    reloads = read(SOURCE / "failure_reload_verification.json")
    require(reloads["passed"], "source failed-model reloads verified")
    for update in UPDATES:
        scenes[update]["fork_probe"] = probes[update]
        old = next(s for row in reloads["results"] if row["update"] == update for s in row["scenes"] if s["seed"] == 10001)
        require(old == scenes[update]["rollout"], "matching independently reloaded source failure")
    snapshots = {}
    for update in UPDATES:
        path = SOURCE / "models" / f"training_state_{update}.pt"
        digest = read(SOURCE / "model_hashes.json")[path.name]
        require(v.file_hash(path) == digest, "frozen model hash")
        bound[str(path)] = digest
        snapshots[update] = torch.load(path, weights_only=True)
        require(not scenes[update]["rollout"]["success"] and scenes[update]["rollout"]["steps"] == 200
                and scenes[update]["rollout"]["actions"][5] == 2
                and scenes[update]["rollout"]["first_reference_action_divergence"] == {"step": 6, "predicted_action": 2, "demonstrated_action": 1}, "predeclared source failure")
    save("provenance.json", {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "script_sha256": v.file_hash(Path(__file__)), "source_hashes": bound, "models": UPDATES, "seed": 10001,
        "forced_step": 6, "forced_action": 1, "optimizer_updates": 0,
        "source_protocol": read(SOURCE / "protocol.json")})
    traces, pairs, cert, checks = {}, [], [], []
    for verification in (False, True):
        for update in UPDATES:
            payload = snapshots[update]
            for mode, intervene in (("greedy", False), ("corrected", True)):
                agent, env = v.prior.independent_restore(payload)
                records, incomplete = [], []
                try:
                    equal(r.snapshot(agent, payload["environment_config"]), payload, "exact initial full snapshot")
                    with v.capture_resets(env, update, records, incomplete):
                        result = independent_rollout(agent, env, part, traces[(update, mode)], intervene) if verification else main_rollout(agent, env, part, scenes[update], intervene)
                    require(not incomplete and len(records) == 1, "complete reset accounting")
                    equal(r.snapshot(agent, payload["environment_config"]), payload, "unchanged complete training state")
                    if verification:
                        expected_cert = next(c["record"] for c in cert if not c["verification"] and c["update"] == update and c["mode"] == mode)
                        require(records[0] == expected_cert, "exact reloaded certification")
                        checks.append({"update": update, "mode": mode, "all_decisions_and_trajectory_match": True, "all_transitions_match": True, "steps": result["steps"]})
                        save("verification.json", checks)
                    else:
                        traces[(update, mode)] = result
                        save(f"td_{update}_{mode}.json", result)
                    cert.append({"verification": verification, "update": update, "mode": mode, "record": records[0]})
                    save("reset_certification.json", cert)
                    outcome = result if verification else result["rollout"]
                    print(json.dumps({"verification": verification, "update": update, "mode": mode, "success": outcome["success"], "steps": outcome["steps"]}), flush=True)
                finally:
                    env.close()
            if not verification:
                fork = paired_fork(traces[(update, "greedy")], traces[(update, "corrected")])
                pairs.append({"update": update, "fork": fork, **{mode: traces[(update, mode)]["rollout"] for mode in ("greedy", "corrected")}})
                save("pairs.json", pairs)
    for path, digest in bound.items():
        require(v.file_hash(Path(path)) == digest, "immutable source after evaluation")
    save("summary.json", {"pairs": pairs, "rescued_models": sum(p["corrected"]["success"] for p in pairs),
        "optimizer_updates": 0, "main_episodes": 4, "verification_episodes": 4,
        "main_policy_actions": sum(t["rollout"]["steps"] for t in traces.values()), "verification_policy_actions": sum(c["steps"] for c in checks),
        "main_certification_actions": sum(c["record"]["certification_environment_steps"] for c in cert if not c["verification"]),
        "verification_certification_actions": sum(c["record"]["certification_environment_steps"] for c in cert if c["verification"]),
        "main_forced_actions": 2, "verification_forced_actions": 2, "verification_passed": True,
        "source_unchanged": True, "scope": "Only step 6 corrected; all later actions, including recovery decisions, remain policy-controlled. Frozen known-scene diagnostic, not learned recovery or generalization."})
    save("status.json", {"status": "completed", "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat()})
    ARCHIVE.mkdir(exist_ok=False)
    files = {}
    for source in sorted(OUTPUT.glob("*.json")):
        target = ARCHIVE / source.name
        shutil.copy2(source, target)
        require(source.read_bytes() == target.read_bytes(), "exact archive bytes")
        files[source.name] = {"sha256": v.file_hash(source), "bytes": source.stat().st_size}
    (ARCHIVE / "archive_manifest.json").write_text(json.dumps({"source_run": str(OUTPUT), "files": files}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        run()
    except BaseException as error:
        if OUTPUT.exists():
            save("failure.json", {"type": type(error).__name__, "message": str(error)})
            save("status.json", {"status": "failed", "finished_at_utc": datetime.now(timezone.utc).isoformat()})
        raise
