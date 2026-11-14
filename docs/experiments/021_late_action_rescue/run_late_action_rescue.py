"""Test exactly one corrected action in two frozen late failure models."""
import hashlib
import importlib.util
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "results/training/late_rollouts_20261114"
PRIOR = ROOT / "results/training/td_weight_20261114"
OUTPUT = ROOT / "results/training/late_action_rescue_20261114"
ARCHIVE = Path(__file__).parent / "evidence"
UPDATES = (9955, 9956)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


rescue = module("rescue021_helpers", ROOT / "docs/experiments/015_single_action_rescue/run_single_action_rescue.py")
v = module("rescue021_verifier", ROOT / "docs/experiments/019_td_weight/verify_td_weight.py")
read, require, equal = v.read_json, v.require, v.require_equal


def save(name, value):
    (OUTPUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def independent_rollout(agent, env, intervene, expected, actions):
    """Execute a separate loop and compare every actual transition with the main trace."""
    obs, _ = env.reset(seed=10001)
    total, stationary, blocked = 0.0, 0, 0
    chosen, path = [], [obs[:2].tolist()]
    for index in range(env.max_steps):
        policy = agent.act(obs, epsilon=0.0)
        encoded = agent.encoder(obs)
        with torch.no_grad():
            values = agent.learner.online(torch.from_numpy(encoded).unsqueeze(0))[0].tolist()
        forced = intervene and index == 20
        action = 0 if forced else policy
        following, reward, terminated, truncated, info = env.step(action)
        stopped = bool(np.linalg.norm(info["executed_motion"]) <= 1e-6)
        row = {"step": index + 1, "position": obs[:2].tolist(),
               "next_position": following[:2].tolist(),
               "encoded_observation_sha256": rescue.diagnostic.array_hash(encoded),
               "q_values": values, "policy_action": policy, "executed_action": action,
               "forced_reference_action": forced, "reward": float(reward),
               "terminated": bool(terminated), "truncated": bool(truncated),
               "stationary": stopped, "collision_blocked": bool(info["collision_blocked"]),
               "executed_motion": np.asarray(info["executed_motion"]).tolist(),
               "distance_to_target": float(info["distance_to_target"]),
               "peak_normalized_deformation": float(info["peak_normalized_deformation"])}
        require(index < len(expected["records"]), "independent replay stays within recorded trace")
        require(all(expected["records"][index][k] == value for k, value in row.items()), "independent exact transition")
        total += float(reward)
        stationary += int(stopped)
        blocked += int(info["collision_blocked"])
        chosen.append(action)
        path.append(following[:2].tolist())
        obs = following
        if terminated or truncated:
            break
    else:
        raise RuntimeError("The independent episode exceeded the unchanged budget.")
    rollout = {"training_scene_only": True, "seed": 10001,
               **v.multi.diagnostic.fit.episode_metrics(total, len(chosen), info, stationary, blocked),
               "first_reference_action_divergence": rescue.diagnostic.first_divergence(chosen, actions),
               "actions": chosen, "robot_path": path}
    require(rollout == expected["rollout"] and len(chosen) == len(expected["records"]), "independent full rollout")
    return rollout


def execute():
    OUTPUT.mkdir(exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    save("status.json", {"status": "running", "started_at_utc": started})
    torch.set_num_threads(1)
    archive = ROOT / "docs/experiments/020_late_rollouts/evidence"
    manifest = read(archive / "archive_manifest.json")
    committed = subprocess.check_output(["git", "show", "HEAD:" + (archive / "archive_manifest.json").relative_to(ROOT).as_posix()], cwd=ROOT)
    require(json.loads(committed) == manifest, "committed attempt-020 evidence manifest")
    bound = {}
    for name, record in manifest["files"].items():
        for path in (SOURCE / name, archive / name):
            require(v.file_hash(path) == record["sha256"], "immutable source evidence")
            bound[str(path)] = record["sha256"]
    require(read(SOURCE / "verification.json")["passed"] and read(SOURCE / "status.json")["status"] == "completed", "verified completed source")
    settings, summary = read(PRIOR / "settings.json"), read(PRIOR / "summary.json")
    v.verify_protocol(settings["training"], settings)
    inherited, _, _, ref013, original = v.verify_inputs(PRIOR, settings["training"])
    batch, _, episodes, _ = v.prior.load_verified_batch(PRIOR, ref013, original, summary)
    episode = next(e for e in episodes if e["seed"] == 10001)
    part = v.multi.episode_batch(batch, episode)
    observations, actions = part.observations.numpy(), part.actions.numpy()
    expected = {row["update"]: next(s["rollout"] for s in row["scenes"] if s["seed"] == 10001)
                for row in read(SOURCE / "observations.json") if row["update"] in UPDATES}
    models = read(SOURCE / "models.json")
    snapshots = {}
    for step in UPDATES:
        name = f"training_state_{step}.pt"
        path = SOURCE / "models" / name
        require(v.file_hash(path) == models[name], "verified reconstructed model hash")
        bound[str(path)] = models[name]
        snapshots[step] = torch.load(path, weights_only=True)
        rescue.validate_intervention(expected[step], actions)
    for name, digest in {**inherited["production_source_sha256"], **inherited["helper_sha256"]}.items():
        require(v.file_hash(ROOT / name) == digest, "immutable source dependency")
        bound[str(ROOT / name)] = digest
    for name in ("docs/experiments/015_single_action_rescue/run_single_action_rescue.py", "docs/experiments/019_td_weight/verify_td_weight.py"):
        bound[str(ROOT / name)] = v.file_hash(ROOT / name)
    save("provenance.json", {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
                             "script_sha256": v.file_hash(Path(__file__)), "source_hashes": bound,
                             "td_updates": UPDATES, "seed": 10001, "forced_step": 21, "forced_action": 0,
                             "optimizer_updates": 0, "source_protocol": settings["training"]})
    pairs, traces, cert, checks = [], {}, [], []
    for verification in (False, True):
        for step in UPDATES:
            snapshot = snapshots[step]
            for mode, intervene in (("greedy", False), ("corrected", True)):
                agent, env = v.prior.independent_restore(snapshot)
                records, incomplete = [], []
                try:
                    equal(rescue.adam.training_snapshot(agent, snapshot["environment_config"]), snapshot, "exact initial full state")
                    with v.capture_resets(env, step, records, incomplete):
                        if verification:
                            result = independent_rollout(agent, env, intervene, traces[(step, mode)], actions)
                        else:
                            result = rescue.diagnostic.trace_policy(agent, env, observations, actions, expected[step], intervene=intervene)
                    require(not incomplete and len(records) == 1, "complete single-reset accounting")
                    equal(rescue.adam.training_snapshot(agent, snapshot["environment_config"]), snapshot, "unchanged full state after episode")
                    if verification:
                        original_cert = next(r["record"] for r in cert if not r["verification"] and r["update"] == step and r["mode"] == mode)
                        require(records[0] == original_cert, "identical certification on independent reload")
                        checks.append({"update": step, "mode": mode, "all_transitions_match": True, "full_state_unchanged": True, "steps": result["steps"]})
                        save("verification.json", checks)
                    else:
                        traces[(step, mode)] = result
                        save(f"td_{step}_{mode}.json", result)
                    cert.append({"verification": verification, "update": step, "mode": mode, "record": records[0]})
                    save("reset_certification.json", cert)
                    outcome = result if verification else result["rollout"]
                    print(json.dumps({"verification": verification, "update": step, "mode": mode, "success": outcome["success"], "steps": outcome["steps"]}), flush=True)
                finally:
                    env.close()
            if not verification:
                fork = rescue.paired_fork(traces[(step, "greedy")], traces[(step, "corrected")])
                pairs.append({"update": step, "fork": fork, **{mode: traces[(step, mode)]["rollout"] for mode in ("greedy", "corrected")}})
                save("pairs.json", pairs)
    for path, digest in bound.items():
        require(v.file_hash(Path(path)) == digest, "all source artifacts unchanged")
    main_actions = sum(trace["rollout"]["steps"] for trace in traces.values())
    summary = {"pairs": pairs, "rescued_models": sum(p["corrected"]["success"] for p in pairs),
               "optimizer_updates": 0, "main_episodes": 4, "verification_episodes": 4,
               "main_policy_actions": main_actions, "verification_policy_actions": sum(r["steps"] for r in checks),
               "main_certification_actions": sum(r["record"]["certification_environment_steps"] for r in cert if not r["verification"]),
               "verification_certification_actions": sum(r["record"]["certification_environment_steps"] for r in cert if r["verification"]),
               "main_forced_actions": 2, "verification_forced_actions": 2, "verification_passed": True,
               "source_unchanged": True, "scope": "Frozen models, one known deterministic scene, one external action correction per treated episode. No policy learning or generalized remedy."}
    save("summary.json", summary)
    save("status.json", {"status": "completed", "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat()})
    ARCHIVE.mkdir(exist_ok=False)
    files = {}
    for source in sorted(OUTPUT.glob("*.json")):
        destination = ARCHIVE / source.name
        shutil.copy2(source, destination)
        require(source.read_bytes() == destination.read_bytes(), "exact archived evidence")
        files[source.name] = {"sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "bytes": source.stat().st_size}
    (ARCHIVE / "archive_manifest.json").write_text(json.dumps({"files": files, "source_run": str(OUTPUT)}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        execute()
    except BaseException as error:
        if OUTPUT.exists():
            save("failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
