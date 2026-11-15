"""Strengthen one early action while retaining the off-reference recovery sample."""
import importlib.util
import json
import shutil
import subprocess
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "results/training/step6_supervision_20261115"
BASE = ROOT / "results/training/recovery_supervision_20261115"
RESCUE = ROOT / "results/training/step25_rescue_20261115"
RELOAD = (2000, 2004, 2009, 2031, 2345, 9956, 10000)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


core = module("step6_027_core", Path(__file__).with_name("focused_update.py"))
b = core.baseline
v, r = b.v, b.r
read, require, equal = b.read, b.require, b.equal
prior = module("step6_027_prior", ROOT / "docs/experiments/025_recovery_supervision/run_recovery_supervision.py")
rescue = module("step6_027_rescue", ROOT / "docs/experiments/024_step25_rescue/run_step25_rescue.py")


def save(name, value):
    (OUTPUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def q_record(agent, recovery):
    agent.learner.online.eval()
    with torch.no_grad():
        q = agent.learner.online(recovery)[0].tolist()
    require(all(np.isfinite(q)), "finite recovery Q values")
    action = int(np.argmax(q))
    return {"q_values": q, "greedy_action": action, "signed_gap": q[4] - max(q[:4] + q[5:])}


def bind_sources():
    hashes = {}
    for name, source in (("025_recovery_supervision", BASE), ("026_step6_rescue", ROOT / "results/training/step6_rescue_20261115")):
        folder = ROOT / "docs/experiments" / name / "evidence"
        path = folder / "archive_manifest.json"
        manifest = read(path)
        committed = subprocess.check_output(["git", "show", "HEAD:" + path.relative_to(ROOT).as_posix()], cwd=ROOT)
        require(json.loads(committed) == manifest, "committed historical manifest")
        hashes[str(path)] = v.file_hash(path)
        for filename, item in manifest["files"].items():
            path = folder / filename
            require(v.file_hash(path) == item["sha256"], "unchanged archived evidence")
            hashes[str(path)] = item["sha256"]
            if filename != "updates_samples.jsonl":
                path = source / filename
                require(v.file_hash(path) == item["sha256"], "unchanged local historical evidence")
                hashes[str(path)] = item["sha256"]
    for name, digest in read(BASE / "provenance.json")["source_hashes"].items():
        require(v.file_hash(Path(name)) == digest, "unchanged inherited dependencies")
        hashes[name] = digest
    manifest = read(ROOT / "docs/experiments/025_recovery_supervision/evidence/archive_manifest.json")
    for filename in ("updates.jsonl", "dense.json"):
        path = BASE / filename
        digest = manifest["raw_log_coverage"][filename]["sha256"]
        require(v.file_hash(path) == digest, "historical raw log")
        hashes[str(path)] = digest
    for path in (Path(__file__), Path(__file__).with_name("focused_update.py"), Path(__file__).parent.parent / "027_step6_supervision.md", ROOT / "tests/test_step6_supervision.py"):
        hashes[str(path)] = v.file_hash(path)
    return hashes


def control_rows_and_models(hashes):
    rows = {row["update"]: row for row in read(BASE / "observations.json")}
    paths = {}
    models = read(BASE / "model_hashes.json")
    for step in rows:
        path = BASE / "models" / f"training_state_{step}.pt"
        require(v.file_hash(path) == models[path.name], "unchanged control model")
        hashes[str(path)] = models[path.name]
        paths[step] = path
    require(len(rows) == 69, "69 historical comparison checkpoints")
    return rows, paths


def conditional_rollout(agent, env, part, recovery, prefix):
    """Replay a fixed prefix; all actions from the recovery state onwards are autonomous."""
    original_act = agent.act
    decisions = []
    def act(obs, epsilon=0.0):
        policy = original_act(obs, epsilon=epsilon)
        index = len(decisions)
        encoded = agent.encoder(obs)
        if index == 24:
            require(np.array_equal(encoded, recovery[0].numpy()), "same conditional recovery input")
        executed = prefix[index] if index < 24 else policy
        decisions.append({"step": index + 1, "policy_action": policy, "executed_action": executed,
                          "scripted_prefix": index < 24, "encoded_observation_sha256": rescue.diagnostic.array_hash(encoded)})
        return executed
    agent.act = act
    try:
        result = v.multi.diagnostic.fit.fixed_scene_rollout(agent, env, 10001, part)
    finally:
        del agent.act
    require(len(decisions) >= 25 and sum(d["scripted_prefix"] for d in decisions) == 24, "fixed 24-action prefix")
    require(all(d["executed_action"] == d["policy_action"] for d in decisions[24:]), "no forced recovery or tail action")
    return {"rollout": result, "decisions": decisions, "scripted_prefix_actions": 24,
            "autonomous_tail_actions": len(decisions) - 24, "recovery_action": decisions[24]["executed_action"]}


def captured(env, step, phase, cert, callback):
    records, incomplete = [], []
    with v.capture_resets(env, step, records, incomplete):
        result = callback()
    require(not incomplete and records, "complete evaluation reset accounting")
    cert.append({"phase": phase, "update": step, "records": records})
    save("reset_certification.json", cert)
    return result


def verify_updates_and_reloads(source, batch, recovery, episodes, env_config, observed, conditional, logs, saved, cert):
    models = OUTPUT / "models"
    independent_count = 0
    for start, end in ((2000, 2201), (9800, 10000)):
        initial = source if start == 2000 else torch.load(models / f"training_state_{start}.pt", weights_only=True)
        agent, env = v.prior.independent_restore(initial)
        try:
            for step in range(start + 1, end + 1):
                stats = core.independent_update(agent.learner, batch, recovery)
                equal({"update": step, **stats}, logs[step - 2001], "independent treatment update log")
                path = models / f"training_state_{step}.pt"
                if path.exists():
                    equal(b.snapshot(agent, env_config), torch.load(path, weights_only=True), "independent treatment full state")
                independent_count += 1
        finally:
            env.close()
    require(sum(x["target_synchronized"] for x in logs) == 40, "forty unchanged synchronizations")
    for name, digest in saved.items():
        path = models / name
        require(v.file_hash(path) == digest, "saved model bytes unchanged")
        payload = torch.load(path, weights_only=True)
        step = payload["update_count"]
        require(payload["optimizer_summary"]["step_min"] == payload["optimizer_summary"]["step_max"] == 20000 + step, "preserved Adam clock")
        target = torch.load(models / f"training_state_{(step // 200) * 200}.pt", weights_only=True)
        equal(payload["target"], target["online"], "target lineage")
    part = v.multi.episode_batch(batch, next(e for e in episodes if e["seed"] == 10001))
    prefix = read(RESCUE / "td_2031_greedy.json")["rollout"]["actions"][:24]
    reloads = []
    for step in RELOAD:
        payload = torch.load(models / f"training_state_{step}.pt", weights_only=True)
        agent, env = v.prior.independent_restore(payload)
        try:
            scenes = []
            for episode, expected in zip(episodes, observed[step]["scenes"], strict=True):
                result = captured(env, step, "verification_full", cert,
                    partial(v.multi.diagnostic.fit.fixed_scene_rollout, agent, env, episode["seed"], v.multi.episode_batch(batch, episode)))
                require(result == expected["rollout"], "exact independent full trajectory")
                scenes.append(result)
            # Separate adapter implementation from the main conditional rollout.
            original_act = agent.act
            decisions = []
            def act(obs, epsilon=0.0, original_act=original_act, decisions=decisions, step=step, agent=agent):
                chosen = original_act(obs, epsilon=epsilon)
                n = len(decisions)
                expected = conditional[step]["decisions"][n]
                actual = prefix[n] if n < len(prefix) else chosen
                require(chosen == expected["policy_action"] and actual == expected["executed_action"]
                        and rescue.diagnostic.array_hash(agent.encoder(obs)) == expected["encoded_observation_sha256"], "independent conditional decision")
                decisions.append(actual)
                return actual
            agent.act = act
            try:
                tail = captured(env, step, "verification_conditional", cert,
                    partial(v.multi.diagnostic.fit.fixed_scene_rollout, agent, env, 10001, part))
            finally:
                del agent.act
            require(tail == conditional[step]["rollout"], "exact independent conditional trajectory")
            equal(b.snapshot(agent, env_config), payload, "reload preserves full state")
            reloads.append({"update": step, "scenes": scenes, "conditional": tail})
            save("reloads.json", reloads)
            print(f"Verified four treatment reload episodes at TD {step}.", flush=True)
        finally:
            env.close()
    return {"passed": True, "independent_updates": independent_count, "gate_updates": 4, "reload_episodes": 28, "reloads": reloads}


def run():
    OUTPUT.mkdir(exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    save("status.json", {"status": "running", "started_at_utc": started})
    torch.set_num_threads(1)
    hashes = bind_sources()
    settings, manifest, batch, returns, episodes, source, inherited = b.inputs()
    hashes.update(inherited)
    controls, paths = control_rows_and_models(hashes)
    equal(source, torch.load(paths[2000], weights_only=True), "same complete starting snapshot")
    cert = []
    old = read(RESCUE / "td_2031_greedy.json")
    sample = read(BASE / "recovery_sample.json")
    recovery = torch.tensor([sample["observation"]], dtype=torch.float32)
    require(sample["action"] == 4 and rescue.diagnostic.array_hash(recovery[0].numpy()) == sample["encoded_observation_sha256"], "unchanged recovery sample")
    require(int(batch.actions[40]) == 1 and int(batch.actions[55]) == 0
            and episodes[1]["start"] + 5 == 40, "selected original step-six identity")
    checkpoints = sorted(set(controls) | set(range(2000, 2051)))
    require(len(checkpoints) == 105, "69 matched plus 36 additional early checkpoints")
    shutil.copy2(BASE / "recovery_sample.json", OUTPUT / "recovery_sample.json")
    save("reset_certification.json", cert)
    save("provenance.json", {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
         "source_hashes": hashes, "inherited_manifest": manifest})
    save("protocol.json", {"start": 2000, "end": 10000, "checkpoints": checkpoints, "matched_checkpoints": sorted(controls), "reloads": RELOAD,
         "td_weight": 0.5, "original_supervision_weight": 1, "original_row55_weight": 2, "original_row40_weight": 2,
         "recovery_weight": 1, "margin": 0.8, "denominator": 117, "mc_weight": 0,
         "settings": settings, "historical_control_reused": True,
         "note": "Embedded settings describe inherited historical training. Only original row 40 supervision weight changes from one to two. Recovery data is reused, not recollected."})
    env_config = source["environment_config"]
    # Four disposable gate updates; no gate state enters treatment.
    agents = [v.prior.independent_restore(source) for _ in range(4)]
    try:
        expected = prior.core.recovery_td_update(agents[0][0].learner, batch, recovery)
        actual = core.recovery_td_update(agents[1][0].learner, batch, recovery, 1.0)
        require({k: actual[k] for k in expected} == expected, "weight-one control exact metrics")
        require({"update": 2001, **expected} == v.read_log(BASE / "updates.jsonl")[0], "historical first control update")
        equal(b.snapshot(agents[0][0], env_config), b.snapshot(agents[1][0], env_config), "weight-one exact full state")
        equal(b.snapshot(agents[0][0], env_config), torch.load(paths[2001], weights_only=True), "historical control full state")
        treatment = core.recovery_td_update(agents[2][0].learner, batch, recovery)
        independent = core.independent_update(agents[3][0].learner, batch, recovery)
        require(treatment == independent, "independent first treatment metrics")
        equal(b.snapshot(agents[2][0], env_config), b.snapshot(agents[3][0], env_config), "independent first treatment full state")
        require(treatment["td_loss_before"] == expected["td_loss_before"] and treatment["recovery_loss_before"] == expected["recovery_loss_before"], "unchanged pre-update TD and recovery terms")
        save("gates.json", {"passed": True, "optimizer_updates": 4, "control": expected, "treatment": treatment})
    finally:
        for _, env in agents:
            env.close()
    part = v.multi.episode_batch(batch, next(e for e in episodes if e["seed"] == 10001))
    prefix = old["rollout"]["actions"][:24]
    save("historical_controls.json", [controls[s] for s in sorted(controls)])
    shutil.copy2(BASE / "conditional.json", OUTPUT / "baseline_conditional.json")
    directory = OUTPUT / "models"
    directory.mkdir()
    agent, env = v.prior.independent_restore(source)
    observations, tails, dense, logs, saved = {}, {}, [], [], {}
    try:
        equal(b.snapshot(agent, env_config), source, "unchanged treatment start")
        with (OUTPUT / "updates.jsonl").open("x") as log:
            for step in range(2000, 10001):
                if step > 2000:
                    stats = core.recovery_td_update(agent.learner, batch, recovery)
                    row = {"update": step, **stats}
                    logs.append(row)
                    log.write(json.dumps(row) + "\n")
                probe = r.extension.sentinel_record(agent, batch, episodes, step, 2000, r.SENTINEL_STEPS)
                with torch.no_grad():
                    correct = int((agent.learner.online(batch.observations).argmax(1) == batch.actions).sum())
                dense.append({"update": step, "original_correct": correct, "reference": probe, "recovery": q_record(agent, recovery)})
                if step in checkpoints:
                    before = b.snapshot(agent, env_config)
                    row = captured(env, step, "treatment_full", cert,
                        partial(r.multi.observe, agent, env, env_config, batch, returns, episodes, "step6_supervision", step, 40000, directory))
                    observations[step] = row
                    if step == 2000:
                        require(row == controls[step], "exact initial autonomous behavior")
                    tails[step] = captured(env, step, "treatment_conditional", cert,
                        partial(conditional_rollout, agent, env, part, recovery, prefix))
                    equal(b.snapshot(agent, env_config), before, "all evaluations preserve complete training state")
                    save("observations.json", list(observations.values()))
                    save("conditional.json", [{"update": u, **value} for u, value in tails.items()])
                    print(json.dumps({"update": step, "autonomous_success": row["aggregate"]["success_count"],
                         "conditional_success": tails[step]["rollout"]["success"], "recovery_action": tails[step]["recovery_action"]}), flush=True)
                if step in checkpoints or step % 200 == 0:
                    path = directory / f"training_state_{step}.pt"
                    torch.save(b.snapshot(agent, env_config), path)
                    saved[path.name] = v.file_hash(path)
                if step % 100 == 0:
                    log.flush()
                    save("progress.json", {"update": step, "checkpoints_completed": len(observations)})
    finally:
        env.close()
        save("reset_certification.json", cert)
    save("dense.json", dense)
    save("model_hashes.json", saved)
    save("main_completed.json", {"optimizer_updates": 8000, "full_episodes": 315, "conditional_episodes": 105})
    checks = verify_updates_and_reloads(source, batch, recovery, episodes, env_config, observations, tails, logs, saved, cert)
    save("verification.json", checks)
    for path, digest in hashes.items():
        require(v.file_hash(Path(path)) == digest, "all source evidence unchanged")
    save("status.json", {"status": "completed", "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    try:
        run()
    except BaseException as error:
        if OUTPUT.exists():
            save("failure.json", {"type": type(error).__name__, "message": str(error)})
            save("status.json", {"status": "failed", "finished_at_utc": datetime.now(timezone.utc).isoformat()})
        raise
