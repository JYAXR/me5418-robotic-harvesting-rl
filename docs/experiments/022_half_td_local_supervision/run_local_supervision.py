"""Test one doubled supervision coefficient while keeping the TD coefficient at 0.5."""
import importlib.util
import json
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from math import isfinite
from numbers import Real
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from me5418_harvesting.agents.dqn_learning import td_loss

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "results/training/td_weight_20261114"
OUTPUT = ROOT / "results/training/half_td_local_supervision_20261114"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


r = module("local022_runner", ROOT / "docs/experiments/019_td_weight/run_td_weight.py")
v = module("local022_verifier", ROOT / "docs/experiments/019_td_weight/verify_td_weight.py")
local = module("local022_margin", ROOT / "docs/experiments/016_local_supervision/run_local_supervision.py")
CHECKPOINTS = sorted(set(r.CHECKPOINTS) | set(range(9946, 9967)))
RELOAD = (2000, 2009, 2200, 2201, 9955, 9956, 10000)
read, require, equal = v.read_json, v.require, v.require_equal


def weighted_td_update(learner, batch, selected_weight=2.0):
    td_weight = 0.5
    if isinstance(selected_weight, bool) or selected_weight not in (1.0, 2.0):
        raise ValueError("The selected coefficient must be 1 or 2.")
    if (isinstance(td_weight, bool) or not isinstance(td_weight, Real)
            or not isfinite(td_weight) or td_weight not in (0.5, 1.0)):
        raise ValueError("TD weight must be the declared treatment 0.5 or control 1.0.")
    learner.online.train()
    learner.optimizer.zero_grad(set_to_none=True)
    before = nn.utils.parameters_to_vector(learner.online.parameters()).detach().clone()
    td_before, _, targets = td_loss(learner.online, learner.target, batch, learner.config.gamma)
    values = learner.online(batch.observations)
    margin_before = local.weighted_margin_loss(values, batch.actions, 55, selected_weight)
    accuracy = float((values.detach().argmax(dim=1) == batch.actions).float().mean())
    # Keep the original graph exactly at coefficient one; no MC term is constructed.
    scaled_td = td_before if td_weight == 1.0 else td_weight * td_before
    loss = scaled_td + 1.0 * margin_before
    if not torch.isfinite(loss):
        raise FloatingPointError("The weighted TD objective is not finite.")
    loss.backward()
    gradient_norm = nn.utils.clip_grad_norm_(
        learner.online.parameters(), learner.config.gradient_clip_norm, error_if_nonfinite=True)
    learner.optimizer.step()
    with torch.no_grad():
        after_values = learner.online(batch.observations)
        after_chosen = after_values.gather(1, batch.actions[:, None]).squeeze(1)
        td_after = F.smooth_l1_loss(after_chosen, targets)
        margin_after = local.weighted_margin_loss(after_values, batch.actions, 55, selected_weight)
        scaled_after = td_after if td_weight == 1.0 else td_weight * td_after
        loss_after = scaled_after + 1.0 * margin_after
        after = nn.utils.parameters_to_vector(learner.online.parameters())
        delta = torch.linalg.vector_norm(after - before)
        if not torch.isfinite(after).all() or not torch.isfinite(loss_after):
            raise FloatingPointError("The weighted TD update produced non-finite values.")
    learner.update_count += 1
    synchronized = learner.update_count % learner.config.target_update_interval == 0
    if synchronized:
        learner.target.load_state_dict(learner.online.state_dict())
    return {
        "loss_before": loss.item(), "loss_after": loss_after.item(),
        "gradient_norm": gradient_norm.item(), "parameter_delta_norm": delta.item(),
        "target_synchronized": synchronized, "td_loss_before": td_before.item(),
        "td_loss_after": td_after.item(), "demonstration_loss_before": margin_before.item(),
        "demonstration_loss_after": margin_after.item(), "demonstration_action_accuracy": accuracy,
        "td_weight": float(td_weight), "selected_weight": float(selected_weight),
    }


def independent_update(learner, batch, weight=2.0):
    """Compute the same objective without the weighted-margin helper."""
    learner.online.train()
    learner.optimizer.zero_grad(set_to_none=True)
    before = nn.utils.parameters_to_vector(learner.online.parameters()).detach().clone()
    chosen = learner.online(batch.observations).gather(1, batch.actions[:, None]).squeeze(1)
    with torch.no_grad():
        following = learner.target(batch.next_observations).max(dim=1).values
        targets = batch.rewards + learner.config.gamma * torch.where(batch.terminated, torch.zeros_like(following), following)
    td = F.smooth_l1_loss(chosen, targets)
    values = learner.online(batch.observations)
    def margin(q):
        hinges = v.anchor.independent_hinges(q, batch.actions)
        coefficients = torch.ones_like(hinges)
        coefficients[55] = weight
        return (hinges * coefficients).mean()
    supervised = margin(values)
    accuracy = float((values.detach().argmax(1) == batch.actions).float().mean())
    loss = 0.5 * td + supervised
    loss.backward()
    norm = nn.utils.clip_grad_norm_(learner.online.parameters(), learner.config.gradient_clip_norm, error_if_nonfinite=True)
    learner.optimizer.step()
    with torch.no_grad():
        after_values = learner.online(batch.observations)
        after_td = F.smooth_l1_loss(after_values.gather(1, batch.actions[:, None]).squeeze(1), targets)
        after_margin = margin(after_values)
        after = nn.utils.parameters_to_vector(learner.online.parameters())
        delta = torch.linalg.vector_norm(after - before)
    learner.update_count += 1
    sync = learner.update_count % learner.config.target_update_interval == 0
    if sync:
        learner.target.load_state_dict(learner.online.state_dict())
    return {"loss_before": loss.item(), "loss_after": (0.5 * after_td + after_margin).item(),
            "gradient_norm": norm.item(), "parameter_delta_norm": delta.item(), "target_synchronized": sync,
            "td_loss_before": td.item(), "td_loss_after": after_td.item(),
            "demonstration_loss_before": supervised.item(), "demonstration_loss_after": after_margin.item(),
            "demonstration_action_accuracy": accuracy, "td_weight": 0.5, "selected_weight": float(weight)}


def save(name, value):
    (OUTPUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def snapshot(agent, environment):
    return deepcopy(r.adam.training_snapshot(agent, environment))


def inputs():
    settings, summary = read(SOURCE / "settings.json"), read(SOURCE / "summary.json")
    v.verify_protocol(settings["training"], settings)
    manifest, _, _, ref013, original = v.verify_inputs(SOURCE, settings["training"])
    batch, returns, episodes, _ = v.prior.load_verified_batch(SOURCE, ref013, original, summary)
    hashes = {}
    for name in ("019_td_weight", "020_late_rollouts", "021_late_action_rescue"):
        directory = ROOT / "docs/experiments" / name / "evidence"
        path = directory / "archive_manifest.json"
        recorded = read(path)
        committed = subprocess.check_output(["git", "show", "HEAD:" + path.relative_to(ROOT).as_posix()], cwd=ROOT)
        require(json.loads(committed) == recorded, "committed source evidence")
        for filename, item in recorded["files"].items():
            require(v.file_hash(directory / filename) == item["sha256"], "immutable archived evidence")
            hashes[str(directory / filename)] = item["sha256"]
    path = SOURCE / "treatment/td/training_state_2000.pt"
    require(v.file_hash(path) == summary["treatment"]["checkpoint_sha256"][path.name], "same full starting model")
    source = torch.load(path, weights_only=True)
    require(int(batch.actions[55]) == 0 and episodes[1]["start"] + 20 == 55, "selected demonstration identity")
    return settings, manifest, batch, returns, episodes, source, hashes


def run():
    OUTPUT.mkdir(exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    save("status.json", {"status": "running", "started_at_utc": started})
    torch.set_num_threads(1)
    settings, manifest, batch, returns, episodes, source, hashes = inputs()
    env_config = settings["environment"]
    save("protocol.json", {"attempt": 22, "selected_row": 55, "selected_weight": 2, "td_weight": 0.5,
         "other_weights": 1, "normalization": 117, "mc_weight": 0, "start": 2000, "end": 10000,
         "checkpoints": CHECKPOINTS, "reloads": RELOAD, "settings": settings,
         "note": "Embedded source training protocol describes historical attempt 019. The sole new factor is selected row weight 2."})
    save("provenance.json", {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
         "script_sha256": v.file_hash(Path(__file__)), "source_manifest": manifest, "archived_source_hashes": hashes})
    # A real-model equivalence gate checks the coefficient-one control before training.
    agents = [v.prior.independent_restore(source) for _ in range(3)]
    try:
        control = r.weighted_td_update(agents[0][0].learner, batch)
        equivalent = weighted_td_update(agents[1][0].learner, batch, 1.0)
        equivalent.pop("selected_weight")
        require(control == equivalent, "exact coefficient-one historical update")
        equal(snapshot(agents[0][0], env_config), snapshot(agents[1][0], env_config), "exact control full state")
        expected = independent_update(agents[2][0].learner, batch, 2.0)
    finally:
        for _, env in agents:
            env.close()
    # Gate updates above are separate verification work, never inherited by treatment.
    agent, env = v.prior.independent_restore(source)
    directory = OUTPUT / "models"
    directory.mkdir()
    recorder = r.ResetCertificationRecorder(env)
    observations, dense, logs = [], [], []
    saved = {}
    try:
        equal(snapshot(agent, env_config), source, "unchanged full restore")
        save("initial_q.json", r.initial_q_equivalence(agent, batch, source))
        with (OUTPUT / "updates.jsonl").open("x") as update_file, (OUTPUT / "sentinels.jsonl").open("x") as probe_file:
            for step in range(2000, 10001):
                if step > 2000:
                    stats = weighted_td_update(agent.learner, batch)
                    if step == 2001:
                        require(stats == expected, "independent first treatment objective")
                    row = {"update": step, **stats}
                    logs.append(row)
                    update_file.write(json.dumps(row) + "\n")
                probe = r.extension.sentinel_record(agent, batch, episodes, step, 2000, r.SENTINEL_STEPS)
                probe_file.write(json.dumps(probe) + "\n")
                with torch.no_grad():
                    correct = int((agent.learner.online(batch.observations).argmax(1) == batch.actions).sum())
                dense.append([step, correct, [s["signed_demonstration_gap"] for s in probe["states"]], [s["greedy_action"] for s in probe["states"]]])
                if step in CHECKPOINTS:
                    before = snapshot(agent, env_config)
                    row = recorder.observe(lambda step=step: r.multi.observe(agent, env, env_config, batch, returns,
                          episodes, "local_weight_2", step, 40000, directory), step)
                    equal(snapshot(agent, env_config), before, "physical evaluation preserves full state")
                    if step == 2000:
                        require(row == read(SOURCE / "treatment/td/observations.json")[0], "exact initial physical behavior")
                    observations.append(row)
                    save("observations.json", observations)
                    recorder.save(OUTPUT)
                if step in CHECKPOINTS or step % 200 == 0:
                    path = directory / f"training_state_{step}.pt"
                    torch.save(snapshot(agent, env_config), path)
                    saved[path.name] = v.file_hash(path)
                if step % 100 == 0:
                    update_file.flush()
                    probe_file.flush()
    finally:
        recorder.save(OUTPUT)
        env.close()
    save("dense.json", {"columns": ["update", "full_batch_correct", "signed_gaps", "greedy_actions"],
                         "reference_steps": list(r.SENTINEL_STEPS), "rows": dense})
    save("model_hashes.json", {p.name: v.file_hash(p) for p in directory.glob("*.pt")})
    save("main_completed.json", {"optimizer_updates": 8000, "episodes": 150, "status": "completed"})
    verify(source, env_config, batch, returns, episodes, observations, logs, saved)
    for path, digest in hashes.items():
        require(v.file_hash(Path(path)) == digest, "historical evidence unchanged")
    save("status.json", {"status": "completed", "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat()})


def verify(source, env_config, batch, returns, episodes, observations, logs, saved):
    directory = OUTPUT / "models"
    probes = {row["update"]: row for row in v.read_log(OUTPUT / "sentinels.jsonl")}
    agent, env = v.prior.independent_restore(source)
    try:
        for step in range(2001, 2202):
            stats = independent_update(agent.learner, batch)
            require({"update": step, **stats} == logs[step - 2001], "independent replay exact log")
            v.prior.verify_sentinel_network(agent, probes[step], batch)
            path = directory / f"training_state_{step}.pt"
            if path.exists():
                equal(snapshot(agent, env_config), torch.load(path, weights_only=True), "independent replay full state")
    finally:
        env.close()
    syncs = 0
    for step in range(2001, 10001):
        entry = logs[step - 2001]
        require(entry["selected_weight"] == 2.0 and entry["td_weight"] == 0.5, "unchanged coefficients")
        require(entry["target_synchronized"] == (step % 200 == 0), "unchanged target schedule")
        syncs += int(entry["target_synchronized"])
    require(syncs == 40, "forty target synchronizations")
    for name, digest in saved.items():
        path = directory / name
        require(v.file_hash(path) == digest, "saved model hash")
        payload = torch.load(path, weights_only=True)
        step = payload["update_count"]
        require(payload["optimizer_summary"]["step_min"] == payload["optimizer_summary"]["step_max"] == 20000 + step,
                "retained Adam clock")
        target_source = (step // 200) * 200
        expected = torch.load(directory / f"training_state_{target_source}.pt", weights_only=True)
        equal(payload["target"], expected["online"], "exact target lineage")
    rows = {r["update"]: r for r in observations}
    cert = read(OUTPUT / "reset_certification.json")["records"]
    reloads = []
    for step in RELOAD:
        payload = torch.load(directory / f"training_state_{step}.pt", weights_only=True)
        agent, env = v.prior.independent_restore(payload)
        records, incomplete, scenes = [], [], []
        try:
            with v.capture_resets(env, step, records, incomplete):
                for entry, expected in zip(episodes, rows[step]["scenes"], strict=True):
                    result = v.multi.diagnostic.fit.fixed_scene_rollout(agent, env, entry["seed"], v.multi.episode_batch(batch, entry))
                    require(result == expected["rollout"], "exact reloaded trajectory")
                    scenes.append(result)
            equal(snapshot(agent, env_config), payload, "reload preserves full state")
            require(not incomplete and records == [r for r in cert if r["update"] == step], "exact reset certification")
            reloads.append({"update": step, "scenes": scenes, "reset_certification": records})
            print(f"Verified three complete reloads at TD {step}.", flush=True)
        finally:
            env.close()
    save("verification.json", {"passed": True, "independent_replay_updates": 201,
         "preflight_gate_updates": 3, "reload_episodes": 21, "reloads": reloads,
         "main_updates": 8000, "main_episodes": 150,
         "main_policy_actions": sum(s["rollout"]["steps"] for r in observations for s in r["scenes"]),
         "main_certification_actions": sum(r["certification_environment_steps"] for r in cert),
         "reload_policy_actions": sum(s["steps"] for r in reloads for s in r["scenes"]),
         "reload_certification_actions": sum(c["certification_environment_steps"] for r in reloads for c in r["reset_certification"]),
         "scope": "First 201 updates numerically replayed; later saved states and target lineage audited. All evaluations autonomous."})


if __name__ == "__main__":
    try:
        run()
    except BaseException as error:
        if OUTPUT.exists():
            save("failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
