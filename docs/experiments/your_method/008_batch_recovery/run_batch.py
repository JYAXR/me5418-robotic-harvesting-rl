"""Bounded recovery-data aggregation and supervised-only policy improvement."""

import argparse
import gzip
import hashlib
import json
import os
import platform
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from me5418_harvesting.agents.dqn_agent import DQNAgent
from me5418_harvesting.agents.optimizer_transition import _canonical, optimizer_state_summary
from me5418_harvesting.agents.optimizer_transition import _digest as json_digest
from me5418_harvesting.agents.training import episode_metrics
from me5418_harvesting.envs.scene_cache import CertifiedSceneCache
from me5418_harvesting.evaluation.scene_feasibility import _copy_environment
from me5418_harvesting.planners.collision_free import NoCollisionFreeRoute, collision_free_actions

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TRAIN = [10000, 10001, 10002, *range(290000, 290030)]
VALIDATION = list(range(590000, 590030))
QUERY_STEPS = (8, 24, 40)
SOURCE = ROOT / "models/your_method_004_targeted.pt"
SOURCE_SHA = "ea42fd1a2d23317c9e15a960c58d90970a38edbcfb7724308a845ead92f6119f"
SOURCE_STATE = HERE.parent / "004_verified_supervision/resume/targeted_state_14000.pt"
SOURCE_STATE_SHA = "ed32355abf23c463a523df6bf00d6a9708b727bcd15fc04f357eaadd6f68fd2c"
BASE_DATA = ROOT / "docs/experiments/029_training_coverage/evidence"
UPDATES = 6000
WORKERS = 4
LR = 0.0001
SAMPLING_SEED = 817008
CACHE = None


def _digest(value):
    return json_digest(_canonical(value))


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def worker_init():
    global CACHE
    torch.set_num_threads(1)
    CACHE = CertifiedSceneCache(128)


def frozen_state(agent):
    return _digest({"online": agent.learner.online.state_dict(),
                    "target": agent.learner.target.state_dict(),
                    "optimizer": agent.learner.optimizer.state_dict(),
                    "updates": agent.learner.update_count, "rng": agent.rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state()})


def scene_state(env):
    return plain({"observation": env._observation().copy(), "rng": deepcopy(env.np_random.bit_generator.state),
                  "steps": env.step_count, "travel": env.total_travel, "peak": env.peak_deformation,
                  "contacts": env.contact_steps, "trajectory": env.trajectory})


def reset_record(env, obs, info):
    return plain({"observation": obs, "info": info, "rng": env.np_random.bit_generator.state})


def packed_trace(path, rows):
    raw = ("\n".join(json.dumps(plain(r), separators=(",", ":"), allow_nan=False) for r in rows) + "\n").encode()
    path.write_bytes(gzip.compress(raw, mtime=0))
    return {"trace": str(path.relative_to(HERE)), "trace_sha256": sha(path),
            "raw_trace_sha256": hashlib.sha256(raw).hexdigest()}


def freeze(agent):
    for net in (agent.learner.online, agent.learner.target):
        net.eval()
        net.requires_grad_(False)

    def forbidden(*args, **kwargs):
        raise RuntimeError("Optimizer steps are forbidden during collection and evaluation.")

    agent.learner.optimizer.step = forbidden


def recovery_probe(env, encoder):
    """Accept a cloned-state demonstration only after its full rollout succeeds."""
    before = scene_state(env)
    probe = _copy_environment(env)
    rows, observations, actions = [], [], []
    result = {"success": False, "reason": "route_exhausted", "planning_seconds": 0.0,
              "physics_seconds": 0.0, "initial_step_count": env.step_count,
              "remaining_budget": env.max_steps - env.step_count}
    try:
        tick = perf_counter()
        try:
            route = collision_free_actions(probe, include_current_shape=True)
        except NoCollisionFreeRoute as exc:
            result.update(reason="planner_found_no_route", detail=str(exc))
            route = []
        result["planning_seconds"] = perf_counter() - tick
        result["planned_actions"] = len(route)
        for action in route:
            obs = probe._observation().copy()
            encoded = encoder(obs)
            rng = deepcopy(probe.np_random.bit_generator.state)
            tick = perf_counter()
            nxt, reward, terminated, truncated, info = probe.step(action)
            result["physics_seconds"] += perf_counter() - tick
            observations.append(encoded)
            actions.append(action)
            rows.append(plain({"step": probe.step_count, "observation": obs, "action": action,
                               "encoded_observation": encoded, "next_observation": nxt,
                               "rng_before": rng, "rng_after": probe.np_random.bit_generator.state,
                               "reward": reward, "terminated": terminated, "truncated": truncated, "info": info}))
            if terminated or truncated:
                result.update(success=bool(info["success"]),
                              reason="success" if info["success"] else "damage" if info["damage"]
                              else "budget_exhausted" if info["budget_exhausted"] else "external_truncation")
                break
        result.update(executed_actions=len(rows), return_sum=sum(r["reward"] for r in rows),
                      final_info=rows[-1]["info"] if rows else None)
        require(scene_state(env) == before, "Recovery query mutated its parent environment.")
        accepted = result["success"] and bool(rows)
        result["accepted_samples"] = len(rows) if accepted else 0
        return result, rows, observations if accepted else [], actions if accepted else []
    finally:
        probe.close()


def collect_task(model, seed, directory):
    require(seed in TRAIN and seed not in VALIDATION, "Recovery collection crossed the training split.")
    directory = Path(directory)
    agent, env, _ = DQNAgent.load(Path(model))
    env.scene_cache, env.profile_reset = CACHE, True
    freeze(agent)
    before = frozen_state(agent)
    tick = perf_counter()
    record = {"seed": seed, "mode": "collection", "queries": [], "prefix_actions": 0}
    samples, labels, prefix = [], [], []
    try:
        try:
            obs, info = env.reset(seed=seed)
        except ValueError as exc:
            record.update(reset_error=str(exc), initial_state=None)
            return record
        record["initial_state"] = reset_record(env, obs, info)
        for step in range(1, max(QUERY_STEPS) + 1):
            if step in QUERY_STEPS:
                query, rows, x, y = recovery_probe(env, agent.encoder)
                query.update(step=step, start=len(samples), stop=len(samples) + len(y))
                query.update(packed_trace(directory / f"recovery_{seed}_{step}.jsonl.gz", rows))
                samples.extend(x)
                labels.extend(y)
                record["queries"].append(query)
            if step == max(QUERY_STEPS):
                break
            action = agent.act(obs, epsilon=0.0)
            old_obs = obs.copy()
            obs, reward, terminated, truncated, info = env.step(action)
            prefix.append(plain({"step": step, "observation": old_obs, "action": action,
                                 "next_observation": obs, "reward": reward,
                                 "terminated": terminated, "truncated": truncated, "info": info}))
            record["prefix_actions"] += 1
            if terminated or truncated:
                record["prefix_termination"] = plain(info)
                break
        record["skipped_query_steps"] = [s for s in QUERY_STEPS if s not in [q["step"] for q in record["queries"]]]
        record.update(packed_trace(directory / f"prefix_{seed}.jsonl.gz", prefix))
        dataset = directory / f"samples_{seed}.npz"
        np.savez_compressed(dataset, observations=np.asarray(samples, dtype=np.float32).reshape(-1, env.observation_size),
                            actions=np.asarray(labels, dtype=np.int64))
        record.update(dataset=str(dataset.relative_to(HERE)), dataset_sha256=sha(dataset), samples=len(labels))
        return record
    finally:
        require(frozen_state(agent) == before, "Collection changed the policy or optimizer state.")
        record.update(elapsed_seconds=perf_counter() - tick, reset_profile=deepcopy(env.last_reset_profile),
                      frozen_state_unchanged=True)
        env.close()


def evaluation_task(model, seed, directory):
    directory = Path(directory)
    agent, env, _ = DQNAgent.load(Path(model))
    env.scene_cache, env.profile_reset = CACHE, True
    freeze(agent)
    before = frozen_state(agent)
    started = perf_counter()
    rows, total, blocked, stationary = [], 0.0, 0, 0
    record = {"seed": seed, "mode": "evaluation", "action_overrides": 0}
    try:
        try:
            obs, info = env.reset(seed=seed)
        except ValueError as exc:
            record.update(reset_error=str(exc), initial_state=None,
                          metrics={"success": False, "damage": False, "budget_exhausted": False,
                                   "return": 0.0, "steps": 0})
            return record
        record["initial_state"] = reset_record(env, obs, info)
        for step in range(1, env.max_steps + 1):
            old = obs.copy()
            with torch.no_grad():
                q = agent.learner.online(torch.from_numpy(agent.encoder(obs)).unsqueeze(0))[0]
            action = agent.act(obs, epsilon=0.0)
            require(action == int(q.argmax()), "Non-greedy evaluation.")
            rng = deepcopy(env.np_random.bit_generator.state)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            blocked += int(info["collision_blocked"])
            stationary += int(np.linalg.norm(info["executed_motion"]) <= 1e-6)
            rows.append(plain({"step": step, "observation": old, "q_values": q.tolist(), "action": action,
                               "next_observation": obs, "reward": reward, "terminated": terminated,
                               "truncated": truncated, "rng_before": rng,
                               "rng_after": env.np_random.bit_generator.state, "info": info}))
            if terminated or truncated:
                break
        require(terminated or truncated, "Evaluation exceeded the task budget.")
        record["metrics"] = episode_metrics(total, len(rows), info, stationary, blocked)
        record.update(packed_trace(directory / f"evaluation_{seed}.jsonl.gz", rows))
        return record
    finally:
        require(frozen_state(agent) == before, "Evaluation changed model, optimizer or RNG state.")
        record.update(elapsed_seconds=perf_counter() - started, reset_profile=deepcopy(env.last_reset_profile),
                      frozen_state_unchanged=True)
        env.close()


def gather(pool, jobs, output):
    records = []
    futures = {pool.submit(fn, str(model), seed, str(directory)): seed for fn, model, seed, directory in jobs}
    for future in as_completed(futures):
        result = future.result()
        records.append(result)
        save(output, sorted(records, key=lambda r: (r["mode"], r["seed"])))
        label = result.get("metrics", {}).get("success", result.get("samples", 0))
        print(f"{output.parent.name}: {len(records)}/{len(jobs)} {result['mode']} seed {result['seed']}: {label}", flush=True)
    return sorted(records, key=lambda r: (r["mode"], r["seed"]))


def aggregate(round_number, records):
    xs, ys, sources = [], [], []
    for number in range(1, round_number + 1):
        current = records if number == round_number else read(HERE / f"round_{number}/collection.json")
        for record in current:
            require(record["seed"] in TRAIN and record["seed"] not in VALIDATION, "Validation leakage.")
            if "dataset" not in record:
                continue
            path = HERE / record["dataset"]
            require(sha(path) == record["dataset_sha256"], "Recovery samples changed.")
            with np.load(path) as data:
                require(len(data["actions"]) == sum(q["accepted_samples"] for q in record["queries"]), "Rejected samples included.")
                xs.append(data["observations"].copy())
                ys.append(data["actions"].copy())
                sources.append({"round": number, "seed": record["seed"], "samples": len(data["actions"])})
    if not xs:
        with np.load(BASE_DATA / "expanded_demonstrations.npz") as data:
            width = data["observations"].shape[1]
        return np.empty((0, width), dtype=np.float32), np.empty(0, dtype=np.int64), sources
    return np.concatenate(xs), np.concatenate(ys), sources


def supervised_step(agent, x, y):
    learner = agent.learner
    learner.online.train()
    learner.optimizer.zero_grad(set_to_none=True)
    logits = learner.online(x)
    loss = F.cross_entropy(logits, y)
    require(bool(torch.isfinite(loss)), "Non-finite supervised loss.")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(learner.online.parameters(), 10.0, error_if_nonfinite=True)
    learner.optimizer.step()
    return float(loss.detach()), float(norm)


def train(round_number, directory, collection):
    torch.set_num_threads(1)
    torch.manual_seed(SAMPLING_SEED)
    agent, env, _ = DQNAgent.load(SOURCE)
    config = torch.load(SOURCE, weights_only=True)["environment_config"]
    source = torch.load(SOURCE_STATE, weights_only=True)
    agent.learner.target.load_state_dict(source["target"])
    agent.learner.update_count = 14000
    agent.learner.optimizer = torch.optim.Adam(agent.learner.online.parameters(), lr=LR)
    rng = np.random.default_rng(SAMPLING_SEED)
    inherited = 0
    if round_number == 2:
        resume = torch.load(HERE / "round_1/resume.pt", weights_only=True)
        agent.learner.online.load_state_dict(resume["online"])
        agent.learner.optimizer.load_state_dict(resume["optimizer"])
        rng.bit_generator.state = resume["sampling_rng"]
        torch.set_rng_state(resume["torch_rng"])
        inherited = resume["supervised_updates"]
        require(inherited == UPDATES, "Wrong round-2 resume boundary.")
    target_hash = _digest(agent.learner.target.state_dict())
    with np.load(BASE_DATA / "expanded_demonstrations.npz") as data:
        base_x, base_y = data["observations"].copy(), data["actions"].copy()
    # Retain the older verified recovery sample as an ordinary base example.
    old = read(ROOT / "docs/experiments/027_step6_supervision/evidence/recovery_sample.json")
    base_x = np.concatenate((base_x, np.asarray([old["observation"]], dtype=np.float32)))
    base_y = np.concatenate((base_y, np.asarray([old["action"]], dtype=np.int64)))
    rec_x, rec_y, sources = aggregate(round_number, collection)
    dataset = directory / "training_data.npz"
    np.savez_compressed(dataset, base_observations=base_x, base_actions=base_y,
                        recovery_observations=rec_x, recovery_actions=rec_y)
    save(directory / "training_data.json", {"base_rows": len(base_y), "recovery_rows": len(rec_y),
         "sources": sources, "sha256": sha(dataset), "validation_samples": 0,
         "sampling": "128 base plus 128 recovery with replacement; 256 base if recovery is empty"})
    tensors = [torch.from_numpy(a) for a in (base_x, base_y, rec_x, rec_y)]
    bx, by, rx, ry = tensors
    before_optimizer = optimizer_state_summary(agent.learner)
    fit = []
    started = perf_counter()
    with (directory / "updates.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        for update in range(1, UPDATES + 1):
            bi = rng.integers(len(by), size=128 if len(ry) else 256)
            x, y = bx[bi], by[bi]
            if len(ry):
                ri = rng.integers(len(ry), size=128)
                x, y = torch.cat((x, rx[ri])), torch.cat((y, ry[ri]))
            loss, norm = supervised_step(agent, x, y)
            handle.write(json.dumps({"update": inherited + update, "cross_entropy": loss, "gradient_norm": norm}) + "\n")
            if update in (1, 1000, 3000, UPDATES):
                with torch.no_grad():
                    stats = {"update": inherited + update, "base_accuracy": float((agent.learner.online(bx).argmax(1) == by).float().mean()),
                             "recovery_accuracy": float((agent.learner.online(rx).argmax(1) == ry).float().mean()) if len(ry) else None}
                fit.append(stats)
                print(f"round {round_number} supervised fit: {stats}", flush=True)
    require(agent.learner.update_count == 14000 and _digest(agent.learner.target.state_dict()) == target_hash,
            "Supervision changed the TD clock or target network.")
    require(all(bool(torch.isfinite(p).all()) for p in agent.learner.online.parameters()), "Non-finite model.")
    model = ROOT / f"models/your_method_008_round{round_number}.pt"
    agent.save(model, config, {"method": "supervised_recovery_aggregation", "round": round_number,
               "supervised_updates": inherited + UPDATES, "new_td_updates": 0,
               "training_seeds": TRAIN, "outputs_are_calibrated_q_values": False})
    torch.save({"online": deepcopy(agent.learner.online.state_dict()), "target": deepcopy(agent.learner.target.state_dict()),
                "optimizer": deepcopy(agent.learner.optimizer.state_dict()), "torch_rng": torch.get_rng_state(),
                "sampling_rng": rng.bit_generator.state, "supervised_updates": inherited + UPDATES,
                "td_updates": 14000, "environment_config": config,
                "dqn_config": torch.load(SOURCE, weights_only=True)["dqn_config"]}, directory / "resume.pt")
    reloaded, reloaded_env, _ = DQNAgent.load(model)
    require(_digest(reloaded.learner.online.state_dict()) == _digest(agent.learner.online.state_dict()), "Export mismatch.")
    reloaded_env.close()
    save(directory / "training.json", {"optimizer_before": before_optimizer,
         "optimizer_after": optimizer_state_summary(agent.learner), "fit": fit,
         "elapsed_seconds": perf_counter() - started, "updates": UPDATES, "new_td_updates": 0,
         "target_unchanged": True, "model": str(model.relative_to(ROOT)), "model_sha256": sha(model),
         "resume_sha256": sha(directory / "resume.pt"), "reload_weights_exact": True})
    env.close()
    return model


def group(records, seeds):
    selected = [r for r in records if r["seed"] in seeds]
    require(len(selected) == len(seeds), "Missing evaluation scenes.")
    return {"episodes": len(selected), "successes": sum(r["metrics"]["success"] for r in selected),
            "success_seeds": sorted(r["seed"] for r in selected if r["metrics"]["success"]),
            "damage": sum(r["metrics"]["damage"] for r in selected),
            "budget_failures": sum(r["metrics"]["budget_exhausted"] for r in selected),
            "reset_errors": sum("reset_error" in r for r in selected),
            "mean_return": float(np.mean([r["metrics"]["return"] for r in selected]))}


def ready_to_stop(summary, baseline_validation):
    return (summary["original_three"]["successes"] == 3 and summary["training"]["successes"] >= 27
            and summary["validation"]["successes"] >= max(9, baseline_validation + 3))


def run(round_number):
    directory = HERE / f"round_{round_number}"
    directory.mkdir(exist_ok=False)
    started, tick = datetime.now(timezone.utc).isoformat(), perf_counter()
    save(directory / "status.json", {"status": "running", "started_at_utc": started})
    try:
        require(round_number in (1, 2), "At most two rounds are allowed.")
        require(not set(TRAIN) & set(VALIDATION), "Split overlap.")
        require(sha(SOURCE) == SOURCE_SHA and sha(SOURCE_STATE) == SOURCE_STATE_SHA, "Source changed.")
        require(sha(BASE_DATA / "expanded_demonstrations.npz") == read(BASE_DATA / "dataset.json")["npz_sha256"], "Base data changed.")
        if round_number == 2:
            require(read(HERE / "round_1/status.json")["status"] == "completed", "Round 1 is incomplete.")
            require(not read(HERE / "round_1/summary.json")["early_stop_target_met"], "Early stop forbids round 2.")
        model = SOURCE if round_number == 1 else ROOT / "models/your_method_008_round1.pt"
        if round_number == 2:
            previous = read(HERE / "round_1/training.json")
            require(sha(model) == previous["model_sha256"] and sha(HERE / "round_1/resume.pt") == previous["resume_sha256"], "Round-1 endpoint changed.")
        paths = list((ROOT / "src").rglob("*.py")) + list(HERE.glob("*.py"))
        paths += [SOURCE, SOURCE_STATE, HERE / "PROTOCOL.md", BASE_DATA / "expanded_demonstrations.npz",
                  BASE_DATA / "dataset.json", ROOT / "docs/experiments/027_step6_supervision/evidence/recovery_sample.json",
                  HERE.parent / "004_verified_supervision/evidence/evaluations.json", model]
        if round_number == 2:
            paths += [p for p in (HERE / "round_1").rglob("*") if p.is_file()]
        hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
        save(directory / "provenance.json", {"input_hashes": hashes,
             "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
             "training_seeds": TRAIN, "validation_seeds": VALIDATION, "query_steps": QUERY_STEPS,
             "workers": WORKERS, "torch_threads_per_process": 1, "cpu_count": os.cpu_count(),
             "python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform(),
             "supervised_updates": UPDATES, "learning_rate": LR, "batch_size": 256,
             "td_loss_used": False, "validation_is_final_test": False})
        collection_dir, evaluation_dir = directory / "collection", directory / "evaluation"
        collection_dir.mkdir()
        evaluation_dir.mkdir()
        with ProcessPoolExecutor(max_workers=WORKERS, initializer=worker_init) as pool:
            jobs = [(collect_task, model, seed, collection_dir) for seed in TRAIN]
            if round_number == 1:
                baseline_dir = directory / "baseline_validation"
                baseline_dir.mkdir()
                jobs += [(evaluation_task, SOURCE, seed, baseline_dir) for seed in VALIDATION]
            combined = gather(pool, jobs, directory / "collection_and_baseline.json")
            collection = [r for r in combined if r["mode"] == "collection"]
            save(directory / "collection.json", collection)
            if round_number == 1:
                baseline_validation = [r for r in combined if r["mode"] == "evaluation"]
                save(directory / "baseline_validation.json", baseline_validation)
            else:
                baseline_validation = read(HERE / "round_1/baseline_validation.json")
            require(len(collection) == 33, "Missing training collection scenes.")
            new_model = train(round_number, directory, collection)
            evaluations = gather(pool, [(evaluation_task, new_model, seed, evaluation_dir) for seed in TRAIN + VALIDATION],
                                 directory / "evaluations.json")
        historical = [r for r in read(HERE.parent / "004_verified_supervision/evidence/evaluations.json")
                      if r["arm"] == "targeted" and r["update"] == 2000]
        initial = {r["seed"]: r["initial_state"] for r in collection + baseline_validation}
        for r in evaluations:
            if r["initial_state"] is not None and initial[r["seed"]] is not None:
                require(r["initial_state"] == initial[r["seed"]], "Evaluation initial state changed.")
        for r in collection:
            if r["initial_state"] is not None:
                old = next(v for v in historical if v["seed"] == r["seed"])["reset_record"]
                require(r["initial_state"] == {k: old[k] for k in ("observation", "info", "rng")}, "Training scene changed.")
        summary = {"round": round_number, "training": group(evaluations, TRAIN),
                   "original_three": group(evaluations, TRAIN[:3]), "validation": group(evaluations, VALIDATION),
                   "baseline_training": group(historical, TRAIN), "baseline_validation": group(baseline_validation, VALIDATION),
                   "accepted_routes": sum(q["success"] for r in collection for q in r["queries"]),
                   "queried_routes": sum(len(r["queries"]) for r in collection),
                   "new_recovery_samples": sum(r.get("samples", 0) for r in collection),
                   "collection_policy_actions": sum(r["prefix_actions"] for r in collection),
                   "recovery_physics_actions": sum(q["executed_actions"] for r in collection for q in r["queries"]),
                   "evaluation_actions": sum(r["metrics"]["steps"] for r in evaluations),
                   "fresh_baseline_actions": sum(r["metrics"]["steps"] for r in baseline_validation) if round_number == 1 else 0,
                   "certification_actions": sum(r["reset_profile"]["certification_actions"] for r in combined + evaluations),
                   "certification_calls": sum(r["reset_profile"]["certification_calls"] for r in combined + evaluations),
                   "cache_hits": sum(r["reset_profile"]["cache_status"] == "hit" for r in combined + evaluations),
                   "supervised_updates": UPDATES, "new_td_updates": 0, "evaluation_action_overrides": 0,
                   "elapsed_seconds": perf_counter() - tick}
        summary["early_stop_target_met"] = ready_to_stop(summary, summary["baseline_validation"]["successes"])
        for path, digest in hashes.items():
            require(sha(ROOT / path) == digest, "Pinned input changed during round.")
        save(directory / "summary.json", summary)
        save(directory / "status.json", {"status": "completed", "started_at_utc": started,
                                         "finished_at_utc": datetime.now(timezone.utc).isoformat()})
        save(directory / "manifest.json", {str(p.relative_to(directory)): sha(p)
             for p in sorted(directory.rglob("*")) if p.is_file()})
        print(json.dumps(summary, indent=2), flush=True)
    except BaseException as exc:
        save(directory / "status.json", {"status": "failed", "started_at_utc": started, "error": repr(exc)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, choices=(1, 2), required=True)
    run(parser.parse_args().round)
