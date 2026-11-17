"""One bounded full-trajectory recovery study with geometry and retention gates."""

import gzip
import importlib.util
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from me5418_harvesting.agents.dqn_agent import DQNAgent
from me5418_harvesting.agents.recovery_policy import GeometryEncoder, RecoveryAgent
from me5418_harvesting.evaluation.scene_feasibility import _copy_environment

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUTPUT = HERE / "evidence"
spec = importlib.util.spec_from_file_location("study009core", HERE.parent / "008_batch_recovery/run_batch.py")
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
TRAIN, VALIDATION = c.TRAIN, c.VALIDATION
SEED = 817009
PILOT, MAX_UPDATES, BLOCK = 400, 2000, 20
SCALES = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625)
MODEL = ROOT / "models/your_method_009_geometry.pt"


class PolicyLoader:
    @staticmethod
    def load(path):
        payload = torch.load(path, weights_only=True)
        return RecoveryAgent.load(path) if payload["format_version"] == 2 else DQNAgent.load(path)


def worker_init():
    c.worker_init()
    c.HERE = HERE
    c.DQNAgent = PolicyLoader


def evaluation_task(model, seed, directory):
    """Keep the process-pool entry point importable under Windows spawn."""
    record = c.evaluation_task(model, seed, directory)
    record["score_semantics"] = "Policy logits; legacy trace field q_values is not a calibrated Q estimate."
    return record


def choose_queries(candidates):
    """Select risk and coverage states in four time bands, deduplicating geometry."""
    chosen, seen = [], set()
    for low, high in ((1, 25), (26, 50), (51, 100), (101, 200)):
        eligible = [v for v in candidates if low <= v["step"] <= high]
        for kind in ("risk", "coverage"):
            eligible = [v for v in eligible if v["fingerprint"] not in seen]
            if not eligible:
                break
            if kind == "risk":
                row = min(eligible, key=lambda v: (-v["blocked_run"], v["clearance"], v["step"]))
            else:
                row = min(eligible, key=lambda v: (abs(v["step"] - (low + high) / 2), v["step"]))
            chosen.append((row, kind))
            seen.add(row["fingerprint"])
    return chosen


def action_probes(env, verified_action):
    """Label immediate damage/stall using physics; allow witnessed successful contact."""
    before = c.scene_state(env)
    checks, bad = [], []
    for action in range(8):
        probe = _copy_environment(env)
        try:
            obs, reward, terminated, truncated, info = probe.step(action)
            stall = np.linalg.norm(info["executed_motion"]) <= 1e-6 and not info["success"]
            checks.append(c.plain({"action": action, "observation": obs, "reward": reward,
                                   "terminated": terminated, "truncated": truncated, "info": info,
                                   "rng_after": probe.np_random.bit_generator.state}))
            bad.append(bool(info["damage"] or stall))
        finally:
            probe.close()
    raw_bad = list(bad)
    if verified_action >= 0:
        c.require(not checks[verified_action]["info"]["damage"], "Successful route starts with damage.")
        # A no-motion action can precede successful relaxation; do not contradict that witness.
        bad[verified_action] = False
    c.require(c.scene_state(env) == before, "Action probes changed the learner state.")
    return raw_bad, bad, checks


def collect_task(model, seed, directory):
    c.require(seed in TRAIN and seed not in VALIDATION, "Validation data collection is forbidden.")
    directory = Path(directory)
    agent, env, _ = DQNAgent.load(Path(model))
    encoder = GeometryEncoder(env)
    env.scene_cache, env.profile_reset = c.CACHE, True
    c.freeze(agent)
    frozen = c.frozen_state(agent)
    started = perf_counter()
    candidates, rows, queries, samples, labels, masks, categories = [], [], [], [], [], [], []
    record = {"seed": seed, "mode": "collection", "queries": queries}
    try:
        try:
            obs, info = env.reset(seed=seed)
        except ValueError as exc:
            record.update(reset_error=str(exc), initial_state=None, samples=0,
                          metrics={"success": False, "damage": False, "budget_exhausted": False, "steps": 0, "return": 0.0})
            return record
        record["initial_state"] = c.reset_record(env, obs, info)
        total, blocked, stationary, run = 0.0, 0, 0, 0
        for step in range(1, env.max_steps + 1):
            candidates.append({"step": step, "env": _copy_environment(env), "blocked_run": run,
                               "clearance": min(b.minimum_distance_to(env.scene.end_effector) for b in env.scene.branches),
                               "fingerprint": np.round(obs[:-1], 3).tobytes()})
            old = obs.copy()
            action = agent.act(obs)
            rng = deepcopy(env.np_random.bit_generator.state)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            blocked += int(info["collision_blocked"])
            stationary += int(np.linalg.norm(info["executed_motion"]) <= 1e-6)
            run = run + 1 if info["collision_blocked"] else 0
            rows.append(c.plain({"step": step, "observation": old, "action": action, "next_observation": obs,
                                 "rng_before": rng, "rng_after": env.np_random.bit_generator.state,
                                 "reward": reward, "terminated": terminated, "truncated": truncated, "info": info}))
            if terminated or truncated:
                break
        record["metrics"] = c.episode_metrics(total, len(rows), info, stationary, blocked)
        record.update(c.packed_trace(directory / f"learner_{seed}.jsonl.gz", rows))
        for candidate, kind in choose_queries(candidates):
            state, step = candidate["env"], candidate["step"]
            query, route_rows, _, actions = c.recovery_probe(state, encoder)
            label = int(actions[0]) if actions else -1
            raw_bad, bad, checks = action_probes(state, label)
            feature = encoder(state._observation())
            query.update(step=step, kind=kind, label=label, sample_index=len(samples),
                         inherited_blocked_run=candidate["blocked_run"], raw_bad_mask=raw_bad,
                         training_bad_mask=bad, action_checks=checks, state=c.scene_state(state),
                         feature=feature.tolist(), used_route_labels=1 if label >= 0 else 0)
            query.update(c.packed_trace(directory / f"witness_{seed}_{step}.jsonl.gz", route_rows))
            samples.append(feature)
            labels.append(label)
            masks.append(bad)
            categories.append(kind)
            queries.append(query)
        dataset = directory / f"states_{seed}.npz"
        np.savez_compressed(dataset, observations=np.asarray(samples, dtype=np.float32).reshape(-1, env.observation_size + 40),
                            actions=np.asarray(labels, dtype=np.int64), bad_actions=np.asarray(masks, dtype=bool).reshape(-1, 8))
        record.update(samples=len(labels), successful_labels=sum(v >= 0 for v in labels), categories=categories,
                      dataset=str(dataset.relative_to(HERE)), dataset_sha256=c.sha(dataset))
        return record
    finally:
        for candidate in candidates:
            candidate["env"].close()
        c.require(c.frozen_state(agent) == frozen, "Collection changed the inherited policy.")
        record.update(elapsed_seconds=perf_counter() - started, reset_profile=deepcopy(env.last_reset_profile),
                      frozen_state_unchanged=True)
        env.close()


def sample_groups(groups, size, rng):
    available = sorted(set(groups.tolist()))
    c.require(bool(available), "Cannot sample an empty pool.")
    return np.asarray([rng.choice(np.flatnonzero(groups == available[int(i)]))
                       for i in rng.integers(len(available), size=size)])


def point_scores(network, observations):
    """Match runtime batch-one inference when checking exact action retention."""
    with torch.no_grad():
        return torch.cat([network(row[None]) for row in observations])


def retain(network, anchor_x, anchor_actions):
    return bool((point_scores(network, anchor_x).argmax(1) == anchor_actions).all())


def project_block(network, optimizer, before_weights, before_optimizer, anchor_x, anchor_actions):
    """Backtrack one proposed block against the fixed successful-action bank."""
    proposed = deepcopy(network.correction.state_dict())
    for scale in SCALES:
        network.correction.load_state_dict({k: before_weights[k] + scale * (proposed[k] - before_weights[k]) for k in proposed})
        if retain(network, anchor_x, anchor_actions):
            return scale
    network.correction.load_state_dict(before_weights)
    optimizer.load_state_dict(before_optimizer)
    c.require(retain(network, anchor_x, anchor_actions), "Rollback did not restore retained actions.")
    return 0.0


def dataset(agent, collection):
    encoder = agent.encoder
    arrays = np.load(c.BASE_DATA / "expanded_demonstrations.npz")
    meta = c.read(c.BASE_DATA / "dataset.json")
    bx = np.stack([encoder.extend(x) for x in arrays["observations"]])
    by = arrays["actions"].copy()
    bg = np.empty(len(by), dtype=np.int64)
    for episode in meta["episodes"]:
        bg[episode["start"]:episode["stop"]] = episode["seed"]
    arrays.close()
    rx, ry, rm, rg, category = [], [], [], [], []
    for record in collection:
        c.require(record["seed"] in TRAIN, "Unexpected data split.")
        if "dataset" not in record:
            continue
        path = HERE / record["dataset"]
        c.require(c.sha(path) == record["dataset_sha256"], "Query samples changed.")
        with np.load(path) as data:
            rx.extend(data["observations"])
            ry.extend(data["actions"])
            rm.extend(data["bad_actions"])
            rg.extend([record["seed"]] * len(data["actions"]))
            category.extend(record["categories"])
    ax, ay, ag = [], [], []
    history = HERE.parent / "004_verified_supervision/evidence"
    for record in c.read(history / "evaluations.json"):
        if record["arm"] != "targeted" or record["update"] != 2000 or not record["metrics"]["success"]:
            continue
        path = history / record["trace"]
        c.require(c.sha(path) == record["trace_compressed_sha256"], "Retention witness changed.")
        raw = gzip.decompress(path.read_bytes())
        c.require(c.hashlib.sha256(raw).hexdigest() == record["trace_sha256"], "Retention trace changed.")
        for line in raw.decode().splitlines():
            row = json.loads(line)
            ax.append(encoder(np.asarray(row["raw_observation"], dtype=np.float32)))
            ay.append(row["action"])
            ag.append(record["seed"])
    values = {"base_x": bx, "base_y": by, "base_groups": bg,
              "query_x": np.asarray(rx, dtype=np.float32), "query_y": np.asarray(ry, dtype=np.int64),
              "bad_masks": np.asarray(rm, dtype=bool), "query_groups": np.asarray(rg),
              "query_categories": np.asarray(category), "anchor_x": np.asarray(ax, dtype=np.float32),
              "anchor_y": np.asarray(ay, dtype=np.int64), "anchor_groups": np.asarray(ag)}
    np.savez_compressed(OUTPUT / "training_data.npz", **values)
    return values


def fit():
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    agent, env, source = RecoveryAgent.from_source(c.SOURCE)
    agent.learner.update_count = 14000
    net, opt = agent.learner.online, agent.learner.optimizer
    data = dataset(agent, c.read(OUTPUT / "collection.json"))
    tensors = {k: torch.from_numpy(v) for k, v in data.items() if not k.endswith("groups") and k != "query_categories"}
    bx, by, rx, ry = [tensors[k] for k in ("base_x", "base_y", "query_x", "query_y")]
    ax, ay, bad = [tensors[k] for k in ("anchor_x", "anchor_y", "bad_masks")]
    valid = np.flatnonzero(data["query_y"] >= 0)
    risk = np.flatnonzero(data["bad_masks"].any(1) & ~data["bad_masks"].all(1))
    c.require(len(valid) > 0, "No verified recovery-entry labels; do not train on fabricated labels.")
    c.require(retain(net, ax, ay), "Initial conversion changed a successful action.")
    inherited_hash = c._digest(net.inherited.state_dict())
    target_hash = c._digest(agent.learner.target.state_dict())
    prior = point_scores(net, ax).softmax(1)
    initial_accuracy = float((point_scores(net, rx[valid]).argmax(1) == ry[valid]).float().mean())
    logs, accepted = [], 0
    timer = perf_counter()
    with (OUTPUT / "updates.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        for start in range(0, MAX_UPDATES, BLOCK):
            before_weights, before_optimizer = deepcopy(net.correction.state_dict()), deepcopy(opt.state_dict())
            losses = []
            for update in range(start + 1, start + BLOCK + 1):
                bi = sample_groups(data["base_groups"], 32, rng)
                ri = valid[sample_groups(data["query_groups"][valid], 64, rng)]
                ai = sample_groups(data["anchor_groups"], 64, rng)
                si = risk[sample_groups(data["query_groups"][risk], 64, rng)] if len(risk) else None
                opt.zero_grad(set_to_none=True)
                root_loss = F.cross_entropy(net(rx[ri]), ry[ri])
                base_loss = F.cross_entropy(net(bx[bi]), by[bi])
                anchor_q = net(ax[ai])
                kl = F.kl_div(anchor_q.log_softmax(1), prior[ai], reduction="batchmean")
                selected = anchor_q.gather(1, ay[ai, None]).squeeze(1)
                others = anchor_q + 0.1
                others = others.scatter(1, ay[ai, None], selected[:, None])
                margin = (others.max(1).values - selected).mean()
                safety = (net(rx[si]).softmax(1) * bad[si]).sum(1).mean() if si is not None else root_loss.new_zeros(())
                loss = root_loss + 0.5 * base_loss + kl + margin + 0.5 * safety
                c.require(bool(torch.isfinite(loss)), "Non-finite policy loss.")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(net.correction.parameters(), 10, error_if_nonfinite=True)
                opt.step()
                row = {"attempted_update": update, "loss": float(loss.detach()), "recovery_ce": float(root_loss.detach()),
                       "base_ce": float(base_loss.detach()), "anchor_kl": float(kl.detach()),
                       "anchor_margin": float(margin.detach()), "risk_mass": float(safety.detach()), "gradient_norm": float(norm)}
                handle.write(json.dumps(row) + "\n")
                losses.append(row["loss"])
            scale = project_block(net, opt, before_weights, before_optimizer, ax, ay)
            accepted += int(scale > 0)
            accuracy = float((point_scores(net, rx[valid]).argmax(1) == ry[valid]).float().mean())
            log = {"attempted_updates": start + BLOCK, "accepted_scale": scale, "root_accuracy": accuracy,
                   "retention_accuracy": 1.0, "mean_proposed_loss": float(np.mean(losses))}
            logs.append(log)
            c.save(OUTPUT / "blocks.json", logs)
            if (start + BLOCK) % 100 == 0:
                print(f"fit {start + BLOCK}: root accuracy={accuracy:.4f}, accepted blocks={accepted}, scale={scale}", flush=True)
            if start + BLOCK == PILOT and (accepted == 0 or accuracy < initial_accuracy + 0.03):
                break
    c.require(c._digest(net.inherited.state_dict()) == inherited_hash, "Inherited network changed.")
    c.require(c._digest(agent.learner.target.state_dict()) == target_hash, "Target changed.")
    c.require(retain(net, ax, ay), "Final retention check failed.")
    metadata = {"method": "geometry_residual_retention", "attempted_updates": logs[-1]["attempted_updates"],
                "accepted_blocks": accepted, "new_td_updates": 0, "historical_td_clock": 14000,
                "training_seeds": TRAIN, "outputs_are_calibrated_q_values": False}
    agent.save(MODEL, source["environment_config"], metadata)
    torch.save({"online": net.state_dict(), "optimizer": opt.state_dict(), "torch_rng": torch.get_rng_state(),
                "sampling_rng": rng.bit_generator.state, "metadata": metadata}, OUTPUT / "resume.pt")
    loaded, loaded_env, _ = RecoveryAgent.load(MODEL)
    c.require(c._digest(loaded.learner.online.state_dict()) == c._digest(net.state_dict()), "Export mismatch.")
    loaded_env.close()
    summary = {**metadata, "initial_root_accuracy": initial_accuracy, "final_root_accuracy": logs[-1]["root_accuracy"],
               "pilot_passed": any(v["attempted_updates"] == PILOT and v["root_accuracy"] >= initial_accuracy + 0.03 for v in logs),
               "retention_actions": len(ay), "retention_seeds": sorted(set(data["anchor_groups"].tolist())),
               "query_states": len(ry), "verified_root_labels": len(valid), "risk_labeled_states": len(risk),
               "base_rows": len(by), "model_sha256": c.sha(MODEL), "resume_sha256": c.sha(OUTPUT / "resume.pt"),
               "training_data_sha256": c.sha(OUTPUT / "training_data.npz"), "elapsed_seconds": perf_counter() - timer,
               "inherited_weights_unchanged": True, "target_unchanged": True, "all_anchor_actions_retained": True}
    summary["optimizer_steps"] = sorted({int(v["step"]) for v in opt.state.values()})
    c.save(OUTPUT / "training.json", summary)
    env.close()
    return summary


def run():
    OUTPUT.mkdir(exist_ok=False)
    started, timer = datetime.now(timezone.utc).isoformat(), perf_counter()
    c.save(OUTPUT / "status.json", {"status": "running", "started_at_utc": started})
    try:
        c.require(c.sha(c.SOURCE) == c.SOURCE_SHA and c.sha(c.SOURCE_STATE) == c.SOURCE_STATE_SHA, "Source changed.")
        c.require(not set(TRAIN) & set(VALIDATION), "Data split overlap.")
        c.require(c.sha(c.BASE_DATA / "expanded_demonstrations.npz") == c.read(c.BASE_DATA / "dataset.json")["npz_sha256"], "Base data changed.")
        paths = list((ROOT / "src").rglob("*.py")) + list(HERE.glob("*.py"))
        history = HERE.parent / "004_verified_supervision/evidence"
        paths += [c.SOURCE, c.SOURCE_STATE, HERE / "PROTOCOL.md", HERE.parent / "008_batch_recovery/run_batch.py",
                  c.BASE_DATA / "expanded_demonstrations.npz", c.BASE_DATA / "dataset.json", history / "evaluations.json",
                  HERE.parent / "008_batch_recovery/round_1/baseline_validation.json"]
        paths += [history / r["trace"] for r in c.read(history / "evaluations.json")
                  if r["arm"] == "targeted" and r["update"] == 2000 and r["metrics"]["success"]]
        hashes = {str(p.relative_to(ROOT)): c.sha(p) for p in paths}
        c.save(OUTPUT / "provenance.json", {"input_hashes": hashes,
               "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
               "training_seeds": TRAIN, "validation_seeds": VALIDATION, "workers": 4,
               "torch_threads": 1, "seed": SEED, "pilot_updates": PILOT, "max_updates": MAX_UPDATES,
               "block_size": BLOCK, "backtracking_scales": SCALES, "new_td_updates": 0})
        collection_dir, evaluation_dir = OUTPUT / "collection", OUTPUT / "evaluation"
        collection_dir.mkdir()
        evaluation_dir.mkdir()
        with ProcessPoolExecutor(max_workers=4, initializer=worker_init) as pool:
            collection = c.gather(pool, [(collect_task, c.SOURCE, s, collection_dir) for s in TRAIN], OUTPUT / "collection.json")
            previous = [r for r in c.read(history / "evaluations.json") if r["arm"] == "targeted" and r["update"] == 2000]
            for r in collection:
                old = next(v for v in previous if v["seed"] == r["seed"])
                if "reset_error" not in r:
                    c.require(r["metrics"] == {k: v for k, v in old["metrics"].items() if k != "seed"}, "Source rollout changed.")
                    c.require(r["initial_state"] == {k: old["reset_record"][k] for k in ("observation", "rng", "info")}, "Source scene changed.")
            training = fit()
            evaluations = c.gather(pool, [(evaluation_task, MODEL, s, evaluation_dir) for s in TRAIN + VALIDATION], OUTPUT / "evaluations.json")
        baseline_validation = c.read(HERE.parent / "008_batch_recovery/round_1/baseline_validation.json")
        initial = {r["seed"]: r["initial_state"] for r in collection + baseline_validation}
        for r in evaluations:
            if r["initial_state"] is not None and initial[r["seed"]] is not None:
                c.require(r["initial_state"] == initial[r["seed"]], "Paired initial state changed.")
        summary = {"training": c.group(evaluations, TRAIN), "original_three": c.group(evaluations, TRAIN[:3]),
                   "validation": c.group(evaluations, VALIDATION), "baseline_training": c.group(previous, TRAIN),
                   "baseline_validation": c.group(baseline_validation, VALIDATION), "fit": training,
                   "query_count": sum(len(r["queries"]) for r in collection),
                   "successful_routes": sum(q["success"] for r in collection for q in r["queries"]),
                   "policy_collection_actions": sum(r["metrics"]["steps"] for r in collection),
                   "recovery_witness_actions": sum(q["executed_actions"] for r in collection for q in r["queries"]),
                   "action_probe_steps": 8 * sum(len(r["queries"]) for r in collection),
                   "evaluation_actions": sum(r["metrics"]["steps"] for r in evaluations),
                   "certification_actions": sum(r["reset_profile"]["certification_actions"] for r in collection + evaluations),
                   "certification_calls": sum(r["reset_profile"]["certification_calls"] for r in collection + evaluations),
                   "cache_hits": sum(r["reset_profile"]["cache_status"] == "hit" for r in collection + evaluations),
                   "evaluation_overrides": 0, "new_td_updates": 0, "elapsed_seconds": perf_counter() - timer,
                   "scope": "Single bounded multi-component intervention; not a component ablation or final held-out test."}
        retained = set(summary["baseline_training"]["success_seeds"]).issubset(summary["training"]["success_seeds"])
        summary["all_source_successes_preserved"] = retained
        summary["replacement_eligible"] = (retained and summary["original_three"]["successes"] == 3
                                            and summary["validation"]["successes"] > summary["baseline_validation"]["successes"])
        for name, digest in hashes.items():
            c.require(c.sha(ROOT / name) == digest, "Pinned input changed.")
        c.save(OUTPUT / "summary.json", summary)
        c.save(OUTPUT / "status.json", {"status": "completed", "started_at_utc": started,
                                       "finished_at_utc": datetime.now(timezone.utc).isoformat()})
        c.save(OUTPUT / "manifest.json", {str(p.relative_to(OUTPUT)): c.sha(p) for p in sorted(OUTPUT.rglob("*")) if p.is_file()})
        print(json.dumps(summary, indent=2), flush=True)
    except BaseException as exc:
        c.save(OUTPUT / "status.json", {"status": "failed", "started_at_utc": started, "error": repr(exc)})
        raise


if __name__ == "__main__":
    run()
