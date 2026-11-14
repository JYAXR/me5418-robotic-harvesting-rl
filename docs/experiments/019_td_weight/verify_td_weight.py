"""Independently verify a reduced TD objective with unchanged margin supervision."""

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import torch
from torch.nn import functional as F

from me5418_harvesting.agents.dqn_agent import DQNAgent
from me5418_harvesting.agents.optimizer_transition import optimizer_state_summary

ROOT = Path(__file__).resolve().parents[3]
PRIOR = ROOT / "docs/experiments/017_mc_anchor/verify_mc_anchor.py"
RUNNER = Path(__file__).with_name("run_td_weight.py")
SPEC = importlib.util.spec_from_file_location("weight019_anchor_verifier", PRIOR)
anchor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(anchor)
local, prior, multi, audit = anchor.local, anchor.prior, anchor.multi, anchor.audit
read_json, read_log, file_hash = anchor.read_json, anchor.read_log, anchor.file_hash
require, require_equal = anchor.require, anchor.require_equal
CHECKPOINTS, TARGET_SOURCES = anchor.CHECKPOINTS, anchor.TARGET_SOURCES
RELOAD_CHECKPOINTS = [2000, 2009, 2200, 2201, 10000]
SENTINEL_STEPS = [6, 18, 20, 21, 22, 23, 24]


def validate_weight(weight):
    require(not isinstance(weight, bool) and weight in (0.5, 1.0),
            "predeclared TD coefficient must be 0.5 or 1")


def independent_update(learner, batch, weight):
    """Reconstruct two separate forwards and the declared weighted TD-plus-margin loss."""
    validate_weight(weight)
    learner.online.train()
    learner.optimizer.zero_grad(set_to_none=True)
    before = torch.nn.utils.parameters_to_vector(learner.online.parameters()).detach().clone()
    chosen = learner.online(batch.observations).gather(1, batch.actions[:, None]).squeeze(1)
    with torch.no_grad():
        following = learner.target(batch.next_observations).max(dim=1).values
        following = torch.where(batch.terminated, torch.zeros_like(following), following)
        targets = batch.rewards + learner.config.gamma * following
    td_before = F.smooth_l1_loss(chosen, targets)
    values = learner.online(batch.observations)
    margin_before = anchor.independent_hinges(values, batch.actions).mean()
    accuracy = float((values.detach().argmax(dim=1) == batch.actions).float().mean())
    # Preserve the original graph when checking the weight-one historical control.
    total_before = td_before if weight == 1.0 else weight * td_before
    total_before = total_before + 1.0 * margin_before
    require(bool(torch.isfinite(total_before)), "finite independent weighted TD objective")
    total_before.backward()
    norm = torch.nn.utils.clip_grad_norm_(
        learner.online.parameters(), learner.config.gradient_clip_norm, error_if_nonfinite=True)
    learner.optimizer.step()
    with torch.no_grad():
        after_values = learner.online(batch.observations)
        after_chosen = after_values.gather(1, batch.actions[:, None]).squeeze(1)
        td_after = F.smooth_l1_loss(after_chosen, targets)
        margin_after = anchor.independent_hinges(after_values, batch.actions).mean()
        total_after = td_after if weight == 1.0 else weight * td_after
        total_after = total_after + 1.0 * margin_after
        after = torch.nn.utils.parameters_to_vector(learner.online.parameters())
        displacement = torch.linalg.vector_norm(after - before)
        require(bool(torch.isfinite(after).all()) and bool(torch.isfinite(total_after)),
                "finite independently updated parameters and loss")
    learner.update_count += 1
    synchronized = learner.update_count % learner.config.target_update_interval == 0
    if synchronized:
        learner.target.load_state_dict(learner.online.state_dict())
    return {
        "loss_before": total_before.item(), "loss_after": total_after.item(),
        "gradient_norm": norm.item(), "parameter_delta_norm": displacement.item(),
        "target_synchronized": synchronized, "td_loss_before": td_before.item(),
        "td_loss_after": td_after.item(), "demonstration_loss_before": margin_before.item(),
        "demonstration_loss_after": margin_after.item(), "demonstration_action_accuracy": accuracy,
        "td_weight": float(weight),
    }


def verify_loss_records(records, weight=0.5):
    """Audit raw terms and their weighted sum without silently rescaling supervision."""
    validate_weight(weight)
    require(prior.verify_update_log(records) == 40, "forty preserved absolute-clock target synchronizations")
    for row in records:
        require(row["td_weight"] == weight, "fixed declared TD coefficient throughout continuation")
        require(row["gradient_norm"] >= 0 and row["parameter_delta_norm"] >= 0,
                "nonnegative gradient and parameter-displacement norms")
        require(not any(name.startswith(("return_loss", "return_anchor", "mc_loss")) for name in row),
                "no MC objective is introduced into the reduced-TD update")
        for when in ("before", "after"):
            td, margin = row[f"td_loss_{when}"], row[f"demonstration_loss_{when}"]
            require(math.isfinite(td) and math.isfinite(margin) and td >= 0 and margin >= 0,
                    "finite nonnegative unweighted TD and margin components")
            require(math.isclose(row[f"loss_{when}"], weight * td + margin,
                                 abs_tol=2e-7, rel_tol=1e-6), "weighted TD plus unscaled margin")


def replay_arm(source, directory, arm, batch, updates, sentinels, end=2201):
    """Independently replay the prescribed first 201 updates with complete Adam state."""
    agent, env = prior.independent_restore(source)
    weight = 1.0 if arm == "historical_control" else 0.5
    count, compared = 0, []
    print(f"Replaying {arm}: TD 2001 through {end} independently.", flush=True)
    try:
        require(agent.learner.update_count == 2000, "preserved replay boundary")
        prior.verify_sentinel_network(agent, sentinels[2000], batch)
        for step in range(2001, end + 1):
            stats = independent_update(agent.learner, batch, weight)
            count += 1
            if weight == 1.0:
                stats.pop("td_weight")
            require({"update": step, "extension_update": step - 2000, **stats} == updates[step],
                    f"{arm} exact independent weighted-objective log at TD {step}")
            prior.verify_sentinel_network(agent, sentinels[step], batch)
            path = directory / f"training_state_{step}.pt"
            if path.exists():
                saved = torch.load(path, map_location="cpu", weights_only=True)
                for name, state in (("online", agent.learner.online.state_dict()),
                                    ("target", agent.learner.target.state_dict()),
                                    ("optimizer", agent.learner.optimizer.state_dict()),
                                    ("agent_rng_state", agent.rng.bit_generator.state),
                                    ("torch_rng_state", torch.get_rng_state())):
                    require_equal(state, saved[name], f"{arm} exact replay TD {step} {name}")
                require(saved["update_count"] == agent.learner.update_count == step
                        and saved["optimizer_summary"] == optimizer_state_summary(agent.learner),
                        "preserved TD clock and Adam steps")
                compared.append(step)
    finally:
        env.close()
    return {"arm": arm, "td_weight": weight, "first_update": 2001, "last_update": end,
            "additional_optimizer_updates": count, "full_snapshot_updates_compared": compared,
            "sentinel_rows_compared": count + 1, "additional_environment_steps": 0,
            "additional_environment_resets": 0,
            "independent_math": "Separate unweighted TD Huber and per-row margin means; only the TD coefficient changes.",
            "logs_sentinels_and_full_snapshots_match_exactly": True}


def verify_treatment_phase(run_dir, phase, settings, source, source_row, batch, episodes):
    """Audit every saved model, target generation and recorded outcome without rollouts."""
    directory = run_dir / "treatment/td"
    require(read_json(directory / "summary.json") == phase, "matching treatment phase summary")
    fixed = {"directory": "treatment/td", "initial_td_update": 2000, "final_td_update": 10000,
             "optimizer_updates": 8000, "prior_updates_on_branch": 40000,
             "inherited_optimizer_updates": 42000, "total_updates_on_branch": 50000,
             "learning_rate": 0.0001, "target_synchronizations": 40,
             "target_source_updates": TARGET_SOURCES, "demonstration_weight": 1.0,
             "normalization_denominator": 117, "return_anchor_weight": 0.0,
             "all_sample_weights": 1.0, "td_weight": 0.5,
             "training_snapshot_file": "treatment/td/training_state.pt"}
    for key, value in fixed.items():
        require(phase[key] == value, f"fixed reduced-TD phase: {key}")
    rows = read_json(directory / "observations.json")
    require([row["update"] for row in rows] == CHECKPOINTS, "all 29 prescribed physical checkpoints")
    require(rows[0] == phase["initial_observation"] == source_row
            and rows[-1] == phase["final_observation"], "unchanged start and recorded endpoint")
    records = read_log(directory / "updates.jsonl")
    verify_loss_records(records)
    sentinel_rows = read_log(directory / "sentinel_q.jsonl")
    require(len(sentinel_rows) == phase["sentinel_records"] == 8001
            and phase["sentinel_environment_steps"] == 0, "complete action-free dense probes")
    for step, row in zip(range(2000, 10001), sentinel_rows, strict=True):
        local.verify_sentinel(row, batch, step)
    sentinels = {row["update"]: row for row in sentinel_rows}
    names = {f"update_{step}.pt" for step in CHECKPOINTS}
    names |= {f"training_state_{step}.pt" for step in CHECKPOINTS}
    names |= {f"target_source_state_{step}.pt" for step in TARGET_SOURCES}
    names.add("training_state.pt")
    require(set(phase["checkpoint_sha256"]) == names == {path.name for path in directory.glob("*.pt")},
            "all 67 prescribed model artifacts")
    for name, digest in phase["checkpoint_sha256"].items():
        require(file_hash(directory / name) == digest, f"treatment model hash: {name}")
    snapshots = {}
    for step in sorted(set(CHECKPOINTS) | set(TARGET_SOURCES)):
        stem = "training_state" if step in CHECKPOINTS else "target_source_state"
        snapshots[step] = torch.load(directory / f"{stem}_{step}.pt", map_location="cpu", weights_only=True)
    prior.verify_resume_payload(snapshots[2000], source)
    for step, payload in snapshots.items():
        target_source = snapshots[step - step % 200]
        stem = "training_state" if step in CHECKPOINTS else "target_source_state"
        multi.verify_snapshot(directory / f"{stem}_{step}.pt", audit.state_hash(payload["online"]), settings,
                              step, audit.state_hash(target_source["online"]), payload["optimizer_summary"])
        require_equal(payload["target"], target_source["online"], f"TD {step} target-source lineage")
        require(payload["optimizer_summary"]["step_min"] == payload["optimizer_summary"]["step_max"] == 20000 + step,
                "unchanged Adam internal steps")
        require_equal(payload["agent_rng_state"], source["agent_rng_state"], "unchanged fixed-data agent RNG")
        require_equal(payload["torch_rng_state"], source["torch_rng_state"], "unchanged fixed-data torch RNG")
        agent, env = prior.independent_restore(payload)
        try:
            prior.verify_sentinel_network(agent, sentinels[step], batch)
        finally:
            env.close()
    for row in rows:
        step = row["update"]
        multi.verify_aggregate(row, episodes)
        require(row["total_updates_on_branch"] == 40000 + step, "cumulative observation update count")
        payload = snapshots[step]
        inference = torch.load(directory / f"update_{step}.pt", map_location="cpu", weights_only=True)
        require_equal(inference["online"], payload["online"], "inference/full online state equality")
        require(json.loads(json.dumps(inference["dqn_config"])) == settings["dqn"]
                and inference["environment_config"] == settings["environment"], "unchanged inference configuration")
        require(row["optimizer_state"] == payload["optimizer_summary"], "observed Adam state equals full snapshot")
        metadata = inference["metadata"]
        require(metadata["phase"] == "multiscene_retention_diagnostic" and metadata["objective"] == "extension/td"
                and metadata["updates"] == step and metadata["prior_updates_on_branch"] == 40000
                and metadata["total_updates_on_branch"] == 40000 + step and metadata["training_scene_only"]
                and metadata["scene_seeds"] == [10000, 10001, 10002]
                and metadata["fit"] == row["fit"] and metadata["aggregate"] == row["aggregate"],
                "unchanged inference metadata and unweighted fit semantics")
    final = torch.load(directory / "training_state.pt", map_location="cpu", weights_only=True)
    require_equal(final, snapshots[10000], "final full snapshot alias")
    require(file_hash(directory / "training_state.pt") == phase["training_snapshot_sha256"]
            and phase["initial_parameter_sha256"] == source["parameter_sha256"]
            and phase["final_parameter_sha256"] == final["parameter_sha256"], "initial/final model identity")
    require(phase["evaluation_environment_steps"] == sum(row["aggregate"]["evaluation_environment_steps"] for row in rows),
            "new treatment policy action count")
    prior.verify_retention(phase["retention"], rows, source_row)
    return rows, records, sentinels


def verify_protocol(protocol, settings):
    """Require the predeclared single coefficient change and unchanged continuation settings."""
    expected = {
        "attempt": 19, "kind": "td_objective_weight_control",
        "scene_seeds": [10000, 10001, 10002], "initial_td_update": 2000, "final_td_update": 10000,
        "new_treatment_optimizer_updates": 8000, "td_checkpoints": CHECKPOINTS,
        "target_source_updates": TARGET_SOURCES, "inherited_shared_optimizer_updates": 42000,
        "learning_rate": 0.0001, "gamma": 0.99, "gradient_clip_norm": 10.0,
        "td_weight": 0.5, "baseline_td_weight": 1.0,
        "demonstration_weight": 1.0, "all_sample_weights": 1.0, "margin": 0.8,
        "return_anchor_weight": 0.0, "baseline_return_anchor_weight": 0.0,
        "target_update_interval": 200, "effective_batch_size": 117, "normalization_denominator": 117,
        "objective": "0.5 * mean TD Huber + 1.0 * mean demonstration hinge",
        "total_loss_rescaled": False, "mc_loss_constructed": False,
        "returns_used_only_for_evaluation": True, "td_targets_mixed_with_returns": False,
        "historical_control_reused_not_rerun": True, "historical_control_arm": "attempt_016/baseline",
        "preserve_adam_moments_and_parameter_steps": True, "preserve_absolute_td_counter": True,
        "synchronize_target_at_resume": False, "early_stopping": False,
        "sentinel_seed": 10001, "sentinel_reference_steps": SENTINEL_STEPS,
        "sentinel_batch_size": 1, "sentinel_updates": "every absolute update from 2000 through 10000 inclusive",
        "sentinel_gap": "Q(demonstrated action) - max Q(other actions)",
        "sentinel_is_physical_evaluation": False, "online_training_steps": 0, "held_out_scenes_accessed": 0,
        "reset_certification_preserved": True,
        "reset_certification_accounting": "An instance-scoped reset observer records returned certification attempts and copied-environment steps without changing reset arguments, results or physics.",
        "update_accuracy_semantics": "Demonstration action accuracy is measured before the update.",
        "update_target_semantics": "Before/after TD losses use the same frozen pre-update Bellman targets.",
    }
    for key, value in expected.items():
        require(protocol[key] == value, f"fixed reduced-TD protocol: {key}")
    dqn = settings["dqn"]
    require(dqn["learning_rate"] == 0.0001 and dqn["gamma"] == 0.99
            and dqn["target_update_interval"] == 200 and dqn["gradient_clip_norm"] == 10
            and dqn["hidden_sizes"] == [256, 256], "unchanged architecture and optimizer settings")
    require(file_hash(RUNNER) == protocol["script_sha256"], "reduced-TD runner source hash")


def verify_inputs(run_dir, protocol):
    """Read-only validation of source archives, historical models and inherited data."""
    manifest = read_json(run_dir / "input_manifest.json")
    reference = Path(manifest["reference_run"])
    anchor.verify_archive(reference, manifest["attempt_016"])
    historical_settings = read_json(reference / "settings.json")
    local.verify_protocol(historical_settings["training"], historical_settings)
    inherited, reference014, reference013, original = local.verify_inputs(reference, historical_settings["training"])
    require(manifest["inherited_manifest"] == inherited
            and Path(manifest["reference_014_run"]) == reference014
            and Path(manifest["reference_013_run"]) == reference013
            and Path(manifest["original_data_run"]) == original, "unbroken original-data and continuation chain")
    require(read_json(reference / "run_status.json")["status"] == "completed"
            and read_json(reference / "reload_checks.json")["passed"], "verified historical experiment")
    source_path = reference / manifest["resumed_snapshot_file"]
    require(manifest["resumed_snapshot_file"] == "baseline/td/training_state_2000.pt"
            and manifest["resumed_snapshot_sha256"] == file_hash(source_path), "same full TD-2000 source file")
    source = torch.load(source_path, map_location="cpu", weights_only=True)
    original_resume = torch.load(reference014 / "td/training_state_2000.pt", map_location="cpu", weights_only=True)
    require_equal(source, original_resume, "same complete TD-2000 training snapshot")
    require(source["update_count"] == 2000 and source["optimizer_summary"]["step_min"]
            == source["optimizer_summary"]["step_max"] == 22000
            and manifest["inherited_td_updates"] == 2000 and manifest["inherited_optimizer_updates"] == 42000,
            "preserved inherited TD, Adam and cumulative clocks")
    require(manifest["dataset_sha256"] == inherited["dataset_sha256"] == file_hash(original / "demonstration.npz"),
            "same ordered original demonstration data")
    helpers = dict(inherited["helper_sha256"])
    for relative in ("docs/experiments/016_local_supervision/run_local_supervision.py",
                     "docs/experiments/017_mc_anchor/run_mc_anchor.py"):
        helpers[relative] = file_hash(ROOT / relative)
    require(protocol["helper_sha256"] == manifest["helper_sha256"] == helpers, "complete immutable helper map")
    provenance = read_json(run_dir / "provenance.json")
    require(provenance["source_sha256"] == manifest["production_source_sha256"]
            == inherited["production_source_sha256"] and len(provenance["source_sha256"]) == 52,
            "all 52 production and configuration sources unchanged")
    for name, digest in {**helpers, **manifest["production_source_sha256"]}.items():
        require(file_hash(ROOT / name) == digest, f"immutable source: {name}")
    launched = subprocess.check_output(
        ["git", "show", f"{provenance['git_head']}:{RUNNER.relative_to(ROOT).as_posix()}"], cwd=ROOT)
    require(launched.replace(b"\r\n", b"\n") == RUNNER.read_bytes().replace(b"\r\n", b"\n"),
            "reduced-TD runner committed before main execution")
    settings = read_json(run_dir / "settings.json")
    require(settings["environment"] == source["environment_config"]
            and settings["dqn"] == json.loads(json.dumps(source["dqn_config"])), "same resolved source configurations")
    old_summary = read_json(reference / "summary.json")
    for arm in ("baseline", "treatment"):
        phase = read_json(reference / arm / "td/summary.json")
        require(phase == old_summary["arms"][arm] and len(phase["checkpoint_sha256"]) == 67,
                "unaltered complete historical model maps")
        for name, digest in phase["checkpoint_sha256"].items():
            require(file_hash(reference / arm / "td" / name) == digest, f"immutable historical model: {arm}/{name}")
    baseline = read_json(run_dir / "baseline_reference.json")
    require(baseline["td_weight"] == 1.0
            and baseline["certification_accounting"] == "Historical policy-action counters exclude reset-time copied-environment certification; no new baseline simulator work is performed.",
            "original control coefficient and explicitly limited historical certification accounting")
    original_baseline = {key: value for key, value in baseline.items()
                         if key not in {"td_weight", "certification_accounting"}}
    anchor.verify_baseline_reference(original_baseline, reference, old_summary["arms"]["baseline"])
    return manifest, reference, reference014, reference013, original


def certification_record(info, seed, update, enabled):
    """Reconstruct completed-reset accounting from the unmodified returned checks."""
    require(info.get("seed", seed) == seed, "reset returns its requested scene seed")
    checks = deepcopy(info.get("reference_route_checks", []))
    require(isinstance(checks, list) and (bool(checks) if enabled else not checks),
            "certification availability matches unchanged environment configuration")
    for index, check in enumerate(checks):
        require(isinstance(check["steps"], int) and not isinstance(check["steps"], bool)
                and check["steps"] >= 0, "finite nonnegative copied-environment action count")
        require(check["success"] is (index == len(checks) - 1),
                "only the last reset candidate is accepted by successful certification")
    return {"update": update, "seed": seed, "certification_enabled": bool(enabled),
            "certification_attempts": len(checks),
            "certification_environment_steps": sum(check["steps"] for check in checks),
            "reference_route_checks": checks}


@contextmanager
def capture_resets(env, update, records, incomplete):
    """Observe one environment instance without modifying reset arguments or return values."""
    original = env.reset
    existed = "reset" in env.__dict__
    previous = env.__dict__.get("reset")

    def tracked(*args, **kwargs):
        try:
            result = original(*args, **kwargs)
        except BaseException:
            incomplete.append({"update": update, "seed": kwargs.get("seed"),
                               "certification_steps_unknown": True})
            raise
        records.append(certification_record(result[1], kwargs.get("seed"), update,
                                            env.verify_reference_route))
        return result

    env.reset = tracked
    try:
        yield
    finally:
        if existed:
            env.reset = previous
        else:
            del env.reset


def verify_certification(record, rows, enabled):
    """Audit all returned reset candidates, including rejected candidates and copied physics."""
    expected_pairs = [(row["update"], scene["seed"]) for row in rows for scene in row["scenes"]]
    require([(row["update"], row["seed"]) for row in record["records"]] == expected_pairs,
            "one complete reset record for each prescribed physical scene evaluation")
    by_seed = {}
    for row in record["records"]:
        rebuilt = certification_record({"seed": row["seed"], "reference_route_checks": row["reference_route_checks"]},
                                       row["seed"], row["update"], enabled)
        require(row == rebuilt, "recorded certification counts equal all returned candidate checks")
        if row["seed"] in by_seed:
            require(row["reference_route_checks"] == by_seed[row["seed"]],
                    "same seeded scene has identical reset certification across checkpoints")
        by_seed[row["seed"]] = row["reference_route_checks"]
    require(record["completed_reset_count"] == len(expected_pairs)
            and record["certification_attempts"] == sum(row["certification_attempts"] for row in record["records"])
            and record["certification_environment_steps"]
            == sum(row["certification_environment_steps"] for row in record["records"])
            and record["incomplete_resets"] == [] and record["accounting_complete"],
            "complete independently summed reset and copied-simulator accounting")


def reload_policy(run_dir, step, row, batch, returns, episodes, expected_certification):
    """Repeat the original inference evaluation while separately observing reset certification."""
    checkpoint = run_dir / "treatment/td" / f"update_{step}.pt"
    agent, env, metadata = DQNAgent.load(checkpoint)
    records, incomplete, verified = [], [], []
    try:
        require(multi.diagnostic.fit.fit_metrics(agent, batch, 0.8) == row["fit"] == metadata["fit"],
                "reloaded pooled demonstration fit")
        require(multi.diagnostic.calibration.return_fit(agent, batch, returns) == row["return_fit"],
                "reloaded pooled return diagnostics")
        with capture_resets(env, step, records, incomplete):
            for entry, scene in zip(episodes, row["scenes"], strict=True):
                part = multi.episode_batch(batch, entry)
                selected_returns = returns[entry["start"]:entry["stop"]]
                require(multi.diagnostic.fit.fit_metrics(agent, part, 0.8) == scene["fit"], "reloaded per-scene fit")
                require(multi.diagnostic.calibration.return_fit(agent, part, selected_returns) == scene["return_fit"],
                        "reloaded per-scene return diagnostics")
                rollout = multi.diagnostic.fit.fixed_scene_rollout(agent, env, entry["seed"], part)
                require(rollout == scene["rollout"], f"TD {step}, scene {entry['seed']} exact reloaded trajectory")
                verified.append({"seed": entry["seed"], "full_rollout_matches_exactly": True,
                                 "fit_and_return_metrics_match_exactly": True,
                                 "success": rollout["success"], "damage": rollout["damage"],
                                 "reload_environment_steps": rollout["steps"]})
        require(not incomplete and records == expected_certification,
                "reloaded certification exactly equals main-run copied physics")
        return {"directory": "treatment/td", "update": step, "scenes": verified,
                "checkpoint_sha256": file_hash(checkpoint), "reset_certification": records}
    finally:
        env.close()


def verify_accounting(accounting, rows, certification):
    """Keep training, reused outcomes, policy actions and copied-simulator actions separate."""
    anchor.verify_accounting(accounting, rows)
    expected = {
        "environment_resets": certification["completed_reset_count"],
        "reset_certification_attempts": certification["certification_attempts"],
        "reset_certification_environment_steps": certification["certification_environment_steps"],
        "total_simulator_environment_steps": accounting["evaluation_environment_steps"] + certification["certification_environment_steps"],
        "reset_accounting_complete": True, "incomplete_resets": [],
    }
    for key, value in expected.items():
        require(accounting[key] == value, f"policy versus certification accounting: {key}")


def verify_initial_q(record, source, batch):
    agent, env = prior.independent_restore(source)
    try:
        agent.learner.online.eval()
        with torch.no_grad():
            values = agent.learner.online(batch.observations)
        digest = hashlib.sha256(values.contiguous().numpy().tobytes()).hexdigest()
        require(record == {"passed": True, "rows": 117, "shape": [117, 8],
                           "source_and_restored_q_sha256": digest, "q_values": values.tolist(),
                           "maximum_absolute_difference": 0.0, "new_optimizer_updates": 0,
                           "environment_actions": 0}, "all 117 ordered initial Q vectors equal original source")
    finally:
        env.close()


def verify(run_dir, output):
    """Independently replay 402 updates and reload 15 episodes after read-only full audits."""
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite independent TD-weight verification: {output}")
    torch.set_num_threads(1)
    require(read_json(run_dir / "run_status.json")["status"] == "completed", "completed reduced-TD treatment")
    settings, summary = read_json(run_dir / "settings.json"), read_json(run_dir / "summary.json")
    protocol = settings["training"]
    require(summary["protocol"] == protocol, "matching declared and summarized protocol")
    print("Auditing protocol, source archives, original data and initial Q vectors.", flush=True)
    verify_protocol(protocol, settings)
    manifest, reference, reference014, reference013, original = verify_inputs(run_dir, protocol)
    source = torch.load(reference / manifest["resumed_snapshot_file"], map_location="cpu", weights_only=True)
    source_row = read_json(reference / "baseline/td/observations.json")[0]
    batch, returns, episodes, return_checks = prior.load_verified_batch(run_dir, reference013, original, summary)
    verify_initial_q(read_json(run_dir / "initial_q_equivalence.json"), source, batch)
    require(summary["historical_control"] == read_json(run_dir / "baseline_reference.json"), "identical historical reference")
    rows, updates, sentinels = verify_treatment_phase(
        run_dir, summary["treatment"], settings, source, source_row, batch, episodes)
    require(read_json(run_dir / "treatment/restore_checks.json") == {
        "exact_full_snapshot": True, "target_synchronized_at_restore": False,
        "optimizer_reset_at_restore": False, "td_counter_reset_at_restore": False,
        "absolute_td_update": 2000, "optimizer_state": source["optimizer_summary"]}, "complete original state restored")
    require(read_json(run_dir / "treatment/initial_equivalence.json") == {
        "passed": True, "exact_previous_final_observation": True,
        "absolute_td_update": 2000, "new_updates_before_gate": 0}, "unchanged initial physical behavior")
    certification = read_json(run_dir / "reset_certification.json")
    require(summary["treatment"]["reset_certification"] == certification, "matching phase certification evidence")
    verify_certification(certification, rows, settings["environment"].get("verify_reference_route", False))
    require(summary["accounting"] == read_json(run_dir / "accounting.json"), "matching completed work accounting")
    verify_accounting(summary["accounting"], rows, certification)
    historical_settings = read_json(reference / "settings.json")
    old_rows, old_updates, _, old_sentinels = local.verify_phase(
        reference, "baseline", read_json(reference / "baseline/td/summary.json"),
        historical_settings, source, source_row, batch, episodes)
    equality = local.verify_baseline(reference, reference014)
    replays = [
        replay_arm(source, reference / "baseline/td", "historical_control", batch,
                   {row["update"]: row for row in old_updates}, old_sentinels),
        replay_arm(source, run_dir / "treatment/td", "treatment", batch,
                   {row["update"]: row for row in updates}, sentinels),
    ]
    by_step = {row["update"]: row for row in rows}
    reloads = []
    for step in RELOAD_CHECKPOINTS:
        print(f"Reloading treatment TD {step}, preserving and counting reset certification.", flush=True)
        expected = [row for row in certification["records"] if row["update"] == step]
        reloads.append(reload_policy(run_dir, step, by_step[step], batch, returns, episodes, expected))
    policy_steps = sum(scene["reload_environment_steps"] for row in reloads for scene in row["scenes"])
    certification_steps = sum(record["certification_environment_steps"] for row in reloads for record in row["reset_certification"])
    certification_attempts = sum(record["certification_attempts"] for row in reloads for record in row["reset_certification"])
    for name, digest in {**manifest["helper_sha256"], **manifest["production_source_sha256"]}.items():
        require(file_hash(ROOT / name) == digest, f"verification preserves historical source: {name}")
    require(file_hash(reference / manifest["resumed_snapshot_file"]) == manifest["resumed_snapshot_sha256"],
            "verification preserves original full snapshot")
    result = {
        "passed": True, "attempt": 19, "verifier_sha256": file_hash(Path(__file__)),
        "prior_verifier_sha256": file_hash(PRIOR), "input_manifest_sha256": file_hash(run_dir / "input_manifest.json"),
        "reference_run": str(reference), "reference_014_run": str(reference014),
        "reference_013_run": str(reference013), "original_data_run": str(original),
        "source_snapshot_sha256": manifest["resumed_snapshot_sha256"],
        "independent_return_checks": return_checks, "historical_baseline_equivalence": equality,
        "verified_new_raw_updates": len(updates), "verified_historical_raw_updates": len(old_updates),
        "verified_new_sentinel_rows": len(sentinels), "verified_historical_sentinel_rows": len(old_sentinels),
        "verified_new_physical_checkpoints": len(rows), "verified_historical_physical_checkpoints": len(old_rows),
        "verified_new_model_files": 67, "verified_historical_control_model_files": 67,
        "independent_replays": replays, "policy_reloads": reloads,
        "additional_optimizer_updates": sum(row["additional_optimizer_updates"] for row in replays),
        "additional_environment_resets": 15, "additional_evaluation_episodes": 15,
        "additional_environment_steps": policy_steps,
        "additional_reset_certification_attempts": certification_attempts,
        "additional_reset_certification_environment_steps": certification_steps,
        "additional_total_simulator_environment_steps": policy_steps + certification_steps,
        "historical_control_evaluations_rerun": False,
        "scope": "Each arm independently replays only TD 2001-2201. All later saved states, logs and target lineage are audited without replaying every main update. Only treatment policies receive 15 reload episodes.",
        "loss_semantics": "Only the TD Huber mean coefficient changes from 1 to 0.5. Every supervised coefficient remains one; no MC objective or total-loss rescaling is present.",
        "physical_accounting": "Policy evaluation actions and reset-time copied-environment certification actions are separate. Certification is preserved and observed without extra resets or planning.",
        "sentinel_interpretation": "Reference-state rankings do not establish physical success at unsampled updates.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.run_dir.resolve(), args.output.resolve())
    print(json.dumps({key: result[key] for key in (
        "passed", "additional_optimizer_updates", "additional_evaluation_episodes", "additional_environment_steps",
        "additional_reset_certification_environment_steps", "additional_total_simulator_environment_steps")}))


if __name__ == "__main__":
    main()
