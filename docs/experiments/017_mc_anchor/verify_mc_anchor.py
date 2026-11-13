"""Independently audit fixed-return anchoring against a verified historical control."""

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
from pathlib import Path

import torch
from torch.nn import functional as F

from me5418_harvesting.agents.optimizer_transition import optimizer_state_summary

ROOT = Path(__file__).resolve().parents[3]
PRIOR = ROOT / "docs/experiments/016_local_supervision/verify_local_supervision.py"
RUNNER = Path(__file__).with_name("run_mc_anchor.py")
SPEC = importlib.util.spec_from_file_location("anchor017_prior_verifier", PRIOR)
local = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(local)
prior, multi, audit = local.prior, local.multi, local.audit
read_json, read_log, file_hash = local.read_json, local.read_log, local.file_hash
require, require_equal = local.require, local.require_equal
CHECKPOINTS, TARGET_SOURCES = local.CHECKPOINTS, local.TARGET_SOURCES
RELOAD_CHECKPOINTS = [2000, 2009, 2200, 2201, 10000]
SENTINEL_STEPS = [6, 18, 20, 21, 22, 23, 24]
EXTRA_FIELDS = {"return_loss_before", "return_loss_after", "return_anchor_weight",
                "td_target_return_rmse", "td_target_return_mean_difference",
                "td_target_return_max_abs_difference"}


def verify_protocol(protocol, settings):
    """Require the declared intervention and every preserved training condition."""
    expected = {
        "attempt": 17, "kind": "persistent_demonstration_return_anchor",
        "scene_seeds": [10000, 10001, 10002], "initial_td_update": 2000, "final_td_update": 10000,
        "new_treatment_optimizer_updates": 8000, "td_checkpoints": CHECKPOINTS,
        "target_source_updates": TARGET_SOURCES, "inherited_shared_optimizer_updates": 42000,
        "learning_rate": 0.0001, "gamma": 0.99, "gradient_clip_norm": 10.0,
        "demonstration_weight": 1.0, "all_sample_weights": 1.0, "margin": 0.8,
        "return_anchor_weight": 1.0, "baseline_return_anchor_weight": 0.0,
        "target_update_interval": 200, "effective_batch_size": 117, "normalization_denominator": 117,
        "objective": "mean TD Huber + mean demonstration hinge + mean fixed-return Huber",
        "return_targets": "Original episodewise actual discounted demonstration returns; detached and unchanged.",
        "total_loss_divided_by_three": False, "td_targets_mixed_with_returns": False,
        "historical_control_reused_not_rerun": True, "historical_control_arm": "attempt_016/baseline",
        "preserve_adam_moments_and_parameter_steps": True, "preserve_absolute_td_counter": True,
        "synchronize_target_at_resume": False, "early_stopping": False,
        "sentinel_seed": 10001, "sentinel_reference_steps": SENTINEL_STEPS,
        "sentinel_batch_size": 1, "sentinel_updates": "every absolute update from 2000 through 10000 inclusive",
        "sentinel_gap": "Q(demonstrated action) - max Q(other actions)",
        "sentinel_is_physical_evaluation": False, "online_training_steps": 0, "held_out_scenes_accessed": 0,
        "update_accuracy_semantics": "Demonstration action accuracy is measured before the update.",
        "update_target_semantics": "Before/after TD losses use the same frozen pre-update Bellman targets.",
    }
    for key, value in expected.items():
        require(protocol[key] == value, f"fixed return-anchor protocol: {key}")
    dqn = settings["dqn"]
    require(dqn["learning_rate"] == 0.0001 and dqn["gamma"] == 0.99
            and dqn["target_update_interval"] == 200 and dqn["gradient_clip_norm"] == 10
            and dqn["hidden_sizes"] == [256, 256], "unchanged architecture and optimizer configuration")
    require(file_hash(RUNNER) == protocol["script_sha256"], "return-anchor runner source hash")


def verify_archive(reference, anchor):
    """Bind historical evidence, full raw logs and samples to committed bytes."""
    relative = anchor["archive_manifest"]
    require(relative == "docs/experiments/016_local_supervision/evidence/archive_manifest.json",
            "fixed attempt-016 evidence archive")
    archive_path = ROOT / relative
    committed = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)
    require(committed == archive_path.read_bytes()
            and hashlib.sha256(committed).hexdigest() == anchor["committed_archive_manifest_sha256"]
            == anchor["archive_manifest_sha256"] and anchor["committed_archive_verified"],
            "exact committed historical archive manifest")
    archive = read_json(archive_path)
    require(archive["status"] == "completed" and Path(archive["source_run"]) == reference
            and len(archive["files"]) == 30 and len(archive["raw_log_coverage"]) == 6,
            "complete historical archive and raw-log coverage")
    raw = {}
    for name, record in archive["files"].items():
        path = archive_path.parent / name
        require(file_hash(path) == record["sha256"] and path.stat().st_size == record["bytes"],
                f"historical archive bytes: {name}")
        blob = subprocess.check_output(["git", "show", f"HEAD:{path.relative_to(ROOT).as_posix()}"], cwd=ROOT)
        require(hashlib.sha256(blob).hexdigest() == record["sha256"], f"committed historical evidence bytes: {name}")
        if not name.endswith("_samples.jsonl"):
            require(file_hash(reference / name) == record["sha256"], f"unchanged historical raw evidence: {name}")
            raw[name] = record["sha256"]
    for name, record in archive["raw_log_coverage"].items():
        lines = (reference / name).read_text(encoding="utf-8").splitlines()
        initial = 2000 if name.endswith("sentinel_q.jsonl") else 2001
        first_count = 11 if initial == 2000 else 10
        require(file_hash(reference / name) == record["raw_sha256"]
                and len(lines) == record["raw_records"]
                and [json.loads(line)["update"] for line in lines] == list(range(initial, 10001)),
                f"complete historical raw log: {name}")
        selected = [line for index, line in enumerate(lines)
                    if index < first_count or index == len(lines) - 1 or json.loads(line)["update"] % 50 == 0]
        sample_name = {"updates.jsonl": "update_samples.jsonl", "supervision.jsonl": "supervision_samples.jsonl",
                       "sentinel_q.jsonl": "sentinel_q_samples.jsonl"}[Path(name).name]
        sample = archive_path.parent / Path(name).parent / sample_name
        require(sample.read_text(encoding="utf-8").splitlines() == selected
                and len(selected) == record["archived_records"], "historical compact samples reproduce raw coverage")
        raw[name] = record["raw_sha256"]
    require(anchor["reference_files_sha256"] == raw
            and anchor["archived_files_sha256"] == {name: row["sha256"] for name, row in archive["files"].items()},
            "complete immutable historical file hash maps")
    return archive


def verify_baseline_reference(record, reference, phase):
    """Ensure reused control evidence is never reported as newly executed work."""
    gate = {"passed": True, "exact_update_records": 8000, "exact_observations": 29,
            "exact_checkpoint_payloads": 67, "exact_inherited_sentinel_rows": 8001,
            "treatment_started_before_gate": False}
    require(record == {
        "reference_run": str(reference), "directory": "baseline/td", "reused_not_rerun": True,
        "return_anchor_weight": 0.0, "demonstration_weight": 1.0, "all_sample_weights": 1.0,
        "optimizer_updates": 8000, "scheduled_evaluation_episodes": 87, "evaluation_environment_steps": 3711,
        "new_optimizer_updates": 0, "new_evaluation_episodes": 0, "new_environment_steps": 0,
        "exact_attempt_014_reproduction": gate,
        "observations_sha256": file_hash(reference / "baseline/td/observations.json"),
        "updates_sha256": file_hash(reference / "baseline/td/updates.jsonl"),
        "sentinel_sha256": file_hash(reference / "baseline/td/sentinel_q.jsonl"),
        "summary_sha256": file_hash(reference / "baseline/td/summary.json"),
        "checkpoint_sha256": phase["checkpoint_sha256"],
    }, "exact historical control identity, original counts and zero new main work")
    require(read_json(reference / "baseline_equivalence.json") == gate, "previous exact baseline reproduction gate")


def verify_inputs(run_dir, protocol):
    """Read-only provenance audit returning the 016, 014, 013 and original 011 paths."""
    manifest = read_json(run_dir / "input_manifest.json")
    reference = Path(manifest["reference_run"])
    verify_archive(reference, manifest["attempt_016"])
    historical_settings = read_json(reference / "settings.json")
    local.verify_protocol(historical_settings["training"], historical_settings)
    inherited, reference014, reference013, original = local.verify_inputs(reference, historical_settings["training"])
    require(manifest["inherited_manifest"] == inherited
            and Path(manifest["reference_014_run"]) == reference014
            and Path(manifest["reference_013_run"]) == reference013
            and Path(manifest["original_data_run"]) == original,
            "unbroken historical continuation and original demonstration chain")
    require(read_json(reference / "run_status.json")["status"] == "completed"
            and read_json(reference / "reload_checks.json")["passed"], "verified historical experiment completion")
    require(manifest["resumed_snapshot_file"] == "baseline/td/training_state_2000.pt"
            and manifest["resumed_snapshot_sha256"] == file_hash(reference / manifest["resumed_snapshot_file"]),
            "exact shared full TD-2000 snapshot file")
    source = torch.load(reference / manifest["resumed_snapshot_file"], map_location="cpu", weights_only=True)
    original_resume = torch.load(reference014 / "td/training_state_2000.pt", map_location="cpu", weights_only=True)
    require_equal(source, original_resume, "same complete initial state as the original all-ones control")
    require(source["update_count"] == 2000 and source["optimizer_summary"]["step_min"]
            == source["optimizer_summary"]["step_max"] == 22000
            and manifest["inherited_td_updates"] == 2000 and manifest["inherited_optimizer_updates"] == 42000,
            "unchanged inherited TD, Adam and cumulative training clocks")
    require(manifest["dataset_sha256"] == inherited["dataset_sha256"] == file_hash(original / "demonstration.npz"),
            "same original ordered demonstration data")
    previous_runner = ROOT / "docs/experiments/016_local_supervision/run_local_supervision.py"
    helpers = {**inherited["helper_sha256"], previous_runner.relative_to(ROOT).as_posix(): file_hash(previous_runner)}
    require(protocol["helper_sha256"] == manifest["helper_sha256"] == helpers, "complete immutable helper map")
    provenance = read_json(run_dir / "provenance.json")
    require(provenance["source_sha256"] == manifest["production_source_sha256"]
            == inherited["production_source_sha256"] and len(provenance["source_sha256"]) == 52,
            "unchanged 52 production and configuration source files")
    for name, digest in {**helpers, **manifest["production_source_sha256"]}.items():
        require(file_hash(ROOT / name) == digest, f"immutable source/helper: {name}")
    launch = subprocess.check_output(
        ["git", "show", f"{provenance['git_head']}:{RUNNER.relative_to(ROOT).as_posix()}"], cwd=ROOT)
    require(launch.replace(b"\r\n", b"\n") == RUNNER.read_bytes().replace(b"\r\n", b"\n"),
            "return-anchor runner committed before launch")
    settings = read_json(run_dir / "settings.json")
    require(settings["environment"] == source["environment_config"]
            and settings["dqn"] == json.loads(json.dumps(source["dqn_config"])), "configuration equals inherited state")
    old_summary = read_json(reference / "summary.json")
    for arm in ("baseline", "treatment"):
        phase = read_json(reference / arm / "td/summary.json")
        require(phase == old_summary["arms"][arm] and len(phase["checkpoint_sha256"]) == 67,
                "complete unchanged historical model maps")
        for name, digest in phase["checkpoint_sha256"].items():
            require(file_hash(reference / arm / "td" / name) == digest, f"historical model bytes: {arm}/{name}")
    baseline = read_json(run_dir / "baseline_reference.json")
    verify_baseline_reference(baseline, reference, old_summary["arms"]["baseline"])
    return manifest, reference, reference014, reference013, original


def independent_hinges(values, actions):
    """Compute each unweighted demonstrated-action margin without runner helpers."""
    require(values.ndim == 2 and actions.shape == (len(values),) and len(values) > 0,
            "one demonstrated action for each nonempty Q row")
    offsets = torch.full_like(values, 0.8)
    offsets.scatter_(1, actions[:, None], 0.0)
    chosen = values.gather(1, actions[:, None]).squeeze(1)
    return (values + offsets).max(dim=1).values - chosen


def validate_anchor(returns, batch, weight):
    """Reject undeclared objectives and malformed targets before touching state."""
    require(not isinstance(weight, bool) and weight in (0.0, 1.0),
            "predeclared MC coefficient must be zero or one")
    require(isinstance(returns, torch.Tensor) and returns.is_floating_point()
            and returns.ndim == 1 and returns.numel() > 0
            and returns.shape == batch.rewards.shape == batch.actions.shape,
            "complete finite return targets match the transition vector")
    require(returns.device == batch.observations.device == batch.rewards.device == batch.actions.device
            and returns.dtype == batch.observations.dtype
            and bool(torch.isfinite(returns).all()), "finite return targets on the batch device")
    require(batch.observations.ndim == 2 and len(batch.observations) == len(returns)
            and batch.actions.dtype == torch.long and bool(torch.isfinite(batch.observations).all()),
            "finite observations and demonstrated actions match the return targets")
    return returns.detach()


def independent_update(learner, batch, returns, weight):
    """Rebuild TD, margin and fixed-return losses without invoking the new updater."""
    fixed_returns = validate_anchor(returns, batch, weight)
    learner.online.train()
    learner.optimizer.zero_grad(set_to_none=True)
    parameters_before = torch.nn.utils.parameters_to_vector(learner.online.parameters()).detach().clone()
    chosen = learner.online(batch.observations).gather(1, batch.actions[:, None]).squeeze(1)
    with torch.no_grad():
        following = learner.target(batch.next_observations).max(dim=1).values
        following = torch.where(batch.terminated, torch.zeros_like(following), following)
        bellman_targets = batch.rewards + learner.config.gamma * following
        target_difference = bellman_targets.double() - fixed_returns.double()
    td_before = F.smooth_l1_loss(chosen, bellman_targets)
    values = learner.online(batch.observations)
    margin_before = independent_hinges(values, batch.actions).mean()
    return_before = F.smooth_l1_loss(chosen, fixed_returns)
    accuracy = float((values.detach().argmax(dim=1) == batch.actions).float().mean())
    total_before = td_before + 1.0 * margin_before
    if weight:
        total_before = total_before + weight * return_before
    require(bool(torch.isfinite(total_before)), "finite independently reconstructed objective")
    total_before.backward()
    norm = torch.nn.utils.clip_grad_norm_(
        learner.online.parameters(), learner.config.gradient_clip_norm, error_if_nonfinite=True)
    learner.optimizer.step()
    with torch.no_grad():
        after_values = learner.online(batch.observations)
        after_chosen = after_values.gather(1, batch.actions[:, None]).squeeze(1)
        td_after = F.smooth_l1_loss(after_chosen, bellman_targets)
        margin_after = independent_hinges(after_values, batch.actions).mean()
        return_after = F.smooth_l1_loss(after_chosen, fixed_returns)
        total_after = td_after + 1.0 * margin_after
        if weight:
            total_after = total_after + weight * return_after
        parameters_after = torch.nn.utils.parameters_to_vector(learner.online.parameters())
        displacement = torch.linalg.vector_norm(parameters_after - parameters_before)
        require(bool(torch.isfinite(parameters_after).all()) and bool(torch.isfinite(total_after)),
                "finite independently updated parameters and loss")
    learner.update_count += 1
    synchronized = learner.update_count % learner.config.target_update_interval == 0
    if synchronized:
        learner.target.load_state_dict(learner.online.state_dict())
    return {
        "loss_before": total_before.item(), "loss_after": total_after.item(),
        "gradient_norm": norm.item(), "parameter_delta_norm": displacement.item(),
        "target_synchronized": synchronized,
        "td_loss_before": td_before.item(), "td_loss_after": td_after.item(),
        "demonstration_loss_before": margin_before.item(), "demonstration_loss_after": margin_after.item(),
        "demonstration_action_accuracy": accuracy,
        "return_loss_before": return_before.item(), "return_loss_after": return_after.item(),
        "return_anchor_weight": float(weight),
        "td_target_return_rmse": float(target_difference.square().mean().sqrt()),
        "td_target_return_mean_difference": float(target_difference.mean()),
        "td_target_return_max_abs_difference": float(target_difference.abs().max()),
    }


def verify_loss_records(records, weight=1.0):
    """Check the three separate means and their unnormalized weighted sum."""
    require(prior.verify_update_log(records) == 40, "forty preserved absolute-clock target synchronizations")
    for row in records:
        require(row["return_anchor_weight"] == weight, "fixed MC coefficient throughout continuation")
        rmse, bias, maximum = (row[key] for key in (
            "td_target_return_rmse", "td_target_return_mean_difference", "td_target_return_max_abs_difference"))
        require(0 <= rmse <= maximum + 1e-12 and abs(bias) <= rmse + 1e-12,
                "consistent finite Bellman-target versus fixed-return diagnostics")
        for timing in ("before", "after"):
            terms = [row[f"{name}_{timing}"] for name in ("td_loss", "demonstration_loss", "return_loss")]
            require(all(math.isfinite(value) and value >= 0 for value in terms), "finite nonnegative loss components")
            expected = float(torch.tensor(terms[0], dtype=torch.float32) + torch.tensor(terms[1], dtype=torch.float32)
                             + weight * torch.tensor(terms[2], dtype=torch.float32))
            require(row[f"loss_{timing}"] == expected, "TD plus margin plus MC, with no total-loss division")
        require(row["parameter_delta_norm"] > 0, "every declared optimizer update changes the parameters")


def replay_arm(source, directory, arm, batch, returns, updates, sentinels, end=2201):
    """Independently replay only the declared first 201 continuation updates."""
    agent, env = prior.independent_restore(source)
    weight = 0.0 if arm == "historical_control" else 1.0
    count, compared = 0, []
    print(f"Replaying {arm}: TD 2001 through {end} independently.", flush=True)
    try:
        require(agent.learner.update_count == 2000, "independent replay starts at the preserved TD boundary")
        prior.verify_sentinel_network(agent, sentinels[2000], batch)
        for step in range(2001, end + 1):
            stats = independent_update(agent.learner, batch, returns, weight)
            count += 1
            if not weight:
                stats = {name: value for name, value in stats.items() if name not in EXTRA_FIELDS}
            require({"update": step, "extension_update": step - 2000, **stats} == updates[step],
                    f"{arm} exact independent objective diagnostics at TD {step}")
            prior.verify_sentinel_network(agent, sentinels[step], batch)
            path = directory / f"training_state_{step}.pt"
            if path.exists():
                saved = torch.load(path, map_location="cpu", weights_only=True)
                for name, state in (("online", agent.learner.online.state_dict()),
                                    ("target", agent.learner.target.state_dict()),
                                    ("optimizer", agent.learner.optimizer.state_dict()),
                                    ("agent_rng_state", agent.rng.bit_generator.state),
                                    ("torch_rng_state", torch.get_rng_state())):
                    require_equal(state, saved[name], f"{arm} exact independent TD {step} {name}")
                require(saved["update_count"] == agent.learner.update_count == step
                        and saved["optimizer_summary"] == optimizer_state_summary(agent.learner),
                        "independent preserved TD and Adam clocks")
                compared.append(step)
    finally:
        env.close()
    return {"arm": arm, "return_anchor_weight": weight, "first_update": 2001, "last_update": end,
            "additional_optimizer_updates": count, "full_snapshot_updates_compared": compared,
            "sentinel_rows_compared": count + 1, "additional_environment_steps": 0,
            "additional_environment_resets": 0,
            "independent_math": "Separately reconstructed TD, equal-per-row margin and fixed-return Huber terms.",
            "logs_sentinels_and_full_snapshots_match_exactly": True}


def verify_treatment_phase(run_dir, phase, settings, source, source_row, batch, episodes):
    """Audit all saved treatment models and records without executing a rollout."""
    directory = run_dir / "treatment/td"
    require(read_json(directory / "summary.json") == phase, "matching treatment phase summary")
    fixed = {"directory": "treatment/td", "initial_td_update": 2000, "final_td_update": 10000,
             "optimizer_updates": 8000, "prior_updates_on_branch": 40000,
             "inherited_optimizer_updates": 42000, "total_updates_on_branch": 50000,
             "learning_rate": 0.0001, "target_synchronizations": 40,
             "target_source_updates": TARGET_SOURCES, "demonstration_weight": 1.0,
             "normalization_denominator": 117, "return_anchor_weight": 1.0, "all_sample_weights": 1.0,
             "training_snapshot_file": "treatment/td/training_state.pt"}
    for key, value in fixed.items():
        require(phase[key] == value, f"fixed treatment phase: {key}")
    rows = read_json(directory / "observations.json")
    require([row["update"] for row in rows] == CHECKPOINTS, "all 29 prescribed physical checkpoints")
    require(rows[0] == phase["initial_observation"] == source_row
            and rows[-1] == phase["final_observation"], "unchanged initial behavior and final observation identity")
    records = read_log(directory / "updates.jsonl")
    verify_loss_records(records)
    sentinel_rows = read_log(directory / "sentinel_q.jsonl")
    require(len(sentinel_rows) == phase["sentinel_records"] == 8001
            and phase["sentinel_environment_steps"] == 0, "all dense probes without physical interactions")
    for step, row in zip(range(2000, 10001), sentinel_rows, strict=True):
        local.verify_sentinel(row, batch, step)
    sentinels = {row["update"]: row for row in sentinel_rows}
    model_names = {f"update_{step}.pt" for step in CHECKPOINTS}
    model_names |= {f"training_state_{step}.pt" for step in CHECKPOINTS}
    model_names |= {f"target_source_state_{step}.pt" for step in TARGET_SOURCES}
    model_names.add("training_state.pt")
    require(set(phase["checkpoint_sha256"]) == model_names == {path.name for path in directory.glob("*.pt")},
            "all 67 prescribed treatment model artifacts")
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
        require_equal(payload["target"], target_source["online"], f"treatment TD {step} target-source lineage")
        require(payload["optimizer_summary"]["step_min"] == payload["optimizer_summary"]["step_max"] == 20000 + step,
                "preserved Adam internal steps")
        require_equal(payload["agent_rng_state"], source["agent_rng_state"], "preserved fixed-data agent RNG")
        require_equal(payload["torch_rng_state"], source["torch_rng_state"], "preserved fixed-data torch RNG")
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
        require(row["optimizer_state"] == payload["optimizer_summary"], "observed optimizer state matches snapshot")
        metadata = inference["metadata"]
        require(metadata["phase"] == "multiscene_retention_diagnostic" and metadata["objective"] == "extension/td"
                and metadata["updates"] == step and metadata["prior_updates_on_branch"] == 40000
                and metadata["total_updates_on_branch"] == 40000 + step and metadata["training_scene_only"]
                and metadata["scene_seeds"] == [10000, 10001, 10002]
                and metadata["fit"] == row["fit"] and metadata["aggregate"] == row["aggregate"],
                "metadata retains original clocks and unweighted fit diagnostics")
    final = torch.load(directory / "training_state.pt", map_location="cpu", weights_only=True)
    require_equal(final, snapshots[10000], "final full snapshot alias")
    require(file_hash(directory / "training_state.pt") == phase["training_snapshot_sha256"]
            and phase["initial_parameter_sha256"] == source["parameter_sha256"]
            and phase["final_parameter_sha256"] == final["parameter_sha256"], "initial/final model identity")
    require(phase["evaluation_environment_steps"] == sum(row["aggregate"]["evaluation_environment_steps"] for row in rows),
            "actual new treatment action count")
    prior.verify_retention(phase["retention"], rows, source_row)
    return rows, records, sentinels


def verify_accounting(accounting, rows):
    """Separate new treatment work, reused control and independent verification."""
    expected = {
        "optimizer_updates_total": 8000, "inherited_shared_optimizer_updates": 42000,
        "absolute_td_update": 10000, "total_updates_on_treatment_branch": 50000,
        "scheduled_evaluation_episodes": 87,
        "evaluation_environment_steps": sum(row["aggregate"]["evaluation_environment_steps"] for row in rows),
        "sentinel_records": 8001, "sentinel_forward_passes": 56007,
        "sentinel_environment_steps": 0, "online_training_steps": 0, "demonstration_collection_steps": 0,
        "held_out_scenes_accessed": 0,
        "reused_historical_control": {"optimizer_updates": 8000, "scheduled_evaluation_episodes": 87,
                                      "evaluation_environment_steps": 3711, "new_optimizer_updates": 0,
                                      "new_evaluation_episodes": 0, "new_environment_steps": 0},
    }
    for key, value in expected.items():
        require(accounting[key] == value, f"main versus reused historical accounting: {key}")


def verify(run_dir, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite independent verification: {output}")
    torch.set_num_threads(1)
    require(read_json(run_dir / "run_status.json")["status"] == "completed", "completed return-anchor treatment")
    settings, summary = read_json(run_dir / "settings.json"), read_json(run_dir / "summary.json")
    protocol = settings["training"]
    require(summary["protocol"] == protocol, "matching declared and summarized protocol")
    print("Auditing protocol, committed historical evidence and immutable inputs.", flush=True)
    verify_protocol(protocol, settings)
    manifest, reference, reference014, reference013, original = verify_inputs(run_dir, protocol)
    source = torch.load(reference / manifest["resumed_snapshot_file"], map_location="cpu", weights_only=True)
    source_row = read_json(reference / "baseline/td/observations.json")[0]
    batch, returns, episodes, return_checks = prior.load_verified_batch(run_dir, reference013, original, summary)
    require(summary["historical_control"] == read_json(run_dir / "baseline_reference.json"),
            "matching historical control reference in summary")
    print("Auditing all treatment snapshots, logs, probes and target lineage.", flush=True)
    rows, updates, sentinels = verify_treatment_phase(
        run_dir, summary["treatment"], settings, source, source_row, batch, episodes)
    require(read_json(run_dir / "treatment/restore_checks.json") == {
        "exact_full_snapshot": True, "target_synchronized_at_restore": False,
        "optimizer_reset_at_restore": False, "td_counter_reset_at_restore": False,
        "absolute_td_update": 2000, "optimizer_state": source["optimizer_summary"],
    }, "recorded complete restoration before treatment")
    require(read_json(run_dir / "treatment/initial_equivalence.json") == {
        "passed": True, "exact_previous_final_observation": True,
        "absolute_td_update": 2000, "new_updates_before_gate": 0,
    }, "initial behavior equality before the first new update")
    print("Auditing reused historical control without re-running its physical evaluations.", flush=True)
    historical_settings = read_json(reference / "settings.json")
    old_rows, old_updates, _, old_sentinels = local.verify_phase(
        reference, "baseline", read_json(reference / "baseline/td/summary.json"),
        historical_settings, source, source_row, batch, episodes)
    equality = local.verify_baseline(reference, reference014)
    require(summary["accounting"] == read_json(run_dir / "accounting.json"), "matching new-work accounting records")
    verify_accounting(summary["accounting"], rows)
    replays = [
        replay_arm(source, reference / "baseline/td", "historical_control", batch, returns,
                   {row["update"]: row for row in old_updates}, old_sentinels),
        replay_arm(source, run_dir / "treatment/td", "treatment", batch, returns,
                   {row["update"]: row for row in updates}, sentinels),
    ]
    by_step = {row["update"]: row for row in rows}
    reloads = []
    for step in RELOAD_CHECKPOINTS:
        print(f"Reloading treatment TD {step} on the three declared scenes.", flush=True)
        reloads.append(multi.reload_policy(run_dir, "treatment/td", step, by_step[step], batch, returns, episodes))
    result = {
        "passed": True, "attempt": 17, "verifier_sha256": file_hash(Path(__file__)),
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
        "additional_environment_resets": sum(len(row["scenes"]) for row in reloads),
        "additional_evaluation_episodes": 15,
        "additional_environment_steps": sum(scene["reload_environment_steps"] for row in reloads for scene in row["scenes"]),
        "historical_control_evaluations_rerun": False,
        "scope": "Each branch independently replays TD 2001-2201 only. Later snapshots, logs, probes, counters and target lineage are audited without replaying all 8000 updates per branch. Only treatment policies receive new reload episodes.",
        "loss_semantics": "All 117 supervised coefficients remain one. Treatment adds one mean fixed-return Huber term to unchanged TD and margin losses, without averaging the total or mixing targets.",
        "sentinel_interpretation": "Reference-state rankings are distinct from measured physical failures, including adjacent steps.",
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
        "passed", "additional_optimizer_updates", "additional_evaluation_episodes", "additional_environment_steps")}))


if __name__ == "__main__":
    main()
