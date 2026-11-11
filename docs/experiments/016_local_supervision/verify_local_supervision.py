"""Independently verify the one-row supervision-weight control and replay."""

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from me5418_harvesting.agents.optimizer_transition import optimizer_state_summary

ROOT = Path(__file__).resolve().parents[3]
PRIOR = ROOT / "docs/experiments/014_td_extension/verify_td_extension.py"
RUNNER = Path(__file__).with_name("run_local_supervision.py")
SPEC = importlib.util.spec_from_file_location("local016_prior_verifier", PRIOR)
prior = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prior)
multi, audit = prior.multi, prior.audit
read_json, read_log, file_hash = prior.read_json, prior.read_log, prior.file_hash
require, require_equal = prior.require, prior.require_equal
CHECKPOINTS = prior.CHECKPOINTS
TARGET_SOURCES = prior.TARGET_SOURCE_UPDATES
RELOAD_CHECKPOINTS = [2000, 2009, 2200, 2201, 10000]
SENTINEL_STEPS = [6, 18, 20, 21, 22, 23, 24]
SENTINEL_INDICES = [35 + step - 1 for step in SENTINEL_STEPS]
SELECTED_ROW = 55
ARMS = ("baseline", "treatment")
RESCUE_VERIFIER = ROOT / "docs/experiments/015_single_action_rescue/verify_single_action_rescue.py"
RESCUE_RUNNER = ROOT / "docs/experiments/015_single_action_rescue/run_single_action_rescue.py"
EXTENSION_RUNNER = ROOT / "docs/experiments/014_td_extension/run_td_extension.py"
SPEC_RESCUE = importlib.util.spec_from_file_location("local016_rescue_verifier", RESCUE_VERIFIER)
rescue_audit = importlib.util.module_from_spec(SPEC_RESCUE)
SPEC_RESCUE.loader.exec_module(rescue_audit)


def verify_protocol(protocol, settings):
    expected = {
        "attempt": 16, "kind": "single_transition_supervision_weight_control",
        "scene_seeds": [10000, 10001, 10002], "initial_td_update": 2000, "final_td_update": 10000,
        "extension_updates_per_arm": 8000, "td_checkpoints": CHECKPOINTS,
        "target_source_updates": TARGET_SOURCES, "inherited_shared_optimizer_updates": 42000,
        "learning_rate": 0.0001, "demonstration_weight": 1.0, "margin": 0.8,
        "target_update_interval": 200, "effective_batch_size": 117,
        "selected_scene_seed": 10001, "selected_reference_step": 21,
        "selected_transition_index": 55, "selected_demonstrated_action": 0,
        "baseline_selected_sample_weight": 1.0, "treatment_selected_sample_weight": 2.0,
        "other_sample_weights": 1.0, "normalization_denominator": 117,
        "normalization_rule": "sum of weighted per-row hinges divided by unchanged batch size 117",
        "td_sample_weights_changed": False, "demonstrations_duplicated": False,
        "preserve_adam_moments_and_parameter_steps": True, "preserve_absolute_td_counter": True,
        "synchronize_target_at_resume": False, "early_stopping": False,
        "baseline_must_exactly_reproduce_attempt_014_before_treatment": True,
        "sentinel_seed": 10001, "sentinel_reference_steps": SENTINEL_STEPS,
        "sentinel_batch_size": 1, "sentinel_updates": "every absolute update from 2000 through 10000 inclusive",
        "sentinel_gap": "Q(demonstrated action) - max Q(other actions)",
        "sentinel_is_physical_evaluation": False, "online_training_steps": 0, "held_out_scenes_accessed": 0,
        "update_demonstration_loss_semantics": "Weighted mean hinge with fixed denominator; unweighted and selected-row terms are in supervision.jsonl.",
    }
    for key, value in expected.items():
        require(protocol[key] == value, f"fixed single-row-control protocol: {key}")
    dqn = settings["dqn"]
    require(dqn["learning_rate"] == 0.0001 and dqn["gamma"] == 0.99
            and dqn["target_update_interval"] == 200 and dqn["gradient_clip_norm"] == 10
            and dqn["hidden_sizes"] == [256, 256], "unchanged network and optimizer configuration")
    require(file_hash(RUNNER) == protocol["script_sha256"], "local-weight runner source hash")


def verify_inputs(run_dir, protocol):
    """Read-only source checks; return the 014, 013 and original 011 run paths."""
    manifest = read_json(run_dir / "input_manifest.json")
    reference = Path(manifest["reference_run"])
    diagnosis = manifest["attempt_015"]
    archive_path = ROOT / diagnosis["archive_manifest"]
    require(archive_path == ROOT / "docs/experiments/015_single_action_rescue/evidence/archive_manifest.json",
            "fixed committed motivating diagnosis")
    committed = subprocess.check_output(
        ["git", "show", f"HEAD:{diagnosis['archive_manifest']}"], cwd=ROOT)
    archive = read_json(archive_path)
    require(json.loads(committed) == archive and archive["status"] == "completed"
            and hashlib.sha256(committed).hexdigest() == diagnosis["committed_archive_manifest_sha256"]
            and file_hash(archive_path) == diagnosis["archive_manifest_sha256"]
            and diagnosis["committed_archive_verified"], "unchanged committed diagnosis archive")
    diagnosis_run = Path(diagnosis["reference_run"])
    require(diagnosis_run == Path(archive["source_run"]), "exact diagnosis source path")
    require(diagnosis["reference_files_sha256"] == {name: record["sha256"] for name, record in archive["files"].items()},
            "complete diagnosis evidence map")
    for name, record in archive["files"].items():
        require(file_hash(archive_path.parent / name) == file_hash(diagnosis_run / name) == record["sha256"],
                f"archived and raw diagnosis: {name}")
    diagnosis_settings = read_json(diagnosis_run / "settings.json")
    rescue_audit.verify_protocol(diagnosis_settings["training"])
    anchored014, _, _, _, _ = rescue_audit.verify_inputs(diagnosis_run, diagnosis_settings)
    require(manifest["attempt_014"] == anchored014 and Path(anchored014["reference_run"]) == reference,
            "identical anchored attempt-014 evidence and provenance")
    diagnosis_summary = read_json(diagnosis_run / "summary.json")
    require(diagnosis_summary["tested_checkpoint_count"] == diagnosis_summary["rescued_checkpoint_count"] == 2
            and diagnosis_summary["td_updates"] == [2009, 10000]
            and read_json(diagnosis_run / "verification.json")["passed"], "verified two fixed step-21 rescues")
    settings014 = read_json(reference / "settings.json")
    prior.verify_protocol(settings014["training"], settings014)
    inherited, reference013, original = prior.verify_inputs(reference, settings014["training"])
    require(manifest["attempt_013"] == inherited, "exact original TD-2000 inheritance manifest")
    require(manifest["resumed_snapshot_file"] == "td/training_state_2000.pt"
            and manifest["resumed_snapshot_sha256"] == file_hash(reference / manifest["resumed_snapshot_file"])
            == read_json(reference / "td/summary.json")["checkpoint_sha256"]["training_state_2000.pt"],
            "fixed resume full snapshot")
    source = torch.load(reference / manifest["resumed_snapshot_file"], map_location="cpu", weights_only=True)
    inherited_source = torch.load(reference013 / inherited["resumed_snapshot_file"], map_location="cpu", weights_only=True)
    require_equal(source, inherited_source, "both branches inherit the exact original full TD-2000 state")
    require(source["update_count"] == 2000 and source["optimizer_summary"]["step_min"]
            == source["optimizer_summary"]["step_max"] == 22000, "unchanged TD and Adam resume clocks")
    require(manifest["inherited_optimizer_updates"] == 42000 and manifest["inherited_td_updates"] == 2000,
            "shared inherited work counted once")
    dataset = read_json(original / "dataset.json")
    require(Path(manifest["original_data_run"]) == original
            and manifest["dataset_sha256"] == dataset["sha256"]
            == file_hash(original / "demonstration.npz"), "original ordered demonstration array")
    with np.load(original / "demonstration.npz", allow_pickle=False) as data:
        selected = data["observations"][55].copy()
        require(int(data["actions"][55]) == 0 and len(data["actions"]) == 117, "selected label and fixed batch size")
    expected_selection = {"scene_seed": 10001, "reference_step": 21, "global_transition_index": 55,
                          "demonstrated_action": 0, "encoded_observation_sha256": hashlib.sha256(selected.tobytes()).hexdigest()}
    require(manifest["selected_transition"] == expected_selection, "exact selected transition identity")
    require(diagnosis["selected_observation_sha256"] == expected_selection["encoded_observation_sha256"],
            "reweighted state is the exact diagnosed fork state")
    for update in (2009, 10000):
        for mode in ("greedy", "counterfactual"):
            fork = read_json(diagnosis_run / f"update_{update}_{mode}.json")["records"][20]
            require(fork["step"] == 21 and fork["encoded_observation_sha256"]
                    == expected_selection["encoded_observation_sha256"], "all four diagnosed fork states match the selected row")
    weights = read_json(run_dir / "supervision_weights.json")
    verify_weight_table(weights, expected_selection)
    expected_helpers = {**anchored014["helper_sha256"],
                        EXTENSION_RUNNER.relative_to(ROOT).as_posix(): file_hash(EXTENSION_RUNNER),
                        RESCUE_RUNNER.relative_to(ROOT).as_posix(): file_hash(RESCUE_RUNNER)}
    require(protocol["helper_sha256"] == manifest["helper_sha256"] == expected_helpers, "complete helper map")
    provenance = read_json(run_dir / "provenance.json")
    require(provenance["source_sha256"] == manifest["production_source_sha256"]
            == anchored014["source_production_sha256"] and len(provenance["source_sha256"]) == 52,
            "unchanged 52 production source files")
    for name, digest in {**manifest["production_source_sha256"], **expected_helpers}.items():
        require(file_hash(ROOT / name) == digest, f"immutable source/helper: {name}")
    launch = subprocess.check_output(
        ["git", "show", f"{provenance['git_head']}:{RUNNER.relative_to(ROOT).as_posix()}"], cwd=ROOT)
    require(launch.replace(b"\r\n", b"\n") == RUNNER.read_bytes().replace(b"\r\n", b"\n"),
            "local-weight code committed before launch")
    settings = read_json(run_dir / "settings.json")
    require(settings["environment"] == source["environment_config"]
            and settings["dqn"] == json.loads(json.dumps(source["dqn_config"])),
            "resolved configuration equals inherited full snapshot")
    return manifest, reference, reference013, original


def verify_weight_table(weights, selected):
    require(weights == {
        "selected_transition": selected, "normalization_denominator": 117,
        "arms": {"baseline": [1.0] * 117, "treatment": [2.0 if index == 55 else 1.0 for index in range(117)]},
        "td_weights_changed": False,
    }, "only one supervised coefficient differs; TD distribution and denominator remain unchanged")


def independent_hinges(values, actions):
    """Build each demonstrated-action hinge independently of the experiment code."""
    require(values.ndim == 2 and actions.shape == (len(values),) and len(values) > 0,
            "one demonstrated action for every nonempty Q row")
    margins = torch.full_like(values, 0.8)
    margins.scatter_(1, actions.unsqueeze(1), 0.0)
    selected = values.gather(1, actions.unsqueeze(1)).squeeze(1)
    return torch.max(values + margins, dim=1).values - selected


def independent_supervision(values, actions, sample_weight, selected_row=SELECTED_ROW):
    """Keep all unaffected row coefficients at exactly 1/N, including TD data."""
    require(sample_weight in (1.0, 2.0) and not isinstance(sample_weight, bool),
            "predeclared local coefficient must be one or two")
    require(0 <= selected_row < len(actions), "selected demonstration index is in range")
    hinges = independent_hinges(values, actions)
    weights = torch.ones_like(hinges)
    weights[selected_row] = sample_weight
    return (hinges * weights).mean(), hinges


def independent_update(learner, batch, sample_weight, selected_row=SELECTED_ROW):
    """Reconstruct weighted TD+margin math without invoking the new updater."""
    learner.online.train()
    learner.optimizer.zero_grad(set_to_none=True)
    before_parameters = torch.nn.utils.parameters_to_vector(learner.online.parameters()).detach().clone()
    chosen = learner.online(batch.observations).gather(1, batch.actions[:, None]).squeeze(1)
    with torch.no_grad():
        following = learner.target(batch.next_observations).max(dim=1).values
        following = torch.where(batch.terminated, torch.zeros_like(following), following)
        targets = batch.rewards + learner.config.gamma * following
    td_before = F.smooth_l1_loss(chosen, targets)
    values = learner.online(batch.observations)
    margin_before, hinges_before = independent_supervision(values, batch.actions, sample_weight, selected_row)
    accuracy = float((values.detach().argmax(dim=1) == batch.actions).float().mean())
    total_before = td_before + margin_before
    require(bool(torch.isfinite(total_before)), "finite independently reconstructed objective")
    total_before.backward()
    norm = torch.nn.utils.clip_grad_norm_(
        learner.online.parameters(), learner.config.gradient_clip_norm, error_if_nonfinite=True)
    learner.optimizer.step()
    with torch.no_grad():
        after_values = learner.online(batch.observations)
        after_chosen = after_values.gather(1, batch.actions[:, None]).squeeze(1)
        td_after = F.smooth_l1_loss(after_chosen, targets)
        margin_after, hinges_after = independent_supervision(after_values, batch.actions, sample_weight, selected_row)
        total_after = td_after + margin_after
        after_parameters = torch.nn.utils.parameters_to_vector(learner.online.parameters())
        displacement = torch.linalg.vector_norm(after_parameters - before_parameters)
        require(bool(torch.isfinite(after_parameters).all()) and bool(torch.isfinite(total_after)),
                "finite independently updated parameters and loss")
    learner.update_count += 1
    synchronized = learner.update_count % learner.config.target_update_interval == 0
    if synchronized:
        learner.target.load_state_dict(learner.online.state_dict())
    stats = {
        "loss_before": total_before.item(), "loss_after": total_after.item(),
        "gradient_norm": norm.item(), "parameter_delta_norm": displacement.item(),
        "target_synchronized": synchronized,
        "td_loss_before": td_before.item(), "td_loss_after": td_after.item(),
        "demonstration_loss_before": margin_before.item(), "demonstration_loss_after": margin_after.item(),
        "demonstration_action_accuracy": accuracy,
    }
    supervision = {
        "unweighted_margin_before": hinges_before.detach().mean().item(),
        "unweighted_margin_after": hinges_after.mean().item(),
        "weighted_margin_before": margin_before.detach().item(),
        "weighted_margin_after": margin_after.item(),
        "selected_hinge_before": hinges_before[selected_row].detach().item(),
        "selected_hinge_after": hinges_after[selected_row].item(),
        "selected_sample_weight": float(sample_weight),
        "normalization_denominator": len(batch.actions),
    }
    return stats, supervision


def verify_sentinel(row, batch, update):
    require(row["update"] == update and row["extension_update"] == update - 2000
            and row["scene_seed"] == 10001, "sentinel absolute identity")
    require(len(row["states"]) == 7, "seven predeclared reference states")
    for state, step, index in zip(row["states"], SENTINEL_STEPS, SENTINEL_INDICES, strict=True):
        action = int(batch.actions[index])
        require(state["reference_step"] == step and state["global_transition_index"] == index
                and state["demonstrated_action"] == action, "sentinel demonstrated row identity")
        scores = np.asarray(state["q_values"], dtype=np.float64)
        require(scores.shape == (8,) and np.isfinite(scores).all(), "finite sentinel Q vector")
        require(state["greedy_action"] == int(scores.argmax()), "sentinel argmax")
        require(state["signed_demonstration_gap"] == float(scores[action] - np.delete(scores, action).max()),
                "signed reference-state Q gap")


def verify_supervision(records, updates, sample_weight):
    """Check weighted/unweighted semantics while allowing float32 reduction rounding."""
    require(len(records) == len(updates), "complete supplemental supervision log")
    for row, update in zip(records, updates, strict=True):
        require(row["update"] == update["update"] and row["extension_update"] == update["extension_update"],
                "aligned supervision clock")
        require(row["selected_sample_weight"] == sample_weight and row["normalization_denominator"] == 117,
                "only the selected row coefficient changes; divisor stays 117")
        for timing in ("before", "after"):
            unweighted, weighted, hinge = (row[f"{name}_{timing}"] for name in
                                           ("unweighted_margin", "weighted_margin", "selected_hinge"))
            require(all(math.isfinite(value) and value >= 0 for value in (unweighted, weighted, hinge)),
                    "finite nonnegative margin diagnostics")
            require(abs(weighted - (unweighted + (sample_weight - 1) * hinge / 117)) <= 2e-7,
                    "weighted objective has exactly one extra row contribution")
            require(weighted == update[f"demonstration_loss_{timing}"], "recorded weighted supervision component")
            if sample_weight == 1:
                require(weighted == unweighted, "baseline supervision remains the original mean")


def verify_baseline(run_dir, reference):
    """Require the entire new control arm to reproduce the unchanged attempt 014."""
    directory = run_dir / "baseline/td"
    original = reference / "td"
    require(read_log(directory / "updates.jsonl") == read_log(original / "updates.jsonl"),
            "all 8000 baseline update metrics exactly reproduce attempt 014")
    require(read_json(directory / "observations.json") == read_json(original / "observations.json"),
            "all 29 baseline observations exactly reproduce attempt 014")
    legacy_probes = read_log(original / "sentinel_q.jsonl")
    expanded = read_log(directory / "sentinel_q.jsonl")
    require(len(expanded) == len(legacy_probes) == 8001, "complete baseline reference probes")
    for old, new in zip(legacy_probes, expanded, strict=True):
        reduced = {**new, "states": [state for state in new["states"] if state["reference_step"] in (6, 18, 22, 24)]}
        require(old == reduced, "expanded baseline probes preserve every old Q vector exactly")
    files = read_json(original / "summary.json")["checkpoint_sha256"]
    for name in files:
        require_equal(torch.load(directory / name, map_location="cpu", weights_only=True),
                      torch.load(original / name, map_location="cpu", weights_only=True),
                      f"baseline complete model payload equals attempt 014: {name}")
    return {"passed": True, "exact_updates": 8000, "exact_observations": 29,
            "exact_legacy_sentinel_rows": 8001, "exact_model_payloads": len(files)}


def verify_phase(run_dir, arm, phase, settings, source, source_row, batch, episodes):
    """Audit complete arm artifacts; this function never executes a rollout."""
    directory = run_dir / arm / "td"
    weight = 1.0 if arm == "baseline" else 2.0
    require(read_json(directory / "summary.json") == phase, "matching arm phase summary")
    require(phase["initial_td_update"] == 2000 and phase["final_td_update"] == 10000
            and phase["optimizer_updates"] == 8000 and phase["prior_updates_on_branch"] == 40000
            and phase["inherited_optimizer_updates"] == 42000 and phase["total_updates_on_branch"] == 50000
            and phase["learning_rate"] == 0.0001 and phase["target_synchronizations"] == 40
            and phase["target_source_updates"] == TARGET_SOURCES, "arm clocks, budgets and controls")
    require(phase["directory"] == f"{arm}/td" and phase["demonstration_weight"] == 1.0
            and phase["selected_sample_weight"] == weight and phase["selected_transition_index"] == 55
            and phase["normalization_denominator"] == 117 and phase["supervision_records"] == 8000
            and phase["training_snapshot_file"] == f"{arm}/td/training_state.pt",
            "arm-specific local weight and fixed supervision denominator")
    rows = read_json(directory / "observations.json")
    require([row["update"] for row in rows] == CHECKPOINTS, "complete fixed observation schedule")
    require(rows[0] == phase["initial_observation"] == source_row
            and rows[-1] == phase["final_observation"], "unchanged initial policy and correct final observation")
    updates = read_log(directory / "updates.jsonl")
    require(prior.verify_update_log(updates) == 40, "forty absolute-clock target synchronizations")
    supplemental = read_log(directory / "supervision.jsonl")
    verify_supervision(supplemental, updates, weight)
    sentinel_rows = read_log(directory / "sentinel_q.jsonl")
    require(len(sentinel_rows) == phase["sentinel_records"] == 8001
            and phase["sentinel_environment_steps"] == 0, "complete seven-state probes without environment actions")
    for step, row in zip(range(2000, 10001), sentinel_rows, strict=True):
        verify_sentinel(row, batch, step)
    sentinels = {row["update"]: row for row in sentinel_rows}
    model_names = {f"update_{step}.pt" for step in CHECKPOINTS}
    model_names |= {f"training_state_{step}.pt" for step in CHECKPOINTS}
    model_names |= {f"target_source_state_{step}.pt" for step in TARGET_SOURCES}
    model_names.add("training_state.pt")
    require(set(phase["checkpoint_sha256"]) == model_names == {path.name for path in directory.glob("*.pt")},
            "exact 67 declared model artifacts per arm")
    for name, digest in phase["checkpoint_sha256"].items():
        require(file_hash(directory / name) == digest, f"{arm} model hash: {name}")
    snapshots = {}
    for step in sorted(set(CHECKPOINTS) | set(TARGET_SOURCES)):
        stem = "training_state" if step in CHECKPOINTS else "target_source_state"
        snapshots[step] = torch.load(directory / f"{stem}_{step}.pt", map_location="cpu", weights_only=True)
    prior.verify_resume_payload(snapshots[2000], source)
    for step, payload in snapshots.items():
        source_target = snapshots[step - step % 200]
        stem = "training_state" if step in CHECKPOINTS else "target_source_state"
        multi.verify_snapshot(directory / f"{stem}_{step}.pt", audit.state_hash(payload["online"]), settings,
                              step, audit.state_hash(source_target["online"]), payload["optimizer_summary"])
        require_equal(payload["target"], source_target["online"], f"{arm} TD {step} recorded target lineage")
        require(payload["optimizer_summary"]["step_min"] == payload["optimizer_summary"]["step_max"] == 20000 + step,
                "preserved Adam parameter step")
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
        require(row["total_updates_on_branch"] == 40000 + step, "observation cumulative branch count")
        payload = snapshots[step]
        inference = torch.load(directory / f"update_{step}.pt", map_location="cpu", weights_only=True)
        require_equal(inference["online"], payload["online"], "inference/full online equality")
        require(json.loads(json.dumps(inference["dqn_config"])) == settings["dqn"]
                and inference["environment_config"] == settings["environment"], "unchanged inference configuration")
        require(row["optimizer_state"] == payload["optimizer_summary"], "observed optimizer state")
        metadata = inference["metadata"]
        require(metadata["phase"] == "multiscene_retention_diagnostic" and metadata["objective"] == "extension/td"
                and metadata["updates"] == step and metadata["prior_updates_on_branch"] == 40000
                and metadata["total_updates_on_branch"] == 40000 + step and metadata["training_scene_only"]
                and metadata["scene_seeds"] == [10000, 10001, 10002]
                and metadata["fit"] == row["fit"] and metadata["aggregate"] == row["aggregate"],
                "checkpoint metadata keeps unweighted fit separate from treatment objective")
    final = torch.load(directory / "training_state.pt", map_location="cpu", weights_only=True)
    require_equal(final, snapshots[10000], "final full snapshot alias")
    require(file_hash(directory / "training_state.pt") == phase["training_snapshot_sha256"]
            and phase["initial_parameter_sha256"] == source["parameter_sha256"]
            and phase["final_parameter_sha256"] == final["parameter_sha256"], "source/final parameter identity")
    require(phase["evaluation_environment_steps"] == sum(row["aggregate"]["evaluation_environment_steps"] for row in rows),
            "actual main-arm physical action count")
    prior.verify_retention(phase["retention"], rows, source_row)
    return rows, updates, supplemental, sentinels


def replay_arm(source, run_dir, arm, batch, updates, supplemental, sentinels, end=2201):
    """Numerically replay only the predeclared first 201 updates of this arm."""
    agent, env = prior.independent_restore(source)
    directory = run_dir / arm / "td"
    count, compared = 0, []
    weight = 1.0 if arm == "baseline" else 2.0
    try:
        require(agent.learner.update_count == 2000, "independent replay begins from the inherited clock")
        prior.verify_sentinel_network(agent, sentinels[2000], batch)
        for step in range(2001, end + 1):
            stats, details = independent_update(agent.learner, batch, weight)
            count += 1
            clock = {"update": step, "extension_update": step - 2000}
            require({**clock, **stats} == updates[step], f"{arm} exact independent update diagnostics at TD {step}")
            require({**clock, **details} == supplemental[step],
                    f"{arm} exact independently reconstructed supervision diagnostics at TD {step}")
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
    return {"arm": arm, "first_update": 2001, "last_update": end, "additional_optimizer_updates": count,
            "full_snapshot_updates_compared": compared, "sentinel_rows_compared": count + 1,
            "additional_environment_steps": 0, "additional_environment_resets": 0,
            "independent_math": "Separately implemented per-row margin weighting; original TD targets, divisor and optimizer.",
            "logs_supplemental_sentinels_and_snapshots_match_exactly": True}


def verify_accounting(accounting, observations):
    expected_arms = {}
    for arm in ARMS:
        rows = observations[arm]
        expected_arms[arm] = {
            "optimizer_updates": 8000, "inherited_optimizer_updates": 42000, "absolute_td_update": 10000,
            "total_updates_on_branch": 50000, "scheduled_evaluation_episodes": 87,
            "evaluation_environment_steps": sum(row["aggregate"]["evaluation_environment_steps"] for row in rows),
            "sentinel_records": 8001, "sentinel_forward_passes": 8001 * 7,
        }
    require(accounting["arms"] == expected_arms, "separate complete arm budgets")
    expected = {
        "optimizer_updates_total": 16000, "inherited_shared_optimizer_updates": 42000,
        "scheduled_evaluation_episodes": 174,
        "evaluation_environment_steps": sum(arm["evaluation_environment_steps"] for arm in expected_arms.values()),
        "sentinel_records": 16002, "sentinel_forward_passes": 16002 * 7,
        "sentinel_environment_steps": 0, "online_training_steps": 0,
        "demonstration_collection_steps": 0, "held_out_scenes_accessed": 0,
    }
    for key, value in expected.items():
        require(accounting[key] == value, f"main experiment accounting: {key}")


def verify(run_dir, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite independent verification: {output}")
    torch.set_num_threads(1)
    require(read_json(run_dir / "run_status.json")["status"] == "completed", "completed two-arm comparison")
    settings, summary = read_json(run_dir / "settings.json"), read_json(run_dir / "summary.json")
    protocol = settings["training"]
    require(summary["protocol"] == protocol, "matching declared and summarized protocol")
    verify_protocol(protocol, settings)
    manifest, reference, reference013, original = verify_inputs(run_dir, protocol)
    source = torch.load(reference / manifest["resumed_snapshot_file"], map_location="cpu", weights_only=True)
    source_row = read_json(reference / "td/observations.json")[0]
    batch, returns, episodes, return_checks = prior.load_verified_batch(run_dir, reference013, original, summary)
    observations, arm_records = {}, {}
    require(set(summary["arms"]) == set(ARMS), "exact baseline and treatment arm set")
    for arm in ARMS:
        rows, updates, supplemental, sentinels = verify_phase(
            run_dir, arm, summary["arms"][arm], settings, source, source_row, batch, episodes)
        require(read_json(run_dir / arm / "restore_checks.json") == {
            "exact_full_snapshot": True, "target_synchronized_at_restore": False,
            "optimizer_reset_at_restore": False, "td_counter_reset_at_restore": False,
            "absolute_td_update": 2000, "optimizer_state": source["optimizer_summary"],
        }, "recorded full restoration before each arm")
        require(read_json(run_dir / arm / "initial_equivalence.json") == {
            "passed": True, "exact_previous_final_observation": True,
            "absolute_td_update": 2000, "new_updates_before_gate": 0,
        }, "recorded initial behavior equality before each arm")
        observations[arm] = rows
        arm_records[arm] = ({row["update"]: row for row in updates},
                           {row["update"]: row for row in supplemental}, sentinels)
    require_equal(torch.load(run_dir / "baseline/td/training_state_2000.pt", map_location="cpu", weights_only=True),
                  torch.load(run_dir / "treatment/td/training_state_2000.pt", map_location="cpu", weights_only=True),
                  "identical complete branch starting states")
    equality = verify_baseline(run_dir, reference)
    expected_gate = {"passed": True, "exact_update_records": 8000, "exact_observations": 29,
                     "exact_checkpoint_payloads": 67, "exact_inherited_sentinel_rows": 8001,
                     "treatment_started_before_gate": False}
    require(read_json(run_dir / "baseline_equivalence.json") == summary["baseline_equivalence"] == expected_gate,
            "baseline reproduction gate recorded before treatment")
    accounting = read_json(run_dir / "accounting.json")
    require(summary["accounting"] == accounting, "matching main accounting records")
    verify_accounting(accounting, observations)
    replays, reloads = [], []
    for arm in ARMS:
        updates, supplemental, sentinels = arm_records[arm]
        replays.append(replay_arm(source, run_dir, arm, batch, updates, supplemental, sentinels))
        by_step = {row["update"]: row for row in observations[arm]}
        for step in RELOAD_CHECKPOINTS:
            reloads.append(multi.reload_policy(run_dir, f"{arm}/td", step, by_step[step], batch, returns, episodes))
    result = {
        "passed": True, "attempt": 16, "verifier_sha256": file_hash(Path(__file__)),
        "prior_verifier_sha256": file_hash(PRIOR), "input_manifest_sha256": file_hash(run_dir / "input_manifest.json"),
        "reference_run": str(reference), "reference_013_run": str(reference013), "original_data_run": str(original),
        "source_snapshot_sha256": manifest["resumed_snapshot_sha256"],
        "independent_return_checks": return_checks, "baseline_equivalence": equality,
        "verified_raw_updates": 16000, "verified_supervision_records": 16000,
        "verified_sentinel_rows": 16002, "verified_physical_checkpoints": 58,
        "verified_full_and_inference_snapshot_files": 134,
        "independent_replays": replays, "policy_reloads": reloads,
        "additional_optimizer_updates": sum(record["additional_optimizer_updates"] for record in replays),
        "additional_environment_resets": sum(len(row["scenes"]) for row in reloads),
        "additional_evaluation_episodes": 30,
        "additional_environment_steps": sum(scene["reload_environment_steps"] for row in reloads for scene in row["scenes"]),
        "scope": "Each arm independently replays TD 2001-2201 only. Later snapshots, logs, probes, counters and target lineage are audited without retraining all 16000 updates.",
        "loss_semantics": "Only row55 supervised hinge is doubled in treatment, with fixed denominator117; TD targets and coefficients remain unchanged.",
        "sentinel_interpretation": "Reference-state rankings are distinct from measured physical failures, including all adjacent steps.",
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
