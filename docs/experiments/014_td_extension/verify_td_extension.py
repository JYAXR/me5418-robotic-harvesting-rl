"""Independently audit the fixed-data continuation from TD 2000 to TD 10000."""

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from me5418_harvesting.agents.dqn_learning import TransitionBatch
from me5418_harvesting.agents.optimizer_transition import optimizer_state_summary

ROOT = Path(__file__).resolve().parents[3]
PRIOR_VERIFIER = ROOT / "docs/experiments/013_td_learning_rate/verify_learning_rate_control.py"
SPEC = importlib.util.spec_from_file_location("extension_prior_verifier", PRIOR_VERIFIER)
prior = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prior)
base, multi, audit = prior.base, prior.base.multi, prior.base.audit
read_json, read_log, file_hash = audit.read_json, audit.read_log, audit.file_hash
require, require_equal = audit.require, audit.require_equal

CHECKPOINTS = [2000, *range(2001, 2011), 2200, 2201, *range(2500, 10001, 500)]
RELOAD_CHECKPOINTS = [2000, 2001, 2200, 2201, 10000]
SENTINEL_STEPS = [6, 18, 22, 24]
SENTINEL_INDICES = [40, 52, 56, 58]
RUNNER = Path(__file__).with_name("run_td_extension.py")
TARGET_SOURCE_UPDATES = list(range(2400, 9401, 1000))


def verify_protocol(protocol, settings):
    expected = {
        "attempt": 14, "kind": "exact_low_rate_td_continuation", "scene_seeds": [10000, 10001, 10002],
        "initial_td_update": 2000, "final_td_update": 10000, "extension_updates": 8000,
        "td_checkpoints": CHECKPOINTS, "target_source_updates": TARGET_SOURCE_UPDATES,
        "inherited_optimizer_updates": 42000, "learning_rate": 0.0001,
        "demonstration_weight": 1.0, "margin": 0.8, "target_update_interval": 200,
        "effective_batch_size": 117, "full_batch_weighting": "equal_per_transition",
        "early_stopping": False, "resume_source": "attempt_013_treatment_final_training_state",
        "preserve_adam_moments_and_parameter_steps": True, "preserve_absolute_td_counter": True,
        "synchronize_target_at_resume": False,
        "initial_gate": "exact_full_training_state_and_previous_final_observation",
        "sentinel_seed": 10001, "sentinel_reference_steps": SENTINEL_STEPS, "sentinel_batch_size": 1,
        "sentinel_updates": "every absolute update from 2000 through 10000 inclusive",
        "sentinel_gap": "Q(demonstrated action) - max Q(other actions)",
        "sentinel_timing": "after the optimizer update and any scheduled target synchronization",
        "sentinel_is_physical_evaluation": False, "online_training_steps": 0, "held_out_scenes_accessed": 0,
    }
    for key, value in expected.items():
        require(protocol[key] == value, f"predeclared continuation protocol: {key}")
    config = settings["dqn"]
    require(config["learning_rate"] == 0.0001 and config["gamma"] == 0.99
            and config["target_update_interval"] == 200 and config["gradient_clip_norm"] == 10
            and config["hidden_sizes"] == [256, 256], "fixed architecture and optimizer configuration")
    require(file_hash(RUNNER) == protocol["script_sha256"], "continuation runner source hash")


def verify_inputs(run_dir, protocol):
    """Read-only verification of committed 013 evidence and its complete 011 chain."""
    manifest = read_json(run_dir / "input_manifest.json")
    reference = Path(manifest["reference_run"])
    inherited = manifest["attempt_013"]
    archive_path = ROOT / inherited["archive_manifest"]
    verify_archive(reference, archive_path, inherited)
    require(read_json(reference / "run_status.json")["status"] == "completed"
            and inherited["committed_archive_verified"], "completed, committed attempt-013 source")
    previous_settings = read_json(reference / "settings.json")
    prior.verify_protocol(previous_settings["training"])
    prior.verify_prior_source_anchor(previous_settings["training"])
    old_manifest, original, old_provenance = base.verify_inputs(reference, previous_settings["training"])
    for key in ("reference_files_sha256", "reference_source_sha256", "config_sha256", "helper_sha256"):
        require(manifest["attempt_011"][key] == old_manifest[key], f"unbroken attempt-011 chain: {key}")
    require(Path(manifest["attempt_011"]["reference_run"]) == original, "unchanged attempt-011 source directory")
    for name, digest in inherited["reference_files_sha256"].items():
        require(file_hash(reference / name) == digest, f"attempt-013 reference hash: {name}")
    for group in ("reference_source_sha256", "helper_sha256", "launch_script_sha256"):
        for name, digest in inherited[group].items():
            require(file_hash(ROOT / name) == digest, f"attempt-013 {group}: {name}")
    require(inherited["reference_source_sha256"] == old_provenance["source_sha256"]
            and inherited["launch_git_head"] == old_provenance["git_head"], "attempt-013 launch provenance")
    for name, digest in inherited["committed_launch_blob_sha256"].items():
        blob = subprocess.check_output(["git", "show", f"{inherited['launch_git_head']}:{name}"], cwd=ROOT)
        require(hashlib.sha256(blob).hexdigest() == digest
                and blob.replace(b"\r\n", b"\n") == (ROOT / name).read_bytes().replace(b"\r\n", b"\n"),
                f"unchanged committed attempt-013 launch source: {name}")
    provenance = read_json(run_dir / "provenance.json")
    require(provenance["source_sha256"] == old_provenance["source_sha256"]
            and len(provenance["source_sha256"]) == 52, "unchanged 52 production source files")
    require(protocol["helper_sha256"] == {**inherited["helper_sha256"], **inherited["launch_script_sha256"]},
            "complete immutable continuation helper map")
    for name, digest in protocol["helper_sha256"].items():
        require(file_hash(ROOT / name) == digest, f"continuation helper hash: {name}")
    launch_blob = subprocess.check_output(
        ["git", "show", f"{provenance['git_head']}:{RUNNER.relative_to(ROOT).as_posix()}"], cwd=ROOT,
    )
    require(launch_blob.replace(b"\r\n", b"\n") == RUNNER.read_bytes().replace(b"\r\n", b"\n"),
            "runner was committed before experiment launch")
    phase = read_json(reference / "treatment/td/summary.json")
    require(read_json(reference / "summary.json")["arms"]["treatment"] == phase,
            "source treatment phase summaries agree")
    for name, digest in phase["checkpoint_sha256"].items():
        require(file_hash(reference / "treatment/td" / name) == digest, f"source treatment model: {name}")
    require(manifest["resumed_snapshot_file"] == phase["training_snapshot_file"] == "treatment/td/training_state.pt"
            and manifest["resumed_snapshot_sha256"] == phase["training_snapshot_sha256"]
            and file_hash(reference / manifest["resumed_snapshot_file"]) == manifest["resumed_snapshot_sha256"],
            "exact source training snapshot hash")
    require(manifest["inherited_td_updates"] == 2000 and manifest["inherited_optimizer_updates"] == 42000
            and manifest["same_frozen_117_transitions"] and manifest["episode_returns_recomputed_exactly"],
            "fixed inherited work and data semantics")
    return manifest, reference, original


def independent_restore(payload):
    """Restore every saved state without invoking the continuation runner."""
    agent, env = audit.independent_restore(payload)
    agent.rng.bit_generator.state = deepcopy(payload["agent_rng_state"])
    torch.set_rng_state(payload["torch_rng_state"].clone())
    return agent, env


def verify_resume_payload(actual, source):
    """An exact continuation has no boundary intervention, including target resync."""
    require_equal(actual, source, "exact full-state continuation boundary")
    agent, env = independent_restore(actual)
    try:
        require_equal(agent.learner.online.state_dict(), source["online"], "restored online network")
        require_equal(agent.learner.target.state_dict(), source["target"], "restored target network")
        require_equal(agent.learner.optimizer.state_dict(), source["optimizer"], "restored Adam state")
        require(agent.learner.update_count == source["update_count"], "preserved absolute TD clock")
        require_equal(agent.rng.bit_generator.state, source["agent_rng_state"], "restored agent RNG")
        require_equal(torch.get_rng_state(), source["torch_rng_state"], "restored torch RNG")
        require(optimizer_state_summary(agent.learner) == source["optimizer_summary"],
                "exact independently restored optimizer summary")
    finally:
        env.close()
    return {"full_snapshot_matches_exactly": True, "target_resynchronized": False,
            "counter_reset": False, "optimizer_reset": False, "rng_reset": False,
            "additional_optimizer_updates": 0, "additional_environment_steps": 0}


def verify_sentinel_row(row, batch, update):
    """Recompute scalar diagnostics from all eight recorded action values."""
    require(row["update"] == update and row["extension_update"] == update - 2000
            and row["scene_seed"] == 10001, "sentinel identity and absolute clock")
    require(len(row["states"]) == len(SENTINEL_STEPS), "four predeclared sentinel states")
    for record, step, index in zip(row["states"], SENTINEL_STEPS, SENTINEL_INDICES):
        action = int(batch.actions[index].item())
        require(record["reference_step"] == step and record["global_transition_index"] == index
                and record["demonstrated_action"] == action, "sentinel demonstrated state identity")
        values = np.asarray(record["q_values"], dtype=np.float64)
        require(values.shape == (8,) and np.isfinite(values).all(), "eight finite sentinel action values")
        other = np.delete(values, action)
        require(record["greedy_action"] == int(np.argmax(values)), "sentinel greedy action")
        require(record["signed_demonstration_gap"] == float(values[action] - np.max(other)),
                "sentinel signed demonstration gap")


def verify_sentinel_network(agent, row, batch):
    """Use batch-one inference, matching policy action selection, independently."""
    before_rng, before_agent_rng = torch.get_rng_state().clone(), deepcopy(agent.rng.bit_generator.state)
    mode = agent.learner.online.training
    agent.learner.online.eval()
    try:
        with torch.no_grad():
            for record in row["states"]:
                index = record["global_transition_index"]
                values = agent.learner.online(batch.observations[index:index + 1])[0].tolist()
                require(values == record["q_values"],
                        f"TD {row['update']} sentinel {record['reference_step']} exact independent forward")
    finally:
        agent.learner.online.train(mode)
    require_equal(torch.get_rng_state(), before_rng, "sentinel preserves torch RNG")
    require_equal(agent.rng.bit_generator.state, before_agent_rng, "sentinel preserves agent RNG")


def verify_update_log(records, initial=2000, final=10000, interval=200):
    require(len(records) == final - initial, "complete continuation update log")
    for step, record in zip(range(initial + 1, final + 1), records):
        require(record["update"] == step and record["extension_update"] == step - initial,
                "sequential absolute and extension update clocks")
        require(record["target_synchronized"] == (step % interval == 0),
                "target synchronization follows the preserved absolute clock")
        for name, value in record.items():
            if name not in {"update", "extension_update", "target_synchronized"}:
                require(isinstance(value, (int, float)) and not isinstance(value, bool)
                        and math.isfinite(value), f"finite update diagnostic {name}")
    return sum(record["target_synchronized"] for record in records)


def replay_continuation(source, run_dir, batch, records, sentinels, end=2201):
    """Replay 201 updates independently; this is additional, separately counted work."""
    agent, env = independent_restore(source)
    count, compared = 0, []
    try:
        require(agent.learner.update_count == 2000, "independent replay begins at inherited TD 2000")
        verify_sentinel_network(agent, sentinels[2000], batch)
        for step in range(2001, end + 1):
            stats = agent.learner.update(batch, demonstration_weight=1.0, demonstration_margin=0.8)
            count += 1
            actual = {"update": step, "extension_update": step - 2000, **asdict(stats)}
            require(actual == records[step], f"independent exact update diagnostics at TD {step}")
            verify_sentinel_network(agent, sentinels[step], batch)
            snapshot = run_dir / "td" / f"training_state_{step}.pt"
            if snapshot.exists():
                saved = torch.load(snapshot, map_location="cpu", weights_only=True)
                for name, state in (("online", agent.learner.online.state_dict()),
                                    ("target", agent.learner.target.state_dict()),
                                    ("optimizer", agent.learner.optimizer.state_dict()),
                                    ("agent_rng_state", agent.rng.bit_generator.state),
                                    ("torch_rng_state", torch.get_rng_state())):
                    require_equal(state, saved[name], f"independent TD {step} {name}")
                require(saved["update_count"] == agent.learner.update_count == step,
                        "independent replay preserves absolute update count")
                require(saved["optimizer_summary"] == optimizer_state_summary(agent.learner),
                        f"independent TD {step} exact optimizer summary")
                compared.append(step)
    finally:
        env.close()
    return {"first_update": 2001, "last_update": end, "additional_optimizer_updates": count,
            "full_snapshot_updates_compared": compared, "sentinel_rows_compared": count + 1,
            "additional_environment_steps": 0, "additional_environment_resets": 0,
            "logs_and_sentinels_match_exactly": True}


def verify_archive(reference, archive_path, manifest):
    """Anchor immutable evidence to Git while tolerating checkout line endings."""
    committed = subprocess.check_output(
        ["git", "show", f"HEAD:{archive_path.relative_to(ROOT).as_posix()}"], cwd=ROOT,
    )
    archive = read_json(archive_path)
    require(json.loads(committed) == archive
            and hashlib.sha256(committed).hexdigest() == manifest["committed_archive_manifest_sha256"]
            and file_hash(archive_path) == manifest["archive_manifest_sha256"],
            "committed attempt-013 archive identity")
    for name, record in archive["files"].items():
        require(file_hash(archive_path.parent / name) == record["sha256"], f"attempt-013 archive {name}")
        if not name.endswith("update_samples.jsonl"):
            require(file_hash(reference / name) == record["sha256"], f"attempt-013 raw evidence {name}")
    for name, record in archive["update_log_coverage"].items():
        require(file_hash(reference / name) == record["raw_sha256"]
                and len(read_log(reference / name)) == record["raw_records"],
                f"attempt-013 complete raw update log {name}")
    return archive


def load_verified_batch(run_dir, reference, original, summary):
    dataset = read_json(run_dir / "dataset.json")
    targets = read_json(run_dir / "mc_targets.json")
    require(dataset == summary["dataset"] == read_json(reference / "dataset.json")
            == read_json(original / "dataset.json"), "unchanged encoded demonstration metadata")
    require(targets == read_json(reference / "mc_targets.json") == read_json(original / "mc_targets.json"),
            "unchanged actual discounted return metadata")
    require(file_hash(original / "demonstration.npz") == dataset["sha256"], "fixed encoded demonstration hash")
    with np.load(original / "demonstration.npz", allow_pickle=False) as encoded:
        batch = TransitionBatch(**{key: torch.from_numpy(encoded[key]) for key in TransitionBatch.__dataclass_fields__})
    episodes = dataset["episodes"]
    require(dataset["transitions"] == dataset["effective_batch_size"] == len(batch.actions) == 117
            and dataset["observation_size"] == batch.observations.shape[1] == 341
            and dataset["seeds"] == [10000, 10001, 10002]
            and [entry["transitions"] for entry in episodes] == [35, 41, 41]
            and dataset["full_batch_weighting"] == "equal_per_transition"
            and dataset["scene_seed_in_network_input"] is False, "fixed ordered 117-transition training batch")
    require(targets["gamma"] == 0.99 and targets["rewards"] == batch.rewards.tolist()
            and targets["network_bootstrapping_used"] is False and targets["episode_boundaries_respected"],
            "non-bootstrapped episode return semantics")
    returns, counts, difference = multi.independent_returns(batch, episodes, 0.99, targets["returns"])
    return batch, returns, episodes, {"episodes": counts, "maximum_float64_difference": difference,
                                    "transitions": len(batch.actions)}


def verify_retention(record, rows, source_row):
    legacy = deepcopy(record)
    for seed, entry in legacy["per_scene"].items():
        early = [row for row in rows if 2001 <= row["update"] <= 2010]
        count = sum(next(scene for scene in row["scenes"] if scene["seed"] == int(seed))["rollout"]["success"]
                    for row in early)
        require(entry.pop("first_ten_extension_updates_observation_count") == len(early) == 10
                and entry.pop("first_ten_extension_updates_success_count") == count,
                "first ten continuation outcomes use absolute TD 2001 through 2010")
        entry["first_ten_updates_observation_count"] = 0
        entry["first_ten_updates_success_count"] = 0
    multi.verify_retention(legacy, rows, source_row)


def verify_phase(run_dir, phase, settings, source, source_row, batch, episodes):
    require(read_json(run_dir / "td/summary.json") == phase, "matching continuation phase summary")
    require(phase["directory"] == "td" and phase["initial_td_update"] == 2000
            and phase["final_td_update"] == 10000 and phase["optimizer_updates"] == 8000
            and phase["prior_updates_on_branch"] == 40000 and phase["inherited_optimizer_updates"] == 42000
            and phase["total_updates_on_branch"] == 50000 and phase["demonstration_weight"] == 1
            and phase["learning_rate"] == 0.0001 and phase["target_synchronizations"] == 40
            and phase["target_source_updates"] == TARGET_SOURCE_UPDATES,
            "continuation phase budgets and unchanged controls")
    rows = read_json(run_dir / "td/observations.json")
    require([row["update"] for row in rows] == CHECKPOINTS, "fixed complete physical observation schedule")
    require(rows[0] == phase["initial_observation"] == source_row
            and rows[-1] == phase["final_observation"], "exact previous final behavior and current final observation")
    records = read_log(run_dir / "td/updates.jsonl")
    require(verify_update_log(records) == 40, "forty ordinary continuation target synchronizations")
    sentinels = read_log(run_dir / "td/sentinel_q.jsonl")
    require(len(sentinels) == phase["sentinel_records"] == 8001 and phase["sentinel_environment_steps"] == 0,
            "complete dense probe coverage without environment actions")
    for step, row in zip(range(2000, 10001), sentinels):
        verify_sentinel_row(row, batch, step)
    probes = {row["update"]: row for row in sentinels}
    expected_files = {f"update_{step}.pt" for step in CHECKPOINTS}
    expected_files |= {f"training_state_{step}.pt" for step in CHECKPOINTS}
    expected_files |= {f"target_source_state_{step}.pt" for step in TARGET_SOURCE_UPDATES}
    expected_files.add("training_state.pt")
    require(set(phase["checkpoint_sha256"]) == expected_files
            == {path.name for path in (run_dir / "td").glob("*.pt")}, "exact declared set of 67 model snapshots")
    for name, digest in phase["checkpoint_sha256"].items():
        require(file_hash(run_dir / "td" / name) == digest, f"continuation model hash: {name}")
    snapshots = {}
    for step in sorted(set(CHECKPOINTS) | set(TARGET_SOURCE_UPDATES)):
        prefix = "training_state" if step in CHECKPOINTS else "target_source_state"
        snapshots[step] = torch.load(run_dir / "td" / f"{prefix}_{step}.pt", map_location="cpu", weights_only=True)
    resume = verify_resume_payload(snapshots[2000], source)
    for step, payload in snapshots.items():
        target_source = snapshots[step - step % 200]
        prefix = "training_state" if step in CHECKPOINTS else "target_source_state"
        path = run_dir / "td" / f"{prefix}_{step}.pt"
        multi.verify_snapshot(path, audit.state_hash(payload["online"]), settings, step,
                              audit.state_hash(target_source["online"]), payload["optimizer_summary"])
        require_equal(payload["target"], target_source["online"], f"TD {step} exact recorded target-source tensors")
        require(payload["optimizer_summary"]["step_min"] == payload["optimizer_summary"]["step_max"]
                == 20000 + step, f"TD {step} preserved cumulative Adam parameter steps")
        require_equal(payload["agent_rng_state"], source["agent_rng_state"], "unchanged fixed-data agent RNG")
        require_equal(payload["torch_rng_state"], source["torch_rng_state"], "unchanged deterministic training RNG")
        agent, env = independent_restore(payload)
        try:
            verify_sentinel_network(agent, probes[step], batch)
        finally:
            env.close()
    for row in rows:
        step = row["update"]
        multi.verify_aggregate(row, episodes)
        require(row["total_updates_on_branch"] == 40000 + step, "physical checkpoint total branch count")
        payload = snapshots[step]
        inference = torch.load(run_dir / "td" / f"update_{step}.pt", map_location="cpu", weights_only=True)
        require_equal(inference["online"], payload["online"], "inference and full-state online tensors agree")
        require(json.loads(json.dumps(inference["dqn_config"])) == settings["dqn"]
                and inference["environment_config"] == settings["environment"], "inference configuration")
        require(row["optimizer_state"] == payload["optimizer_summary"], "observed Adam state")
        metadata = inference["metadata"]
        require(metadata["phase"] == "multiscene_retention_diagnostic" and metadata["objective"] == "extension/td"
                and metadata["updates"] == step and metadata["prior_updates_on_branch"] == 40000
                and metadata["total_updates_on_branch"] == 40000 + step and metadata["training_scene_only"]
                and metadata["scene_seeds"] == [10000, 10001, 10002]
                and metadata["fit"] == row["fit"] and metadata["aggregate"] == row["aggregate"],
                "inference checkpoint metadata")
    final = torch.load(run_dir / "td/training_state.pt", map_location="cpu", weights_only=True)
    require_equal(final, snapshots[10000], "final training-state alias payload")
    require(phase["training_snapshot_file"] == "td/training_state.pt"
            and file_hash(run_dir / phase["training_snapshot_file"]) == phase["training_snapshot_sha256"]
            and phase["initial_parameter_sha256"] == source["parameter_sha256"]
            and phase["final_parameter_sha256"] == final["parameter_sha256"], "initial and final training identities")
    require(phase["evaluation_environment_steps"] == sum(row["aggregate"]["evaluation_environment_steps"] for row in rows),
            "physical evaluation action accounting")
    verify_retention(phase["retention"], rows, source_row)
    return rows, records, probes, resume


def verify_accounting(accounting, rows):
    expected = {"optimizer_updates_total": 8000, "td_optimizer_updates": 8000,
                "inherited_optimizer_updates": 42000, "inherited_supervised_updates": 20000,
                "inherited_calibration_updates": 20000, "inherited_td_updates": 2000,
                "total_updates_on_branch": 50000, "absolute_td_update": 10000,
                "evaluation_environment_steps": sum(row["aggregate"]["evaluation_environment_steps"] for row in rows),
                "scheduled_evaluation_episodes": 87, "sentinel_records": 8001,
                "sentinel_forward_passes": 32004, "sentinel_environment_steps": 0,
                "demonstration_collection_steps": 0, "online_training_steps": 0, "held_out_scenes_accessed": 0}
    for key, value in expected.items():
        require(accounting[key] == value, f"main experiment accounting: {key}")


def verify(run_dir, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite continuation verification evidence: {output}")
    torch.set_num_threads(1)
    settings, summary = read_json(run_dir / "settings.json"), read_json(run_dir / "summary.json")
    require(read_json(run_dir / "run_status.json")["status"] == "completed", "completed continuation run")
    protocol = summary["protocol"]
    require(protocol == settings["training"], "resolved and summarized protocol agree")
    verify_protocol(protocol, settings)
    manifest, reference, original = verify_inputs(run_dir, protocol)
    source = torch.load(reference / manifest["resumed_snapshot_file"], map_location="cpu", weights_only=True)
    source_phase = read_json(reference / "treatment/td/summary.json")
    require(source["update_count"] == 2000 and source["optimizer_summary"]["step_min"]
            == source["optimizer_summary"]["step_max"] == 22000
            and source_phase["optimizer_updates"] == 2000 and source_phase["total_updates_on_branch"] == 42000
            and source_phase["learning_rate"] == 0.0001 and source_phase["demonstration_weight"] == 1.0
            and all(scene["rollout"]["success"] for scene in source_phase["final_observation"]["scenes"]),
            "successful inherited treatment and exact TD/Adam counters")
    require(json.loads(json.dumps(source["dqn_config"])) == settings["dqn"]
            and source["environment_config"] == settings["environment"], "unchanged resumed configuration")
    batch, returns, episodes, return_checks = load_verified_batch(run_dir, reference, original, summary)
    rows, records, sentinels, resume = verify_phase(
        run_dir, summary["extension"], settings, source, source_phase["final_observation"], batch, episodes,
    )
    require(read_json(run_dir / "initial_equivalence.json") == {
        "passed": True, "exact_previous_final_observation": True,
        "absolute_td_update": 2000, "new_updates_before_gate": 0,
    }, "resume behavior gate passed before new training")
    require(read_json(run_dir / "restore_checks.json") == {
        "exact_full_snapshot": True, "target_synchronized_at_restore": False,
        "optimizer_reset_at_restore": False, "td_counter_reset_at_restore": False,
        "absolute_td_update": 2000, "optimizer_state": source["optimizer_summary"],
    }, "recorded exact continuation restoration")
    require(summary["accounting"] == read_json(run_dir / "accounting.json"), "matching main accounting records")
    verify_accounting(summary["accounting"], rows)
    replay = replay_continuation(source, run_dir, batch, {row["update"]: row for row in records}, sentinels)
    by_step = {row["update"]: row for row in rows}
    reloads = [multi.reload_policy(run_dir, "td", step, by_step[step], batch, returns, episodes)
               for step in RELOAD_CHECKPOINTS]
    result = {
        "passed": True, "attempt": 14, "verifier_sha256": file_hash(Path(__file__)),
        "reference_run": str(reference), "original_data_run": str(original),
        "source_snapshot_sha256": manifest["resumed_snapshot_sha256"], "resume": resume,
        "independent_return_checks": return_checks, "verified_raw_updates": len(records),
        "verified_sentinel_rows": len(sentinels), "verified_physical_checkpoints": len(rows),
        "verified_full_and_inference_snapshot_files": len(summary["extension"]["checkpoint_sha256"]),
        "independent_replay": replay, "policy_reloads": reloads,
        "additional_optimizer_updates": replay["additional_optimizer_updates"],
        "additional_environment_resets": sum(len(row["scenes"]) for row in reloads),
        "additional_environment_steps": sum(scene["reload_environment_steps"] for row in reloads for scene in row["scenes"]),
        "additional_evaluation_episodes": 15,
        "scope": "Replay covers TD 2001-2201 only; later model hashes, target lineage, clocks and saved-checkpoint probes are audited without replaying all 8000 updates.",
        "sentinel_interpretation": "Action rankings on fixed demonstration states are separate from measured physical rollout failures.",
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
    print(json.dumps({key: result[key] for key in ("passed", "additional_optimizer_updates",
                                                  "additional_evaluation_episodes", "additional_environment_steps")}))


if __name__ == "__main__":
    main()
