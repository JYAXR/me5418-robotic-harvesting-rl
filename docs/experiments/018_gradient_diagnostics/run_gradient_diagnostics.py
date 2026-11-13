"""Diagnose historical TD updates without extending a policy-training trajectory."""

import argparse
import hashlib
import importlib.util
import json
import subprocess
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from me5418_harvesting.agents.dqn import DQNConfig
from me5418_harvesting.agents.dqn_learning import DQNUpdater
from me5418_harvesting.agents.run_record import begin_run, finish_run

ROOT = Path(__file__).resolve().parents[3]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


anchor = load_module("gradient018_anchor", ROOT / "docs/experiments/017_mc_anchor/run_mc_anchor.py")
core = load_module("gradient018_core", Path(__file__).with_name("gradient_core.py"))
trajectory = load_module("gradient018_trajectory", Path(__file__).with_name("trajectory_probe.py"))
fit, extension, adam = anchor.fit, anchor.extension, anchor.adam
require, require_equal = anchor.require, anchor.require_equal
CASES = (2001, 2008, 2009, 2010, 2018, 2152, 2158, 2161, 2200, 2201,
         3000, 4500, 8500, 10000)
PROBES = ((10001, 6), (10001, 18), (10001, 20), (10001, 21), (10001, 22),
          (10001, 23), (10001, 24), (10002, 37))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def restore_agent(snapshot, input_size, action_count):
    """Restore all saved training state without constructing or resetting an environment."""
    learner = DQNUpdater(input_size, action_count, DQNConfig(**snapshot["dqn_config"]))
    learner.online.load_state_dict(deepcopy(snapshot["online"]))
    learner.target.load_state_dict(deepcopy(snapshot["target"]))
    learner.optimizer.load_state_dict(deepcopy(snapshot["optimizer"]))
    learner.update_count = snapshot["update_count"]
    rng = np.random.default_rng(0)
    rng.bit_generator.state = deepcopy(snapshot["agent_rng_state"])
    torch.set_rng_state(snapshot["torch_rng_state"].clone())
    agent = SimpleNamespace(learner=learner, rng=rng)
    require_equal(adam.training_snapshot(agent, snapshot["environment_config"]), snapshot,
                  "environment-free restoration preserves the complete source state")
    return agent


def make_probes(batch, episodes):
    probes, metadata = [], []
    for seed, step in PROBES:
        episode = next(item for item in episodes if item["seed"] == seed)
        index = episode["start"] + step - 1
        require(episode["start"] <= index < episode["stop"], "probe belongs to its original episode")
        name = f"scene_{seed}_step_{step}"
        probes.append({"name": name, "observation": batch.observations[index:index + 1],
                       "demonstrated_action": int(batch.actions[index])})
        metadata.append({"name": name, "scene_seed": seed, "reference_step": step,
                         "global_transition_index": index,
                         "demonstrated_action": int(batch.actions[index])})
    return probes, metadata


def audit_inputs(reference):
    """Bind both historical arms and their data to previously committed experiment evidence."""
    archive = ROOT / "docs/experiments/017_mc_anchor/evidence"
    manifest_path = archive / "archive_manifest.json"
    manifest = read_json(manifest_path)
    committed = subprocess.check_output(
        ["git", "show", f"HEAD:{manifest_path.relative_to(ROOT).as_posix()}"], cwd=ROOT)
    require(committed == manifest_path.read_bytes() and manifest["status"] == "completed"
            and Path(manifest["source_run"]) == reference, "committed completed attempt-017 archive")
    hashes = {str(manifest_path): digest(manifest_path)}
    for name, info in manifest["files"].items():
        archived = archive / name
        require(digest(archived) == info["sha256"] and archived.stat().st_size == info["bytes"],
                f"unchanged archived evidence: {name}")
        hashes[str(archived)] = info["sha256"]
        if not name.endswith("_samples.jsonl"):
            require(digest(reference / name) == info["sha256"], f"unchanged historical raw evidence: {name}")
            hashes[str(reference / name)] = info["sha256"]
    for name, info in manifest["raw_log_coverage"].items():
        require(digest(reference / name) == info["raw_sha256"], f"unchanged historical complete log: {name}")
        hashes[str(reference / name)] = info["raw_sha256"]
    source_manifest = read_json(reference / "input_manifest.json")
    control = Path(source_manifest["reference_run"])
    inherited = anchor.load_inputs(control, ROOT / "docs/experiments/016_local_supervision/evidence")
    require_equal(inherited["manifest"], source_manifest, "unchanged complete demonstration and source chain")
    require(read_json(reference / "reload_checks.json")["passed"], "independent attempt-017 verification passed")
    require(digest(ROOT / "docs/experiments/017_mc_anchor/run_mc_anchor.py")
            == read_json(reference / "settings.json")["training"]["script_sha256"], "unchanged original updater")
    for name, expected in {**source_manifest["helper_sha256"],
                           **source_manifest["production_source_sha256"]}.items():
        hashes[str(ROOT / name)] = expected
    original = Path(source_manifest["original_data_run"]) / "demonstration.npz"
    hashes[str(original)] = source_manifest["dataset_sha256"]
    directories = {"baseline": control / "baseline/td", "mc_anchor": reference / "treatment/td"}
    arms = {}
    for arm, directory in directories.items():
        phase = read_json(directory / "summary.json")
        snapshots = {}
        for name, expected in phase["checkpoint_sha256"].items():
            path = directory / name
            require(digest(path) == expected, f"unchanged historical model: {arm}/{name}")
            hashes[str(path)] = expected
            if name.startswith(("training_state_", "target_source_state_")):
                snapshots[int(path.stem.rsplit("_", 1)[1])] = path
        for name in ("summary.json", "updates.jsonl", "sentinel_q.jsonl", "observations.json"):
            hashes[str(directory / name)] = digest(directory / name)
        updates = {row["update"]: row for row in anchor.read_updates(directory / "updates.jsonl")}
        sentinels = {row["update"]: row for row in anchor.read_updates(directory / "sentinel_q.jsonl")}
        require(sorted(updates) == list(range(2001, 10001))
                and sorted(sentinels) == list(range(2000, 10001)), "complete historical clocks")
        arms[arm] = {"directory": directory, "snapshots": snapshots,
                     "updates": updates, "sentinels": sentinels, "weight": float(arm == "mc_anchor")}
    return inherited, arms, hashes


def source_before(snapshots, update):
    candidates = [step for step in snapshots if step < update]
    if not candidates:
        raise ValueError("No complete snapshot precedes the requested diagnostic update.")
    return max(candidates)


def replay_groups(snapshots, cases=CASES):
    """Share overlapping historical prefixes while keeping every diagnostic branch isolated."""
    groups = {}
    for update in cases:
        groups.setdefault(source_before(snapshots, update), []).append(update)
    return groups


def compare_historical_step(agent, inputs, history, stats, update):
    """Require exact original log, reference Q values and available complete post-update state."""
    expected = history["updates"][update]
    actual = {"update": update, "extension_update": update - 2000, **stats}
    require_equal(actual, expected, f"exact historical update log at TD {update}")
    probe = extension.sentinel_record(agent, inputs["batch"], inputs["episodes"], update,
                                      reference_steps=anchor.SENTINEL_STEPS)
    require_equal(probe, history["sentinels"][update], f"exact historical dense Q row at TD {update}")
    checked = False
    if update in history["snapshots"]:
        snapshot = torch.load(history["snapshots"][update], map_location="cpu", weights_only=True)
        require_equal(adam.training_snapshot(agent, snapshot["environment_config"]), snapshot,
                      f"exact historical full state at TD {update}")
        checked = True
    return checked


def historical_update(agent, inputs, history):
    if history["weight"]:
        return anchor.anchored_update(agent.learner, inputs["batch"], inputs["returns"])
    return asdict(agent.learner.update(inputs["batch"], demonstration_weight=1.0,
                                     demonstration_margin=0.8))


def run(reference, output):
    reference, output = reference.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite gradient diagnostic evidence: {output}")
    torch.set_num_threads(1)
    inputs, arms, source_hashes = audit_inputs(reference)
    probes, metadata = make_probes(inputs["batch"], inputs["episodes"])
    source_paths = [Path(__file__), Path(__file__).with_name("gradient_core.py"),
                    Path(__file__).with_name("trajectory_probe.py")]
    protocol = {
        "attempt": 18, "kind": "retrospective_gradient_and_adam_diagnosis",
        "incoming_td_updates": list(CASES), "arms": ["baseline", "mc_anchor"],
        "reference_probes": deepcopy(metadata),
        "additional_probe": "Scene 10002 actual incoming step 40, reconstructed from the common 39-action prefix of treatment policies TD 7500, 8500 and 9000. Reference action south_small comes from the two successful policies, not a demonstration label.",
        "selection": "Fixed before diagnostics using attempt-017 failures, early reversals, minima, target-sync adjacency and endpoint. Retrospective, not a random sample.",
        "effective_batch_size": 117, "margin": 0.8, "gamma": 0.99,
        "learning_rate": 0.0001, "gradient_clip_norm": 10.0,
        "baseline_mc_weight": 0.0, "treatment_mc_weight": 1.0,
        "supervision_weights": "All 117 coefficients remain one.",
        "loss_target_semantics": "TD targets are frozen from the pre-update target network; MC targets are fixed actual episode returns.",
        "gap_semantics": "Derivative uses the strongest non-demonstrated action fixed before the update; actual max-other gap is also measured after each branch.",
        "counterfactual_semantics": "Independent one-step clones, never chained or adopted as trained policies.",
        "adam_decomposition": "Old first moment and current objective contributions use the same actual new second-moment denominator and clip coefficient. Numerical remainder is recorded.",
        "environment_resets": 1, "environment_steps": 39, "completed_evaluation_episodes": 0,
        "new_policy_training_updates": 0,
        "new_demonstrations": 0, "held_out_scenes": 0,
        "scripts_sha256": {path.relative_to(ROOT).as_posix(): digest(path) for path in source_paths},
    }
    output.mkdir(parents=True, exist_ok=False)
    begin_run(output, inputs["snapshot"]["environment_config"], inputs["snapshot"]["dqn_config"], protocol)
    save_json(output / "input_manifest.json", {"reference_run": str(reference), "files_sha256": source_hashes})
    accounting = {"historical_replay_optimizer_updates": 0, "diagnostic_branch_optimizer_updates": 0,
                  "historical_post_update_verification_steps": 0, "completed_cases": 0,
                  "new_policy_training_updates": 0, "environment_resets": 0, "environment_steps": 0,
                  "completed_evaluation_episodes": 0}
    cases = []
    try:
        extra_probe, extra_metadata, trajectory_evidence = trajectory.build_trajectory_probe(
            reference, inputs["batch"], inputs["episodes"])
        probes.append(extra_probe)
        metadata.append(extra_metadata)
        save_json(output / "trajectory_probe.json", trajectory_evidence)
        save_json(output / "probe_metadata.json", metadata)
        save_json(output / "probe_observations.json", {
            row["name"]: row["observation"].tolist() for row in probes})
        accounting.update(environment_resets=1, environment_steps=39)
        save_json(output / "accounting.json", accounting)
        for arm, history in arms.items():
            for start, group_cases in replay_groups(history["snapshots"]).items():
                path = history["snapshots"][start]
                payload = torch.load(path, map_location="cpu", weights_only=True)
                agent = restore_agent(payload, inputs["batch"].observations.shape[1], 8)
                checked = []
                for update in range(start + 1, max(group_cases) + 1):
                    if update not in group_cases:
                        stats = historical_update(agent, inputs, history)
                        accounting["historical_replay_optimizer_updates"] += 1
                        if compare_historical_step(agent, inputs, history, stats, update):
                            checked.append(update)
                        continue
                    before = adam.training_snapshot(agent, payload["environment_config"])
                    result, actual_learner = core.diagnose_update(
                        agent.learner, inputs["batch"], inputs["returns"], probes, mc_weight=history["weight"])
                    require_equal(adam.training_snapshot(agent, payload["environment_config"]), before,
                                  "diagnostics leave the source model, optimizer and RNG unchanged")
                    stats = historical_update(agent, inputs, history)
                    accounting["historical_post_update_verification_steps"] += 1
                    if compare_historical_step(agent, inputs, history, stats, update):
                        checked.append(update)
                    for name in ("online", "target", "optimizer"):
                        require_equal(getattr(actual_learner, name).state_dict(),
                                      getattr(agent.learner, name).state_dict(),
                                      f"diagnostic actual branch exactly reproduces original {name}")
                    require(actual_learner.update_count == agent.learner.update_count == update,
                            "diagnostic actual branch preserves the original TD counter")
                    result.update(arm=arm, incoming_td_update=update, pre_update_td=update - 1,
                                  source_snapshot=str(path), source_snapshot_sha256=digest(path),
                                  restored_td_update=start, logical_prefix_length=update - start - 1,
                                  shared_replay_group=group_cases, checked_full_state_updates=list(checked),
                                  exact_historical_log_and_sentinel=True, exact_original_updater_state=True)
                    name = f"cases/{arm}_{update}.json"
                    save_json(output / name, result)
                    cases.append({"arm": arm, "incoming_td_update": update, "file": name,
                                  "sha256": digest(output / name)})
                    accounting["diagnostic_branch_optimizer_updates"] += result["independent_optimizer_steps"]
                    accounting["completed_cases"] += 1
                    save_json(output / "accounting.json", accounting)
                    print(f"Completed {arm} incoming TD {update}: restored {start}; exact historical update verified.", flush=True)
        for name, expected in source_hashes.items():
            require(digest(Path(name)) == expected, f"source remains immutable: {name}")
        save_json(output / "summary.json", {
            "protocol": protocol, "cases": cases, "accounting": accounting,
            "diagnostic_artifacts_sha256": {name: digest(output / name) for name in (
                "probe_metadata.json", "probe_observations.json", "trajectory_probe.json")},
            "source_hashes_unchanged": True,
            "limitations": ["Retrospective, selected updates from one initialization and three known layouts.",
                            "Reference-state gaps do not establish complete rollout success.",
                            "Per-loss gradient alignment is local; Adam counterfactuals alter moments and clipping.",
                            "One-step ablations are diagnostic evidence, not tested long-term remedies.",
                            "Scene 10002 step 37 is a first divergence shared by successful runs, not a proven cause."]})
    except BaseException as error:
        save_json(output / "accounting.json", accounting)
        finish_run(output, "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error)
        raise
    else:
        finish_run(output, "completed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.reference_run, args.output_dir)


if __name__ == "__main__":
    main()
