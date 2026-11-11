"""Test one fixed action correction in two frozen failed attempt-014 policies."""

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from me5418_harvesting.agents.run_record import begin_run, finish_run

ROOT = Path(__file__).resolve().parents[3]
UPDATES = (2009, 10000)
SEED, STEP, ACTION, ORIGINAL_ACTION = 10001, 21, 0, 4


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DIAGNOSTIC = ROOT / "docs/experiments/012_divergence_control/diagnose_divergence.py"
VERIFIER = ROOT / "docs/experiments/014_td_extension/verify_td_extension.py"
SNAPSHOT_HELPER = ROOT / "docs/experiments/010_adam_state/run_adam_state.py"
diagnostic = import_file("rescue015_diagnostic", DIAGNOSTIC)
previous = import_file("rescue015_previous", VERIFIER)
adam = import_file("rescue015_snapshot", SNAPSHOT_HELPER)
read_json, save_json, file_hash = diagnostic.read_json, diagnostic.save_json, diagnostic.file_hash
require, require_equal = previous.require, previous.require_equal


def validate_intervention(expected, actions):
    """Bind the existing trace helper to exactly step 21 and action north_small."""
    declared = {"step": STEP, "predicted_action": ORIGINAL_ACTION, "demonstrated_action": ACTION}
    require(diagnostic.first_divergence(expected["actions"], actions) == declared,
            "the only permitted intervention is the fixed step-21 north_small correction")
    require(expected["first_reference_action_divergence"] == declared,
            "the archived first divergence must match the declared fixed intervention")
    require(expected["seed"] == SEED and expected["training_scene_only"]
            and not expected["success"] and not expected["damage"]
            and expected["budget_exhausted"] and expected["steps"] == 200,
            "the source must be the recorded scene-10001 budget failure")


def verify_archive(reference, archive):
    """Check every committed archive entry and its complete local source log."""
    manifest_path = archive / "archive_manifest.json"
    relative = manifest_path.relative_to(ROOT).as_posix()
    committed = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)
    manifest = read_json(manifest_path)
    require(json.loads(committed) == manifest and manifest["status"] == "completed",
            "the attempt-014 archive must match committed completed evidence")
    raw, archived = {}, {}
    for name, record in manifest["files"].items():
        diagnostic.require_hash(archive / name, record["sha256"])
        archived[name] = record["sha256"]
        if not name.endswith("_samples.jsonl"):
            diagnostic.require_hash(reference / name, record["sha256"])
            raw[name] = record["sha256"]
    for name, record in manifest["raw_log_coverage"].items():
        diagnostic.require_hash(reference / name, record["raw_sha256"])
        require(len(previous.read_log(reference / name)) == record["raw_records"],
                f"complete attempt-014 raw record count: {name}")
        raw[name] = record["raw_sha256"]
    return {
        "reference_run": str(reference), "archive_dir": str(archive),
        "archive_manifest": relative, "archive_manifest_sha256": file_hash(manifest_path),
        "committed_archive_manifest_blob_sha256": hashlib.sha256(committed).hexdigest(),
        "committed_archive_verified": True, "archived_files_sha256": archived,
        "reference_files_sha256": raw,
    }


def load_inputs(reference, archive):
    """Audit all immutable provenance without a reset, step, or optimizer update."""
    manifest = verify_archive(reference, archive)
    settings, summary = read_json(reference / "settings.json"), read_json(reference / "summary.json")
    require(read_json(reference / "run_status.json")["status"] == "completed"
            and read_json(reference / "reload_checks.json")["passed"], "verified completed source run")
    require(summary["protocol"] == settings["training"], "matching source resolved protocol")
    previous.verify_protocol(settings["training"], settings)
    inherited, prior_reference, data_reference = previous.verify_inputs(reference, settings["training"])
    require(file_hash(VERIFIER) == read_json(reference / "reload_checks.json")["verifier_sha256"],
            "unchanged independently verified attempt-014 verifier")
    require(file_hash(DIAGNOSTIC) == read_json(reference / "q_analysis.json")["diagnostic_helper_sha256"],
            "unchanged attempt-012 action tracing helper")
    phase = summary["extension"]
    require(phase == read_json(reference / "td/summary.json"), "matching source phase summaries")
    for name, digest in phase["checkpoint_sha256"].items():
        diagnostic.require_hash(reference / "td" / name, digest)
        manifest["reference_files_sha256"][f"td/{name}"] = digest
    dataset = read_json(reference / "dataset.json")
    require(dataset == read_json(data_reference / "dataset.json")
            and dataset["transitions"] == 117 and dataset["seeds"] == [10000, 10001, 10002],
            "unchanged three-scene demonstration dataset")
    data_path = data_reference / "demonstration.npz"
    diagnostic.require_hash(data_path, dataset["sha256"])
    with np.load(data_path, allow_pickle=False) as data:
        episode = next(row for row in dataset["episodes"] if row["seed"] == SEED)
        observations = data["observations"][episode["start"]:episode["stop"]].copy()
        actions = data["actions"][episode["start"]:episode["stop"]].copy()
    require(episode["start"] == 35 and episode["stop"] == 76
            and observations.shape == (41, 341) and len(actions) == 41
            and int(actions[STEP - 1]) == ACTION
            and diagnostic.array_hash(observations[0]) == episode["initial_observation_sha256"],
            "fixed scene-10001 demonstration and intervention state")
    source_rows = {row["update"]: row for row in read_json(reference / "td/observations.json")}
    snapshots, expected = {}, {}
    for update in UPDATES:
        row = source_rows[update]
        expected[update] = next(scene["rollout"] for scene in row["scenes"] if scene["seed"] == SEED)
        validate_intervention(expected[update], actions)
        snapshot = torch.load(reference / "td" / f"training_state_{update}.pt",
                              map_location="cpu", weights_only=True)
        inference = torch.load(reference / "td" / f"update_{update}.pt",
                               map_location="cpu", weights_only=True)
        require_equal(inference["online"], snapshot["online"], "full and inference online weights agree")
        require(snapshot["update_count"] == update
                and snapshot["optimizer_summary"]["step_min"]
                == snapshot["optimizer_summary"]["step_max"] == 20000 + update
                and row["total_updates_on_branch"] == 40000 + update
                and snapshot["environment_config"] == settings["environment"]
                and json.loads(json.dumps(snapshot["dqn_config"])) == settings["dqn"]
                and row["optimizer_state"] == snapshot["optimizer_summary"],
                f"unchanged full training snapshot at TD {update}")
        snapshots[update] = snapshot
    manifest.update({
        "attempt_014_input_manifest_sha256": file_hash(reference / "input_manifest.json"),
        "attempt_013_reference_run": str(prior_reference),
        "original_data_run": str(data_reference), "dataset_file": str(data_path),
        "dataset_sha256": dataset["sha256"], "source_chain_checked_through_attempt_011": True,
        "source_production_sha256": read_json(reference / "provenance.json")["source_sha256"],
        "helper_sha256": {
            **settings["training"]["helper_sha256"],
            previous.RUNNER.relative_to(ROOT).as_posix(): file_hash(previous.RUNNER),
            DIAGNOSTIC.relative_to(ROOT).as_posix(): file_hash(DIAGNOSTIC),
            VERIFIER.relative_to(ROOT).as_posix(): file_hash(VERIFIER),
            SNAPSHOT_HELPER.relative_to(ROOT).as_posix(): file_hash(SNAPSHOT_HELPER),
        },
        "source_snapshots": {str(update): {
            "file": f"td/training_state_{update}.pt",
            "sha256": phase["checkpoint_sha256"][f"training_state_{update}.pt"],
            "parameter_sha256": snapshots[update]["parameter_sha256"],
            "absolute_td_update": update, "adam_parameter_step": 20000 + update,
            "historical_updates_on_branch": 40000 + update,
        } for update in UPDATES},
        "inherited_attempt_014_manifest": inherited,
    })
    return {"manifest": manifest, "settings": settings, "snapshots": snapshots,
            "expected": expected, "observations": observations, "actions": actions}


def frozen_trace(snapshot, observations, actions, expected, *, intervene, progress=None):
    """Freshly restore one frozen arm, run it once, and require exact state preservation."""
    validate_intervention(expected, actions)
    agent, env = previous.independent_restore(snapshot)
    try:
        require_equal(adam.training_snapshot(agent, snapshot["environment_config"]), snapshot,
                      "exact full state before diagnostic rollout")
        trace = diagnostic.trace_policy(agent, env, observations, actions, expected,
                                        intervene=intervene, progress=progress)
        require_equal(adam.training_snapshot(agent, snapshot["environment_config"]), snapshot,
                      "diagnostic inference leaves the complete full training state unchanged")
        trace["full_training_state_checks"] = {
            "exact_snapshot_at_restore": True, "exact_snapshot_after_rollout": True,
            "absolute_td_update": snapshot["update_count"],
            "adam_parameter_step": snapshot["optimizer_summary"]["step_min"],
            "parameter_sha256": snapshot["parameter_sha256"],
            "optimizer_state": snapshot["optimizer_summary"],
        }
        return trace
    finally:
        env.close()


def paired_fork(greedy, counterfactual):
    """Check that only the declared executed action changes at the fork."""
    left, right = greedy["records"], counterfactual["records"]
    require(left[:STEP - 1] == right[:STEP - 1], "identical first 20 actual transitions")
    a, b = left[STEP - 1], right[STEP - 1]
    require(a["step"] == b["step"] == STEP
            and all(a[name] == b[name] for name in (
                "position", "encoded_observation_sha256", "q_values", "policy_action"))
            and a["policy_action"] == ORIGINAL_ACTION
            and a["executed_action"] == ORIGINAL_ACTION and not a["forced_reference_action"]
            and b["executed_action"] == ACTION and b["forced_reference_action"],
            "same decision state and Q scores, with exactly the declared fork action changed")
    for trace, correction in ((greedy, False), (counterfactual, True)):
        require(trace["forced_action_count"] == int(correction), "declared forced-action count")
        for row in trace["records"]:
            forced = correction and row["step"] == STEP
            require(row["forced_reference_action"] == forced
                    and row["executed_action"] == (ACTION if forced else row["policy_action"]),
                    "all actions outside step 21 remain the model's own greedy decisions")
    return {
        "step": STEP, "prefix_records_equal_before_intervention": True,
        "decision_state_and_q_equal": True, "policy_action": ORIGINAL_ACTION,
        "greedy_executed_action": ORIGINAL_ACTION, "counterfactual_executed_action": ACTION,
        "signed_demonstration_gap": diagnostic.q_metrics(a["q_values"], ACTION)["signed_demonstration_margin"],
    }


def run(reference, archive, output):
    reference, archive, output = reference.resolve(), archive.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite the diagnostic run: {output}")
    inputs = load_inputs(reference, archive)
    protocol = {
        "attempt": 15, "kind": "fixed_step_single_action_rescue", "scene_seed": SEED,
        "td_updates": list(UPDATES), "intervention_step": STEP,
        "intervention_action": ACTION, "intervention_action_name": "north_small",
        "original_policy_action": ORIGINAL_ACTION, "original_policy_action_name": "north_large",
        "episode_order": ["2009/greedy", "2009/counterfactual", "10000/greedy", "10000/counterfactual"],
        "episodes": 4, "optimizer_updates": 0, "online_training_steps": 0,
        "held_out_scenes_accessed": 0, "fresh_full_restore_for_each_arm": True,
        "same_environment_reset_seed": True, "epsilon": 0.0,
        "intervention_rule": "Override only one-based step 21 with action 0, then immediately resume the unchanged greedy neural policy.",
        "script_sha256": file_hash(Path(__file__)), "helper_sha256": inputs["manifest"]["helper_sha256"],
        "verification_scope": "Read-only trace, source, and shared-prefix Q audit; zero additional physical episodes.",
    }
    accounting = {
        "optimizer_updates": 0, "started_episodes": 0, "completed_episodes": 0,
        "greedy_environment_steps": 0, "counterfactual_environment_steps": 0,
        "environment_steps": 0, "forced_actions": 0, "online_training_steps": 0,
        "demonstration_collection_steps": 0, "held_out_scenes_accessed": 0,
        "inherited_optimizer_updates_by_checkpoint": {str(update): 40000 + update for update in UPDATES},
        "note": "Historical model training is reused, not rerun or summed across checkpoints. Each actual diagnostic env.step action is counted; reset feasibility probes are excluded.",
    }
    output.mkdir(parents=True, exist_ok=False)
    try:
        begin_run(output, inputs["settings"]["environment"], inputs["settings"]["dqn"], protocol)
        save_json(output / "input_manifest.json", inputs["manifest"])
        save_json(output / "accounting.json", accounting)
        torch.set_num_threads(1)
        pairs = []
        for update in UPDATES:
            pair, complete = {"update": update}, {}
            for mode, intervene in (("greedy", False), ("counterfactual", True)):
                trace_path = output / f"update_{update}_{mode}_steps.jsonl"
                accounting["started_episodes"] += 1
                save_json(output / "accounting.json", accounting)
                with trace_path.open("x", encoding="utf-8") as stream:
                    def progress(record, current_mode=mode):
                        accounting[f"{current_mode}_environment_steps"] += 1
                        accounting["environment_steps"] += 1
                        accounting["forced_actions"] += int(record["forced_reference_action"])
                        stream.write(json.dumps(record, allow_nan=False) + "\n")
                        stream.flush()
                        save_json(output / "accounting.json", accounting)

                    trace = frozen_trace(inputs["snapshots"][update], inputs["observations"],
                                         inputs["actions"], inputs["expected"][update],
                                         intervene=intervene, progress=progress)
                accounting["completed_episodes"] += 1
                trace_file = output / f"update_{update}_{mode}.json"
                save_json(trace_file, trace)
                save_json(output / "accounting.json", accounting)
                complete[mode] = trace
                pair[mode] = {key: value for key, value in trace.items() if key != "records"}
                pair[mode].update(trace_file=trace_file.name, steps_file=trace_path.name,
                                  trace_sha256=file_hash(trace_file), steps_sha256=file_hash(trace_path))
                print(json.dumps({"update": update, "mode": mode,
                                  "success": trace["rollout"]["success"],
                                  "steps": trace["rollout"]["steps"],
                                  "forced_actions": trace["forced_action_count"]}), flush=True)
            pair["paired_fork"] = paired_fork(complete["greedy"], complete["counterfactual"])
            pair["single_action_rescued_success"] = pair["counterfactual"]["rollout"]["success"]
            pairs.append(pair)
            save_json(output / "replays.json", pairs)
        require(load_inputs(reference, archive)["manifest"] == inputs["manifest"],
                "all source inputs remain unchanged after the four diagnostic episodes")
        save_json(output / "source_unchanged.json", {
            "passed": True, "all_input_manifest_hashes_rechecked": True,
            "source_models_and_production_files_unchanged": True,
        })
        summary = {
            "protocol": protocol, "scene_seed": SEED, "td_updates": list(UPDATES),
            "replays": pairs, "accounting": accounting,
            "rescued_checkpoint_count": sum(pair["single_action_rescued_success"] for pair in pairs),
            "tested_checkpoint_count": len(pairs),
            "interpretation": "A single externally corrected action tests these two failures on one known deterministic scene. It changes neither the learned policy nor its unassisted success rate.",
        }
        save_json(output / "summary.json", summary)
    except BaseException as error:
        save_json(output / "accounting.json", accounting)
        if (output / "run_status.json").exists():
            finish_run(output, "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error)
        raise
    else:
        finish_run(output, "completed")
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.reference_run, args.archive_dir, args.output_dir)


if __name__ == "__main__":
    main()
