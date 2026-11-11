"""Audit the single-action rescue evidence without new episodes or learning."""

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import torch

from me5418_harvesting.agents.dqn import DQNConfig, build_q_network

ROOT = Path(__file__).resolve().parents[3]
RUNNER = Path(__file__).with_name("run_single_action_rescue.py")
PRIOR = ROOT / "docs/experiments/014_td_extension/verify_td_extension.py"
SPEC = importlib.util.spec_from_file_location("rescue015_prior_verifier", PRIOR)
prior = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prior)
read_json, file_hash, require = prior.read_json, prior.file_hash, prior.require
UPDATES = (2009, 10000)
SEED, STEP, ACTION = 10001, 21, 0
ACTION_NAMES = ("north_small", "south_small", "east_small", "west_small",
                "north_large", "south_large", "east_large", "west_large")


def array_hash(array):
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def first_divergence(actions, demonstration):
    for index, action in enumerate(actions):
        if index >= len(demonstration) or action != int(demonstration[index]):
            return {"step": index + 1, "predicted_action": action,
                    "demonstrated_action": int(demonstration[index]) if index < len(demonstration) else None}
    return None


def verify_trace(trace, expected, observations, actions, network, *, intervene):
    """Audit all logged choices and independently infer only the known prefix."""
    rows, rollout = trace["records"], trace["rollout"]
    require(len(rows) == rollout["steps"] == len(rollout["actions"])
            and len(rollout["robot_path"]) == len(rows) + 1 and STEP <= len(rows) <= 200,
            "trace action, path and episode lengths")
    require(trace["mode"] == ("single_reference_action_then_neural" if intervene else "greedy_reproduction")
            and trace["diagnostic_only"] and trace["optimizer_updates"] == 0
            and trace["forced_action_count"] == int(intervene)
            and trace["intervention_step"] == (STEP if intervene else None)
            and trace["original_greedy_exactly_reproduced"] == (not intervene),
            "fixed trace mode and intervention count")
    require(rollout["training_scene_only"] and rollout["seed"] == SEED,
            "only the specified known training scene")
    if not intervene:
        require(rollout == expected, "unmodified greedy reproduction equals attempt 014")
    require(first_divergence(expected["actions"], actions)
            == {"step": STEP, "predicted_action": 4, "demonstrated_action": ACTION},
            "fixed original first divergence")
    forced, checked_q = 0, 0
    reward = 0.0
    for index, row in enumerate(rows):
        require(row["step"] == index + 1, "one-based sequential trace steps")
        q = np.asarray(row["q_values"], dtype=np.float64)
        require(q.shape == (8,) and np.isfinite(q).all(), "eight finite recorded Q values")
        policy = int(q.argmax())
        is_forced = intervene and index + 1 == STEP
        executed = ACTION if is_forced else policy
        require(row["policy_action"] == policy and row["executed_action"] == executed
                and row["forced_reference_action"] == is_forced,
                "only step 21 is overridden; every other action is greedy")
        require(row["policy_action_name"] == ACTION_NAMES[policy]
                and row["executed_action_name"] == ACTION_NAMES[executed],
                "action names match their indices")
        require(row["executed_action"] == rollout["actions"][index]
                and row["position"] == rollout["robot_path"][index]
                and row["next_position"] == rollout["robot_path"][index + 1],
                "step records equal the stored action and position trajectory")
        require(math.isfinite(row["reward"]), "finite recorded reward")
        require(not (row["terminated"] or row["truncated"]) if index < len(rows) - 1
                else row["terminated"] or row["truncated"], "episode stops at its first end flag")
        require(row["stationary"] == bool(np.linalg.norm(row["executed_motion"]) <= 1e-6),
                "stationary flag follows actual executed motion")
        forced += int(row["forced_reference_action"])
        reward += row["reward"]
        if index < STEP:
            known = np.asarray(observations[index], dtype=np.float32)
            require(row["encoded_observation_sha256"] == array_hash(known),
                    "common prefix states match the demonstration")
            require(row["policy_action"] == expected["actions"][index]
                    and row["position"] == expected["robot_path"][index],
                    "prefix and fork match the original greedy route")
            with torch.no_grad():
                values = network(torch.from_numpy(known.copy()).unsqueeze(0))[0].tolist()
            require(values == row["q_values"], "exact independently recomputed prefix Q values")
            checked_q += 1
    require(forced == int(intervene), "exactly one intervention in the corrected episode")
    require(reward == rollout["return"]
            and sum(row["stationary"] for row in rows) == rollout["stationary_steps"]
            and sum(row["collision_blocked"] for row in rows) == rollout["blocked_steps"],
            "episode reward, stationary and blocked totals")
    require(rows[-1]["distance_to_target"] == rollout["distance_to_target"]
            and rows[-1]["peak_normalized_deformation"] == rollout["peak_deformation"],
            "final recorded distance and deformation")
    require(rollout["first_reference_action_divergence"] == first_divergence(rollout["actions"], actions),
            "executed trajectory first demonstration mismatch")
    if intervene:
        require(rows[STEP - 1]["policy_action"] == 4 and rows[STEP - 1]["executed_action"] == ACTION,
                "the fork replaces north_large with north_small")
        require(len(rows) > STEP
                and rows[STEP]["encoded_observation_sha256"] == array_hash(
                    np.asarray(observations[STEP], dtype=np.float32)),
                "the corrected action reaches demonstration state 22")
    return {"steps": len(rows), "success": rollout["success"], "forced_actions": forced,
            "prefix_q_vectors_recomputed": checked_q,
            "all_logged_argmax_and_action_rules_valid": True}


def verify_pair(greedy, corrected, paired_fork):
    """Compare the two pre-intervention histories without rerunning the physics."""
    before, after = greedy["records"], corrected["records"]
    require(before[:STEP - 1] == after[:STEP - 1], "identical complete records before the intervention")
    fork_a, fork_b = before[STEP - 1], after[STEP - 1]
    for field in ("step", "position", "encoded_observation_sha256", "q_values",
                  "policy_action", "policy_action_name"):
        require(fork_a[field] == fork_b[field], f"identical fork decision input: {field}")
    q = np.asarray(fork_a["q_values"], dtype=np.float64)
    gap = float(q[ACTION] - np.max(q[1:]))
    expected = {
        "step": STEP, "prefix_records_equal_before_intervention": True,
        "decision_state_and_q_equal": True, "policy_action": 4,
        "greedy_executed_action": 4, "counterfactual_executed_action": ACTION,
        "signed_demonstration_gap": gap,
    }
    require(paired_fork == expected, "paired fork summary matches the two raw traces")
    require(gap < 0 and fork_a["executed_action"] == 4
            and fork_b["executed_action"] == ACTION, "single changed decision at the shared state")
    return expected


def verify_state_claim(trace, snapshot, update):
    expected = {
        "exact_snapshot_at_restore": True, "exact_snapshot_after_rollout": True,
        "absolute_td_update": update, "adam_parameter_step": 20000 + update,
        "parameter_sha256": snapshot["parameter_sha256"],
        "optimizer_state": snapshot["optimizer_summary"],
    }
    require(trace["full_training_state_checks"] == expected,
            "full-state invariance checks agree with the source snapshot")
    require(trace["parameter_sha256"] == snapshot["parameter_sha256"],
            "trace policy parameter identity")


def verify_protocol(protocol):
    expected = {
        "attempt": 15, "kind": "fixed_step_single_action_rescue", "scene_seed": SEED,
        "td_updates": list(UPDATES), "intervention_step": STEP, "intervention_action": ACTION,
        "intervention_action_name": "north_small", "original_policy_action": 4,
        "original_policy_action_name": "north_large",
        "episode_order": ["2009/greedy", "2009/counterfactual", "10000/greedy", "10000/counterfactual"],
        "episodes": 4, "optimizer_updates": 0, "online_training_steps": 0,
        "held_out_scenes_accessed": 0, "fresh_full_restore_for_each_arm": True,
        "same_environment_reset_seed": True, "epsilon": 0.0,
    }
    for key, value in expected.items():
        require(protocol[key] == value, f"fixed single-action protocol: {key}")
    require(protocol["script_sha256"] == file_hash(RUNNER), "unchanged experiment runner")


def verify_inputs(run_dir, settings):
    """Independently bind the experiment to committed 014 evidence and its chain."""
    manifest = read_json(run_dir / "input_manifest.json")
    reference, archive = Path(manifest["reference_run"]), Path(manifest["archive_dir"])
    archive_path = archive / "archive_manifest.json"
    require(archive_path.relative_to(ROOT).as_posix() == manifest["archive_manifest"],
            "source archive path")
    committed = subprocess.check_output(["git", "show", f"HEAD:{manifest['archive_manifest']}"], cwd=ROOT)
    archived = read_json(archive_path)
    require(json.loads(committed) == archived and archived["status"] == "completed"
            and hashlib.sha256(committed).hexdigest() == manifest["committed_archive_manifest_blob_sha256"]
            and file_hash(archive_path) == manifest["archive_manifest_sha256"]
            and manifest["committed_archive_verified"], "committed source archive identity")
    require(manifest["archived_files_sha256"]
            == {name: record["sha256"] for name, record in archived["files"].items()},
            "complete source archived-file map")
    source_files = {}
    for name, record in archived["files"].items():
        require(file_hash(archive / name) == record["sha256"], f"source archive hash: {name}")
        if not name.endswith("_samples.jsonl"):
            require(file_hash(reference / name) == record["sha256"], f"source raw evidence: {name}")
            source_files[name] = record["sha256"]
    for name, record in archived["raw_log_coverage"].items():
        require(file_hash(reference / name) == record["raw_sha256"]
                and len(prior.read_log(reference / name)) == record["raw_records"],
                f"complete raw source log: {name}")
        source_files[name] = record["raw_sha256"]
    source_settings = read_json(reference / "settings.json")
    source_summary = read_json(reference / "summary.json")
    require(read_json(reference / "run_status.json")["status"] == "completed"
            and read_json(reference / "reload_checks.json")["passed"],
            "source experiment and independent verification completed")
    require(settings["environment"] == source_settings["environment"]
            and settings["dqn"] == source_settings["dqn"], "unchanged environment and DQN configuration")
    prior.verify_protocol(source_settings["training"], source_settings)
    inherited, inherited_reference, original = prior.verify_inputs(reference, source_settings["training"])
    require(manifest["inherited_attempt_014_manifest"] == inherited
            and manifest["attempt_014_input_manifest_sha256"] == file_hash(reference / "input_manifest.json")
            and manifest["attempt_013_reference_run"] == str(inherited_reference)
            and manifest["original_data_run"] == str(original)
            and manifest["source_chain_checked_through_attempt_011"], "complete source inheritance chain")
    phase = source_summary["extension"]
    require(phase == read_json(reference / "td/summary.json"), "source phase summary identity")
    for name, digest in phase["checkpoint_sha256"].items():
        relative = f"td/{name}"
        require(file_hash(reference / relative) == digest, f"source model identity: {relative}")
        source_files[relative] = digest
    require(source_files == manifest["reference_files_sha256"], "complete raw source and model hash map")
    provenance = read_json(run_dir / "provenance.json")
    require(provenance["source_sha256"] == manifest["source_production_sha256"]
            == read_json(reference / "provenance.json")["source_sha256"]
            and len(provenance["source_sha256"]) == 52, "unchanged 52 production source files")
    require(settings["training"]["helper_sha256"] == manifest["helper_sha256"], "declared immutable helper map")
    for name, digest in {**manifest["source_production_sha256"], **manifest["helper_sha256"]}.items():
        require(file_hash(ROOT / name) == digest, f"source/helper hash: {name}")
    require(file_hash(PRIOR) == read_json(reference / "reload_checks.json")["verifier_sha256"],
            "unchanged previously verified source verifier")
    launch = subprocess.check_output(
        ["git", "show", f"{provenance['git_head']}:{RUNNER.relative_to(ROOT).as_posix()}"], cwd=ROOT)
    require(launch.replace(b"\r\n", b"\n") == RUNNER.read_bytes().replace(b"\r\n", b"\n"),
            "runner committed before diagnostic launch")
    dataset = read_json(reference / "dataset.json")
    data_path = original / "demonstration.npz"
    require(manifest["dataset_file"] == str(data_path)
            and file_hash(data_path) == manifest["dataset_sha256"] == dataset["sha256"],
            "original unchanged demonstration array")
    episode = next(row for row in dataset["episodes"] if row["seed"] == SEED)
    require(episode["start"] == 35 and episode["stop"] == 76, "fixed 41-step demonstration slice")
    with np.load(data_path, allow_pickle=False) as data:
        observations = data["observations"][35:76].copy()
        actions = data["actions"][35:76].copy()
    require(observations.shape == (41, 341) and int(actions[STEP - 1]) == ACTION
            and array_hash(observations[0]) == episode["initial_observation_sha256"],
            "fixed demonstration state and action identity")
    expected, snapshots = {}, {}
    by_update = {row["update"]: row for row in read_json(reference / "td/observations.json")}
    require(set(manifest["source_snapshots"]) == set(map(str, UPDATES)), "exact two source checkpoints")
    for update in UPDATES:
        relative = f"td/training_state_{update}.pt"
        snapshot = torch.load(reference / relative, map_location="cpu", weights_only=True)
        record = {"file": relative, "sha256": phase["checkpoint_sha256"][Path(relative).name],
                  "parameter_sha256": snapshot["parameter_sha256"], "absolute_td_update": update,
                  "adam_parameter_step": 20000 + update, "historical_updates_on_branch": 40000 + update}
        require(record == manifest["source_snapshots"][str(update)]
                and snapshot["update_count"] == update
                and snapshot["optimizer_summary"]["step_min"] == snapshot["optimizer_summary"]["step_max"]
                == 20000 + update, "source snapshot and preserved clocks")
        expected[update] = next(scene["rollout"] for scene in by_update[update]["scenes"] if scene["seed"] == SEED)
        require(not expected[update]["success"] and expected[update]["budget_exhausted"]
                and not expected[update]["damage"] and expected[update]["steps"] == 200,
                "two original budget failures")
        snapshots[update] = snapshot
    require(read_json(run_dir / "source_unchanged.json") == {
        "passed": True, "all_input_manifest_hashes_rechecked": True,
        "source_models_and_production_files_unchanged": True,
    }, "recorded final immutable-source recheck")
    return manifest, snapshots, expected, observations, actions


def verify(run_dir, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite verification evidence: {output}")
    torch.set_num_threads(1)
    require(read_json(run_dir / "run_status.json")["status"] == "completed", "completed four-episode diagnostic")
    settings, summary = read_json(run_dir / "settings.json"), read_json(run_dir / "summary.json")
    require(summary["protocol"] == settings["training"], "matching declared and summarized protocol")
    verify_protocol(settings["training"])
    manifest, snapshots, expected, observations, actions = verify_inputs(run_dir, settings)
    require(summary["scene_seed"] == SEED and summary["td_updates"] == list(UPDATES)
            and summary["replays"] == read_json(run_dir / "replays.json")
            and [pair["update"] for pair in summary["replays"]] == list(UPDATES),
            "exact paired replay set and scene")
    checked, counts, forced = [], {"greedy": 0, "counterfactual": 0}, 0
    for pair in summary["replays"]:
        update, traces = pair["update"], {}
        snapshot = snapshots[update]
        network = build_q_network(observations.shape[1], 8, DQNConfig(**snapshot["dqn_config"]))
        network.load_state_dict(snapshot["online"])
        network.eval()
        vector = torch.nn.utils.parameters_to_vector(network.parameters()).detach().numpy()
        require(array_hash(vector) == snapshot["parameter_sha256"], "independent network parameter identity")
        decisions = {}
        for mode, intervene in (("greedy", False), ("counterfactual", True)):
            entry = pair[mode]
            require(entry["trace_file"] == f"update_{update}_{mode}.json"
                    and entry["steps_file"] == f"update_{update}_{mode}_steps.jsonl",
                    "fixed per-arm trace paths")
            trace_path, steps_path = run_dir / entry["trace_file"], run_dir / entry["steps_file"]
            require(file_hash(trace_path) == entry["trace_sha256"]
                    and file_hash(steps_path) == entry["steps_sha256"], "trace and raw-step hashes")
            trace = read_json(trace_path)
            require(trace["records"] == prior.read_log(steps_path), "complete raw step log equals trace records")
            brief = {key: value for key, value in trace.items() if key != "records"}
            require(brief == {key: value for key, value in entry.items()
                              if key not in {"trace_file", "steps_file", "trace_sha256", "steps_sha256"}},
                    "paired summary exactly matches complete trace")
            verify_state_claim(trace, snapshot, update)
            decisions[mode] = verify_trace(trace, expected[update], observations, actions, network, intervene=intervene)
            traces[mode] = trace
            counts[mode] += len(trace["records"])
            forced += trace["forced_action_count"]
        fork = verify_pair(traces["greedy"], traces["counterfactual"], pair["paired_fork"])
        require(pair["single_action_rescued_success"] == traces["counterfactual"]["rollout"]["success"],
                "rescue label follows the actual corrected outcome")
        checked.append({"update": update, "fork": fork, "arms": decisions,
                        "single_action_rescued_success": pair["single_action_rescued_success"]})
    accounting = read_json(run_dir / "accounting.json")
    require(accounting == summary["accounting"], "matching completed accounting")
    expected_counts = {
        "optimizer_updates": 0, "started_episodes": 4, "completed_episodes": 4,
        "greedy_environment_steps": counts["greedy"], "counterfactual_environment_steps": counts["counterfactual"],
        "environment_steps": sum(counts.values()), "forced_actions": 2, "online_training_steps": 0,
        "demonstration_collection_steps": 0, "held_out_scenes_accessed": 0,
        "inherited_optimizer_updates_by_checkpoint": {str(update): 40000 + update for update in UPDATES},
    }
    for key, value in expected_counts.items():
        require(accounting[key] == value, f"actual diagnostic accounting: {key}")
    require(forced == 2 and counts["greedy"] == 400, "exact two interventions and two original failures")
    rescued = sum(pair["single_action_rescued_success"] for pair in checked)
    require(summary["rescued_checkpoint_count"] == rescued and summary["tested_checkpoint_count"] == 2,
            "observed rescue count")
    result = {
        "passed": True, "attempt": 15, "verifier_sha256": file_hash(Path(__file__)),
        "prior_verifier_sha256": file_hash(PRIOR), "input_manifest_sha256": file_hash(run_dir / "input_manifest.json"),
        "reference_run": manifest["reference_run"], "checked_pairs": checked,
        "main_episodes_checked": 4, "main_environment_actions_checked": sum(counts.values()),
        "independent_prefix_q_vectors": sum(arm["prefix_q_vectors_recomputed"] for pair in checked
                                            for arm in pair["arms"].values()),
        "additional_optimizer_updates": 0, "additional_evaluation_episodes": 0,
        "additional_environment_steps": 0, "additional_environment_resets": 0,
        "scope": "All recorded argmax/actions, paired prefix, source hashes and accounting audited. Only demonstration-prefix steps 1-21 receive independent Q inference; no off-route physics or full off-route Q replay is performed.",
        "state_invariance_scope": "The committed runner asserted exact full-state equality before and after every arm. Recorded assertions are checked against the immutable source snapshot; no additional episode is executed.",
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
    print(json.dumps({key: result[key] for key in ("passed", "main_episodes_checked",
                                                  "main_environment_actions_checked",
                                                  "independent_prefix_q_vectors",
                                                  "additional_environment_steps")}))


if __name__ == "__main__":
    main()

