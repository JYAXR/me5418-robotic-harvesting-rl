"""Inspect stored policies and test one-action interventions without any learning."""

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from me5418_harvesting.agents.dqn_agent import DQNAgent
from me5418_harvesting.agents.run_record import begin_run, finish_run
from me5418_harvesting.agents.training import episode_metrics

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "docs/experiments/011_multiscene_retention/evidence"
SEED = 10001
CHECKPOINTS = (*range(11), 20, 50, 100, 200, 500, 1000, 2000)
FAILED_UPDATES = (1, 6, 9, 10, 200)
MARGIN = 0.8
ACTION_NAMES = (
    "north_small", "south_small", "east_small", "west_small",
    "north_large", "south_large", "east_large", "west_large",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(value):
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def parameter_hash(agent):
    vector = torch.nn.utils.parameters_to_vector(agent.learner.online.parameters())
    return array_hash(vector.detach().cpu().numpy())


def require_hash(path, expected):
    actual = file_hash(path)
    if actual != expected:
        raise ValueError(f"Input hash mismatch: {path}")
    return actual


def verify_reference(reference, archive=ARCHIVE, root=ROOT):
    """Bind local inputs to the archived attempt; never edit or rerun that attempt."""
    manifest = read_json(archive / "archive_manifest.json")
    committed_path = (archive / "archive_manifest.json").relative_to(root).as_posix()
    committed_bytes = subprocess.check_output(
        ["git", "-C", str(root), "show", f"HEAD:{committed_path}"],
        stderr=subprocess.PIPE, timeout=10,
    )
    if json.loads(committed_bytes) != manifest:
        raise ValueError("The local archive manifest differs from committed Git HEAD evidence.")
    names = (
        "dataset.json", "provenance.json", "input_manifest.json", "settings.json",
        "calibration/summary.json", "preserve/td/summary.json", "preserve/td/observations.json",
    )
    verified = {}
    for name in names:
        expected = manifest["files"][name]["sha256"]
        require_hash(archive / name, expected)
        verified[name] = require_hash(reference / name, expected)
    provenance = read_json(reference / "provenance.json")
    inputs = read_json(reference / "input_manifest.json")
    source_hashes = {**provenance["source_sha256"], **inputs["config_sha256"],
                     **inputs["helper_sha256"]}
    for name, expected in source_hashes.items():
        require_hash(root / name, expected)
    dataset = read_json(reference / "dataset.json")
    verified["demonstration.npz"] = require_hash(reference / "demonstration.npz", dataset["sha256"])
    baseline = read_json(reference / "calibration/summary.json")
    stage = read_json(reference / "preserve/td/summary.json")
    checkpoints = {"calibration/update_20000.pt": baseline["checkpoint_sha256"]["update_20000.pt"]}
    checkpoints.update({f"preserve/td/update_{update}.pt": stage["checkpoint_sha256"][f"update_{update}.pt"]
                        for update in CHECKPOINTS})
    for name, expected in checkpoints.items():
        verified[name] = require_hash(reference / name, expected)
    return {
        "reference_run": str(reference.resolve()), "archive": str(archive.resolve()),
        "archive_manifest_sha256": file_hash(archive / "archive_manifest.json"),
        "committed_archive_manifest_blob_sha256": hashlib.sha256(committed_bytes).hexdigest(),
        "reference_git_head": provenance["git_head"], "verified_files": verified,
        "verified_source_files": source_hashes, "source_file_count": len(source_hashes),
        "checkpoint_count": len(checkpoints), "dataset": dataset,
    }


def single_state_q(agent, encoded):
    """Match DQNAgent.act's float32 batch-of-one inference, including tie order."""
    value = np.asarray(encoded, dtype=np.float32)
    if value.ndim != 1:
        raise ValueError("A diagnostic forward pass requires exactly one encoded state.")
    agent.learner.online.eval()
    with torch.no_grad():
        scores = agent.learner.online(torch.from_numpy(value.copy()).unsqueeze(0))[0]
    if scores.shape != (8,) or not torch.isfinite(scores).all():
        raise ValueError("Expected eight finite action values.")
    return scores.detach().cpu().numpy().copy()


def q_metrics(values, demonstrated_action):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (8,) or not np.isfinite(values).all():
        raise ValueError("Expected eight finite action values.")
    chosen = int(values.argmax())
    other = values.copy()
    other[demonstrated_action] = -np.inf
    competitor = int(other.argmax())
    signed_margin = float(values[demonstrated_action] - values[competitor])
    ranked = np.sort(values)
    return {
        "q_values": values.tolist(), "chosen_action": chosen,
        "chosen_action_name": ACTION_NAMES[chosen], "demonstrated_action": int(demonstrated_action),
        "demonstrated_action_name": ACTION_NAMES[demonstrated_action],
        "strongest_other_action": competitor, "strongest_other_action_name": ACTION_NAMES[competitor],
        "signed_demonstration_margin": signed_margin,
        "top1_minus_top2": float(ranked[-1] - ranked[-2]),
        "hinge_margin_shortfall": max(0.0, MARGIN - signed_margin),
        "demonstration_is_greedy": chosen == demonstrated_action,
    }


def first_divergence(actions, demonstrated):
    """Locate a reference mismatch without assigning it a causal failure label."""
    for index, action in enumerate(actions):
        if index >= len(demonstrated) or action != int(demonstrated[index]):
            return {"step": index + 1, "predicted_action": action,
                    "demonstrated_action": int(demonstrated[index]) if index < len(demonstrated) else None}
    return None


def label_context(observations, actions, index):
    previous = int(actions[index - 1]) if index else None
    following = int(actions[index + 1]) if index + 1 < len(actions) else None
    opposite = np.flatnonzero(actions != actions[index])
    distances = np.linalg.norm(observations[opposite] - observations[index], axis=1)
    nearest = int(opposite[int(distances.argmin())]) if len(opposite) else None
    return {
        "previous_action": previous, "next_action": following,
        "label_transition_from_previous": previous is not None and previous != int(actions[index]),
        "previous_state_l2_distance": float(np.linalg.norm(observations[index] - observations[index - 1]))
        if index else None,
        "nearest_different_label_step": nearest + 1 if nearest is not None else None,
        "nearest_different_label_action": int(actions[nearest]) if nearest is not None else None,
        "nearest_different_label_l2_distance": float(distances.min()) if len(distances) else None,
        "neighbor_labels": [
            {"step": i + 1, "action": int(actions[i]), "name": ACTION_NAMES[int(actions[i])]}
            for i in range(max(0, index - 2), min(len(actions), index + 3))
        ],
    }


def build_q_analysis(reference, inputs):
    """Read all sampled policies without calling reset, step, or an optimizer."""
    data = np.load(reference / "demonstration.npz", allow_pickle=False)
    full_observations, full_actions = data["observations"].copy(), data["actions"].copy()
    data.close()
    episode = next(row for row in inputs["dataset"]["episodes"] if row["seed"] == SEED)
    observations = full_observations[episode["start"]:episode["stop"]]
    actions = full_actions[episode["start"]:episode["stop"]]
    if array_hash(observations[0]) != episode["initial_observation_sha256"]:
        raise ValueError("The scene episode offset does not match its recorded initial state.")
    unique, inverse, counts = np.unique(full_observations, axis=0, return_inverse=True, return_counts=True)
    conflicts = sum(len(np.unique(full_actions[inverse == i])) > 1 for i in range(len(unique)))
    baseline, baseline_env, _ = DQNAgent.load(reference / "calibration/update_20000.pt")
    baseline_hash = parameter_hash(baseline)
    try:
        baseline_q = [single_state_q(baseline, state) for state in observations]
    finally:
        baseline_env.close()
    rows = read_json(reference / "preserve/td/observations.json")
    if tuple(row["update"] for row in rows) != CHECKPOINTS:
        raise ValueError("The preserved checkpoint schedule differs from attempt 011.")
    checkpoints = []
    for row in rows:
        update = row["update"]
        rollout = next(scene["rollout"] for scene in row["scenes"] if scene["seed"] == SEED)
        divergence = first_divergence(rollout["actions"], actions)
        if divergence != rollout["first_reference_action_divergence"]:
            raise ValueError(f"Stored first divergence disagrees with actual actions at update {update}.")
        agent, env, _ = DQNAgent.load(reference / f"preserve/td/update_{update}.pt")
        before = parameter_hash(agent)
        try:
            states = []
            for index, observation in enumerate(observations):
                current = q_metrics(single_state_q(agent, observation), int(actions[index]))
                initial = q_metrics(baseline_q[index], int(actions[index]))
                states.append({"step": index + 1, "dataset_row": episode["start"] + index,
                               "encoded_observation_sha256": array_hash(observation),
                               **current, "baseline_same_state": initial,
                               "margin_change_from_calibration": current["signed_demonstration_margin"]
                               - initial["signed_demonstration_margin"],
                               "label_context": label_context(observations, actions, index)})
            divergent = states[divergence["step"] - 1] if divergence else None
            if divergent and divergent["chosen_action"] != divergence["predicted_action"]:
                raise ValueError(f"Batch-of-one inference differs from stored action at update {update}.")
            if before != parameter_hash(agent) or agent.learner.update_count != 0:
                raise RuntimeError("Q analysis unexpectedly changed the policy.")
        finally:
            env.close()
        margins = [state["signed_demonstration_margin"] for state in states]
        checkpoints.append({
            "update": update, "parameter_sha256": before, "stored_rollout": rollout,
            "first_divergence": divergent, "states": states,
            "margins_summary": {
                "positive_count": sum(value > 0 for value in margins),
                "zero_count": sum(value == 0 for value in margins),
                "negative_count": sum(value < 0 for value in margins),
                "negative_steps": [i + 1 for i, value in enumerate(margins) if value < 0],
                "minimum": min(margins), "mean": float(np.mean(margins)),
                "below_required_margin_count": sum(value < MARGIN for value in margins),
            },
        })
    failures = tuple(row["update"] for row in checkpoints if not row["stored_rollout"]["success"])
    if failures != FAILED_UPDATES:
        raise ValueError("The failed checkpoint set differs from the declared diagnostic scope.")
    if checkpoints[0]["parameter_sha256"] != baseline_hash:
        raise ValueError("The initial preserved policy differs from the calibrated baseline.")
    return {
        "scene_seed": SEED, "scene_episode": episode, "action_names": ACTION_NAMES,
        "dataset_sha256": inputs["dataset"]["sha256"], "margin_requirement": MARGIN,
        "inference_batch_size": 1, "optimizer_updates": 0, "environment_steps": 0,
        "baseline_parameter_sha256": baseline_hash,
        "duplicate_observation_rows": int(np.sum(counts - 1)),
        "conflicting_duplicate_label_groups": conflicts, "checkpoints": checkpoints,
        "interpretation": "A reference-action mismatch is not necessarily harmful; successful deviations are retained.",
    }, observations, actions


def motion_characteristics(path, records):
    """Report position recurrences, not repeated full states (the budget changes)."""
    positions = np.asarray(path, dtype=np.float64)
    quantized = [tuple(np.round(position, 6)) for position in positions]
    visits = Counter(quantized)
    stationary = [bool(row["stationary"]) for row in records]
    longest, run = 0, 0
    for stopped in stationary:
        run = run + 1 if stopped else 0
        longest = max(longest, run)
    returns = [i for i in range(2, len(positions))
               if np.linalg.norm(positions[i] - positions[i - 2]) <= 1e-6
               and np.linalg.norm(positions[i] - positions[i - 1]) > 1e-6]
    first_seen, first_repeat = {}, None
    for index, position in enumerate(quantized):
        if position in first_seen and first_repeat is None:
            first_repeat = {"position_index": index, "first_position_index": first_seen[position]}
        first_seen.setdefault(position, index)
    return {
        "position_tolerance": 1e-6, "position_quantization_decimals": 6,
        "unique_quantized_positions": len(visits),
        "repeated_position_visits": len(quantized) - len(visits),
        "maximum_position_visits": max(visits.values()), "first_repeated_position": first_repeat,
        "longest_stationary_run": longest, "two_step_position_returns": returns,
        "note": "Position recurrence alone does not imply an identical encoded state or a permanent cycle.",
    }


def trace_policy(agent, env, observations, actions, expected, *, intervene=False, progress=None):
    """Force at most one reference action, then immediately resume the neural policy."""
    divergence = first_divergence(expected["actions"], actions)
    if divergence is None or divergence["demonstrated_action"] is None:
        raise ValueError("The requested intervention needs a divergence within the demonstration.")
    fork_index = divergence["step"] - 1
    observation, _ = env.reset(seed=SEED)
    if not np.array_equal(agent.encoder(observation), observations[0]):
        raise ValueError("The reset does not match the recorded demonstration scene.")
    if observation[:2].tolist() != expected["robot_path"][0]:
        raise ValueError("The reset position differs from the original rollout.")
    before = parameter_hash(agent)
    before_rng = json.dumps(agent.rng.bit_generator.state, sort_keys=True)
    before_updates = agent.learner.update_count
    before_optimizer_states = len(agent.learner.optimizer.state)
    executed, path, records = [], [observation[:2].tolist()], []
    reward_sum, stationary, blocked, forced_count = 0.0, 0, 0, 0
    for index in range(env.max_steps):
        encoded = agent.encoder(observation)
        if index <= fork_index and not np.array_equal(encoded, observations[index]):
            raise RuntimeError("The common prefix state differs from the demonstration.")
        values = single_state_q(agent, encoded)
        policy_action = agent.act(observation, epsilon=0.0)
        if policy_action != int(values.argmax()):
            raise RuntimeError("Diagnostic inference differs from DQNAgent.act.")
        if index <= fork_index or not intervene:
            if index >= len(expected["actions"]) or policy_action != expected["actions"][index]:
                raise RuntimeError("The replay policy action differs from the original trajectory.")
            if observation[:2].tolist() != expected["robot_path"][index]:
                raise RuntimeError("The replay position differs from the original trajectory.")
        forced = intervene and index == fork_index
        action = int(actions[index]) if forced else policy_action
        forced_count += int(forced)
        following, reward, terminated, truncated, info = env.step(action)
        stopped = bool(np.linalg.norm(info["executed_motion"]) <= 1e-6)
        record = {
            "step": index + 1, "position": observation[:2].tolist(),
            "next_position": following[:2].tolist(), "encoded_observation_sha256": array_hash(encoded),
            "q_values": values.tolist(), "policy_action": policy_action,
            "policy_action_name": ACTION_NAMES[policy_action], "executed_action": action,
            "executed_action_name": ACTION_NAMES[action], "forced_reference_action": forced,
            "reward": float(reward), "terminated": bool(terminated), "truncated": bool(truncated),
            "stationary": stopped, "collision_blocked": bool(info["collision_blocked"]),
            "executed_motion": np.asarray(info["executed_motion"]).tolist(),
            "distance_to_target": float(info["distance_to_target"]),
            "peak_normalized_deformation": float(info["peak_normalized_deformation"]),
        }
        if progress:
            progress(record)
        records.append(record)
        executed.append(action)
        path.append(following[:2].tolist())
        reward_sum += float(reward)
        stationary += int(stopped)
        blocked += int(info["collision_blocked"])
        if not intervene and path[-1] != expected["robot_path"][index + 1]:
            raise RuntimeError("The replay next position differs from the original trajectory.")
        if (forced and index + 1 < len(observations)
                and not np.array_equal(agent.encoder(following), observations[index + 1])):
            raise RuntimeError("The forced action did not reach the next demonstration state.")
        observation = following
        if terminated or truncated:
            break
    else:
        raise RuntimeError("The diagnostic rollout exceeded the declared episode budget.")
    result = {
        "training_scene_only": True, "seed": SEED,
        **episode_metrics(reward_sum, len(executed), info, stationary, blocked),
        "first_reference_action_divergence": first_divergence(executed, actions),
        "actions": executed, "robot_path": path,
    }
    if not intervene and result != expected:
        raise RuntimeError("The greedy replay differs from the original recorded outcome.")
    if forced_count != int(intervene):
        raise RuntimeError("The intervention did not execute exactly its declared single action.")
    if (parameter_hash(agent) != before or agent.learner.update_count != before_updates
            or len(agent.learner.optimizer.state) != before_optimizer_states
            or json.dumps(agent.rng.bit_generator.state, sort_keys=True) != before_rng):
        raise RuntimeError("Inference changed parameters, optimizer state, or exploration state.")
    return {
        "mode": "single_reference_action_then_neural" if intervene else "greedy_reproduction",
        "diagnostic_only": True, "optimizer_updates": 0, "forced_action_count": forced_count,
        "intervention_step": fork_index + 1 if intervene else None,
        "parameter_sha256": before, "rollout": result, "records": records,
        "motion_characteristics": motion_characteristics(path, records),
        "original_greedy_exactly_reproduced": not intervene,
    }


def run(reference, output, *, q_only=False):
    output.mkdir(parents=True, exist_ok=False)
    accounting = {"optimizer_updates": 0, "completed_episodes": 0,
                  "greedy_environment_steps": 0, "counterfactual_environment_steps": 0,
                  "environment_steps": 0, "held_out_scenes_accessed": 0,
                  "note": "Reset feasibility/planner probes are excluded; every diagnostic env.step action is counted."}
    started = False
    try:
        settings = read_json(reference / "settings.json")
        begin_run(output, settings["environment"], settings["dqn"], {
            "kind": "read_only_divergence_diagnostic", "reference_run": str(reference.resolve()),
            "scene_seed": SEED, "failed_updates": FAILED_UPDATES, "all_checkpoints": CHECKPOINTS,
            "q_only": q_only, "script_sha256": file_hash(Path(__file__)),
            "counterfactual": "Override only the first divergent action, then resume the greedy policy.",
        })
        started = True
        torch.set_num_threads(1)
        inputs = verify_reference(reference)
        save_json(output / "verified_inputs.json", inputs)
        analysis, observations, actions = build_q_analysis(reference, inputs)
        save_json(output / "q_analysis.json", analysis)
        for row in analysis["checkpoints"]:
            divergent = row["first_divergence"]
            print(json.dumps({"update": row["update"], "success": row["stored_rollout"]["success"],
                              "divergence_step": divergent["step"] if divergent else None,
                              "signed_margin": divergent["signed_demonstration_margin"] if divergent else None}), flush=True)
        replay_summaries = []
        if not q_only:
            for row in analysis["checkpoints"]:
                update = row["update"]
                if update not in FAILED_UPDATES:
                    continue
                checkpoint = reference / f"preserve/td/update_{update}.pt"
                agent, env, _ = DQNAgent.load(checkpoint)
                pair = {"update": update}
                try:
                    for intervene in (False, True):
                        mode = "counterfactual" if intervene else "greedy"
                        trace_path = output / f"update_{update}_{mode}_steps.jsonl"
                        with trace_path.open("x", encoding="utf-8") as trace_file:
                            def progress(record, current_mode=mode):
                                accounting[f"{current_mode}_environment_steps"] += 1
                                accounting["environment_steps"] += 1
                                trace_file.write(json.dumps(record, allow_nan=False) + "\n")
                                trace_file.flush()

                            trace = trace_policy(agent, env, observations, actions, row["stored_rollout"],
                                                 intervene=intervene, progress=progress)
                        accounting["completed_episodes"] += 1
                        save_json(output / f"update_{update}_{mode}.json", trace)
                        pair[mode] = {key: value for key, value in trace.items() if key != "records"}
                        pair[mode]["steps_file_sha256"] = file_hash(trace_path)
                        save_json(output / "accounting.json", accounting)
                    pair["single_action_rescued_success"] = pair["counterfactual"]["rollout"]["success"]
                    replay_summaries.append(pair)
                    save_json(output / "replays.json", replay_summaries)
                    print(json.dumps({"update": update, "single_action_rescued_success": pair["single_action_rescued_success"],
                                      "counterfactual_steps": pair["counterfactual"]["rollout"]["steps"]}), flush=True)
                finally:
                    env.close()
        # Recheck all frozen inputs after the diagnostic, including every loaded model file.
        verify_reference(reference)
        save_json(output / "accounting.json", accounting)
        save_json(output / "summary.json", {
            "scene_seed": SEED, "q_analysis_sha256": file_hash(output / "q_analysis.json"),
            "failed_updates": FAILED_UPDATES, "replays": replay_summaries, "accounting": accounting,
            "interpretation": "Counterfactual success tests one action on this known scene; it is not a deployed policy or held-out evaluation.",
        })
        finish_run(output, "completed")
    except BaseException as error:
        save_json(output / "accounting.json", accounting)
        if started:
            finish_run(output, "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--q-only", action="store_true", help="Inspect stored Q values without any environment steps.")
    args = parser.parse_args()
    run(args.reference_run, args.output_dir, q_only=args.q_only)


if __name__ == "__main__":
    main()
