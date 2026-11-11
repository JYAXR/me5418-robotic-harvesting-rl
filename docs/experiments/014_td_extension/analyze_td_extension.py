"""Audit sampled retention and dense reference-state rankings without new rollouts."""

import argparse
import importlib.util
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = Path(__file__).resolve().parent
PREVIOUS = ROOT / "docs/experiments/012_divergence_control"
START, END = 2000, 10000
SEEDS = (10000, 10001, 10002)
FOCAL_SEED, FOCAL_STEPS = 10001, (6, 18, 22, 24)
CHECKPOINTS = (START, *range(2001, 2011), 2200, 2201, *range(2500, END + 1, 500))
HISTORICAL_CHECKPOINTS = (*range(11), 20, 50, 100, 200, 500, 1000, 2000)


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helper = import_file("extension014_q_helpers", PREVIOUS / "analyze_control_q.py")
diagnostic = helper.diagnostic
read_json, file_hash, require = helper.read_json, helper.file_hash, helper.require


def validate_protocol(protocol):
    """Reject a different objective, resume boundary, state or observation schedule."""
    expected = {
        "attempt": 14, "kind": "exact_low_rate_td_continuation",
        "initial_td_update": START, "final_td_update": END, "extension_updates": END - START,
        "inherited_optimizer_updates": 42000, "learning_rate": 0.0001,
        "demonstration_weight": 1.0, "margin": 0.8, "target_update_interval": 200,
        "effective_batch_size": 117, "full_batch_weighting": "equal_per_transition",
        "early_stopping": False, "resume_source": "attempt_013_treatment_final_training_state",
        "preserve_adam_moments_and_parameter_steps": True, "preserve_absolute_td_counter": True,
        "synchronize_target_at_resume": False, "sentinel_seed": FOCAL_SEED,
        "sentinel_batch_size": 1, "sentinel_is_physical_evaluation": False,
        "online_training_steps": 0, "held_out_scenes_accessed": 0,
    }
    require(all(protocol.get(name) == value for name, value in expected.items()),
            "The declared continuation protocol changes a fixed condition.")
    require(tuple(protocol["scene_seeds"]) == SEEDS and tuple(protocol["td_checkpoints"]) == CHECKPOINTS
            and tuple(protocol["sentinel_reference_steps"]) == FOCAL_STEPS
            and tuple(protocol["target_source_updates"]) == tuple(range(2400, 9401, 1000)),
            "The declared scene, checkpoint or sentinel schedule differs.")


def intervals(updates, mask, gaps, checkpoints):
    """Describe contiguous reference-state events, never inferred rollout outcomes."""
    require(len(updates) == len(mask) == len(gaps), "Interval inputs have different lengths.")
    result, current = [], []

    def finish():
        if not current:
            return
        indices = np.asarray(current, dtype=int)
        begin, end = int(updates[indices[0]]), int(updates[indices[-1]])
        minimum = int(indices[int(np.argmin(np.asarray(gaps)[indices]))])
        observed = [int(step) for step in checkpoints if begin <= step <= end]
        preceding = [int(step) for step in checkpoints if step < begin]
        following = [int(step) for step in checkpoints if step > end]
        result.append({
            "start_update": begin, "end_update": end, "update_count": len(current),
            "minimum_gap": float(gaps[minimum]), "minimum_gap_update": int(updates[minimum]),
            "start_gap": float(gaps[indices[0]]), "end_gap": float(gaps[indices[-1]]),
            "scheduled_physical_checkpoints_within_interval": observed,
            "preceding_physical_checkpoint": preceding[-1] if preceding else None,
            "following_physical_checkpoint": following[0] if following else None,
            "entirely_between_physical_checkpoints": not observed,
            "classification": "REFERENCE-STATE RANK REVERSAL",
            "actual_failed_rollout_inferred": False,
        })
        current.clear()

    for index, selected in enumerate(mask):
        if current and updates[index] != updates[current[-1]] + 1:
            finish()
        if selected:
            current.append(index)
        else:
            finish()
    finish()
    return result


def sentinel_series_summary(updates, gaps, greedy_actions, demonstrated_action, checkpoints):
    """Distinguish strict negative gaps from argmax disagreements at exact ties."""
    gaps = np.asarray(gaps, dtype=np.float64)
    chosen = np.asarray(greedy_actions, dtype=int)
    require(len(updates) == len(gaps) == len(chosen) and len(updates) > 0,
            "Sentinel series must be nonempty and aligned.")
    require(np.isfinite(gaps).all(), "Sentinel gaps must be finite.")
    negative, mismatches = gaps < 0, chosen != demonstrated_action
    minimum = int(gaps.argmin())
    maximum = int(gaps.argmax())
    negative_indices = np.flatnonzero(negative)
    mismatch_indices = np.flatnonzero(mismatches)
    transitions = []
    for index in range(1, len(updates)):
        if np.sign(gaps[index]) != np.sign(gaps[index - 1]) or chosen[index] != chosen[index - 1]:
            transitions.append({
                "previous_update": int(updates[index - 1]), "update": int(updates[index]),
                "previous_gap": float(gaps[index - 1]), "gap": float(gaps[index]),
                "previous_greedy_action": int(chosen[index - 1]), "greedy_action": int(chosen[index]),
                "observed_at_physical_checkpoint": updates[index] in checkpoints,
                "classification": "REFERENCE-STATE RANK TRANSITION",
                "actual_failed_rollout_inferred": False,
            })
    negative_intervals = intervals(updates, negative, gaps, checkpoints)
    return {
        "demonstrated_action": int(demonstrated_action),
        "records_including_resume_boundary": len(updates),
        "negative_gap_count_including_resume_boundary": int(negative.sum()),
        "negative_gap_count_new_updates": int(negative[1:].sum()),
        "zero_gap_count": int((gaps == 0).sum()),
        "greedy_disagreement_count": int(mismatches.sum()),
        "greedy_disagreement_at_zero_gap_count": int(((gaps == 0) & mismatches).sum()),
        "below_margin_0_8_count": int((gaps < 0.8).sum()),
        "first_negative_gap_update": int(updates[negative_indices[0]]) if len(negative_indices) else None,
        "first_greedy_disagreement_update": int(updates[mismatch_indices[0]]) if len(mismatch_indices) else None,
        "minimum_gap": float(gaps[minimum]), "minimum_gap_update": int(updates[minimum]),
        "maximum_gap": float(gaps[maximum]), "maximum_gap_update": int(updates[maximum]),
        "negative_gap_intervals": negative_intervals,
        "negative_intervals_entirely_between_physical_checks": sum(
            row["entirely_between_physical_checkpoints"] for row in negative_intervals),
        "greedy_disagreement_intervals": intervals(updates, mismatches, gaps, checkpoints),
        "rank_transitions": transitions,
        "signed_demonstration_gaps": gaps.tolist(), "greedy_actions": chosen.tolist(),
        "interpretation": "Reference-state rankings only; a negative gap or different greedy action does not establish an actual failed rollout.",
    }


def analyze_sentinels(path, observations, actions, episode, checkpoint_q,
                      start=START, end=END, checkpoints=CHECKPOINTS):
    """Recompute each recorded gap and compare every saved checkpoint to batch-one Q."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    updates = list(range(start, end + 1))
    require([row["update"] for row in rows] == updates, "Dense sentinel update coverage differs.")
    series = {step: {"gaps": [], "actions": []} for step in FOCAL_STEPS}
    crosschecks = 0
    for row in rows:
        require(row["extension_update"] == row["update"] - start and row["scene_seed"] == FOCAL_SEED,
                "A sentinel has a reset clock or a different scene.")
        require(tuple(state["reference_step"] for state in row["states"]) == FOCAL_STEPS,
                "The fixed reference-state set differs.")
        for state in row["states"]:
            step = state["reference_step"]
            index = episode["start"] + step - 1
            require(state["global_transition_index"] == index
                    and state["demonstrated_action"] == int(actions[index]),
                    "A sentinel refers to a different demonstrated state or action.")
            metrics = diagnostic.q_metrics(state["q_values"], state["demonstrated_action"])
            require(metrics["chosen_action"] == state["greedy_action"]
                    and metrics["signed_demonstration_margin"] == state["signed_demonstration_gap"],
                    "A stored sentinel gap or greedy action disagrees with its raw Q values.")
            if row["update"] in checkpoint_q:
                require(np.array_equal(np.asarray(state["q_values"]), checkpoint_q[row["update"]][index]),
                        "A dense sentinel differs from independently reloaded checkpoint Q values.")
                crosschecks += 1
            series[step]["gaps"].append(metrics["signed_demonstration_margin"])
            series[step]["actions"].append(metrics["chosen_action"])
    details = {}
    for step, values in series.items():
        index = episode["start"] + step - 1
        details[str(step)] = {
            "reference_step": step, "global_transition_index": index,
            "encoded_observation_sha256": diagnostic.array_hash(observations[index]),
            "role": "Known successful reference-action deviation control" if step == 24
            else "Previously implicated failure reference state",
            **sentinel_series_summary(updates, values["gaps"], values["actions"],
                                      int(actions[index]), checkpoints),
        }
    return {
        "raw_file": str(path.resolve()), "raw_sha256": file_hash(path), "raw_records": len(rows),
        "updates": updates, "new_update_records": end - start, "resume_boundary_records": 1,
        "inference_batch_size": 1, "reloaded_checkpoint_state_crosschecks": crosschecks,
        "reference_states": details, "new_environment_actions": 0,
    }


def update_statistics(path, clip_norm, start=START, end=END):
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    require([row["update"] for row in records] == list(range(start + 1, end + 1)),
            "The continuation update log is incomplete or its absolute clock differs.")
    require(all(row["extension_update"] == row["update"] - start for row in records),
            "The extension update clock differs from the absolute TD clock.")
    fields = ("loss_before", "loss_after", "td_loss_before", "td_loss_after",
              "demonstration_loss_before", "demonstration_loss_after", "gradient_norm", "parameter_delta_norm")
    statistics = {}
    for name in fields:
        values = np.asarray([row[name] for row in records], dtype=np.float64)
        require(np.isfinite(values).all(), f"The update log has non-finite {name} values.")
        statistics[name] = {"mean": float(values.mean()), "minimum": float(values.min()),
                            "maximum": float(values.max())}
    require(all(row["parameter_delta_norm"] >= 0 for row in records),
            "Parameter displacement norms cannot be negative.")
    return {
        "raw_sha256": file_hash(path), "records": len(records), "statistics": statistics,
        "first_update": records[0], "last_update": records[-1],
        "nonzero_parameter_update_count": sum(row["parameter_delta_norm"] > 0 for row in records),
        "gradient_clip_norm": clip_norm,
        "gradient_norm_above_clip_count": sum(row["gradient_norm"] > clip_norm for row in records),
        "gradient_norm_semantics": "Pre-clipping global norm returned by clip_grad_norm_.",
        "component_loss_semantics": "TD and demonstration losses are unweighted; demonstration weight remains 1 and margin remains 0.8.",
    }


def retention_summary(rows):
    require(all(tuple(scene["seed"] for scene in row["scenes"]) == SEEDS for row in rows),
            "Retention observations omit or reorder scenes.")
    failures = {str(seed): [row["update"] for row in rows if not next(
        scene["rollout"]["success"] for scene in row["scenes"] if scene["seed"] == seed)] for seed in SEEDS}
    return {
        "checkpoint_count": len(rows), "scene_check_count": len(rows) * len(SEEDS),
        "all_scene_success_checkpoint_count": sum(all(scene["rollout"]["success"]
                                                       for scene in row["scenes"]) for row in rows),
        "successful_scene_check_count": sum(scene["rollout"]["success"] for row in rows for scene in row["scenes"]),
        "failed_updates_by_scene": failures,
        "first_sampled_failure_by_scene": {seed: steps[0] if steps else None for seed, steps in failures.items()},
        "sampled_updates": [row["update"] for row in rows],
        "recorded_actions_across_these_observations": sum(scene["rollout"]["steps"] for row in rows for scene in row["scenes"]),
    }


def combined_retention(historical, continuation):
    require(tuple(row["update"] for row in historical) == HISTORICAL_CHECKPOINTS,
            "The attempt-013 treatment observation schedule differs.")
    require(tuple(row["update"] for row in continuation) == CHECKPOINTS,
            "The attempt-014 physical observation schedule differs.")
    require(historical[-1]["scenes"] == continuation[0]["scenes"]
            and historical[-1]["aggregate"] == continuation[0]["aggregate"],
            "The repeated update-2000 physical evaluation differs from its source.")
    combined = historical[:-1] + continuation
    require(len(combined) == 46, "Expected 46 distinct historical-plus-continuation checkpoints.")
    return {
        "historical_attempt_013": retention_summary(historical),
        "new_attempt_014_evaluations": retention_summary(continuation),
        "combined_distinct_updates": retention_summary(combined),
        "repeated_boundary_update": START,
        "accounting_note": "Only the 87 attempt-014 episodes are newly executed here, including the repeated update-2000 evaluation. The 138 combined scene checks reuse historical evidence and count update 2000 once.",
        "scope": "Success and failure refer to the fixed sampled physical evaluations, not every intervening optimizer update or unseen scenes.",
    }


def analyze(run_dir):
    require(read_json(run_dir / "run_status.json")["status"] == "completed",
            "Extension analysis requires a completed run; running or failed runs are refused.")
    verifier = import_file("extension014_readonly_input_audit", DIRECTORY / "verify_td_extension.py")
    settings, summary = read_json(run_dir / "settings.json"), read_json(run_dir / "summary.json")
    protocol = settings["training"]
    require(summary["protocol"] == protocol, "Summary and launch protocol differ.")
    validate_protocol(protocol)
    verifier.verify_protocol(protocol, settings)
    manifest, reference013, reference011 = verifier.verify_inputs(run_dir, protocol)
    dataset = read_json(run_dir / "dataset.json")
    require(dataset == summary["dataset"] == read_json(reference011 / "dataset.json"),
            "The ordered demonstration dataset differs from its source.")
    dataset_path = reference011 / "demonstration.npz"
    diagnostic.require_hash(dataset_path, dataset["sha256"])
    with np.load(dataset_path, allow_pickle=False) as arrays:
        observations, actions = arrays["observations"].copy(), arrays["actions"].copy()
    require(len(actions) == dataset["transitions"] == 117, "Expected 117 fixed transitions.")
    episodes = {row["seed"]: row for row in dataset["episodes"]}
    require(tuple(episodes) == SEEDS, "The demonstration scene order differs.")
    for episode in episodes.values():
        require(diagnostic.array_hash(observations[episode["start"]]) == episode["initial_observation_sha256"],
                "An episode has a different initial observation.")
    torch.set_num_threads(1)
    initial013 = read_json(reference013 / "treatment/td/summary.json")
    original = read_json(reference011 / "calibration/summary.json")
    dqn = deepcopy(settings["dqn"])
    dqn["learning_rate"] = 0.0001
    source_models = (
        ("resume", reference013 / "treatment/td/update_2000.pt", initial013["checkpoint_sha256"]["update_2000.pt"], dqn),
        ("calibration", reference011 / "calibration/update_20000.pt", original["checkpoint_sha256"]["update_20000.pt"],
         read_json(reference011 / "settings.json")["dqn"]),
    )
    baselines, verified_models = {}, {}
    for name, path, expected, config in source_models:
        agent, env, _ = helper.load_verified_agent(path, expected, settings["environment"], config)
        try:
            values, parameters_sha = helper.all_q(agent, observations)
            baselines[name] = {"q": values, "parameter_sha256": parameters_sha,
                               "parameters": torch.nn.utils.parameters_to_vector(agent.learner.online.parameters()).detach().clone()}
            low, span = agent.encoder.low.copy(), agent.encoder.span.copy()
            verified_models[str(path.resolve())] = expected
        finally:
            env.close()
    directory = run_dir / "td"
    phase = read_json(directory / "summary.json")
    require(phase == summary["extension"] and phase["optimizer_updates"] == END - START
            and phase["initial_td_update"] == START and phase["final_td_update"] == END
            and phase["prior_updates_on_branch"] == 40000 and phase["total_updates_on_branch"] == 50000
            and phase["target_synchronizations"] == 40
            and phase["learning_rate"] == 0.0001 and phase["demonstration_weight"] == 1.0,
            "The continuation budget, target schedule or cumulative update accounting differs.")
    expected_models = {f"update_{step}.pt" for step in CHECKPOINTS}
    expected_models |= {f"training_state_{step}.pt" for step in CHECKPOINTS} | {"training_state.pt"}
    expected_models |= {f"target_source_state_{step}.pt" for step in range(2400, 9401, 1000)}
    require(set(phase["checkpoint_sha256"]) == expected_models,
            "The continuation inference, full-state or target-generation model coverage differs.")
    for name, expected in phase["checkpoint_sha256"].items():
        verified_models[f"td/{name}"] = diagnostic.require_hash(directory / name, expected)
    rows = read_json(directory / "observations.json")
    require(tuple(row["update"] for row in rows) == CHECKPOINTS, "The physical observation schedule differs.")
    results, checkpoint_q = [], {}
    final_distances = {}
    for row in rows:
        step, name = row["update"], f"update_{row['update']}.pt"
        agent, env, metadata = helper.load_verified_agent(
            directory / name, phase["checkpoint_sha256"][name], settings["environment"], dqn)
        try:
            require(metadata["updates"] == step and metadata["objective"] == "extension/td"
                    and metadata["aggregate"] == row["aggregate"]
                    and metadata["prior_updates_on_branch"] == 40000
                    and metadata["total_updates_on_branch"] == 40000 + step,
                    "Checkpoint metadata differs from the recorded observation or cumulative clock.")
            values, parameters_sha = helper.all_q(agent, observations)
            checkpoint_q[step] = values
            if step == START:
                require(parameters_sha == baselines["resume"]["parameter_sha256"]
                        and np.array_equal(values, baselines["resume"]["q"]),
                        "The continuation initial policy differs from the exact attempt-013 source.")
            if step == END:
                parameters = torch.nn.utils.parameters_to_vector(agent.learner.online.parameters()).detach()
                final_distances = {f"from_{name}": torch.linalg.vector_norm(parameters - baseline["parameters"]).item()
                                   for name, baseline in baselines.items()}
            require(tuple(scene["seed"] for scene in row["scenes"]) == SEEDS,
                    "An observation omitted or reordered training scenes.")
            scenes = []
            for scene in row["scenes"]:
                episode, rollout = episodes[scene["seed"]], scene["rollout"]
                divergence, prefix = helper.validate_shared_prefix(rollout, episode, observations, actions, values, low, span)
                detail = helper.state_comparison(values, baselines["calibration"]["q"], observations, actions,
                                                 episode, divergence["step"]) if divergence else None
                scenes.append({"seed": scene["seed"], "success": rollout["success"], "damage": rollout["damage"],
                               "budget_exhausted": rollout["budget_exhausted"], "steps": rollout["steps"],
                               "return": rollout["return"], "stored_first_divergence": divergence,
                               "first_divergence_q": detail, "shared_state_action_checks": prefix,
                               "shared_state_position_checks": prefix,
                               "interpretation": "First deviation from the demonstrated action, including successful deviations; no causal failure label is assigned."})
            focal = next(scene for scene in scenes if scene["seed"] == FOCAL_SEED)
            fixed_states = []
            for focal_step in FOCAL_STEPS:
                detail = helper.state_comparison(values, baselines["calibration"]["q"], observations, actions,
                                                 episodes[FOCAL_SEED], focal_step)
                index = episodes[FOCAL_SEED]["start"] + focal_step - 1
                detail["resume_same_state"] = diagnostic.q_metrics(baselines["resume"]["q"][index], int(actions[index]))
                detail["on_verified_common_prefix"] = focal_step <= focal["shared_state_action_checks"]
                fixed_states.append(detail)
            results.append({"update": step, "parameter_sha256": parameters_sha,
                            "focal_reference_states": fixed_states, "scenes": scenes,
                            "success_count": sum(scene["success"] for scene in scenes)})
        finally:
            env.close()
    history = read_json(reference013 / "treatment/td/observations.json")
    verified = {name: file_hash(run_dir / name) for name in (
        "settings.json", "summary.json", "input_manifest.json", "dataset.json", "provenance.json",
        "td/summary.json", "td/observations.json", "td/updates.jsonl", "td/sentinel_q.jsonl")}
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "run_dir": str(run_dir.resolve()),
        "script_sha256": file_hash(Path(__file__)), "analysis_helper_sha256": file_hash(PREVIOUS / "analyze_control_q.py"),
        "diagnostic_helper_sha256": file_hash(PREVIOUS / "diagnose_divergence.py"),
        "input_verifier_sha256": file_hash(DIRECTORY / "verify_td_extension.py"),
        "input_manifest_sha256": file_hash(run_dir / "input_manifest.json"),
        "source_resume_snapshot_sha256": manifest["resumed_snapshot_sha256"],
        "committed_reference_evidence_verified": True, "dataset_sha256": dataset["sha256"],
        "verified_files_sha256": verified, "verified_model_files_sha256": verified_models,
        "verified_model_file_count": len(verified_models),
        "resume_parameter_sha256": baselines["resume"]["parameter_sha256"],
        "calibration_parameter_sha256": baselines["calibration"]["parameter_sha256"],
        "initial_q_vectors_identical_to_source_on_all_117_states": True,
        "environment_steps": 0, "environment_resets": 0, "optimizer_updates": 0, "inference_batch_size": 1,
        "checkpoints": results, "action_names": diagnostic.ACTION_NAMES,
        "final_parameter_displacement_norm": final_distances,
        "update_statistics": update_statistics(directory / "updates.jsonl", dqn["gradient_clip_norm"]),
        "dense_sentinels": analyze_sentinels(directory / "sentinel_q.jsonl", observations, actions,
                                             episodes[FOCAL_SEED], checkpoint_q),
        "sampled_retention": combined_retention(history, rows),
        "limitations": [
            "Every dense sentinel is evaluated at a fixed demonstrated state, which may not be visited by the current greedy policy.",
            "Reference-state rank reversals between scheduled checkpoints do not establish actual rollout failure or its exact onset.",
            "Success at all sampled checkpoints cannot establish success at unsampled updates or on unseen scenes.",
            "The known successful step-24 action mismatch is retained as a control, together with every successful or failed first divergence.",
            "This analysis performs no environment reset, environment step, optimizer update or counterfactual rescue.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite analysis: {args.output}")
    result = analyze(args.run_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output.resolve()),
                      "new_sampled_retention": result["sampled_retention"]["new_attempt_014_evaluations"],
                      "new_environment_steps": 0, "new_optimizer_updates": 0}), flush=True)


if __name__ == "__main__":
    main()
