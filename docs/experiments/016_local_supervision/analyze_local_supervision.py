"""Compare local supervision, neighboring Q rankings and measured scene retention."""

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = Path(__file__).resolve().parent
START, END = 2000, 10000
ARMS = ("baseline", "treatment")
SEEDS = (10000, 10001, 10002)
FOCAL_STEPS = (6, 18, 20, 21, 22, 23, 24)
NEIGHBORS = (20, 22, 23)
CHECKPOINTS = (START, *range(2001, 2011), 2200, 2201, *range(2500, END + 1, 500))


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREVIOUS = ROOT / "docs/experiments/014_td_extension/analyze_td_extension.py"
previous = import_file("local016_previous_analysis", PREVIOUS)
helper, diagnostic = previous.helper, previous.diagnostic
read_json, file_hash, require = previous.read_json, previous.file_hash, previous.require


def compact_intervals(updates, selected, gaps, checkpoints):
    """Keep every contiguous interval without duplicating verbose per-row keys."""
    result = []
    for row in previous.intervals(updates, selected, gaps, checkpoints):
        result.append([row["start_update"], row["end_update"], row["update_count"],
                       row["minimum_gap"], row["minimum_gap_update"],
                       row["scheduled_physical_checkpoints_within_interval"]])
    return result


def series_summary(updates, gaps, greedy, demonstrated, checkpoints):
    """Separate negative scores, exact ties and action disagreements from episodes."""
    gaps, greedy = np.asarray(gaps, dtype=np.float64), np.asarray(greedy, dtype=int)
    require(len(updates) == len(gaps) == len(greedy) and len(updates) > 0,
            "Dense series must be nonempty and aligned.")
    require(np.isfinite(gaps).all(), "Dense Q gaps must be finite.")
    negative, disagree, ties = gaps < 0, greedy != demonstrated, gaps == 0
    minimum, maximum = int(gaps.argmin()), int(gaps.argmax())
    negative_indices, disagree_indices = np.flatnonzero(negative), np.flatnonzero(disagree)
    intervals = compact_intervals(updates, negative, gaps, checkpoints)
    mismatches = compact_intervals(updates, disagree, gaps, checkpoints)
    changes = [int(updates[i]) for i in range(1, len(updates))
               if np.sign(gaps[i]) != np.sign(gaps[i - 1]) or greedy[i] != greedy[i - 1]]
    return {
        "demonstrated_action": int(demonstrated),
        "records_including_resume_boundary": len(updates),
        "negative_gap_count_including_resume_boundary": int(negative.sum()),
        "negative_gap_count_new_updates": int(negative[1:].sum()),
        "greedy_disagreement_count_including_resume_boundary": int(disagree.sum()),
        "greedy_disagreement_count_new_updates": int(disagree[1:].sum()),
        "zero_gap_count": int(ties.sum()),
        "greedy_disagreement_at_zero_gap_count": int((ties & disagree).sum()),
        "below_margin_0_8_count": int((gaps < 0.8).sum()),
        "first_negative_gap_update": int(updates[negative_indices[0]]) if len(negative_indices) else None,
        "first_greedy_disagreement_update": int(updates[disagree_indices[0]]) if len(disagree_indices) else None,
        "minimum_gap": float(gaps[minimum]), "minimum_gap_update": int(updates[minimum]),
        "maximum_gap": float(gaps[maximum]), "maximum_gap_update": int(updates[maximum]),
        "negative_interval_count": len(intervals),
        "negative_intervals_entirely_between_physical_checks": sum(not row[5] for row in intervals),
        "interval_columns": ["start_update", "end_update", "record_count", "minimum_gap",
                             "minimum_gap_update", "scheduled_physical_checkpoints"],
        "negative_gap_intervals": intervals, "greedy_disagreement_intervals": mismatches,
        "rank_transition_updates": changes,
        "classification": "REFERENCE-STATE RANKINGS, NOT PHYSICAL EPISODE OUTCOMES",
        "actual_failed_rollouts_inferred": 0,
    }


def analyze_sentinels(path, observations, actions, episode, checkpoint_q,
                      start=START, end=END, checkpoints=CHECKPOINTS):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    updates = list(range(start, end + 1))
    require([row["update"] for row in rows] == updates, "Dense Q coverage differs from the fixed update clock.")
    series = {step: {"gaps": [], "greedy": []} for step in FOCAL_STEPS}
    crosschecks = 0
    for row in rows:
        require(row["extension_update"] == row["update"] - start and row["scene_seed"] == 10001,
                "Dense Q probe has a changed clock or scene.")
        require(tuple(state["reference_step"] for state in row["states"]) == FOCAL_STEPS,
                "The seven fixed Q-probe states differ.")
        for state in row["states"]:
            step = state["reference_step"]
            index = episode["start"] + step - 1
            require(state["global_transition_index"] == index
                    and state["demonstrated_action"] == int(actions[index]),
                    "Dense Q probe has a changed demonstration row or action.")
            metrics = diagnostic.q_metrics(state["q_values"], int(actions[index]))
            require(state["greedy_action"] == metrics["chosen_action"]
                    and state["signed_demonstration_gap"] == metrics["signed_demonstration_margin"],
                    "Dense Q probe disagrees with its raw scores.")
            if row["update"] in checkpoint_q:
                require(np.array_equal(np.asarray(state["q_values"]), checkpoint_q[row["update"]][index]),
                        "Dense Q probe disagrees with the reloaded checkpoint.")
                crosschecks += 1
            series[step]["gaps"].append(metrics["signed_demonstration_margin"])
            series[step]["greedy"].append(metrics["chosen_action"])
    summaries, compact = {}, {}
    for step, values in series.items():
        index = episode["start"] + step - 1
        role = "Locally reweighted state" if step == 21 else (
            "Adjacent decision" if step in NEIGHBORS else
            "Known successful-deviation control" if step == 24 else "Earlier failure sentinel")
        detail = {
            "reference_step": step, "global_transition_index": index, "role": role,
            "encoded_observation_sha256": diagnostic.array_hash(observations[index]),
            **series_summary(updates, values["gaps"], values["greedy"], int(actions[index]), checkpoints),
        }
        dense_keys = ("negative_gap_intervals", "greedy_disagreement_intervals", "rank_transition_updates")
        summaries[str(step)] = {key: value for key, value in detail.items() if key not in dense_keys}
        compact[str(step)] = {"demonstrated_action": int(actions[index]),
                              "signed_demonstration_gaps": values["gaps"],
                              "greedy_actions": values["greedy"],
                              "interval_columns": detail["interval_columns"],
                              **{key: detail[key] for key in dense_keys}}
    common = {
        "raw_file": str(path.resolve()), "raw_sha256": file_hash(path), "raw_records": len(rows),
        "initial_update": start, "final_update": end, "new_update_records": end - start,
        "resume_boundary_records": 1, "inference_batch_size": 1,
        "reloaded_checkpoint_state_crosschecks": crosschecks,
        "new_environment_actions": 0, "reference_states": summaries,
    }
    return common, {"updates": updates, "reference_states": compact,
                    "interpretation": "All records are fixed demonstration-state rankings, not additional evaluated episodes."}


def compare_dense(series):
    """Count local ranking improvements and neighboring regressions at matched updates."""
    baseline, treatment = series["baseline"], series["treatment"]
    require(baseline["updates"] == treatment["updates"], "Matched dense-series clocks differ.")
    updates = np.asarray(baseline["updates"])
    paired, neighbor_regression = {}, np.zeros(len(updates), dtype=bool)
    for step in FOCAL_STEPS:
        left, right = (arm["reference_states"][str(step)] for arm in (baseline, treatment))
        require(left["demonstrated_action"] == right["demonstrated_action"],
                "Paired dense series use different demonstrated actions.")
        gaps_a, gaps_b = np.asarray(left["signed_demonstration_gaps"]), np.asarray(right["signed_demonstration_gaps"])
        action = left["demonstrated_action"]
        correct_a, correct_b = np.asarray(left["greedy_actions"]) == action, np.asarray(right["greedy_actions"]) == action
        require(all(len(values) == len(updates) for values in (gaps_a, gaps_b, correct_a, correct_b)),
                "A paired dense series has missing values.")
        delta, regression, improvement = gaps_b - gaps_a, correct_a & ~correct_b, ~correct_a & correct_b
        if step in NEIGHBORS:
            neighbor_regression |= regression
        paired[str(step)] = {
            "demonstrated_action": int(action), "new_update_count": len(updates) - 1,
            "mean_gap_change_new_updates": float(delta[1:].mean()),
            "minimum_gap_change_new_updates": float(delta[1:].min()),
            "maximum_gap_change_new_updates": float(delta[1:].max()),
            "higher_gap_new_update_count": int((delta[1:] > 0).sum()),
            "lower_gap_new_update_count": int((delta[1:] < 0).sum()),
            "demonstration_action_regression_new_update_count": int(regression[1:].sum()),
            "demonstration_action_improvement_new_update_count": int(improvement[1:].sum()),
            "first_demonstration_action_regression_update": int(updates[np.flatnonzero(regression)[0]]) if regression.any() else None,
            "negative_gap_count_baseline_new_updates": int((gaps_a[1:] < 0).sum()),
            "negative_gap_count_treatment_new_updates": int((gaps_b[1:] < 0).sum()),
            "initial_gap_baseline": float(gaps_a[0]), "initial_gap_treatment": float(gaps_b[0]),
            "final_gap_baseline": float(gaps_a[-1]), "final_gap_treatment": float(gaps_b[-1]),
        }
    selected_a, selected_b = (np.asarray(arm["reference_states"]["21"]["signed_demonstration_gaps"])
                               for arm in (baseline, treatment))
    return {
        "reference_states": paired, "neighbor_steps": list(NEIGHBORS),
        "any_neighbor_action_regression_new_update_count": int(neighbor_regression[1:].sum()),
        "higher_step21_gap_with_neighbor_action_regression_new_update_count": int(
            ((selected_b > selected_a) & neighbor_regression)[1:].sum()),
        "interpretation": "Improvements and regressions refer to demonstration-action rankings on matched reference inputs, not inferred physical successes or failures.",
    }


def update_statistics(directory, clip_norm, weight, start=START, end=END):
    result = previous.update_statistics(directory / "updates.jsonl", clip_norm, start, end)
    records = [json.loads(line) for line in (directory / "updates.jsonl").read_text(encoding="utf-8").splitlines()]
    diagnostics = [json.loads(line) for line in (directory / "supervision.jsonl").read_text(encoding="utf-8").splitlines()]
    require([row["update"] for row in diagnostics] == list(range(start + 1, end + 1)),
            "Supplemental supervision coverage differs.")
    require(all(row["target_synchronized"] == (row["update"] % 200 == 0) for row in records),
            "Target synchronization changed the absolute schedule.")
    fields = ("unweighted_margin_before", "unweighted_margin_after", "weighted_margin_before",
              "weighted_margin_after", "selected_hinge_before", "selected_hinge_after")
    for metrics, row in zip(records, diagnostics, strict=True):
        require(row["extension_update"] == row["update"] - start
                and row["selected_sample_weight"] == weight and row["normalization_denominator"] == 117,
                "Supplemental supervision changes the one-row coefficient or denominator.")
        require(all(np.isfinite(row[key]) and row[key] >= 0 for key in fields),
                "Supplemental supervision contains invalid loss values.")
        for position in ("before", "after"):
            expected = row[f"unweighted_margin_{position}"] + (weight - 1) * row[f"selected_hinge_{position}"] / 117
            require(np.isclose(row[f"weighted_margin_{position}"], expected, atol=2e-7, rtol=1e-6),
                    "Weighted supervision is not one extra selected-row hinge divided by 117.")
            require(metrics[f"demonstration_loss_{position}"] == row[f"weighted_margin_{position}"],
                    "Update loss differs from supplemental weighted supervision.")
    supplemental = {}
    for name in fields:
        values = np.asarray([row[name] for row in diagnostics], dtype=np.float64)
        supplemental[name] = {"mean": float(values.mean()), "minimum": float(values.min()), "maximum": float(values.max())}
    result.update({
        "supervision_raw_sha256": file_hash(directory / "supervision.jsonl"),
        "supervision_statistics": supplemental, "first_supervision_record": diagnostics[0],
        "last_supervision_record": diagnostics[-1], "selected_sample_weight": weight,
        "normalization_denominator": 117, "target_synchronizations": sum(row["target_synchronized"] for row in records),
        "component_loss_semantics": "TD loss is unchanged. demonstration_loss is the coefficient-weighted hinge sum divided by 117; supplemental unweighted_margin uses all coefficients 1.",
        "comparison_warning": "The two weighted objectives differ. Compare unweighted margin and physical outcomes explicitly; later TD targets also evolve separately.",
    })
    return result


def compare_retention(rows):
    baseline, treatment = rows["baseline"], rows["treatment"]
    require([row["update"] for row in baseline] == [row["update"] for row in treatment],
            "Matched physical checkpoint schedules differ.")
    comparisons = []
    for a, b in zip(baseline, treatment, strict=True):
        require(tuple(scene["seed"] for scene in a["scenes"]) == SEEDS
                and tuple(scene["seed"] for scene in b["scenes"]) == SEEDS,
                "Matched physical scenes differ.")
        for left, right in zip(a["scenes"], b["scenes"], strict=True):
            comparisons.append({
                "update": a["update"], "seed": left["seed"],
                "baseline_success": left["rollout"]["success"], "treatment_success": right["rollout"]["success"],
                "baseline_steps": left["rollout"]["steps"], "treatment_steps": right["rollout"]["steps"],
            })
    return {
        "arms": {arm: previous.retention_summary(values) for arm, values in rows.items()},
        "first_ten_continuation_updates": {arm: previous.retention_summary(
            [row for row in values if 2001 <= row["update"] <= 2010]) for arm, values in rows.items()},
        "new_episode_count": len(comparisons) * 2,
        "treatment_rescued_scene_checks": [row for row in comparisons if not row["baseline_success"] and row["treatment_success"]],
        "treatment_regressed_scene_checks": [row for row in comparisons if row["baseline_success"] and not row["treatment_success"]],
        "both_failed_scene_checks": [row for row in comparisons if not row["baseline_success"] and not row["treatment_success"]],
        "all_paired_scene_checks": comparisons,
        "interpretation": "These are repeated known-scene checks, not independent generalization trials.",
    }


def analyze(run_dir):
    require(read_json(run_dir / "run_status.json")["status"] == "completed",
            "Local-supervision analysis requires a completed run.")
    verifier = import_file("local016_readonly_inputs", DIRECTORY / "verify_local_supervision.py")
    settings, summary = read_json(run_dir / "settings.json"), read_json(run_dir / "summary.json")
    protocol = settings["training"]
    require(summary["protocol"] == protocol and protocol["attempt"] == 16, "The recorded experiment protocol differs.")
    verifier.verify_protocol(protocol, settings)
    manifest, reference014, reference013, reference011 = verifier.verify_inputs(run_dir, protocol)
    dataset = read_json(run_dir / "dataset.json")
    require(dataset == summary["dataset"] == read_json(reference011 / "dataset.json"),
            "The ordered demonstration dataset changed.")
    data_path = reference011 / "demonstration.npz"
    diagnostic.require_hash(data_path, dataset["sha256"])
    with np.load(data_path, allow_pickle=False) as arrays:
        observations, actions = arrays["observations"].copy(), arrays["actions"].copy()
    episodes = {row["seed"]: row for row in dataset["episodes"]}
    require(len(actions) == 117 and tuple(episodes) == SEEDS
            and episodes[10001]["start"] == 35 and int(actions[55]) == 0,
            "The targeted demonstration row is not scene 10001 step 21, north_small.")
    require(manifest["selected_transition"] == {
        "scene_seed": 10001, "reference_step": 21, "global_transition_index": 55,
        "demonstrated_action": 0, "encoded_observation_sha256": diagnostic.array_hash(observations[55]),
    }, "The source manifest identifies a different reweighted state.")
    torch.set_num_threads(1)
    source_phases = {
        "resume": read_json(reference013 / "treatment/td/summary.json"),
        "calibration": read_json(reference011 / "calibration/summary.json"),
    }
    source_models = (
        ("resume", reference013 / "treatment/td/update_2000.pt", "update_2000.pt", settings["dqn"]),
        ("calibration", reference011 / "calibration/update_20000.pt", "update_20000.pt",
         read_json(reference011 / "settings.json")["dqn"]),
    )
    sources, model_hashes = {}, {}
    for name, path, key, config in source_models:
        digest = source_phases[name]["checkpoint_sha256"][key]
        agent, env, _ = helper.load_verified_agent(path, digest, settings["environment"], config)
        try:
            values, identity = helper.all_q(agent, observations)
            sources[name] = {"q": values, "parameter_sha256": identity,
                             "parameters": torch.nn.utils.parameters_to_vector(agent.learner.online.parameters()).detach().clone()}
            low, span = agent.encoder.low.copy(), agent.encoder.span.copy()
            model_hashes[str(path.resolve())] = digest
        finally:
            env.close()
    results, dense, rows_by_arm, verified = {}, {}, {}, {}
    expected_models = {f"update_{step}.pt" for step in CHECKPOINTS}
    expected_models |= {f"training_state_{step}.pt" for step in CHECKPOINTS} | {"training_state.pt"}
    expected_models |= {f"target_source_state_{step}.pt" for step in range(2400, 9401, 1000)}
    for arm in ARMS:
        directory = run_dir / arm / "td"
        phase = read_json(directory / "summary.json")
        require(phase == summary["arms"][arm] and phase["optimizer_updates"] == 8000
                and phase["initial_td_update"] == START and phase["final_td_update"] == END
                and phase["target_synchronizations"] == 40 and phase["learning_rate"] == 0.0001
                and phase["demonstration_weight"] == 1.0 and phase["selected_transition_index"] == 55
                and phase["selected_sample_weight"] == (1 if arm == "baseline" else 2)
                and phase["normalization_denominator"] == 117, "An arm changed the shared training conditions.")
        require(set(phase["checkpoint_sha256"]) == expected_models, "Expected all 67 model artifacts per arm.")
        for name, digest in phase["checkpoint_sha256"].items():
            model_hashes[f"{arm}/td/{name}"] = diagnostic.require_hash(directory / name, digest)
            if arm == "baseline":
                verifier.require_equal(torch.load(directory / name, map_location="cpu", weights_only=True),
                                       torch.load(reference014 / "td" / name, map_location="cpu", weights_only=True),
                                       f"baseline exact attempt-014 model payload {name}")
        rows = read_json(directory / "observations.json")
        require(tuple(row["update"] for row in rows) == CHECKPOINTS, "The physical checkpoint schedule differs.")
        rows_by_arm[arm] = rows
        if arm == "baseline":
            require(rows == read_json(reference014 / "td/observations.json"), "Baseline observations differ from attempt 014.")
            require([json.loads(line) for line in (directory / "updates.jsonl").read_text(encoding="utf-8").splitlines()]
                    == [json.loads(line) for line in (reference014 / "td/updates.jsonl").read_text(encoding="utf-8").splitlines()],
                    "Baseline update metrics differ from attempt 014.")
        checkpoints, checkpoint_q, distances = [], {}, {}
        for row in rows:
            step, key = row["update"], f"update_{row['update']}.pt"
            agent, env, metadata = helper.load_verified_agent(
                directory / key, phase["checkpoint_sha256"][key], settings["environment"], settings["dqn"])
            try:
                require(metadata["updates"] == step and metadata["objective"] == "extension/td"
                        and metadata["aggregate"] == row["aggregate"]
                        and metadata["prior_updates_on_branch"] == 40000
                        and metadata["total_updates_on_branch"] == 40000 + step,
                        "Checkpoint metadata differs from the preserved absolute clocks.")
                values, identity = helper.all_q(agent, observations)
                checkpoint_q[step] = values
                if step == START:
                    require(identity == sources["resume"]["parameter_sha256"]
                            and np.array_equal(values, sources["resume"]["q"]), "An arm changed the initial policy.")
                if step == END:
                    parameters = torch.nn.utils.parameters_to_vector(agent.learner.online.parameters()).detach()
                    distances = {f"from_{name}": float(torch.linalg.vector_norm(parameters - source["parameters"]))
                                 for name, source in sources.items()}
                scenes = []
                for scene in row["scenes"]:
                    episode, rollout = episodes[scene["seed"]], scene["rollout"]
                    divergence, prefix = helper.validate_shared_prefix(rollout, episode, observations, actions, values, low, span)
                    detail = helper.state_comparison(values, sources["calibration"]["q"], observations, actions,
                                                     episode, divergence["step"]) if divergence else None
                    scenes.append({
                        "seed": scene["seed"], "success": rollout["success"], "damage": rollout["damage"],
                        "budget_exhausted": rollout["budget_exhausted"], "steps": rollout["steps"], "return": rollout["return"],
                        "stored_first_divergence": divergence, "first_divergence_q": detail,
                        "shared_state_action_checks": prefix, "shared_state_position_checks": prefix,
                        "interpretation": "A first reference-action deviation may occur in either a successful or failed episode.",
                    })
                focal = next(scene for scene in scenes if scene["seed"] == 10001)
                fixed = []
                for focal_step in FOCAL_STEPS:
                    detail = helper.state_comparison(values, sources["calibration"]["q"], observations, actions,
                                                     episodes[10001], focal_step)
                    index = 35 + focal_step - 1
                    detail["resume_same_state"] = diagnostic.q_metrics(sources["resume"]["q"][index], int(actions[index]))
                    detail["on_verified_common_prefix"] = focal_step <= focal["shared_state_action_checks"]
                    fixed.append(detail)
                checkpoints.append({
                    "update": step, "parameter_sha256": identity, "focal_reference_states": fixed,
                    "scenes": scenes, "success_count": sum(scene["success"] for scene in scenes),
                    "batch_one_demonstration_correct": int((values.argmax(axis=1) == actions).sum()),
                })
            finally:
                env.close()
        sentinels, dense[arm] = analyze_sentinels(directory / "sentinel_q.jsonl", observations, actions, episodes[10001], checkpoint_q)
        results[arm] = {
            "checkpoints": checkpoints, "dense_sentinels": sentinels,
            "update_statistics": update_statistics(directory, settings["dqn"]["gradient_clip_norm"], 1 if arm == "baseline" else 2),
            "final_parameter_displacement_norm": distances,
            "initial_full_batch_fit": rows[0]["fit"], "final_full_batch_fit": rows[-1]["fit"],
            "initial_return_fit_rmse": rows[0]["return_fit"]["rmse"], "final_return_fit_rmse": rows[-1]["return_fit"]["rmse"],
        }
        for name in ("summary.json", "observations.json", "updates.jsonl", "supervision.jsonl", "sentinel_q.jsonl"):
            verified[f"{arm}/td/{name}"] = file_hash(directory / name)
    first_a, first_b = (results[arm]["update_statistics"]["first_supervision_record"] for arm in ARMS)
    require(first_a["unweighted_margin_before"] == first_b["unweighted_margin_before"]
            and first_a["selected_hinge_before"] == first_b["selected_hinge_before"],
            "The first update does not begin from identical unweighted supervision.")
    comparison = {
        "weighted_margin_before_baseline": first_a["weighted_margin_before"],
        "weighted_margin_before_treatment": first_b["weighted_margin_before"],
        "expected_extra_selected_hinge_over_117": first_a["selected_hinge_before"] / 117,
        "actual_weighted_margin_difference": first_b["weighted_margin_before"] - first_a["weighted_margin_before"],
        "scope": "This isolates the new row coefficient at the first update; later trajectories evolve separately.",
    }
    for name in ("settings.json", "summary.json", "input_manifest.json", "dataset.json", "provenance.json"):
        verified[name] = file_hash(run_dir / name)
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "run_dir": str(run_dir.resolve()),
        "script_sha256": file_hash(Path(__file__)), "prior_analysis_sha256": file_hash(PREVIOUS),
        "input_verifier_sha256": file_hash(DIRECTORY / "verify_local_supervision.py"),
        "dataset_sha256": dataset["sha256"], "verified_files_sha256": verified,
        "verified_model_files_sha256": model_hashes, "verified_model_file_count": len(model_hashes),
        "committed_reference_evidence_verified": True, "baseline_exact_attempt_014_reproduction": True,
        "initial_q_vectors_identical_to_source_on_all_117_states": True,
        "source_input_manifest_sha256": file_hash(run_dir / "input_manifest.json"),
        "source_resume_parameter_sha256": sources["resume"]["parameter_sha256"],
        "source_resume_snapshot_sha256": manifest["resumed_snapshot_sha256"],
        "environment_steps": 0, "environment_resets": 0, "optimizer_updates": 0, "inference_batch_size": 1,
        "arms": results, "sampled_retention": compare_retention(rows_by_arm),
        "paired_dense_rankings": compare_dense(dense), "first_update_supervision_comparison": comparison,
        "limitations": [
            "Local weight 2 changes only the selected supervised hinge coefficient; the denominator stays 117 and TD weighting is unchanged.",
            "Dense reference-state rankings are not additional physical episode evaluations and may concern states the policy no longer visits.",
            "A more positive step-21 gap can coexist with new neighboring-action errors; these are reported separately.",
            "The data contain three known scenes and one initialization. No held-out or online-learning claim is supported.",
            "The analyzer performs no environment reset, environment step, optimizer update or forced-action rescue.",
        ],
    }, dense


def write_outputs(output, result, dense):
    paths = {arm: output.with_name(f"{output.stem}_{arm}_dense.json") for arm in ARMS}
    require(not output.exists() and all(not path.exists() for path in paths.values()),
            "Refusing to overwrite existing analysis evidence.")
    encoded = {arm: json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n" for arm, value in dense.items()}
    require(all(len(value.encode("utf-8")) < 8 * 1024 * 1024 for value in encoded.values()),
            "A dense analysis artifact exceeds the compact archive limit.")
    output.parent.mkdir(parents=True, exist_ok=True)
    for arm, path in paths.items():
        path.write_text(encoded[arm], encoding="utf-8")
        result["arms"][arm]["dense_series_file"] = path.name
        result["arms"][arm]["dense_series_sha256"] = file_hash(path)
    main_text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    require(len(main_text.encode("utf-8")) < 8 * 1024 * 1024, "The main analysis exceeds the archive limit.")
    output.write_text(main_text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite analysis: {args.output}")
    result, dense = analyze(args.run_dir.resolve())
    write_outputs(args.output.resolve(), result, dense)
    print(json.dumps({"output": str(args.output.resolve()),
                      "sampled_retention": result["sampled_retention"]["arms"],
                      "new_environment_steps": 0, "new_optimizer_updates": 0}))


if __name__ == "__main__":
    main()
