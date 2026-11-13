"""Audit return-anchored TD against a verified historical control without rollouts."""

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = Path(__file__).resolve().parent
PREVIOUS = ROOT / "docs/experiments/016_local_supervision/analyze_local_supervision.py"
START, END = 2000, 10000
ARMS = ("baseline", "treatment")
SEEDS = (10000, 10001, 10002)
FOCAL_STEPS = (6, 18, 20, 21, 22, 23, 24)
CHECKPOINTS = (START, *range(2001, 2011), 2200, 2201, *range(2500, END + 1, 500))


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


previous = import_file("anchor017_previous_analysis", PREVIOUS)
helper, diagnostic = previous.helper, previous.diagnostic
read_json, file_hash, require = previous.read_json, previous.file_hash, previous.require
compare_dense, series_summary = previous.compare_dense, previous.series_summary


def analyze_sentinels(path, observations, actions, episode, checkpoint_q,
                      start=START, end=END, checkpoints=CHECKPOINTS):
    """Reuse complete ranking validation, without suggesting any local reweighting."""
    summary, dense = previous.analyze_sentinels(
        path, observations, actions, episode, checkpoint_q, start, end, checkpoints)
    summary["reference_states"]["21"]["role"] = "Previous first-divergence sentinel"
    summary["all_supervised_sample_coefficients"] = 1.0
    return summary, dense


def compare_retention(rows):
    """Compare matched observations while counting only the new treatment episodes."""
    result = previous.compare_retention(rows)
    new_count = sum(len(row["scenes"]) for row in rows["treatment"])
    reused_count = sum(len(row["scenes"]) for row in rows["baseline"])
    result.update({
        "new_episode_count": new_count,
        "reused_historical_control_episode_count": reused_count,
        "total_compared_episode_records": new_count + reused_count,
        "baseline_origin": "Verified attempt-016 baseline, reused without new training or episodes.",
        "interpretation": "Matched known-scene observations from one inherited initialization; historical control episodes are reused, not newly executed or independent generalization trials.",
    })
    return result


def statistics(values):
    values = np.asarray(values, dtype=np.float64)
    require(values.ndim == 1 and len(values) > 0 and np.isfinite(values).all(),
            "Diagnostic statistics require a finite nonempty vector.")
    return {"mean": float(values.mean()), "minimum": float(values.min()), "maximum": float(values.max())}


def update_statistics(directory, clip_norm, mc_weight, start=START, end=END):
    """Check every objective component and keep the reused control's missing MC log explicit."""
    require(mc_weight in (0.0, 1.0), "This comparison declares only MC coefficients zero and one.")
    path = directory / "updates.jsonl"
    result = previous.previous.update_statistics(path, clip_norm, start, end)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    require(all(row["target_synchronized"] == (row["update"] % 200 == 0) for row in records),
            "Target synchronization differs from the original absolute clock.")
    fields = ("loss_before", "loss_after", "td_loss_before", "td_loss_after",
              "demonstration_loss_before", "demonstration_loss_after", "gradient_norm")
    require(all(row[key] >= 0 for row in records for key in fields), "Losses and norms must be nonnegative.")
    for row in records:
        if mc_weight:
            require(row["return_anchor_weight"] == mc_weight, "The logged MC coefficient differs.")
        for position in ("before", "after"):
            expected = row[f"td_loss_{position}"] + row[f"demonstration_loss_{position}"]
            if mc_weight:
                value = row[f"return_loss_{position}"]
                require(np.isfinite(value) and value >= 0, "The MC term is not a finite nonnegative loss.")
                expected += mc_weight * value
            require(np.isclose(row[f"loss_{position}"], expected, atol=2e-7, rtol=1e-6),
                    "Combined loss is not TD plus unweighted margin plus the declared MC term.")
    result.update({
        "mc_weight": mc_weight, "normalization_denominator": 117,
        "all_supervised_sample_coefficients": 1.0,
        "target_synchronizations": sum(row["target_synchronized"] for row in records),
        "historical_control_reused": not bool(mc_weight),
        "component_loss_semantics": "Each TD, supervised hinge and active MC-return term is separately averaged over 117 rows. The total is their sum, without division by three.",
        "comparison_warning": "The combined objectives differ; compare their shared TD/margin diagnostics, return RMSE and physical outcomes separately. Later target networks evolve separately.",
    })
    if mc_weight:
        for field in ("return_loss_before", "return_loss_after", "td_target_return_rmse",
                      "td_target_return_mean_difference", "td_target_return_max_abs_difference"):
            result["statistics"][field] = statistics([row[field] for row in records])
        require(all(row["td_target_return_rmse"] >= 0 and row["td_target_return_max_abs_difference"] >= 0
                    and abs(row["td_target_return_mean_difference"]) <= row["td_target_return_max_abs_difference"] + 1e-12
                    for row in records), "TD-target versus return diagnostics are inconsistent.")
        result["mc_log_availability"] = "The new treatment records its active MC loss before and after every update."
        result["td_target_return_gap_semantics"] = "Frozen pre-update one-step Bellman target minus the fixed actual demonstration return, evaluated in float64. At a target synchronization this is the old target generation, not the saved post-update target network."
    else:
        result["mc_log_availability"] = "The historical control did not log per-update MC loss. Its checkpoint return RMSE is retained; absent MC losses are not replaced with zeros."
    return result


def state_comparison(values, calibrated, observations, actions, episode, step):
    """Name the calibration reference explicitly so it cannot be mistaken for the control arm."""
    result = helper.state_comparison(values, calibrated, observations, actions, episode, step)
    result["calibration_reference_same_state"] = result.pop("baseline_same_state")
    return result


def expected_model_names():
    return ({f"update_{step}.pt" for step in CHECKPOINTS}
            | {f"training_state_{step}.pt" for step in CHECKPOINTS}
            | {f"target_source_state_{step}.pt" for step in range(2400, 9401, 1000)}
            | {"training_state.pt"})


def validate_phase(phase, arm, summary):
    """Bind each phase to its own role and forbid unnoticed extra objective changes."""
    require(arm in ARMS and phase["optimizer_updates"] == 8000 and phase["initial_td_update"] == START
            and phase["final_td_update"] == END and phase["target_synchronizations"] == 40
            and phase["learning_rate"] == 0.0001 and phase["demonstration_weight"] == 1.0
            and phase["normalization_denominator"] == 117,
            "An arm changed the shared training conditions.")
    if arm == "baseline":
        control = summary["historical_control"]
        require(phase["selected_sample_weight"] == 1.0
                and control["return_anchor_weight"] == 0.0 and control["all_sample_weights"] == 1.0
                and control["new_optimizer_updates"] == control["new_evaluation_episodes"] == control["new_environment_steps"] == 0
                and control["checkpoint_sha256"] == phase["checkpoint_sha256"],
                "Historical control is not the verified reused all-ones baseline.")
    else:
        require(phase == summary["treatment"] and phase["return_anchor_weight"] == 1.0
                and phase["all_sample_weights"] == 1.0,
                "Treatment differs from the sole coefficient-one return-anchor intervention.")
    require(set(phase["checkpoint_sha256"]) == expected_model_names(), "Expected all 67 model artifacts per arm.")


def checkpoint_analysis(directory, phase, rows, settings, observations, actions, episodes, sources):
    """Reload inference weights and validate recorded reference prefixes without stepping an environment."""
    checkpoints, checkpoint_q, displacement = [], {}, {}
    for row in rows:
        step, key = row["update"], f"update_{row['update']}.pt"
        agent, env, metadata = helper.load_verified_agent(
            directory / key, phase["checkpoint_sha256"][key], settings["environment"], settings["dqn"])
        try:
            require(metadata["updates"] == step and metadata["objective"] == "extension/td"
                    and metadata["aggregate"] == row["aggregate"]
                    and metadata["prior_updates_on_branch"] == 40000
                    and metadata["total_updates_on_branch"] == 40000 + step,
                    "Checkpoint metadata differs from the preserved TD clocks and outcomes.")
            values, identity = helper.all_q(agent, observations)
            checkpoint_q[step] = values
            if step == START:
                require(identity == sources["resume"]["parameter_sha256"]
                        and np.array_equal(values, sources["resume"]["q"]), "Initial policy differs from the shared resume.")
            if step == END:
                parameters = torch.nn.utils.parameters_to_vector(agent.learner.online.parameters()).detach()
                displacement = {f"from_{name}": float(torch.linalg.vector_norm(parameters - source["parameters"]))
                                for name, source in sources.items()}
            scenes = []
            for scene in row["scenes"]:
                episode, rollout = episodes[scene["seed"]], scene["rollout"]
                divergence, prefix = helper.validate_shared_prefix(
                    rollout, episode, observations, actions, values, agent.encoder.low, agent.encoder.span)
                detail = state_comparison(values, sources["calibration"]["q"], observations, actions,
                                          episode, divergence["step"]) if divergence else None
                scenes.append({
                    "seed": scene["seed"], "success": rollout["success"], "damage": rollout["damage"],
                    "budget_exhausted": rollout["budget_exhausted"], "steps": rollout["steps"], "return": rollout["return"],
                    "stored_first_divergence": divergence, "first_divergence_q": detail,
                    "shared_state_action_checks": prefix, "shared_state_position_checks": prefix,
                    "interpretation": "A first reference-action deviation can occur in successful or failed episodes; isolated-action causality is not established here.",
                })
            focal = next(scene for scene in scenes if scene["seed"] == 10001)
            fixed = []
            for point in FOCAL_STEPS:
                detail = state_comparison(values, sources["calibration"]["q"], observations, actions, episodes[10001], point)
                index = episodes[10001]["start"] + point - 1
                detail["resume_same_state"] = diagnostic.q_metrics(sources["resume"]["q"][index], int(actions[index]))
                detail["on_verified_common_prefix"] = point <= focal["shared_state_action_checks"]
                fixed.append(detail)
            checkpoints.append({
                "update": step, "parameter_sha256": identity, "focal_reference_states": fixed, "scenes": scenes,
                "success_count": sum(scene["success"] for scene in scenes),
                "batch_one_demonstration_correct": int((values.argmax(axis=1) == actions).sum()),
            })
        finally:
            env.close()
    return checkpoints, checkpoint_q, displacement


def analyze(run_dir):
    require(read_json(run_dir / "run_status.json")["status"] == "completed",
            "MC-anchor analysis requires a completed run.")
    verifier = import_file("anchor017_readonly_inputs", DIRECTORY / "verify_mc_anchor.py")
    settings, summary = read_json(run_dir / "settings.json"), read_json(run_dir / "summary.json")
    protocol = settings["training"]
    require(summary["protocol"] == protocol and protocol["attempt"] == 17, "The recorded MC-anchor protocol differs.")
    verifier.verify_protocol(protocol, settings)
    manifest, reference016, reference014, reference013, reference011 = verifier.verify_inputs(run_dir, protocol)
    dataset = read_json(run_dir / "dataset.json")
    require(dataset == summary["dataset"] == read_json(reference011 / "dataset.json"),
            "The ordered demonstration dataset changed.")
    targets = read_json(run_dir / "mc_targets.json")
    require(targets == read_json(reference016 / "mc_targets.json") and targets["gamma"] == 0.99
            and len(targets["returns"]) == 117 and np.isfinite(targets["returns"]).all(),
            "The original fixed demonstration returns differ.")
    data_path = reference011 / "demonstration.npz"
    diagnostic.require_hash(data_path, dataset["sha256"])
    with np.load(data_path, allow_pickle=False) as arrays:
        observations, actions = arrays["observations"].copy(), arrays["actions"].copy()
    episodes = {row["seed"]: row for row in dataset["episodes"]}
    require(len(actions) == 117 and tuple(episodes) == SEEDS and episodes[10001]["start"] == 35,
            "The 117 ordered training transitions differ.")
    torch.set_num_threads(1)
    sources, model_hashes = {}, {}
    for name, source_run, phase_name, step, config in (
        ("resume", reference013, "treatment/td", START, settings["dqn"]),
        ("calibration", reference011, "calibration", 20000, read_json(reference011 / "settings.json")["dqn"]),
    ):
        phase = read_json(source_run / phase_name / "summary.json")
        key = f"update_{step}.pt"
        path, digest = source_run / phase_name / key, phase["checkpoint_sha256"][key]
        agent, env, _ = helper.load_verified_agent(path, digest, settings["environment"], config)
        try:
            values, identity = helper.all_q(agent, observations)
            sources[name] = {"q": values, "parameter_sha256": identity,
                             "parameters": torch.nn.utils.parameters_to_vector(agent.learner.online.parameters()).detach().clone()}
            model_hashes[str(path.resolve())] = digest
        finally:
            env.close()
    results, dense, rows_by_arm, verified = {}, {}, {}, {}
    for arm in ARMS:
        directory = (reference016 / "baseline/td") if arm == "baseline" else (run_dir / "treatment/td")
        phase = read_json(directory / "summary.json")
        validate_phase(phase, arm, summary)
        for name, digest in phase["checkpoint_sha256"].items():
            model_hashes[f"{arm}/td/{name}"] = diagnostic.require_hash(directory / name, digest)
        rows = read_json(directory / "observations.json")
        require(tuple(row["update"] for row in rows) == CHECKPOINTS, "The physical checkpoint schedule differs.")
        rows_by_arm[arm] = rows
        checkpoints, checkpoint_q, displacement = checkpoint_analysis(
            directory, phase, rows, settings, observations, actions, episodes, sources)
        sentinels, dense[arm] = analyze_sentinels(
            directory / "sentinel_q.jsonl", observations, actions, episodes[10001], checkpoint_q)
        results[arm] = {
            "origin": "reused_verified_attempt_016_baseline" if arm == "baseline" else "new_attempt_017_treatment",
            "new_optimizer_updates": 0 if arm == "baseline" else 8000,
            "new_physical_episodes": 0 if arm == "baseline" else 87,
            "checkpoints": checkpoints, "dense_sentinels": sentinels,
            "update_statistics": update_statistics(directory, settings["dqn"]["gradient_clip_norm"], 0.0 if arm == "baseline" else 1.0),
            "final_parameter_displacement_norm": displacement,
            "initial_full_batch_fit": rows[0]["fit"], "final_full_batch_fit": rows[-1]["fit"],
            "initial_return_fit_rmse": rows[0]["return_fit"]["rmse"], "final_return_fit_rmse": rows[-1]["return_fit"]["rmse"],
        }
        for name in ("summary.json", "observations.json", "updates.jsonl", "sentinel_q.jsonl"):
            verified[f"{arm}/td/{name}"] = file_hash(directory / name)
    first_a, first_b = (results[arm]["update_statistics"]["first_update"] for arm in ARMS)
    require(first_a["td_loss_before"] == first_b["td_loss_before"]
            and first_a["demonstration_loss_before"] == first_b["demonstration_loss_before"],
            "The first update does not share the original TD and supervision components.")
    for name in ("settings.json", "summary.json", "input_manifest.json", "dataset.json", "mc_targets.json", "provenance.json"):
        verified[name] = file_hash(run_dir / name)
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "run_dir": str(run_dir.resolve()),
        "script_sha256": file_hash(Path(__file__)), "prior_analysis_sha256": file_hash(PREVIOUS),
        "input_verifier_sha256": file_hash(DIRECTORY / "verify_mc_anchor.py"),
        "dataset_sha256": dataset["sha256"], "verified_files_sha256": verified,
        "verified_model_files_sha256": model_hashes, "verified_model_file_count": len(model_hashes),
        "committed_reference_evidence_verified": True,
        "initial_q_vectors_identical_to_source_on_all_117_states": True,
        "source_input_manifest_sha256": file_hash(run_dir / "input_manifest.json"),
        "source_resume_parameter_sha256": sources["resume"]["parameter_sha256"],
        "source_resume_snapshot_sha256": manifest["resumed_snapshot_sha256"],
        "historical_control_run": str(reference016.resolve()), "historical_control_arm": "baseline",
        "prior_control_reference_run": str(reference014.resolve()),
        "environment_steps": 0, "environment_resets": 0, "optimizer_updates": 0, "inference_batch_size": 1,
        "arms": results, "sampled_retention": compare_retention(rows_by_arm),
        "paired_dense_rankings": compare_dense(dense),
        "first_update_objective_comparison": {
            "shared_td_loss_before": first_a["td_loss_before"],
            "shared_margin_loss_before": first_a["demonstration_loss_before"],
            "added_return_loss_before": first_b["return_loss_before"],
            "combined_loss_difference": first_b["loss_before"] - first_a["loss_before"],
            "scope": "The first objective differs only by the coefficient-one MC term; later optimization paths and target values evolve separately.",
        },
        "limitations": [
            "Only the ongoing fixed-return Huber term is added; all demonstration coefficients remain one, with denominator117 per term.",
            "The control's 8000 updates and 87 episodes are historical reused evidence, not new attempt-017 computation.",
            "Actual demonstration returns describe the demonstration policy and are not assumed to be optimal Q values.",
            "Dense reference-state rank changes do not imply measured episode failure, particularly after a trajectory leaves the shared prefix.",
            "Three known scenes and one initialization do not establish held-out generalization or online learning.",
            "This analyzer performs no environment reset, environment step, optimizer update or forced-action rescue.",
        ],
    }, dense


def write_outputs(output, result, dense):
    """Retain complete compact series using the already validated bounded writer."""
    previous.write_outputs(output, result, dense)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite analysis: {args.output}")
    result, dense = analyze(args.run_dir.resolve())
    write_outputs(args.output.resolve(), result, dense)
    print(json.dumps({"output": str(args.output.resolve()), "sampled_retention": result["sampled_retention"]["arms"],
                      "new_environment_steps": 0, "new_optimizer_updates": 0}))


if __name__ == "__main__":
    main()
