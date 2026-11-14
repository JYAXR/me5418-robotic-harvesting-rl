"""Audit a lower TD coefficient against the unchanged historical no-MC control."""

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = Path(__file__).resolve().parent
PREVIOUS = ROOT / "docs/experiments/017_mc_anchor/analyze_mc_anchor.py"
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


previous = import_file("td019_previous_analysis", PREVIOUS)
helper, diagnostic = previous.helper, previous.diagnostic
read_json, file_hash, require = previous.read_json, previous.file_hash, previous.require
analyze_sentinels, compare_retention = previous.analyze_sentinels, previous.compare_retention
compare_dense, state_comparison = previous.compare_dense, previous.state_comparison
checkpoint_analysis, expected_model_names = previous.checkpoint_analysis, previous.expected_model_names
write_outputs = previous.write_outputs


def update_statistics(directory, clip_norm, td_weight, start=START, end=END):
    """Check the weighted objective while retaining its unweighted component diagnostics."""
    require(not isinstance(td_weight, bool) and td_weight in (0.5, 1.0),
            "Only the declared TD coefficients 0.5 and 1 are supported.")
    path = directory / "updates.jsonl"
    result = previous.previous.previous.update_statistics(path, clip_norm, start, end)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    require(all(row["target_synchronized"] == (row["update"] % 200 == 0) for row in records),
            "Target synchronization differs from the original absolute clock.")
    fields = ("loss_before", "loss_after", "td_loss_before", "td_loss_after",
              "demonstration_loss_before", "demonstration_loss_after", "gradient_norm")
    require(all(row[key] >= 0 for row in records for key in fields), "Losses and norms must be nonnegative.")
    for row in records:
        require(not any(key.startswith(("return_loss_", "td_target_return_")) for key in row)
                and "return_anchor_weight" not in row, "No MC objective or MC update diagnostics are declared.")
        require(row.get("td_weight", 1.0) == td_weight
                and (td_weight == 1.0 or "td_weight" in row), "The logged TD coefficient differs.")
        for position in ("before", "after"):
            expected = td_weight * row[f"td_loss_{position}"] + row[f"demonstration_loss_{position}"]
            require(np.isclose(row[f"loss_{position}"], expected, atol=2e-7, rtol=1e-6),
                    "The objective is not weighted mean TD Huber plus unchanged mean supervised hinge.")
    result.update({
        "td_weight": td_weight, "mc_weight": 0.0, "normalization_denominator": 117,
        "all_supervised_sample_coefficients": 1.0,
        "target_synchronizations": sum(row["target_synchronized"] for row in records),
        "historical_control_reused": td_weight == 1.0,
        "component_loss_semantics": "TD and supervised hinge are each averaged over 117 rows. Only the TD mean is scaled; the total is not renormalized. Both logged components remain unweighted.",
        "comparison_warning": "Combined objectives differ and are not directly comparable performance scores. Shared unweighted TD/margin, calibration RMSE and actual outcomes are reported separately. Later target networks evolve separately.",
        "mc_log_availability": "Neither arm optimizes MC returns or logs per-update MC losses. Checkpoint actual-return RMSE is a read-only diagnostic; absent losses are not replaced with zeros.",
    })
    return result


def validate_phase(phase, arm, summary):
    """Require the same complete continuation while distinguishing new and reused work."""
    require(arm in ARMS and phase["optimizer_updates"] == 8000 and phase["initial_td_update"] == START
            and phase["final_td_update"] == END and phase["target_synchronizations"] == 40
            and phase["learning_rate"] == 0.0001 and phase["demonstration_weight"] == 1.0
            and phase["normalization_denominator"] == 117, "An arm changed shared training conditions.")
    if arm == "baseline":
        control = summary["historical_control"]
        require(phase["selected_sample_weight"] == 1.0
                and control["return_anchor_weight"] == 0.0 and control["all_sample_weights"] == 1.0
                and control.get("td_weight", 1.0) == 1.0
                and control["new_optimizer_updates"] == control["new_evaluation_episodes"] == control["new_environment_steps"] == 0
                and control["checkpoint_sha256"] == phase["checkpoint_sha256"],
                "Historical control is not the verified reused all-ones baseline.")
    else:
        require(phase == summary["treatment"] and phase["td_weight"] == 0.5
                and phase["return_anchor_weight"] == 0.0 and phase["all_sample_weights"] == 1.0,
                "Treatment differs from the sole TD-coefficient intervention.")
    require(set(phase["checkpoint_sha256"]) == expected_model_names(), "Expected all 67 model artifacts per arm.")


def first_objective_comparison(first_control, first_treatment):
    """Bind the first coefficient change to identical starting TD and margin losses."""
    require(first_control["td_loss_before"] == first_treatment["td_loss_before"]
            and first_control["demonstration_loss_before"] == first_treatment["demonstration_loss_before"],
            "The first update does not share the original TD and supervised components.")
    difference = first_treatment["loss_before"] - first_control["loss_before"]
    expected = -0.5 * first_control["td_loss_before"]
    require(np.isclose(difference, expected, atol=2e-7, rtol=1e-6),
            "The first combined-loss change is not solely the halved TD term.")
    return {"shared_td_loss_before": first_control["td_loss_before"],
            "shared_margin_loss_before": first_control["demonstration_loss_before"],
            "baseline_td_weight": 1.0, "treatment_td_weight": 0.5, "mc_weight": 0.0,
            "combined_loss_difference": difference, "expected_float64_difference": expected,
            "float32_arithmetic_residual": difference - expected,
            "scope": "The first objective differs only in its TD coefficient. Later states, Adam moments and Bellman targets evolve on separate trajectories."}


def simulator_accounting(accounting, certification, rows):
    """Keep copied-environment reset checks separate from autonomous policy evaluation."""
    expected = [(row["update"], scene["seed"]) for row in rows for scene in row["scenes"]]
    records = certification["records"]
    require([(row["update"], row["seed"]) for row in records] == expected
            and certification["accounting_complete"] and not certification["incomplete_resets"],
            "Complete reset certification must correspond to every scheduled evaluation.")
    for row in records:
        checks = row["reference_route_checks"]
        require(all(isinstance(check["steps"], int) and not isinstance(check["steps"], bool)
                    and check["steps"] >= 0 for check in checks), "Valid certification step counts are required.")
        require(row["certification_attempts"] == len(checks)
                and row["certification_environment_steps"] == sum(check["steps"] for check in checks),
                "A reset certification record disagrees with its retained candidate checks.")
    attempts = sum(row["certification_attempts"] for row in records)
    copied_steps = sum(row["certification_environment_steps"] for row in records)
    policy_steps = sum(row["aggregate"]["evaluation_environment_steps"] for row in rows)
    require(certification["completed_reset_count"] == accounting["environment_resets"] == len(records)
            and certification["certification_attempts"] == accounting["reset_certification_attempts"] == attempts
            and certification["certification_environment_steps"] == accounting["reset_certification_environment_steps"] == copied_steps
            and accounting["evaluation_environment_steps"] == policy_steps
            and accounting["total_simulator_environment_steps"] == policy_steps + copied_steps
            and accounting["reset_accounting_complete"] and not accounting["incomplete_resets"],
            "Policy and reset-certification simulator work must be counted separately and summed exactly.")
    return {"new_policy_evaluation_steps": policy_steps, "new_reset_certification_steps": copied_steps,
            "new_total_simulator_steps": policy_steps + copied_steps, "new_environment_resets": len(records),
            "historical_policy_evaluation_steps": accounting["reused_historical_control"]["evaluation_environment_steps"],
            "historical_reset_certification_steps": None,
            "semantics": "New policy actions and copied-simulator reset certification are distinct. Historical certification counts were not logged and are not invented as zeros or new work."}


def analyze(run_dir):
    require(read_json(run_dir / "run_status.json")["status"] == "completed",
            "TD-weight analysis requires a completed run.")
    verifier = import_file("td019_readonly_inputs", DIRECTORY / "verify_td_weight.py")
    settings, summary = read_json(run_dir / "settings.json"), read_json(run_dir / "summary.json")
    protocol = settings["training"]
    require(summary["protocol"] == protocol and protocol["attempt"] == 19, "The recorded TD-weight protocol differs.")
    verifier.verify_protocol(protocol, settings)
    manifest, reference016, reference014, reference013, reference011 = verifier.verify_inputs(run_dir, protocol)
    dataset = read_json(run_dir / "dataset.json")
    require(dataset == summary["dataset"] == read_json(reference011 / "dataset.json"),
            "The ordered demonstration dataset changed.")
    targets = read_json(run_dir / "mc_targets.json")
    require(targets == read_json(reference016 / "mc_targets.json") and targets["gamma"] == 0.99
            and len(targets["returns"]) == 117 and np.isfinite(targets["returns"]).all(),
            "The original demonstration-return evaluation targets differ.")
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
            "origin": "reused_verified_attempt_016_baseline" if arm == "baseline" else "new_attempt_019_treatment",
            "new_optimizer_updates": 0 if arm == "baseline" else 8000,
            "new_physical_episodes": 0 if arm == "baseline" else 87,
            "checkpoints": checkpoints, "dense_sentinels": sentinels,
            "update_statistics": update_statistics(directory, settings["dqn"]["gradient_clip_norm"],
                                                   1.0 if arm == "baseline" else 0.5),
            "final_parameter_displacement_norm": displacement,
            "initial_full_batch_fit": rows[0]["fit"], "final_full_batch_fit": rows[-1]["fit"],
            "initial_return_fit_rmse": rows[0]["return_fit"]["rmse"], "final_return_fit_rmse": rows[-1]["return_fit"]["rmse"],
        }
        for name in ("summary.json", "observations.json", "updates.jsonl", "sentinel_q.jsonl"):
            verified[f"{arm}/td/{name}"] = file_hash(directory / name)
    comparison = first_objective_comparison(
        results["baseline"]["update_statistics"]["first_update"],
        results["treatment"]["update_statistics"]["first_update"])
    accounting, certification = read_json(run_dir / "accounting.json"), read_json(run_dir / "reset_certification.json")
    require(accounting == summary["accounting"] and certification == summary["treatment"]["reset_certification"],
            "Stored simulator accounting differs from the completed treatment summary.")
    simulator = simulator_accounting(accounting, certification, rows_by_arm["treatment"])
    for name in ("settings.json", "summary.json", "input_manifest.json", "dataset.json", "mc_targets.json", "provenance.json",
                 "accounting.json", "reset_certification.json", "baseline_reference.json", "initial_q_equivalence.json"):
        verified[name] = file_hash(run_dir / name)
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "run_dir": str(run_dir.resolve()),
        "script_sha256": file_hash(Path(__file__)), "prior_analysis_sha256": file_hash(PREVIOUS),
        "input_verifier_sha256": file_hash(DIRECTORY / "verify_td_weight.py"),
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
        "main_simulator_accounting": simulator,
        "paired_dense_rankings": compare_dense(dense), "first_update_objective_comparison": comparison,
        "limitations": [
            "Only the mean TD Huber coefficient changes from one to one half; all supervised coefficients remain one and MC is inactive.",
            "The control's 8000 updates and 87 episodes are historical reused evidence, not new attempt-019 computation.",
            "Actual demonstration-return RMSE is measured only, not optimized or assumed to represent optimal Q values.",
            "Reset-time certification uses simulator copies and is reported separately from policy actions; unavailable historical certification counts remain unknown.",
            "Dense reference-state rank changes do not imply measured episode failure, particularly after a trajectory leaves the shared prefix.",
            "Three known scenes and one initialization do not establish held-out generalization or online learning.",
            "This analyzer performs no environment reset, environment step, optimizer update or forced-action rescue.",
        ],
    }, dense


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
