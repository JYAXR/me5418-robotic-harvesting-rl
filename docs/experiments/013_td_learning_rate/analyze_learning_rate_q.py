"""Compare fixed-state Q rankings and update sizes in the learning-rate control."""

import argparse
import importlib.util
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
PREVIOUS = ROOT / "docs/experiments/012_divergence_control"


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helper = import_file("lr013_fixed_state_helpers", PREVIOUS / "analyze_control_q.py")
audit = import_file("lr013_input_audit", PREVIOUS / "verify_control.py")
diagnostic = helper.diagnostic
read_json, file_hash, require = helper.read_json, helper.file_hash, helper.require
ARMS, SEEDS, CHECKPOINTS = helper.ARMS, helper.SEEDS, helper.CHECKPOINTS
FOCAL_SEED, FOCAL_STEPS = 10001, (6, 18, 22, 24)
BASELINE_RATE, TREATMENT_RATE = 3e-4, 1e-4
ENTRYPOINT = "docs/experiments/013_td_learning_rate/run_learning_rate_control.py"
BASE_RUNNER = "docs/experiments/012_divergence_control/run_control.py"


def validate_protocol(protocol):
    """Reject another factor or objective before inspecting any model."""
    require(protocol["attempt"] == 13 and protocol["factor"] == "learning_rate",
            "Expected the declared attempt-013 learning-rate comparison.")
    require(protocol["baseline_value"] == BASELINE_RATE
            and protocol["treatment_value"] == TREATMENT_RATE,
            "The predeclared learning rates are 0.0003 and 0.0001.")
    require(protocol["baseline_settings"] == {
        "learning_rate": BASELINE_RATE, "demonstration_weight": 1.0,
    } and protocol["treatment_settings"] == {
        "learning_rate": TREATMENT_RATE, "demonstration_weight": 1.0,
    } and protocol["margin"] == 0.8, "Only the learning rate may differ between arms.")
    require(tuple(protocol["arms"]) == ARMS and tuple(protocol["scene_seeds"]) == SEEDS
            and tuple(protocol["td_checkpoints"]) == CHECKPOINTS
            and protocol["td_updates"] == 2000 and not protocol["early_stopping"],
            "Both arms must complete the fixed update budget and observation schedule.")
    require(protocol["preserve_adam_moments_and_parameter_steps"]
            and protocol["inherited_optimizer_updates"] == 40000
            and protocol["effective_batch_size"] == 117
            and protocol["full_batch_weighting"] == "equal_per_transition",
            "The calibrated Adam state and complete demonstration batch must be retained.")
    require(protocol["entrypoint_script"] == ENTRYPOINT
            and protocol["base_runner_script"] == BASE_RUNNER
            and protocol["base_runner_sha256"] == protocol["script_sha256"],
            "The strict entrypoint and immutable generic runner must be identified.")


def first_update_comparison(arms, changes):
    """Measure the first displacement ratio without assuming later updates scale alike."""
    baseline = arms["baseline"]["update_statistics"]["first_update"]
    treatment = arms["treatment"]["update_statistics"]["first_update"]
    fields = ("loss_before", "td_loss_before", "demonstration_loss_before", "gradient_norm")
    agreements = {name: baseline[name] == treatment[name] for name in fields}
    require(all(agreements.values()), "The first update does not share its initial objective and gradient norm.")
    require(baseline["parameter_delta_norm"] > 0 and treatment["parameter_delta_norm"] > 0,
            "A nonzero parameter change is required for the learning-rate ratio diagnostic.")
    rate_ratio = TREATMENT_RATE / BASELINE_RATE
    before, after = changes["baseline"].double(), changes["treatment"].double()
    expected = before * rate_ratio
    residual = after - expected
    expected_norm = torch.linalg.vector_norm(expected).item()
    require(expected_norm > 0, "The stored first parameter displacement is zero.")
    return {
        "equal_initial_fields": agreements,
        "learning_rate_ratio": rate_ratio,
        "logged_parameter_change_norm_ratio": treatment["parameter_delta_norm"]
        / baseline["parameter_delta_norm"],
        "stored_parameter_change_norm_ratio": torch.linalg.vector_norm(after).item()
        / torch.linalg.vector_norm(before).item(),
        "maximum_absolute_proportionality_residual": residual.abs().max().item(),
        "relative_l2_proportionality_residual": torch.linalg.vector_norm(residual).item() / expected_norm,
        "note": "Equal initial weights, targets, gradients and Adam state suggest a one-third first displacement; float32 parameter rounding can prevent an exact ratio. Later gradients and moments may diverge.",
    }


def update_statistics(path, clip_norm):
    result = helper.update_statistics(path, 2000, clip_norm)
    result.pop("combined_loss_warning")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    require(all(row["parameter_delta_norm"] >= 0 for row in records),
            "Parameter-displacement norms cannot be negative.")
    result["nonzero_parameter_update_count"] = sum(row["parameter_delta_norm"] > 0 for row in records)
    result["objective_comparison_note"] = (
        "Both arms use demonstration weight 1 and margin 0.8. Initial TD targets agree; "
        "later target networks and optimizer states may differ, so aggregate loss comparisons are descriptive."
    )
    return result


def analyze(run_dir):
    require(read_json(run_dir / "run_status.json")["status"] == "completed",
            "Learning-rate analysis requires a completed run; running or failed runs are refused.")
    settings, summary = read_json(run_dir / "settings.json"), read_json(run_dir / "summary.json")
    protocol = settings["training"]
    require(summary["protocol"] == protocol, "The summary and recorded protocol differ.")
    validate_protocol(protocol)
    for name, expected in ((ENTRYPOINT, protocol["entrypoint_sha256"]),
                           (BASE_RUNNER, protocol["base_runner_sha256"])):
        diagnostic.require_hash(ROOT / name, expected)
    manifest, reference, provenance = audit.verify_inputs(run_dir, protocol)
    reference_hashes = manifest["reference_files_sha256"]
    inherited = read_json(reference / "settings.json")
    require(settings["environment"] == inherited["environment"] and settings["dqn"] == inherited["dqn"],
            "The base environment and learner must match the calibrated reference.")
    dataset = read_json(reference / "dataset.json")
    require(dataset == read_json(run_dir / "dataset.json") == summary["dataset"]
            and dataset["sha256"] == reference_hashes["demonstration.npz"],
            "The frozen demonstration dataset differs from its reference.")
    with np.load(reference / "demonstration.npz", allow_pickle=False) as arrays:
        observations, actions = arrays["observations"].copy(), arrays["actions"].copy()
    require(len(actions) == dataset["transitions"] == 117, "Expected all 117 fixed training transitions.")
    episodes = {episode["seed"]: episode for episode in dataset["episodes"]}
    require(tuple(episodes) == SEEDS, "The demonstration episode order differs.")
    for episode in episodes.values():
        require(diagnostic.array_hash(observations[episode["start"]])
                == episode["initial_observation_sha256"], "The demonstration episode offset differs.")
    torch.set_num_threads(1)
    calibrated, env, _ = helper.load_verified_agent(
        reference / "calibration/update_20000.pt", reference_hashes["calibration/update_20000.pt"],
        settings["environment"], settings["dqn"],
    )
    try:
        initial_q, initial_hash = helper.all_q(calibrated, observations)
        initial_parameters = torch.nn.utils.parameters_to_vector(
            calibrated.learner.online.parameters()).detach().clone()
        low, span = calibrated.encoder.low.copy(), calibrated.encoder.span.copy()
    finally:
        env.close()
    verified = {name: file_hash(run_dir / name) for name in (
        "settings.json", "summary.json", "input_manifest.json", "dataset.json", "provenance.json",
    )}
    arms, verified_models, first_changes = {}, {}, {}
    for arm in ARMS:
        directory = run_dir / arm / "td"
        phase = read_json(directory / "summary.json")
        require(phase == summary["arms"][arm] and phase["optimizer_updates"] == 2000,
                f"The {arm} summary differs or its update budget is incomplete.")
        rate = BASELINE_RATE if arm == "baseline" else TREATMENT_RATE
        require(phase["demonstration_weight"] == 1.0 and phase["learning_rate"] == rate,
                f"The {arm} phase changed another objective setting.")
        names = {f"update_{step}.pt" for step in CHECKPOINTS}
        names |= {f"training_state_{step}.pt" for step in CHECKPOINTS} | {"training_state.pt"}
        require(set(phase["checkpoint_sha256"]) == names, f"The {arm} checkpoint coverage differs.")
        for name, expected in phase["checkpoint_sha256"].items():
            verified_models[f"{arm}/td/{name}"] = diagnostic.require_hash(directory / name, expected)
        rows = read_json(directory / "observations.json")
        require(tuple(row["update"] for row in rows) == CHECKPOINTS,
                f"The {arm} observation schedule differs.")
        for name in ("summary.json", "observations.json"):
            verified[f"{arm}/td/{name}"] = file_hash(directory / name)
        dqn = deepcopy(settings["dqn"])
        dqn["learning_rate"] = rate
        results = []
        final_distance = None
        for row in rows:
            step, name = row["update"], f"update_{row['update']}.pt"
            agent, env, metadata = helper.load_verified_agent(
                directory / name, phase["checkpoint_sha256"][name], settings["environment"], dqn,
            )
            try:
                require(metadata["updates"] == step and metadata["objective"] == f"{arm}/td"
                        and metadata["aggregate"] == row["aggregate"],
                        f"The {arm} checkpoint metadata differs from its observation.")
                values, parameters = helper.all_q(agent, observations)
                if step == 0:
                    require(parameters == initial_hash and np.array_equal(values, initial_q),
                            f"The {arm} initial policy or Q vectors differ from calibration.")
                if step == 1:
                    first_changes[arm] = torch.nn.utils.parameters_to_vector(
                        agent.learner.online.parameters()).detach() - initial_parameters
                if step == 2000:
                    final_parameters = torch.nn.utils.parameters_to_vector(
                        agent.learner.online.parameters()).detach()
                    final_distance = torch.linalg.vector_norm(final_parameters - initial_parameters).item()
                require(tuple(scene["seed"] for scene in row["scenes"]) == SEEDS,
                        "An observation omitted or reordered the declared training scenes.")
                scenes = []
                for scene in row["scenes"]:
                    episode, rollout = episodes[scene["seed"]], scene["rollout"]
                    divergence, prefix = helper.validate_shared_prefix(
                        rollout, episode, observations, actions, values, low, span,
                    )
                    detail = helper.state_comparison(values, initial_q, observations, actions,
                                                     episode, divergence["step"]) if divergence else None
                    scenes.append({
                        "seed": scene["seed"], "success": rollout["success"],
                        "damage": rollout["damage"], "budget_exhausted": rollout["budget_exhausted"],
                        "steps": rollout["steps"], "return": rollout["return"],
                        "stored_first_divergence": divergence, "first_divergence_q": detail,
                        "shared_state_action_checks": prefix, "shared_state_position_checks": prefix,
                    })
                focal = next(scene for scene in scenes if scene["seed"] == FOCAL_SEED)
                fixed_states = []
                for focal_step in FOCAL_STEPS:
                    detail = helper.state_comparison(values, initial_q, observations, actions,
                                                     episodes[FOCAL_SEED], focal_step)
                    detail["on_verified_common_prefix"] = focal_step <= focal["shared_state_action_checks"]
                    fixed_states.append(detail)
                results.append({"update": step, "parameter_sha256": parameters,
                                "focal_reference_states": fixed_states, "scenes": scenes,
                                "success_count": sum(scene["success"] for scene in scenes)})
            finally:
                env.close()
        arms[arm] = {
            "demonstration_weight": 1.0, "learning_rate": rate, "checkpoints": results,
            "final_parameter_displacement_norm_from_calibration": final_distance,
            "failed_updates_by_scene": {
                str(seed): [row["update"] for row in results if not next(
                    scene for scene in row["scenes"] if scene["seed"] == seed)["success"]]
                for seed in SEEDS
            },
            "update_statistics": update_statistics(directory / "updates.jsonl",
                                                     settings["dqn"]["gradient_clip_norm"]),
        }
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "run_dir": str(run_dir.resolve()),
        "script_sha256": file_hash(Path(__file__)),
        "analysis_helper_sha256": file_hash(PREVIOUS / "analyze_control_q.py"),
        "diagnostic_helper_sha256": file_hash(PREVIOUS / "diagnose_divergence.py"),
        "input_verifier_sha256": file_hash(PREVIOUS / "verify_control.py"),
        "launch_entrypoint_sha256": protocol["entrypoint_sha256"],
        "launch_base_runner_sha256": protocol["base_runner_sha256"],
        "source_files_verified": len(provenance["source_sha256"]),
        "committed_reference_evidence_verified": True,
        "dataset_sha256": dataset["sha256"], "verified_files_sha256": verified,
        "verified_model_files_sha256": verified_models, "verified_model_file_count": len(verified_models),
        "baseline_parameter_sha256": initial_hash, "initial_q_vectors_identical_on_all_117_states": True,
        "inference_batch_size": 1, "environment_steps": 0, "environment_resets": 0, "optimizer_updates": 0,
        "focal_seed": FOCAL_SEED, "focal_reference_steps": FOCAL_STEPS,
        "action_names": diagnostic.ACTION_NAMES, "arms": arms,
        "first_update_comparison": first_update_comparison(arms, first_changes),
        "limitations": [
            "Fixed reference-state Q values need not describe states visited after a trajectory diverges.",
            "Common states follow the same seed and exact recorded action prefix; no new physical replay is performed.",
            "All failures and successful reference-action disagreements remain in the analysis.",
            "Later gradients, Adam moments and TD targets may differ despite the unchanged objective weights.",
            "No new counterfactual rescue is performed for any learning-rate treatment failure.",
            "Success at sampled checkpoints does not establish success at unsampled updates or on unseen scenes.",
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
    print(json.dumps({"output": str(args.output.resolve()), "verified_models": result["verified_model_file_count"],
                      "failed_updates": {arm: result["arms"][arm]["failed_updates_by_scene"] for arm in ARMS},
                      "first_update_comparison": result["first_update_comparison"],
                      "environment_steps": 0, "optimizer_updates": 0}), flush=True)


if __name__ == "__main__":
    main()
