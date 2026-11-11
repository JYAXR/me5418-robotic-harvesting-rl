"""Analyze fixed-state Q margins and observed divergences after the control finishes."""

import argparse
import importlib.util
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from me5418_harvesting.agents.dqn_agent import DQNAgent

ROOT = Path(__file__).resolve().parents[3]
HELPER = Path(__file__).with_name("diagnose_divergence.py")
SPEC = importlib.util.spec_from_file_location("control_q_diagnostic_helpers", HELPER)
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)
ARMS = ("baseline", "treatment")
FOCAL_SEED = 10001
FOCAL_STEPS = (6, 18, 24)
SEEDS = (10000, 10001, 10002)
CHECKPOINTS = (*range(11), 20, 50, 100, 200, 500, 1000, 2000)
read_json = diagnostic.read_json
file_hash = diagnostic.file_hash


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_verified_agent(path, expected_sha, environment, dqn):
    diagnostic.require_hash(path, expected_sha)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(payload["environment_config"] == environment, f"Environment settings differ: {path}")
    require(json.loads(json.dumps(payload["dqn_config"])) == dqn, f"DQN settings differ: {path}")
    return DQNAgent.load(path)


def all_q(agent, observations):
    """Use the exact greedy action inference batch size on every fixed reference state."""
    before = diagnostic.parameter_hash(agent)
    values = np.stack([diagnostic.single_state_q(agent, state) for state in observations])
    require(diagnostic.parameter_hash(agent) == before, "Inference changed policy parameters.")
    require(agent.learner.update_count == 0 and not agent.learner.optimizer.state,
            "The inference-only agent performed an optimizer update.")
    return values, before


def state_comparison(values, baseline, observations, actions, episode, step):
    index = episode["start"] + step - 1
    action = int(actions[index])
    current = diagnostic.q_metrics(values[index], action)
    calibrated = diagnostic.q_metrics(baseline[index], action)
    return {
        "scene_seed": episode["seed"], "step": step, "dataset_row": index,
        "encoded_observation_sha256": diagnostic.array_hash(observations[index]),
        **current, "baseline_same_state": calibrated,
        "margin_change_from_calibration": current["signed_demonstration_margin"]
        - calibrated["signed_demonstration_margin"],
        "q_change_from_calibration": (values[index].astype(np.float64)
                                      - baseline[index].astype(np.float64)).tolist(),
    }


def validate_shared_prefix(rollout, episode, observations, actions, values, low, span):
    """Check each recorded action at the shared reference states, including the fork."""
    start, stop = episode["start"], episode["stop"]
    scene_actions = actions[start:stop]
    require(rollout["seed"] == episode["seed"], "The rollout seed differs from its episode.")
    require(rollout["steps"] == len(rollout["actions"]), "The stored rollout step count differs.")
    require(len(rollout["robot_path"]) == len(rollout["actions"]) + 1,
            "The stored robot path length differs.")
    divergence = diagnostic.first_divergence(rollout["actions"], scene_actions)
    require(divergence == rollout["first_reference_action_divergence"],
            "The stored first divergence differs from the action trace.")
    prefix_steps = divergence["step"] if divergence else len(rollout["actions"])
    require(prefix_steps <= stop - start,
            "The first divergence lies beyond the reference; a new replay would be required.")
    for index in range(prefix_steps):
        require(int(values[start + index].argmax()) == rollout["actions"][index],
                f"Batch-of-one Q inference differs from the recorded action at step {index + 1}.")
        position = observations[start + index, :2] * span + low
        require(np.array_equal(position, np.asarray(rollout["robot_path"][index], dtype=np.float32)),
                f"The recorded position differs from the shared reference state at step {index + 1}.")
    return divergence, prefix_steps


def update_statistics(path, expected_count, clip_norm):
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    require([row["update"] for row in records] == list(range(1, expected_count + 1)),
            "The update log is incomplete or has a different schedule.")
    fields = ("td_loss_before", "td_loss_after", "demonstration_loss_before",
              "demonstration_loss_after", "gradient_norm", "parameter_delta_norm")
    statistics = {}
    for name in fields:
        values = np.asarray([row[name] for row in records], dtype=np.float64)
        require(np.isfinite(values).all(), f"The update log has non-finite {name} values.")
        statistics[name] = {"mean": float(values.mean()), "minimum": float(values.min()),
                            "maximum": float(values.max())}
    return {
        "raw_sha256": file_hash(path), "records": len(records), "statistics": statistics,
        "gradient_norm_semantics": "Pre-clipping global gradient norm returned by clip_grad_norm_.",
        "gradient_clip_norm": clip_norm,
        "gradient_norm_above_clip_count": sum(row["gradient_norm"] > clip_norm for row in records),
        "gradient_norm_above_10_count": sum(row["gradient_norm"] > 10.0 for row in records),
        "demonstration_loss_semantics": "Unweighted hinge loss; both arms use the same margin.",
        "first_update": records[0],
        "combined_loss_warning": "Combined losses have different demonstration weights and must not be compared as the same objective.",
    }


def analyze(run_dir):
    require(read_json(run_dir / "run_status.json")["status"] == "completed",
            "Control analysis requires a completed run; running or failed controls are refused.")
    manifest = read_json(run_dir / "input_manifest.json")
    settings = read_json(run_dir / "settings.json")
    summary = read_json(run_dir / "summary.json")
    protocol = settings["training"]
    require(summary["protocol"] == protocol, "The summary and recorded protocol differ.")
    require(tuple(protocol["scene_seeds"]) == SEEDS and tuple(protocol["arms"]) == ARMS
            and tuple(protocol["td_checkpoints"]) == CHECKPOINTS and protocol["td_updates"] == 2000,
            "The control schedule differs from the declared attempt 012 scope.")
    require(protocol["factor"] == "demonstration_weight" and protocol["baseline_value"] == 1.0
            and protocol["treatment_value"] == 2.0 and protocol["margin"] == 0.8,
            "This analysis expects the declared demonstration-weight 1-to-2 comparison.")
    reference = Path(manifest["reference_run"])
    reference_hashes = manifest["reference_files_sha256"]
    verified = {"control_input_manifest": file_hash(run_dir / "input_manifest.json"),
                "control_settings": file_hash(run_dir / "settings.json"),
                "control_summary": file_hash(run_dir / "summary.json")}
    for name in ("settings.json", "dataset.json", "demonstration.npz", "calibration/update_20000.pt"):
        verified[f"reference/{name}"] = diagnostic.require_hash(reference / name, reference_hashes[name])
    inherited_settings = read_json(reference / "settings.json")
    require(settings["environment"] == inherited_settings["environment"]
            and settings["dqn"] == inherited_settings["dqn"],
            "The control base settings differ from its verified inherited settings.")
    dataset = read_json(reference / "dataset.json")
    require(dataset == read_json(run_dir / "dataset.json") == summary["dataset"],
            "The reference, local and summary dataset records differ.")
    require(dataset["sha256"] == reference_hashes["demonstration.npz"],
            "The NPZ digest differs from dataset metadata.")
    for group in ("reference_source_sha256", "config_sha256", "helper_sha256"):
        for name, expected in manifest[group].items():
            diagnostic.require_hash(ROOT / name, expected)
    with np.load(reference / "demonstration.npz", allow_pickle=False) as arrays:
        observations, actions = arrays["observations"].copy(), arrays["actions"].copy()
    require(len(actions) == dataset["transitions"] == 117, "Expected 117 fixed training transitions.")
    episodes = {episode["seed"]: episode for episode in dataset["episodes"]}
    require(tuple(episodes) == SEEDS, "The demonstration episode order differs.")
    for episode in episodes.values():
        require(diagnostic.array_hash(observations[episode["start"]])
                == episode["initial_observation_sha256"], "The demonstration episode offset differs.")
    torch.set_num_threads(1)
    calibrated, env, _ = load_verified_agent(
        reference / "calibration/update_20000.pt", reference_hashes["calibration/update_20000.pt"],
        settings["environment"], settings["dqn"],
    )
    try:
        baseline_values, baseline_parameter_sha = all_q(calibrated, observations)
        low, span = calibrated.encoder.low.copy(), calibrated.encoder.span.copy()
    finally:
        env.close()
    arms, verified_models, initial_values = {}, {}, {}
    for arm in ARMS:
        directory = run_dir / arm / "td"
        phase = read_json(directory / "summary.json")
        require(phase == summary["arms"][arm], f"The {arm} phase summary differs from the root summary.")
        require(phase["optimizer_updates"] == 2000, f"The {arm} phase did not finish all updates.")
        require(phase["demonstration_weight"] == protocol[f"{arm}_settings"]["demonstration_weight"]
                and phase["learning_rate"] == protocol[f"{arm}_settings"]["learning_rate"],
                f"The {arm} phase settings differ from the declared intervention.")
        verified[f"{arm}/td/summary.json"] = file_hash(directory / "summary.json")
        for name, expected in phase["checkpoint_sha256"].items():
            verified_models[f"{arm}/td/{name}"] = diagnostic.require_hash(directory / name, expected)
        rows = read_json(directory / "observations.json")
        require(tuple(row["update"] for row in rows) == CHECKPOINTS,
                f"The {arm} observed checkpoint schedule differs.")
        verified[f"{arm}/td/observations.json"] = file_hash(directory / "observations.json")
        dqn = deepcopy(settings["dqn"])
        if arm == "treatment" and protocol["factor"] == "learning_rate":
            dqn["learning_rate"] = protocol["treatment_value"]
        results = []
        for row in rows:
            update = row["update"]
            name = f"update_{update}.pt"
            agent, env, metadata = load_verified_agent(
                directory / name, phase["checkpoint_sha256"][name], settings["environment"], dqn,
            )
            try:
                require(metadata["updates"] == update and metadata["objective"] == f"{arm}/td"
                        and metadata["aggregate"] == row["aggregate"],
                        f"The {arm} checkpoint metadata differs from its observation.")
                values, parameters = all_q(agent, observations)
                if update == 0:
                    require(parameters == baseline_parameter_sha and np.array_equal(values, baseline_values),
                            f"The {arm} initial Q values differ from the shared calibrated policy.")
                    initial_values[arm] = values.copy()
                scenes = []
                require(tuple(scene["seed"] for scene in row["scenes"]) == SEEDS,
                        "The observation omitted or reordered a training scene.")
                for scene in row["scenes"]:
                    episode = episodes[scene["seed"]]
                    rollout = scene["rollout"]
                    divergence, prefix = validate_shared_prefix(
                        rollout, episode, observations, actions, values, low, span,
                    )
                    detail = state_comparison(values, baseline_values, observations, actions,
                                              episode, divergence["step"]) if divergence else None
                    scenes.append({
                        "seed": scene["seed"], "success": rollout["success"],
                        "damage": rollout["damage"], "budget_exhausted": rollout["budget_exhausted"],
                        "steps": rollout["steps"], "return": rollout["return"],
                        "stored_first_divergence": divergence, "first_divergence_q": detail,
                        "shared_state_action_checks": prefix,
                        "shared_state_position_checks": prefix,
                    })
                focal = next(scene for scene in scenes if scene["seed"] == FOCAL_SEED)
                fixed_states = []
                for step in FOCAL_STEPS:
                    detail = state_comparison(values, baseline_values, observations, actions,
                                              episodes[FOCAL_SEED], step)
                    detail["on_verified_common_prefix"] = step <= focal["shared_state_action_checks"]
                    fixed_states.append(detail)
                results.append({"update": update, "parameter_sha256": parameters,
                                "focal_reference_states": fixed_states, "scenes": scenes,
                                "success_count": sum(scene["success"] for scene in scenes)})
            finally:
                env.close()
        arms[arm] = {
            "demonstration_weight": phase["demonstration_weight"], "learning_rate": phase["learning_rate"],
            "checkpoints": results,
            "failed_updates_by_scene": {
                str(seed): [row["update"] for row in results
                            if not next(scene for scene in row["scenes"] if scene["seed"] == seed)["success"]]
                for seed in SEEDS
            },
            "update_statistics": update_statistics(directory / "updates.jsonl", 2000,
                                                     settings["dqn"]["gradient_clip_norm"]),
        }
    require(np.array_equal(initial_values["baseline"], initial_values["treatment"]),
            "The two control arms have different initial Q vectors.")
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "run_dir": str(run_dir.resolve()),
        "script_sha256": file_hash(Path(__file__)), "helper_sha256": file_hash(HELPER),
        "dataset_sha256": dataset["sha256"], "verified_files_sha256": verified,
        "verified_model_files_sha256": verified_models, "verified_model_file_count": len(verified_models),
        "baseline_parameter_sha256": baseline_parameter_sha,
        "initial_q_vectors_identical_on_all_117_states": True,
        "inference_batch_size": 1, "environment_steps": 0, "environment_resets": 0,
        "optimizer_updates": 0, "focal_seed": FOCAL_SEED, "focal_reference_steps": FOCAL_STEPS,
        "action_names": diagnostic.ACTION_NAMES, "arms": arms,
        "limitations": [
            "Fixed reference-state Q values need not describe states visited after a trajectory diverges.",
            "Common states are established from the same seed and exact recorded action prefix; no new physical replay is performed.",
            "All failures and successful reference-action deviations remain in the analysis.",
            "Combined objective losses differ in weighting across arms; compare their unweighted components separately.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New JSON output file; existing files are refused.")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite analysis: {args.output}")
    result = analyze(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "verified_models": result["verified_model_file_count"],
                      "failed_updates": {arm: result["arms"][arm]["failed_updates_by_scene"] for arm in ARMS},
                      "environment_steps": 0, "optimizer_updates": 0}), flush=True)


if __name__ == "__main__":
    main()
