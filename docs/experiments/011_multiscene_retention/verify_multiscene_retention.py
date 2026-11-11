"""Independently audit multi-scene retention, episode returns and resumed Adam state."""

import argparse
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from me5418_harvesting.agents.dqn_agent import DQNAgent
from me5418_harvesting.agents.dqn_learning import TransitionBatch
from me5418_harvesting.agents.optimizer_transition import optimizer_state_summary

ROOT = Path(__file__).resolve().parents[3]
RUNNER = Path(__file__).with_name("run_multiscene_retention.py")
INDEPENDENT_HELPER = ROOT / "docs/experiments/010_adam_state/verify_adam_state.py"


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = import_file("multiscene_independent_adam", INDEPENDENT_HELPER)
diagnostic = import_file("multiscene_retention_diagnostic", RUNNER)
read_json, read_log = audit.read_json, audit.read_log
file_hash, require, require_equal = audit.file_hash, audit.require, audit.require_equal


def episode_batch(batch, entry):
    """Slice without using the production episode-boundary helper."""
    return TransitionBatch(**{
        name: getattr(batch, name)[entry["start"]:entry["stop"]].detach().clone()
        for name in TransitionBatch.__dataclass_fields__
    })


def independent_returns(batch, episodes, gamma, recorded):
    """Reset float32 and float64 accumulators at each independently checked terminal."""
    require(batch.rewards.dtype == torch.float32, "expected stored float32 rewards")
    accumulated, double_precision, counts, cursor = [], [], [], 0
    seen = set()
    for entry in episodes:
        start, stop, seed = entry["start"], entry["stop"], entry["seed"]
        require(seed not in seen and start == cursor and stop > start,
                "unique ordered episode offsets with no missing or duplicated transitions")
        seen.add(seed)
        cursor = stop
        part = episode_batch(batch, entry)
        require(len(part.actions) == entry["transitions"] == stop - start,
                f"scene {seed} episode length")
        require(part.terminated.tolist() == [False] * (stop - start - 1) + [True]
                and not part.truncated.any().item(), f"scene {seed} complete terminal episode")
        require(torch.equal(part.next_observations[:-1], part.observations[1:]),
                f"scene {seed} within-episode continuity")
        rewards = part.rewards.numpy()
        first = np.empty(len(rewards), dtype=np.float32)
        second = np.empty(len(rewards), dtype=np.float64)
        tail32, tail64 = np.float32(0), 0.0
        for index in range(len(rewards) - 1, -1, -1):
            tail32 = np.float32(rewards[index] + np.float32(gamma) * tail32)
            tail64 = float(rewards[index]) + gamma * tail64
            first[index], second[index] = tail32, tail64
        require(float(first[-1]) == float(rewards[-1]), f"scene {seed} final reward counted once")
        require(np.isfinite(first).all() and np.isfinite(second).all(),
                f"scene {seed} finite return recurrence")
        accumulated.extend(first.tolist())
        double_precision.extend(second.tolist())
        counts.append({"seed": seed, "transitions": len(rewards),
                       "initial_discounted_return": float(first[0]),
                       "maximum_float64_difference": float(np.max(np.abs(first.astype(float) - second)))})
    require(cursor == len(batch.actions) == len(recorded), "complete multi-scene return coverage")
    values = np.asarray(accumulated, dtype=np.float32)
    expected = np.asarray(recorded, dtype=np.float32)
    require(np.array_equal(values, expected), "independently computed float32 returns match exactly")
    difference = float(np.max(np.abs(values.astype(float) - np.asarray(double_precision))))
    require(difference <= 1e-4, "independent float64 return tolerance")
    return torch.from_numpy(values), counts, difference


def verify_aggregate(row, episodes):
    seeds = [entry["seed"] for entry in episodes]
    require([scene["seed"] for scene in row["scenes"]] == seeds,
            "all predeclared scenes appear in every observation")
    for scene in row["scenes"]:
        require(scene["rollout"]["training_scene_only"]
                and scene["rollout"]["seed"] == scene["seed"], "training-scene-only rollout identity")
    expected = {
        "scene_count": len(seeds),
        "success_count": sum(scene["rollout"]["success"] for scene in row["scenes"]),
        "success_seeds": [scene["seed"] for scene in row["scenes"] if scene["rollout"]["success"]],
        "damage_count": sum(scene["rollout"]["damage"] for scene in row["scenes"]),
        "budget_exhausted_count": sum(scene["rollout"]["budget_exhausted"] for scene in row["scenes"]),
        "evaluation_environment_steps": sum(scene["rollout"]["steps"] for scene in row["scenes"]),
    }
    require(row["aggregate"] == expected, "aggregate outcomes reproduce per-scene evidence")
    require(row["fit"]["transitions"] == sum(scene["fit"]["transitions"] for scene in row["scenes"]),
            "pooled action agreement covers every transition")
    chosen = [value for scene in row["scenes"] for value in scene["return_fit"]["chosen_values"]]
    targets = [value for scene in row["scenes"] for value in scene["return_fit"]["mc_targets"]]
    # Different GEMM batch shapes can round differently; reloads use exact shapes.
    require(np.allclose(row["return_fit"]["chosen_values"], chosen, rtol=1e-5, atol=1e-5)
            and row["return_fit"]["mc_targets"] == targets,
            "pooled value errors preserve episode ordering")


def verify_retention(record, rows, preceding):
    if preceding is None:
        require(record is None, "pretraining is acquisition, not retention of an earlier stage")
        return
    eligible = {scene["seed"]: bool(scene["rollout"]["success"])
                for scene in preceding["scenes"]}
    acquired = [seed for seed, success in eligible.items() if success]
    require(record["eligible_seeds"] == acquired
            and record["all_scene_retention_eligible"] == all(eligible.values()),
            "retention eligibility uses each preceding final rollout")
    retained = {}
    require(set(record["per_scene"]) == {str(seed) for seed in eligible},
            "every scene receives an explicit retention classification")
    for seed, success in eligible.items():
        outcomes = [(row["update"], next(scene for scene in row["scenes"]
                                        if scene["seed"] == seed)["rollout"]["success"])
                    for row in rows]
        failures = [step for step, outcome in outcomes if not outcome]
        retained[seed] = success and not failures
        expected = {
            "retention_eligible": success, "failed_sampled_updates": failures,
            "first_sampled_failure_update": min(failures) if failures else None,
            "retained_at_all_sampled_checkpoints": retained[seed],
            "interpretation": "retention_of_acquired_scene" if success
            else "downstream_behavior_after_previously_failed_scene",
            "first_ten_updates_observation_count": sum(1 <= step <= 10 for step, _ in outcomes),
            "first_ten_updates_success_count": sum(outcome for step, outcome in outcomes
                                                   if 1 <= step <= 10),
        }
        require(record["per_scene"][str(seed)] == expected,
                f"scene {seed} acquisition, recovery and retention are separated")
    require(record["all_scenes_retained_at_all_sampled_checkpoints"] == all(retained.values())
            and record["acquired_subset_retained_at_all_sampled_checkpoints"]
            == (bool(acquired) and all(retained[seed] for seed in acquired)),
            "an empty acquired subset cannot satisfy retention")


def verify_snapshot(path, expected_hash, settings, updates, target_hash, optimizer_record):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(payload["format_version"] == 1 and payload["kind"] == "fixed_data_optimizer_boundary"
            and payload["update_count"] == updates, f"{path.name} full snapshot format and clock")
    require(json.loads(json.dumps(payload["dqn_config"])) == settings["dqn"]
            and payload["environment_config"] == settings["environment"],
            f"{path.name} snapshot configuration")
    require(audit.state_hash(payload["online"]) == payload["parameter_sha256"] == expected_hash,
            f"{path.name} online parameters")
    require(audit.state_hash(payload["target"]) == target_hash, f"{path.name} target parameters")
    agent, env = audit.independent_restore(payload)
    try:
        require(optimizer_state_summary(agent.learner) == payload["optimizer_summary"] == optimizer_record,
                f"{path.name} Adam settings, moments and internal steps")
        require_equal(agent.learner.optimizer.state_dict(), payload["optimizer"],
                      f"{path.name} independently restored Adam tensors")
    finally:
        env.close()
    return payload


def verify_phase(run_dir, record, settings, episodes, checkpoints, count, prior,
                 expected_start_hash, preceding=None):
    directory = record["directory"]
    path = run_dir / directory
    require(read_json(path / "summary.json") == record, f"{directory} summary copy")
    logs, rows = read_log(path / "updates.jsonl"), read_json(path / "observations.json")
    require([row["update"] for row in logs] == list(range(1, count + 1)),
            f"{directory} complete recorded update sequence")
    require([row["update"] for row in rows] == checkpoints, f"{directory} complete observation schedule")
    is_td = directory.endswith("/td")
    expected_sync = [is_td and step % settings["dqn"]["target_update_interval"] == 0
                     for step in range(1, count + 1)]
    require([row["target_synchronized"] for row in logs] == expected_sync
            and record["target_synchronizations"] == sum(expected_sync),
            f"{directory} target synchronization schedule")
    require(record["optimizer_updates"] == count and record["prior_updates_on_branch"] == prior
            and record["total_updates_on_branch"] == prior + count
            and record["initial_parameter_sha256"] == expected_start_hash,
            f"{directory} initial weights and update history")
    require(rows[0] == record["initial_observation"] and rows[-1] == record["final_observation"],
            f"{directory} endpoint observations")
    require(set(record["checkpoint_sha256"]) == {f"update_{step}.pt" for step in checkpoints}
            | {"training_state.pt"}, f"{directory} complete saved checkpoint inventory")
    for name, expected in record["checkpoint_sha256"].items():
        require(file_hash(path / name) == expected, f"{directory}/{name} checkpoint hash")
    for row in rows:
        step = row["update"]
        verify_aggregate(row, episodes)
        require(row["total_updates_on_branch"] == prior + step, f"{directory} observation update history")
        payload = torch.load(path / f"update_{step}.pt", map_location="cpu", weights_only=True)
        require(json.loads(json.dumps(payload["dqn_config"])) == settings["dqn"]
                and payload["environment_config"] == settings["environment"],
                f"{directory}/{step} inference configuration")
        metadata = payload["metadata"]
        require(metadata["phase"] == "multiscene_retention_diagnostic"
                and metadata["objective"] == directory and metadata["updates"] == step
                and metadata["prior_updates_on_branch"] == prior
                and metadata["total_updates_on_branch"] == prior + step
                and metadata["training_scene_only"]
                and metadata["scene_seeds"] == [entry["seed"] for entry in episodes]
                and metadata["fit"] == row["fit"] and metadata["aggregate"] == row["aggregate"],
                f"{directory}/{step} inference metadata")
        if step == 0:
            require(audit.state_hash(payload["online"]) == expected_start_hash,
                    f"{directory} saved starting weights")
        if step == count:
            require(audit.state_hash(payload["online"]) == record["final_parameter_sha256"],
                    f"{directory} saved ending weights")
        adam_step = step + (20000 if directory == "preserve/td" else 0)
        optimizer = row["optimizer_state"]
        if adam_step:
            require(optimizer["initialized_parameter_count"] == optimizer["parameter_count"]
                    and optimizer["step_min"] == optimizer["step_max"] == adam_step,
                    f"{directory}/{step} Adam bias-correction clock")
        else:
            require(optimizer["initialized_parameter_count"] == 0 and optimizer["steps"] == {},
                    f"{directory} initially empty Adam state")
    if preceding is not None:
        for key in ("fit", "return_fit", "scenes", "aggregate"):
            require(rows[0][key] == preceding[key], f"{directory} unchanged initial {key}")
    verify_retention(record["retention"], rows, preceding)
    require(record["evaluation_environment_steps"] == sum(
        row["aggregate"]["evaluation_environment_steps"] for row in rows),
        f"{directory} all-scene evaluation accounting")
    snapshot_path = run_dir / record["training_snapshot_file"]
    require(snapshot_path == path / "training_state.pt"
            and file_hash(snapshot_path) == record["training_snapshot_sha256"],
            f"{directory} full training snapshot identity")
    target_hash = record["final_parameter_sha256"] if is_td else expected_start_hash
    payload = verify_snapshot(snapshot_path, record["final_parameter_sha256"], settings,
                              count, target_hash, rows[-1]["optimizer_state"])
    return rows, logs, payload


def verify_boundary(boundary, previous, initial, expected_hash, prior, preserve):
    require(boundary["previous_phase_updates"] == 20000 and boundary["phase_updates"] == 0
            and boundary["target_synchronized"] and boundary["target_equals_online"]
            and boundary["policy_parameter_sha256"] == expected_hash
            and boundary["total_prior_updates_on_branch"] == prior
            and boundary["optimizer_reset"] == (not preserve), "controlled phase boundary")
    before, after = boundary["optimizer_before"], boundary["optimizer_after"]
    require(before == previous and after == initial
            and after["settings_sha256"] == before["settings_sha256"],
            "phase boundary preserves optimizer options and declared incoming state")
    if preserve:
        require(after == before, "preserved Adam state is unchanged at the boundary")
    else:
        require(after["initialized_parameter_count"] == 0 and after["steps"] == {}
                and after["first_moment_l2"] == after["second_moment_l2"] == 0,
                "reset state has no moments or per-parameter steps")


def reload_policy(run_dir, directory, step, row, batch, returns, episodes):
    checkpoint = run_dir / directory / f"update_{step}.pt"
    agent, env, metadata = DQNAgent.load(checkpoint)
    try:
        require(diagnostic.fit.fit_metrics(agent, batch, 0.8) == row["fit"] == metadata["fit"],
                f"{directory}/{step} reloaded pooled fit")
        require(diagnostic.calibration.return_fit(agent, batch, returns) == row["return_fit"],
                f"{directory}/{step} reloaded pooled return metrics")
        verified = []
        for entry, scene in zip(episodes, row["scenes"]):
            part = episode_batch(batch, entry)
            selected_returns = returns[entry["start"]:entry["stop"]]
            require(diagnostic.fit.fit_metrics(agent, part, 0.8) == scene["fit"],
                    f"{directory}/{step}/{entry['seed']} reloaded scene fit")
            require(diagnostic.calibration.return_fit(agent, part, selected_returns) == scene["return_fit"],
                    f"{directory}/{step}/{entry['seed']} reloaded scene return metrics")
            rollout = diagnostic.fit.fixed_scene_rollout(agent, env, entry["seed"], part)
            require(rollout == scene["rollout"],
                    f"{directory}/{step}/{entry['seed']} exact reloaded trajectory")
            verified.append({"seed": entry["seed"], "full_rollout_matches_exactly": True,
                             "fit_and_return_metrics_match_exactly": True,
                             "success": rollout["success"], "damage": rollout["damage"],
                             "reload_environment_steps": rollout["steps"]})
        return {"directory": directory, "update": step, "scenes": verified,
                "checkpoint_sha256": file_hash(checkpoint)}
    finally:
        env.close()


def verify(run_dir, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite multi-scene verification evidence: {output}")
    torch.set_num_threads(1)
    summary, settings = read_json(run_dir / "summary.json"), read_json(run_dir / "settings.json")
    protocol = summary["protocol"]
    require(read_json(run_dir / "run_status.json")["status"] == "completed", "completed experiment")
    require(protocol == settings["training"] and protocol["scene_seeds"] == [10000, 10001, 10002]
            and protocol["initialization_seed"] == 0 and protocol["fresh_initialization"]
            and protocol["arms"] == ["reset", "preserve"], "predeclared scenes and initialization")
    require(protocol["pretraining_updates"] == protocol["calibration_updates"] == 20000
            and protocol["td_updates"] == 2000 and not protocol["early_stopping"]
            and protocol["continue_after_unsuccessful_calibration"], "fixed budgets and honest continuation")
    require(protocol["pretraining_checkpoints"] == [0, 1000, 5000, 10000, 20000]
            and protocol["calibration_checkpoints"] == [0, 1, 10, 100, 1000, 5000, 10000, 20000]
            and protocol["td_checkpoints"] == [*range(11), 20, 50, 100, 200, 500, 1000, 2000],
            "predeclared complete observation schedules")
    require(protocol["gamma"] == settings["dqn"]["gamma"] == 0.99
            and protocol["learning_rate"] == settings["dqn"]["learning_rate"] == 0.0003
            and protocol["margin"] == 0.8 and protocol["demonstration_weight"] == 1.0
            and settings["dqn"]["target_update_interval"] == 200
            and settings["dqn"]["hidden_sizes"] == [256, 256]
            and settings["dqn"]["gradient_clip_norm"] == 10,
            "unchanged declared model and learning hyperparameters")
    require(protocol["loss_weighting"] == "equal_per_transition_not_equal_per_scene"
            and protocol["seed_id_in_network_input"] is False, "declared weighting and input semantics")
    require(file_hash(RUNNER) == protocol["script_sha256"], "experiment runner hash")
    provenance = read_json(run_dir / "provenance.json")
    for name, expected in provenance["source_sha256"].items():
        require(file_hash(ROOT / name) == expected, f"launch source: {name}")
    manifest = read_json(run_dir / "input_manifest.json")
    require(manifest["inherited_checkpoint_used"] is False
            and manifest["helper_sha256"] == protocol["helper_sha256"], "fresh model input provenance")
    for group in ("config_sha256", "helper_sha256"):
        for name, expected in manifest[group].items():
            require(file_hash(ROOT / name) == expected, f"{group}: {name}")
    require(diagnostic.read_mapping(ROOT / "configs/environment_multiroute.yaml") == settings["environment"]
            and json.loads(json.dumps(asdict(audit.DQNConfig(**diagnostic.read_mapping(
                ROOT / "configs/dqn_multiroute_pilot.yaml"))))) == settings["dqn"],
            "resolved configurations match hashed input files")
    dataset, targets = read_json(run_dir / "dataset.json"), read_json(run_dir / "mc_targets.json")
    require(dataset == summary["dataset"] and dataset["seeds"] == protocol["scene_seeds"]
            and dataset["observation_size"] == 341
            and dataset["full_batch_weighting"] == "equal_per_transition"
            and dataset["scene_seed_in_network_input"] is False
            and file_hash(run_dir / "demonstration.npz") == dataset["sha256"], "frozen encoded dataset")
    with np.load(run_dir / "demonstration.npz", allow_pickle=False) as encoded:
        batch = TransitionBatch(**{name: torch.from_numpy(encoded[name])
                                   for name in TransitionBatch.__dataclass_fields__})
    episodes = dataset["episodes"]
    require([entry["seed"] for entry in episodes] == protocol["scene_seeds"]
            and dataset["transitions"] == dataset["effective_batch_size"] == len(batch.actions),
            "all declared scene trajectories in the full batch")
    require(targets["gamma"] == protocol["gamma"] and targets["rewards"] == batch.rewards.tolist()
            and targets["network_bootstrapping_used"] is False and targets["episode_boundaries_respected"],
            "observed rewards and non-bootstrapped targets")
    returns, return_checks, precision_difference = independent_returns(
        batch, episodes, protocol["gamma"], targets["returns"],
    )
    require(len(targets["episodes"]) == len(episodes), "one return check per episode")
    initial_hashes = []
    for entry, target_episode in zip(episodes, targets["episodes"]):
        part = episode_batch(batch, entry)
        initial_hash = audit.hashlib.sha256(part.observations[0].numpy().tobytes()).hexdigest()
        initial_hashes.append(initial_hash)
        require(entry["initial_observation_sha256"] == initial_hash
                and entry["action_label_counts"] == torch.bincount(part.actions, minlength=8).tolist(),
                f"scene {entry['seed']} encoded inputs and action labels")
        require(all(target_episode[name] == entry[name] for name in target_episode if name != "numerical_check"),
                f"scene {entry['seed']} return boundary metadata")
        check = target_episode["numerical_check"]
        require(check["rewards"] == part.rewards.tolist()
                and check["returns"] == returns[entry["start"]:entry["stop"]].tolist()
                and check["complete_ordered_terminal_trajectory_verified"]
                and check["network_bootstrapping_used"] is False,
                f"scene {entry['seed']} recorded complete return check")
    require(len(set(initial_hashes)) == dataset["distinct_initial_observations"] == 3,
            "three distinct scene initial observations")
    require(dataset["action_label_counts"] == torch.bincount(batch.actions, minlength=8).tolist(),
            "combined demonstration action counts")
    unique, inverse, counts = np.unique(batch.observations.numpy(), axis=0,
                                       return_inverse=True, return_counts=True)
    conflicts = sum(len(np.unique(batch.actions.numpy()[inverse == index])) > 1
                    for index in range(len(unique)) if counts[index] > 1)
    require(dataset["unique_observations"] == len(unique)
            and dataset["duplicate_observation_rows"] == len(batch.actions) - len(unique)
            and dataset["conflicting_duplicate_observation_groups"] == conflicts,
            "exact duplicate and conflicting-label diagnostics")
    collection = read_json(run_dir / "demonstrations.json")
    require(collection == summary["demonstration_collection"]
            and collection["requested_seeds"] == protocol["scene_seeds"]
            and set(collection["collections"]) == {str(seed) for seed in protocol["scene_seeds"]},
            "all collection attempts recorded without replacement seeds")
    collection_actions = 0
    for entry in episodes:
        data = collection["collections"][str(entry["seed"])]
        require(data["requested_episodes"] == data["successful_episodes"] == 1
                and data["accepted_transitions"] == entry["transitions"]
                and data["optimizer_updates"] == 0 and len(data["records"]) == 1,
                "one physically executed successful demonstration per seed")
        episode = data["records"][0]
        require(episode["seed"] == entry["seed"] and episode["success"] and not episode["damage"]
                and episode["steps"] == data["collection_environment_steps"] == entry["transitions"],
                "physical collection action accounting")
        collection_actions += data["collection_environment_steps"]
    require(collection["successful_episodes"] == 3 and collection["accepted_transitions"] == len(batch.actions)
            and collection["collection_environment_steps"] == collection_actions,
            "combined collection accounting")

    initial = read_json(run_dir / "initial_training_snapshot.json")
    initial_path = run_dir / initial["file"]
    require(file_hash(initial_path) == initial["sha256"], "initial full snapshot hash")
    initial_payload = verify_snapshot(initial_path, initial["parameter_sha256"], settings, 0,
                                      initial["parameter_sha256"], initial["optimizer_summary"])
    require(initial_payload["optimizer_summary"]["initialized_parameter_count"] == 0,
            "fresh model has no inherited Adam state")
    torch.manual_seed(protocol["initialization_seed"])
    init_env = audit.HarvestingEnv(settings["environment"])
    try:
        fresh = DQNAgent(init_env, audit.DQNConfig(**settings["dqn"]), seed=protocol["initialization_seed"])
        require(diagnostic.fit.parameter_hash(fresh) == initial["parameter_sha256"],
                "independently reproduced fresh seed-zero model")
    finally:
        init_env.close()
    pretraining = summary["pretraining"]
    pre_rows, _, _ = verify_phase(run_dir, pretraining, settings, episodes,
                                  protocol["pretraining_checkpoints"], 20000, 0,
                                  initial["parameter_sha256"])
    calibrated = summary["calibration"]
    calibration_rows, _, calibrated_payload = verify_phase(
        run_dir, calibrated, settings, episodes, protocol["calibration_checkpoints"], 20000, 20000,
        pretraining["final_parameter_sha256"], pre_rows[-1],
    )
    calibration_boundary = read_json(run_dir / "calibration_boundary.json")
    require(calibration_boundary == summary["calibration_boundary"], "calibration boundary copy")
    verify_boundary(calibration_boundary, pre_rows[-1]["optimizer_state"],
                    calibration_rows[0]["optimizer_state"], pretraining["final_parameter_sha256"],
                    20000, False)
    verified_arms, policy_reloads = {}, []
    for directory, rows in (("pretraining", pre_rows), ("calibration", calibration_rows)):
        policy_reloads.append(reload_policy(run_dir, directory, 20000, rows[-1], batch, returns, episodes))
    evaluation_actions = pretraining["evaluation_environment_steps"] + calibrated["evaluation_environment_steps"]
    for branch in protocol["arms"]:
        record = summary["arms"][branch]
        rows, _, _ = verify_phase(run_dir, record, settings, episodes,
                                  protocol["td_checkpoints"], 2000, 40000,
                                  calibrated["final_parameter_sha256"], calibration_rows[-1])
        boundary = read_json(run_dir / branch / "td_boundary.json")
        require(boundary == summary["td_boundaries"][branch], f"{branch} boundary copy")
        verify_boundary(boundary, calibration_rows[-1]["optimizer_state"], rows[0]["optimizer_state"],
                        calibrated["final_parameter_sha256"], 40000, branch == "preserve")
        early = audit.replay_early_updates(calibrated_payload, branch, run_dir, batch, protocol,
                                          rows, rows[0]["optimizer_state"])
        for step in (1, 2000):
            row = next(item for item in rows if item["update"] == step)
            policy_reloads.append(reload_policy(run_dir, f"{branch}/td", step, row, batch, returns, episodes))
        verified_arms[branch] = {
            "checkpoints_verified": len(rows), "update_records_verified": record["optimizer_updates"],
            "full_training_snapshot_verified": True,
            "independent_early_replay": early, "per_scene_retention_verified": record["retention"],
        }
        evaluation_actions += record["evaluation_environment_steps"]
    accounting = read_json(run_dir / "accounting.json")
    require(all(summary[name] == value for name, value in accounting.items()), "copied run accounting")
    actual_logs = {path.relative_to(run_dir).as_posix(): len(read_log(path))
                   for path in run_dir.rglob("updates.jsonl")}
    require(accounting["update_records_per_log"] == actual_logs
            and accounting["optimizer_updates_total"] == sum(actual_logs.values()) == 44000
            and accounting["td_optimizer_updates"] == 4000, "actual optimizer update accounting")
    require(accounting["evaluation_environment_steps"] == evaluation_actions
            and accounting["demonstration_collection_steps"] == collection_actions
            and accounting["inherited_optimizer_updates"] == accounting["online_training_steps"]
            == accounting["held_out_scenes_accessed"] == 0, "action and data access accounting")
    result = {
        "source_files_verified": len(provenance["source_sha256"]),
        "config_input_files_verified": len(manifest["config_sha256"]),
        "runner_and_helper_hashes_verified": True, "dataset_hash_verified": True,
        "independently_verified_episode_returns": return_checks,
        "mc_float64_maximum_difference": precision_difference,
        "fresh_initialization_verified": True, "full_training_snapshots_verified": 5,
        "pretraining_checkpoints_verified": len(pre_rows),
        "calibration_checkpoints_verified": len(calibration_rows),
        "all_update_records_verified": 44000, "arms": verified_arms,
        "reloaded_policies": policy_reloads,
        "additional_verification_optimizer_updates": 20,
        "additional_verification_episodes": sum(len(item["scenes"]) for item in policy_reloads),
        "additional_verification_environment_steps": sum(
            scene["reload_environment_steps"] for item in policy_reloads for scene in item["scenes"]),
        "training_scene_only": True,
        "note": "Independent verification work; reset feasibility probes are excluded from action counts.",
        "target_metric_limit": "Inference reloads verify action and MC metrics and full trajectories; historical TD targets are checked by the independent early update replay.",
        "verifier_sha256": file_hash(Path(__file__)),
        "independent_adam_helper_sha256": file_hash(INDEPENDENT_HELPER),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.run_dir.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
