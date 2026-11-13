"""Reconstruct one historical shared trajectory prefix for local Q diagnostics."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from me5418_harvesting.agents.dqn import DQNConfig, build_q_network
from me5418_harvesting.agents.dqn_agent import ObservationEncoder
from me5418_harvesting.envs.harvesting_env import HarvestingEnv

CHECKPOINTS = (7500, 8500, 9000)
SCENE_SEED = 10002
PREFIX_STEPS = 39
REFERENCE_ACTION = 1
ALTERNATIVE_ACTION = 6
PROBE_NAME = "scene_10002_actual_step_40"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observation_hash(values):
    """Hash the exact float32 network input, independently of JSON formatting."""
    array = np.ascontiguousarray(values, dtype=np.float32)
    return hashlib.sha256(array.tobytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_inputs(reference):
    """Read unchanged inference checkpoints and their historical rollout records."""
    directory = reference / "treatment/td"
    summary_path, log_path = directory / "summary.json", directory / "observations.json"
    summary, observations = read_json(summary_path), read_json(log_path)
    hashes = {str(path): file_hash(path) for path in (summary_path, log_path)}
    payloads, rollouts = {}, {}
    for update in CHECKPOINTS:
        path = directory / f"update_{update}.pt"
        expected = summary["checkpoint_sha256"][path.name]
        require(file_hash(path) == expected, f"Historical checkpoint changed: {path}")
        hashes[str(path)] = expected
        payload = torch.load(path, map_location="cpu", weights_only=True)
        require(payload["format_version"] == 1
                and payload["encoder_version"] == ObservationEncoder.VERSION,
                "Unsupported historical inference checkpoint format.")
        rows = [row for row in observations if row["update"] == update]
        require(len(rows) == 1, "A historical checkpoint must have exactly one observation record.")
        scenes = [scene for scene in rows[0]["scenes"] if scene["seed"] == SCENE_SEED]
        require(len(scenes) == 1, "The historical observation must identify scene 10002 exactly once.")
        rollout = scenes[0]["rollout"]
        require(len(rollout["actions"]) == rollout["steps"] > PREFIX_STEPS
                and len(rollout["robot_path"]) == rollout["steps"] + 1,
                "The source must retain complete actions and positions beyond the shared prefix.")
        require(rollout["seed"] == SCENE_SEED, "The historical rollout scene seed differs.")
        require(rollout["success"] is (update != 8500), "The selected historical outcomes differ.")
        require(rollout["actions"][PREFIX_STEPS]
                == (ALTERNATIVE_ACTION if update == 8500 else REFERENCE_ACTION),
                "The recorded step-40 fork differs from the declared diagnostic case.")
        payloads[update], rollouts[update] = payload, rollout
    first = payloads[CHECKPOINTS[0]]
    prefix = rollouts[8500]["actions"][:PREFIX_STEPS]
    for update in CHECKPOINTS:
        require(payloads[update]["environment_config"] == first["environment_config"]
                and payloads[update]["dqn_config"] == first["dqn_config"],
                "The three historical models must use identical environment and network settings.")
        require(rollouts[update]["actions"][:PREFIX_STEPS] == prefix,
                "The source policies do not share all 39 historical actions.")
        require(rollouts[update]["robot_path"][:PREFIX_STEPS + 1]
                == rollouts[8500]["robot_path"][:PREFIX_STEPS + 1],
                "The source policies do not share all 40 historical positions.")
    config = first["environment_config"]
    require(config.get("execution_noise_fraction", 0.0) == 0.0,
            "This diagnostic requires the unchanged deterministic historical dynamics.")
    return payloads, rollouts, hashes


def policy_values(model, encoded):
    """Perform exactly one batch-one inference without changing model state."""
    with torch.no_grad():
        values = model(torch.from_numpy(encoded).unsqueeze(0))[0]
    require(values.shape == (8,) and bool(torch.isfinite(values).all()),
            "A historical policy must return eight finite Q values.")
    return values.tolist()


def assert_position(observation, rollouts, step):
    """Match logged float32 coordinates exactly, without rounding the source log."""
    position = np.asarray(observation)[:2]
    for update, rollout in rollouts.items():
        expected = np.asarray(rollout["robot_path"][step], dtype=np.float64)
        require(expected.shape == (2,) and np.array_equal(position, expected),
                f"Reconstructed position differs at completed step {step}, checkpoint {update}.")
    return position.tolist()


def build_trajectory_probe(reference017: Path, batch, episodes):
    """Replay 39 shared historical actions once; never complete or correct a rollout.

    The returned action label is supported by two successful historical policies.
    The core API calls this field ``demonstrated_action``; it is not a new teacher
    label, an optimality assertion, or an instruction to change the demonstration.
    Historical logs contain actions and positions, but no per-step Q vectors.
    All Q vectors in the returned evidence are newly measured policy inferences.
    """
    reference = Path(reference017).resolve()
    payloads, rollouts, hashes = load_inputs(reference)
    episode = [item for item in episodes if item["seed"] == SCENE_SEED]
    require(len(episode) == 1, "The original demonstration must identify scene 10002 exactly once.")
    index = episode[0]["start"] + PREFIX_STEPS
    require(episode[0]["start"] <= index < episode[0]["stop"]
            and index < len(batch.observations), "Demonstration step 40 is unavailable.")
    require(int(batch.actions[index]) == REFERENCE_ACTION,
            "The original demonstration step-40 action differs from the comparison label.")
    first = payloads[CHECKPOINTS[0]]
    environment = deepcopy(first["environment_config"])
    # Certification runs physics on a separate copy. Skip this repeated check,
    # then require the full initial state to equal the original accepted scene.
    certification_was_enabled = bool(environment.get("verify_reference_route", False))
    environment["verify_reference_route"] = False
    env = HarvestingEnv(environment)
    encoder = ObservationEncoder(env)
    require(batch.observations.shape[1] == env.observation_size,
            "The demonstration and reconstructed observation dimensions differ.")
    # Network construction consumes random numbers; preserve the caller's state.
    with torch.random.fork_rng(devices=[]):
        models = {}
        for update, payload in payloads.items():
            model = build_q_network(env.observation_size, 8, DQNConfig(**payload["dqn_config"]))
            model.load_state_dict(payload["online"])
            model.eval()
            models[update] = model
    prefix, step40 = [], []
    try:
        observation, _ = env.reset(seed=SCENE_SEED)
        initial = encoder(observation)
        original_initial = batch.observations[episode[0]["start"]].detach().cpu().numpy()
        require(np.array_equal(initial, original_initial),
                "Skipping repeated certification did not recover the exact original accepted scene.")
        for offset, action in enumerate(rollouts[8500]["actions"][:PREFIX_STEPS]):
            before = assert_position(observation, rollouts, offset)
            encoded = encoder(observation)
            q_values = {}
            for update, model in models.items():
                values = policy_values(model, encoded)
                require(int(np.argmax(values)) == rollouts[update]["actions"][offset] == action,
                        f"Recomputed policy action differs at step {offset + 1}, checkpoint {update}.")
                q_values[str(update)] = values
            observation, _, terminated, truncated, _ = env.step(action)
            require(not terminated and not truncated,
                    "The historical prefix unexpectedly completed an episode.")
            after = assert_position(observation, rollouts, offset + 1)
            prefix.append({"step": offset + 1, "action": action, "position_before": before,
                           "position_after": after, "encoded_observation_sha256": observation_hash(encoded),
                           "newly_inferred_q_values": q_values})
        encoded = encoder(observation)
        require(env.step_count == PREFIX_STEPS, "The reconstruction did not stop before action 40.")
        for update, model in models.items():
            values = policy_values(model, encoded)
            action = int(np.argmax(values))
            require(action == rollouts[update]["actions"][PREFIX_STEPS],
                    f"The reconstructed step-40 fork differs at checkpoint {update}.")
            step40.append({"checkpoint_update": update, "newly_inferred_q_values": values,
                           "greedy_action": action, "logged_action": action,
                           "south_small_minus_east_large": values[REFERENCE_ACTION] - values[ALTERNATIVE_ACTION],
                           "historical_rollout_success": rollouts[update]["success"]})
    finally:
        env.close()
    demonstration = batch.observations[index].detach().cpu().numpy()
    require(demonstration.shape == encoded.shape and np.isfinite(demonstration).all(),
            "The original demonstration comparison state is invalid.")
    difference = np.abs(encoded.astype(np.float64) - demonstration.astype(np.float64))
    metadata = {
        "name": PROBE_NAME, "scene_seed": SCENE_SEED, "reference_step": PREFIX_STEPS + 1,
        "source_trajectory_checkpoint": 8500, "demonstrated_action": REFERENCE_ACTION,
        "reference_action": REFERENCE_ACTION, "reference_action_kind": "successful_policy_action",
        "reference_action_sources": [7500, 9000], "fixed_comparison_action": ALTERNATIVE_ACTION,
        "encoded_observation_shape": [1, len(encoded)],
        "encoded_observation_sha256": observation_hash(encoded),
        "label_semantics": "The core API field demonstrated_action carries a successful-policy reference action; no new demonstration or optimality claim is made.",
    }
    evidence = {
        "metadata": metadata, "source_files_sha256": hashes,
        "shared_prefix_actions": rollouts[8500]["actions"][:PREFIX_STEPS],
        "prefix": prefix, "encoded_observation": encoded.tolist(),
        "encoded_observation_sha256": observation_hash(encoded), "step40_policies": step40,
        "demonstration_state_comparison": {
            "global_transition_index": index, "scene_seed": SCENE_SEED, "reference_step": 40,
            "encoded_observation_sha256": observation_hash(demonstration),
            "exactly_equal": bool(np.array_equal(encoded, demonstration)),
            "allclose_atol_1e_7_rtol_0": bool(np.allclose(encoded, demonstration, atol=1e-7, rtol=0)),
            "maximum_absolute_difference": float(difference.max()),
            "different_feature_count": int(np.count_nonzero(difference)),
        },
        "verification": {"all_logged_actions_exact": True, "all_logged_positions_exact": True,
                         "three_policies_queried_on_one_common_state": True,
                         "historical_per_step_q_available": False,
                         "q_values_semantics": "Newly recomputed from unchanged historical models; original rollout logs did not retain per-step Q vectors."},
        "reset_certification": {
            "original_verify_reference_route": certification_was_enabled,
            "reconstruction_verify_reference_route": False,
            "original_initial_global_transition_index": episode[0]["start"],
            "original_initial_encoded_observation_sha256": observation_hash(original_initial),
            "reconstructed_initial_encoded_observation_sha256": observation_hash(initial),
            "initial_encoded_observation_exact": True,
            "semantics": "Only repeated reset-time certification is disabled to avoid hidden copied-environment actions. The entire original accepted initial state is required to match exactly; dynamics and source files are unchanged.",
        },
        "accounting": {"environment_resets": 1, "environment_steps": PREFIX_STEPS,
                       "completed_episodes": 0, "prefix_policy_forwards": 3 * PREFIX_STEPS,
                       "step40_policy_forwards": 3, "optimizer_updates": 0,
                       "new_demonstrations": 0},
        "limitations": ["This is a retrospective local decision probe, not a forced rescue or a new success evaluation.",
                        "A later common-state action fork is not by itself proof of the cause of failure."],
    }
    for name, expected in hashes.items():
        require(file_hash(Path(name)) == expected, f"Historical input changed during reconstruction: {name}")
    core_probe = {"name": PROBE_NAME, "observation": torch.from_numpy(encoded.copy()).unsqueeze(0),
                  "demonstrated_action": REFERENCE_ACTION}
    return core_probe, metadata, evidence
