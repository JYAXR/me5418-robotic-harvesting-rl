"""Verify bounded positional diversity and a usable introductory training scene."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from me5418_harvesting import HarvestingEnv
from me5418_harvesting.envs.scene_generator import _sample_scene, generate_scene
from me5418_harvesting.envs.scene_validation import scene_issues

ROOT = Path(__file__).resolve().parents[1]
CONFIGURATIONS = (
    "environment_easy.yaml", "environment.yaml",
    "environment_inclined.yaml", "environment_complex.yaml",
)


def load_environment(name):
    return HarvestingEnv(yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8")))


@pytest.mark.parametrize("name", CONFIGURATIONS)
def test_configured_positions_vary_within_bounds_and_reproduce(name):
    env = load_environment(name)
    positions = []
    gates = []
    try:
        mixed = name == "environment_complex.yaml"
        base_x = [0.24, 0.42, 0.61, 0.79] if mixed else (
            [0.5] if name == "environment_easy.yaml" else [0.33, 0.67]
        )
        for seed in range(20):
            observation, info = env.reset(seed=seed)
            first = env.scene
            repeated, other_info = env.reset(seed=seed)
            np.testing.assert_array_equal(observation, repeated)
            assert info == other_info
            assert scene_issues(first, env.workspace_low, env.workspace_high,
                                env.contact_radius, env.target_radius) == []
            assert abs(first.end_effector[0] - (0.08 if mixed else 0.10)) <= 0.015001
            assert abs(first.end_effector[1] - 0.5) <= (0.060001 if mixed else 0.030001)
            assert abs(first.target[0] - (0.92 if mixed else 0.90)) <= 0.015001
            assert abs(first.target[1] - 0.5) <= (0.060001 if mixed else 0.050001)
            gate_x = [float(first.branches[pair[0]].rest_nodes[-1, 0])
                      for pair in first.passage_pairs]
            amplitude = env.scene_parameters["gate_x_jitter"]
            assert np.all(np.abs(np.array(gate_x) - base_x) <= amplitude + 1e-6)
            assert np.all(np.diff(gate_x) > 0)
            positions.append(np.concatenate([first.end_effector, first.target]))
            gates.append(gate_x)
        # Both coordinates of start/target and every gate position must actually vary.
        assert np.all(np.ptp(np.array(positions), axis=0) > 0.005)
        assert np.all(np.ptp(np.array(gates), axis=0) > 0.005)
    finally:
        env.close()


def test_zero_jitter_preserves_original_seeded_sampling():
    original = _sample_scene(np.random.default_rng(13))
    current = generate_scene(np.random.default_rng(13))
    np.testing.assert_array_equal(original.end_effector, current.end_effector)
    np.testing.assert_array_equal(original.target, current.target)
    for old, new in zip(original.branches, current.branches):
        np.testing.assert_array_equal(old.rest_nodes, new.rest_nodes)


def test_pair_translation_preserves_gap_geometry_and_twig_shape():
    settings = {"layout": "mixed_canopy", "max_branches": 14}
    original = generate_scene(np.random.default_rng(0), **settings)
    shifted = generate_scene(np.random.default_rng(0), **settings,
                             gate_x_jitter=0.015, twig_jitter=(0.01, 0.01))
    assert shifted.generation_attempts == 1
    assert shifted.passage_kinds == original.passage_kinds
    for pair in shifted.passage_pairs:
        first, second = pair
        old_gap = original.branches[second].rest_nodes[-1] - original.branches[first].rest_nodes[-1]
        new_gap = shifted.branches[second].rest_nodes[-1] - shifted.branches[first].rest_nodes[-1]
        np.testing.assert_allclose(old_gap, new_gap, atol=1e-7)
    for index, (old, new) in enumerate(zip(original.branches, shifted.branches)):
        np.testing.assert_allclose(np.diff(old.rest_nodes, axis=0),
                                   np.diff(new.rest_nodes, axis=0), atol=1e-7)
        assert old.stiffness == new.stiffness and old.safe_limit == new.safe_limit
        if index >= 8:
            offset = new.rest_nodes[0] - old.rest_nodes[0]
            assert np.all(np.abs(offset) <= 0.010001)
            assert np.linalg.norm(offset) > 0


def test_randomized_inclined_layout_retains_seed_matched_base_scene():
    upright = load_environment("environment.yaml")
    inclined = load_environment("environment_inclined.yaml")
    try:
        upright.reset(seed=0)
        inclined.reset(seed=0)
        np.testing.assert_array_equal(upright.scene.end_effector, inclined.scene.end_effector)
        np.testing.assert_array_equal(upright.scene.target, inclined.scene.target)
        for base, tilted in zip(upright.scene.branches, inclined.scene.branches):
            np.testing.assert_array_equal(base.rest_nodes[-1], tilted.rest_nodes[-1])
            assert base.stiffness == tilted.stiffness
    finally:
        upright.close()
        inclined.close()


def test_easy_scene_can_be_solved_without_contact_and_keeps_observation_slots():
    env = load_environment("environment_easy.yaml")
    try:
        for seed in range(20):
            observation, _ = env.reset(seed=seed)
            assert observation.shape == (125,)
            assert len(env.scene.branches) == 2
            assert env.scene.passage_kinds == ("open",)
            assert np.all(observation[4 + 2 * 24:-1] == 0)
            while True:
                # A geometric baseline establishes task feasibility, not learned success.
                delta = env.scene.target - env.scene.end_effector
                axis = int(np.argmax(np.abs(delta)))
                direction = (2 if delta[0] >= 0 else 3) if axis == 0 else (
                    0 if delta[1] >= 0 else 1
                )
                action = direction + (4 if abs(delta[axis]) > env.large_step else 0)
                _, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    assert info["success"] and not info["damage"]
                    assert info["contact_steps"] == 0
                    assert env.step_count < env.max_steps
                    break
    finally:
        env.close()


@pytest.mark.parametrize("parameters", [
    {"start_jitter": (-0.01, 0)}, {"target_jitter": (0, float("nan"))},
    {"twig_jitter": (0.01,)}, {"gate_x_jitter": -1},
    {"gate_x_jitter": float("inf")}, {"layout": "easy_gate"},
])
def test_invalid_randomization_and_narrow_easy_passages_are_rejected(parameters):
    with pytest.raises(ValueError):
        generate_scene(np.random.default_rng(0), **parameters)
