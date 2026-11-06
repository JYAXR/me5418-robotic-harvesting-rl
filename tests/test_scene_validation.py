"""Geometry rejection, reproducibility, and bounded sampling regression tests."""

from dataclasses import replace

import numpy as np
import pytest

from me5418_harvesting.envs import scene_generator
from me5418_harvesting.envs.branch_physics import Branch
from me5418_harvesting.envs.harvesting_env import HarvestingEnv
from me5418_harvesting.envs.scene_generator import Scene, generate_scene
from me5418_harvesting.envs.scene_validation import scene_issues


def make_scene(*polylines, closed=False):
    branches = [Branch(np.array(nodes), np.array(nodes), 1.0, 0.1) for nodes in polylines]
    return Scene(
        np.array([0.1, 0.5]), np.array([0.9, 0.5]), branches,
        passage_pairs=((0, 1),) if closed else (),
        passage_kinds=("closed",) if closed else (),
    )


def issues(scene):
    return scene_issues(scene, [0, 0], [1, 1], 0.035, 0.05)


@pytest.mark.parametrize("nodes, expected", [
    ([(0.1, 0.4), (0.1, 0.6)], "Start clearance"),
    ([(0.94, 0.4), (0.94, 0.6)], "Target clearance"),
    ([(0.5, -0.1), (0.5, 0.6)], "outside the workspace"),
    ([(0.5, 0.4), (0.5, 0.4)], "zero-length"),
    ([(0.5, float("nan")), (0.5, 0.6)], "invalid node"),
    ([(0.3, 0.3), (0.7, 0.7), (0.7, 0.3), (0.3, 0.7)], "intersect"),
    ([(0.3, 0.3), (0.6, 0.3), (0.4, 0.3)], "intersect"),
])
def test_invalid_branch_geometry(nodes, expected):
    assert any(expected in reason for reason in issues(make_scene(nodes)))


@pytest.mark.parametrize("first, second", [
    ([(0.3, 0.3), (0.7, 0.7)], [(0.3, 0.7), (0.7, 0.3)]),
    ([(0.3, 0.3), (0.7, 0.3)], [(0.5, 0.3), (0.5, 0.7)]),
    ([(0.3, 0.3), (0.6, 0.3)], [(0.4, 0.3), (0.7, 0.3)]),
])
def test_crossings_t_junctions_and_collinear_overlap_are_rejected(first, second):
    assert any("intersect" in reason for reason in issues(make_scene(first, second)))


def test_closed_passage_allows_only_opposing_tip_contact():
    lower = [(0.5, 0.0), (0.5, 0.25), (0.5, 0.5)]
    upper = [(0.5, 1.0), (0.5, 0.75), (0.5, 0.5)]
    assert issues(make_scene(lower, upper, closed=True)) == []
    assert issues(make_scene(lower, upper))
    overlap = [(0.5, 0.1), (0.5, 0.3), (0.5, 0.5)]
    assert issues(make_scene(lower, overlap, closed=True))


def test_invalid_start_target_and_material_parameters():
    scene = make_scene([(0.5, 0.2), (0.5, 0.8)])
    assert issues(replace(scene, end_effector=np.array([0.01, 0.5])))
    assert issues(replace(scene, target=np.array([0.99, 0.5])))
    assert issues(replace(scene, target=scene.end_effector.copy()))
    scene.branches[0].stiffness = float("inf")
    assert any("material" in reason for reason in issues(scene))


def test_retries_invalid_sample_and_reports_attempt_count(monkeypatch):
    valid = make_scene([(0.5, 0.2), (0.5, 0.8)])
    invalid = replace(valid, end_effector=np.array([0.5, 0.5]))
    samples = iter([invalid, valid])
    monkeypatch.setattr(scene_generator, "_sample_scene", lambda *args: next(samples))
    env = HarvestingEnv()
    _, info = env.reset(seed=7)
    assert info["scene_generation_attempts"] == 2
    assert env.scene is valid
    env.close()


def test_retry_limit_fails_with_rejection_reason(monkeypatch):
    invalid = make_scene([(0.1, 0.4), (0.1, 0.6)])
    calls = []

    def sample(*args):
        calls.append(1)
        return invalid

    monkeypatch.setattr(scene_generator, "_sample_scene", sample)
    with pytest.raises(ValueError, match="after 3 attempts.*Start clearance"):
        generate_scene(np.random.default_rng(0), max_attempts=3)
    assert len(calls) == 3


@pytest.mark.parametrize("parameters", [
    {"max_attempts": 0}, {"contact_radius": -0.1}, {"target_radius": float("nan")},
    {"workspace_high": [0, 1]}, {"stiffness_range": [0, 1]},
    {"safe_limit_range": [0.1, float("inf")]}, {"tip_gap_range": [0.1, 0.01]},
])
def test_invalid_sampling_configuration(parameters):
    with pytest.raises(ValueError):
        generate_scene(np.random.default_rng(0), **parameters)


@pytest.mark.parametrize("parameters", [
    {}, {"branch_tilt_degrees_range": (12, 22)},
    {"layout": "mixed_canopy", "max_branches": 14},
])
def test_seed_sweep_returns_valid_reproducible_scenes(parameters):
    for seed in range(30):
        first = generate_scene(np.random.default_rng(seed), **parameters)
        second = generate_scene(np.random.default_rng(seed), **parameters)
        assert issues(first) == []
        assert first.generation_attempts == second.generation_attempts
        np.testing.assert_array_equal(first.target, second.target)
        for left, right in zip(first.branches, second.branches):
            np.testing.assert_array_equal(left.current_nodes, right.current_nodes)
