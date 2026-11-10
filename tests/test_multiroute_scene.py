"""Check multi-region route choices, seeded geometry, and actual passage behaviour."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from me5418_harvesting import HarvestingEnv
from me5418_harvesting.envs.scene_generator import Scene, generate_scene
from me5418_harvesting.envs.scene_validation import scene_issues

ROOT = Path(__file__).resolve().parents[1]


def environment():
    config = yaml.safe_load(
        (ROOT / "configs/environment_multiroute.yaml").read_text(encoding="utf-8")
    )
    # Geometry and local contact tests do not need full-scene route certification.
    config["verify_reference_route"] = False
    return HarvestingEnv(config)


def test_multi_route_has_distinct_choices_in_every_region_and_seeded_diversity():
    patterns = set()
    for seed in range(20):
        scene = generate_scene(np.random.default_rng(seed), max_branches=14, layout="multi_route")
        repeated = generate_scene(np.random.default_rng(seed), max_branches=14, layout="multi_route")
        assert len(scene.branches) == 14
        assert scene.passage_pairs == tuple((i, i + 1) for i in range(0, 12, 2))
        assert scene.passage_kinds.count("open") == 3
        assert "pushable" in scene.passage_kinds and "closed" in scene.passage_kinds
        assert scene.passage_kinds == repeated.passage_kinds
        assert scene.generation_attempts == repeated.generation_attempts
        assert scene_issues(scene, [0, 0], [1, 1], 0.035, 0.05) == []
        open_sides = []
        for region in range(3):
            kinds = scene.passage_kinds[2 * region:2 * region + 2]
            assert kinds.count("open") == 1
            assert set(kinds) <= {"open", "pushable", "closed"}
            open_sides.append(kinds.index("open"))
            lower_tip = scene.branches[4 * region].rest_nodes[-1]
            upper_tip = scene.branches[4 * region + 2].rest_nodes[-1]
            assert lower_tip[1] < 0.5 < upper_tip[1]
        assert len(set(open_sides)) == 2
        patterns.add(scene.passage_kinds)
        np.testing.assert_array_equal(scene.end_effector, repeated.end_effector)
        np.testing.assert_array_equal(scene.target, repeated.target)
        for first, second in zip(scene.branches, repeated.branches):
            np.testing.assert_array_equal(first.rest_nodes, second.rest_nodes)
            assert first.stiffness == second.stiffness
        for pair, kind in zip(scene.passage_pairs, scene.passage_kinds):
            first, second = (scene.branches[index] for index in pair)
            gap = float(np.linalg.norm(second.rest_nodes[-1] - first.rest_nodes[-1]))
            if kind == "open":
                assert gap > 0.07
            elif kind == "closed":
                assert gap == 0
                assert min(first.stiffness, second.stiffness) >= 1e6
            else:
                assert 0 < gap < 0.07
                assert max(first.stiffness, second.stiffness) < 10
    assert len(patterns) >= 4


def test_multi_route_observation_contains_all_branches_and_uses_full_budget():
    env = environment()
    try:
        observation, _ = env.reset(seed=7)
        assert env.max_steps == 200
        assert observation.shape == (341,)
        assert env.observation_space.contains(observation)
        for index, branch in enumerate(env.scene.branches):
            start = 4 + index * 24
            np.testing.assert_array_equal(observation[start:start + 10], branch.current_nodes.ravel())
            assert observation[start + 23] == 1
    finally:
        env.close()


@pytest.mark.parametrize("kind", ["open", "closed", "pushable"])
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_multi_route_passage_types_have_distinct_physical_behaviour(kind, seed):
    env = environment()
    try:
        env.reset(seed=seed)
        pair = env.scene.passage_pairs[env.scene.passage_kinds.index(kind)]
        branches = [env.scene.branches[index] for index in pair]
        center = 0.5 * (branches[0].rest_nodes[-1] + branches[1].rest_nodes[-1])
        env.scene = Scene(
            center - np.array([0.08, 0], dtype=np.float32),
            np.array([0.95, center[1]], dtype=np.float32), branches,
            passage_pairs=((0, 1),), passage_kinds=(kind,),
        )
        for _ in range(10):
            _, _, terminated, truncated, info = env.step(2)
            assert not info["damage"]
            assert info["minimum_substep_clearance"] >= env.contact_radius - 2e-5
            for branch in branches:
                np.testing.assert_array_equal(branch.current_nodes[0], branch.rest_nodes[0])
                assert branch.max_segment_strain <= 0.0301
            if terminated or truncated:
                break
        if kind == "open":
            assert env.scene.end_effector[0] > center[0] + env.contact_radius
            assert info["contact_steps"] == 0
            assert info["peak_normalized_deformation"] == 0
        elif kind == "pushable":
            assert env.scene.end_effector[0] > center[0] + env.contact_radius
            assert info["contact_steps"] > 0
            assert 0 < info["peak_normalized_deformation"] < 1
        else:
            assert env.scene.end_effector[0] < center[0]
            assert info["collision_blocked"]
            assert all(branch.max_displacement < 1e-5 for branch in branches)
    finally:
        env.close()


def test_central_anchor_gap_does_not_create_an_unintended_straight_shortcut():
    env = environment()
    try:
        env.reset(seed=0)
        # Isolate the two branches anchored beside the central gap in region one.
        branches = [env.scene.branches[1], env.scene.branches[2]]
        anchors = [branch.rest_nodes[0] for branch in branches]
        assert np.linalg.norm(anchors[1] - anchors[0]) < 2 * env.contact_radius
        center = 0.5 * (anchors[0] + anchors[1])
        env.scene = Scene(
            center - np.array([0.08, 0], dtype=np.float32),
            center + np.array([0.20, 0], dtype=np.float32), branches,
        )
        for _ in range(8):
            _, _, terminated, truncated, info = env.step(6)
            assert info["minimum_substep_clearance"] >= env.contact_radius - 2e-5
            if terminated or truncated:
                break
        assert env.scene.end_effector[0] < center[0]
        assert not info["success"]
    finally:
        env.close()


def test_multi_route_rejects_insufficient_observation_capacity():
    with pytest.raises(ValueError):
        generate_scene(np.random.default_rng(0), max_branches=13, layout="multi_route")
