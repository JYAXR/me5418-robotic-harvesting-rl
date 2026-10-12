import numpy as np
import pytest

from me5418_harvesting import HarvestingEnv
from me5418_harvesting.envs.branch_physics import Branch
from me5418_harvesting.envs.scene_generator import Scene, generate_scene


def test_reset_and_step_follow_gymnasium_contract() -> None:
    env = HarvestingEnv()
    observation, info = env.reset(seed=7)
    assert observation.shape == env.observation_space.shape
    assert observation.dtype == np.float32
    assert info["seed"] == 7

    next_observation, reward, terminated, truncated, step_info = env.step(0)
    assert next_observation.shape == observation.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "max_normalized_deformation" in step_info


def test_deterministic_mode_is_seed_reproducible() -> None:
    first = HarvestingEnv({"execution_noise_fraction": 0.0})
    second = HarvestingEnv({"execution_noise_fraction": 0.0})
    first.reset(seed=11)
    second.reset(seed=11)
    for action in [2, 2, 4, 0, 6]:
        first_transition = first.step(action)
        second_transition = second.step(action)
        np.testing.assert_allclose(first_transition[0], second_transition[0])
        assert first_transition[1:4] == second_transition[1:4]
        np.testing.assert_allclose(
            first_transition[4]["executed_motion"],
            second_transition[4]["executed_motion"],
        )


def test_rgb_array_render_produces_image() -> None:
    env = HarvestingEnv(render_mode="rgb_array")
    env.reset(seed=3)
    frame = env.render()
    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3
    assert frame.shape[2] == 3
    assert frame.dtype == np.uint8


def test_trajectory_records_executed_positions() -> None:
    env = HarvestingEnv()
    env.reset(seed=5)
    env.step(0)
    env.step(2)
    assert len(env.trajectory) == 3


def test_stiff_branch_blocks_tunnelling_motion() -> None:
    env = HarvestingEnv({"execution_noise_fraction": 0.0})
    env.reset(seed=0)
    nodes = np.column_stack(
        (
            np.full(5, 0.30, dtype=np.float32),
            np.linspace(0.20, 0.80, 5, dtype=np.float32),
        )
    )
    branch = Branch(nodes.copy(), nodes.copy(), stiffness=1e6, safe_limit=0.10)
    env.scene = Scene(
        end_effector=np.array([0.26, 0.50], dtype=np.float32),
        target=np.array([0.90, 0.50], dtype=np.float32),
        branches=[branch],
    )
    env.trajectory = [env.scene.end_effector.copy()]

    _, _, _, _, info = env.step(6)

    assert info["collision_blocked"] is True
    assert env.scene.end_effector[0] < 0.27


def test_generated_scene_has_narrow_internal_gaps_and_alternating_outer_routes() -> None:
    scene = generate_scene(np.random.default_rng(13), max_branches=5, nodes_per_branch=5)
    assert len(scene.branches) == 4
    for lower, upper in zip(scene.branches[::2], scene.branches[1::2]):
        tip_gap = float(upper.rest_nodes[-1, 1] - lower.rest_nodes[-1, 1])
        assert 0.028 <= tip_gap <= 0.043
        assert tip_gap < 2 * HarvestingEnv().contact_radius
    assert scene.branches[0].rest_nodes[0, 1] == 0.0
    assert scene.branches[3].rest_nodes[0, 1] == 1.0
    assert scene.branches[1].rest_nodes[0, 1] < 0.85
    assert scene.branches[2].rest_nodes[0, 1] > 0.15


@pytest.mark.parametrize("scene_parameters", [{}, {"branch_tilt_degrees_range": (12.0, 22.0)}])
def test_soft_tip_gap_opens_without_robot_penetration_or_unsafe_strain(scene_parameters) -> None:
    env = HarvestingEnv({"scene": scene_parameters})
    env.reset(seed=0)
    # Isolate contact physics from randomized gap placement.
    for first, second in zip(env.scene.branches[::2], env.scene.branches[1::2]):
        offset = 0.5 - 0.5 * (first.rest_nodes[-1, 1] + second.rest_nodes[-1, 1])
        for branch in (first, second):
            branch.rest_nodes[:, 1] += np.linspace(0.0, offset, len(branch.rest_nodes))
            branch.current_nodes = branch.rest_nodes.copy()
    lower, upper = env.scene.branches[:2]
    initial_gap = float(upper.current_nodes[-1, 1] - lower.current_nodes[-1, 1])
    widest_gap = initial_gap
    peak_deformation = 0.0
    for _ in range(60):
        _, _, terminated, truncated, info = env.step(2)
        assert info["minimum_clearance"] >= env.contact_radius - 2e-5
        assert not info["damage"]
        peak_deformation = max(peak_deformation, info["peak_normalized_deformation"])
        for branch in env.scene.branches:
            np.testing.assert_array_equal(branch.current_nodes[0], branch.rest_nodes[0])
            assert branch.max_segment_strain < 0.0301
        widest_gap = max(
            widest_gap, float(upper.current_nodes[-1, 1] - lower.current_nodes[-1, 1])
        )
        if terminated or truncated:
            break
    assert env.scene.end_effector[0] > 0.85
    assert widest_gap > initial_gap + 0.02
    assert 0.0 < peak_deformation < 1.0


def test_rigid_version_of_same_gap_cannot_be_crossed() -> None:
    env = HarvestingEnv()
    env.reset(seed=0)
    for branch in env.scene.branches:
        branch.stiffness = 1e6
    for _ in range(30):
        _, _, _, _, info = env.step(6)
        assert info["minimum_clearance"] >= env.contact_radius - 2e-5
    assert env.scene.end_effector[0] < 0.33


def test_branches_recover_after_contact_is_released() -> None:
    env = HarvestingEnv()
    env.reset(seed=0)
    branch = env.scene.branches[0]
    branch.current_nodes[1:, 0] += 0.02
    initial_displacement = branch.max_displacement
    assert initial_displacement > 0.0
    for _ in range(80):
        branch.update_contact(
            np.array([0.10, 0.90], dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            env.contact_radius,
            env.relaxation_rate,
        )
    assert branch.max_displacement < 0.1 * initial_displacement


def test_inclined_branches_keep_gap_size_and_seeded_tilt_geometry() -> None:
    parameters = {"branch_tilt_degrees_range": (12.0, 22.0)}
    first = generate_scene(np.random.default_rng(0), **parameters)
    second = generate_scene(np.random.default_rng(0), **parameters)
    upright = generate_scene(np.random.default_rng(0))
    np.testing.assert_array_equal(first.target, upright.target)
    for branch, repeated, original in zip(first.branches, second.branches, upright.branches):
        np.testing.assert_array_equal(branch.rest_nodes, repeated.rest_nodes)
        np.testing.assert_array_equal(branch.rest_nodes[:, 1], original.rest_nodes[:, 1])
        np.testing.assert_array_equal(branch.rest_nodes[-1], original.rest_nodes[-1])
        assert branch.stiffness == original.stiffness
        assert branch.safe_limit == original.safe_limit
        delta = branch.rest_nodes[-1] - branch.rest_nodes[0]
        angle = np.rad2deg(np.arctan2(abs(float(delta[0])), abs(float(delta[1]))))
        assert 12.0 - 1e-4 <= angle <= 22.0 + 1e-4
        assert np.all((branch.rest_nodes >= 0.0) & (branch.rest_nodes <= 1.0))
    for lower, upper in zip(first.branches[::2], first.branches[1::2]):
        assert lower.rest_nodes[-1, 0] == upper.rest_nodes[-1, 0]
        assert 0.028 <= np.linalg.norm(upper.rest_nodes[-1] - lower.rest_nodes[-1]) <= 0.043


@pytest.mark.parametrize("tilt_range", [(-1.0, 10.0), (22.0, 12.0), (0.0, 26.0)])
def test_invalid_tilt_ranges_are_rejected(tilt_range) -> None:
    with pytest.raises(ValueError, match="tilt"):
        generate_scene(np.random.default_rng(0), branch_tilt_degrees_range=tilt_range)


def test_inclined_stiff_branch_blocks_motion_without_tunnelling() -> None:
    env = HarvestingEnv()
    env.reset(seed=0)
    nodes = np.array([[0.20, 0.20], [0.40, 0.80]], dtype=np.float32)
    branch = Branch(nodes.copy(), nodes.copy(), stiffness=1e6, safe_limit=0.10)
    env.scene = Scene(
        end_effector=np.array([0.24, 0.50], dtype=np.float32),
        target=np.array([0.90, 0.50], dtype=np.float32),
        branches=[branch],
    )
    env.trajectory = [env.scene.end_effector.copy()]
    for _ in range(4):
        _, _, _, _, info = env.step(6)
        assert info["minimum_substep_clearance"] >= env.contact_radius - 2e-5
        np.testing.assert_array_equal(branch.current_nodes[0], branch.rest_nodes[0])
    assert info["collision_blocked"]
    assert env.scene.end_effector[0] < 0.30


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_mixed_scene_contains_open_closed_pushable_passages_and_varied_twigs(seed) -> None:
    first = generate_scene(np.random.default_rng(seed), max_branches=14, layout="mixed_canopy")
    repeated = generate_scene(np.random.default_rng(seed), max_branches=14, layout="mixed_canopy")
    assert len(first.branches) == 14
    assert sorted(first.passage_kinds) == ["closed", "open", "pushable", "pushable"]
    lengths = []
    for branch, other in zip(first.branches, repeated.branches):
        np.testing.assert_array_equal(branch.rest_nodes, other.rest_nodes)
        assert np.all((branch.rest_nodes >= 0) & (branch.rest_nodes <= 1))
        assert branch.minimum_distance_to(first.end_effector) > 0.035
        assert branch.minimum_distance_to(first.target) > 0.035
        lengths.append(float(np.linalg.norm(np.diff(branch.rest_nodes, axis=0), axis=1).sum()))
    assert max(lengths) > 5 * min(lengths)
    for (left, right), kind in zip(first.passage_pairs, first.passage_kinds):
        lower, upper = first.branches[left], first.branches[right]
        gap = np.linalg.norm(upper.rest_nodes[-1] - lower.rest_nodes[-1])
        if kind == "closed":
            assert gap == 0
            assert min(lower.stiffness, upper.stiffness) >= 1e6
        elif kind == "open":
            assert gap > 2 * 0.035
        else:
            assert 0 < gap < 2 * 0.035
            assert max(lower.stiffness, upper.stiffness) < 10
    extra_vectors = [branch.rest_nodes[-1] - branch.rest_nodes[0] for branch in first.branches[8:]]
    assert any(abs(vector[0]) > 3 * abs(vector[1]) for vector in extra_vectors)
    assert any(abs(vector[1]) > abs(vector[0]) for vector in extra_vectors)


def test_mixed_scene_observation_covers_every_branch_without_truncating_state() -> None:
    env = HarvestingEnv({"max_branches": 14, "scene": {"layout": "mixed_canopy"}})
    observation, _ = env.reset(seed=0)
    assert observation.shape == (341,)
    assert env.observation_space.contains(observation)
    for index, branch in enumerate(env.scene.branches):
        np.testing.assert_array_equal(observation[4 + index * 24:14 + index * 24], branch.current_nodes.ravel())


def test_closed_mixed_passage_cannot_be_forced_open() -> None:
    env = HarvestingEnv({"max_branches": 14, "scene": {"layout": "mixed_canopy"}})
    env.reset(seed=0)
    passage_index = env.scene.passage_kinds.index("closed")
    pair = env.scene.passage_pairs[passage_index]
    center = env.scene.branches[pair[0]].rest_nodes[-1].copy()
    env.scene.end_effector = center - np.array([0.06, 0], dtype=np.float32)
    for _ in range(10):
        _, _, _, _, info = env.step(6)
        assert not info["damage"]
        assert info["minimum_substep_clearance"] >= env.contact_radius - 2e-5
    assert env.scene.end_effector[0] < center[0]
    assert info["collision_blocked"]
    for index in pair:
        assert env.scene.branches[index].max_displacement < 1e-5


@pytest.mark.parametrize("kind", ["open", "pushable"])
def test_mixed_passage_types_have_distinct_actual_contact_behaviour(kind) -> None:
    env = HarvestingEnv({"max_branches": 14, "scene": {"layout": "mixed_canopy"}})
    env.reset(seed=0)
    pair = env.scene.passage_pairs[env.scene.passage_kinds.index(kind)]
    branches = [env.scene.branches[index] for index in pair]
    center = 0.5 * (branches[0].rest_nodes[-1] + branches[1].rest_nodes[-1])
    env.scene = Scene(
        center - np.array([0.08, 0], dtype=np.float32),
        np.array([0.95, center[1]], dtype=np.float32), branches,
        passage_pairs=((0, 1),), passage_kinds=(kind,),
    )
    for _ in range(10):
        _, _, terminated, _, info = env.step(2)
        assert not info["damage"]
        assert info["minimum_substep_clearance"] >= env.contact_radius - 2e-5
        for branch in branches:
            np.testing.assert_array_equal(branch.current_nodes[0], branch.rest_nodes[0])
            assert branch.max_segment_strain <= 0.0301
        if terminated:
            break
    assert env.scene.end_effector[0] > center[0] + env.contact_radius
    if kind == "open":
        assert info["contact_steps"] == 0
        assert info["peak_normalized_deformation"] == 0
    else:
        assert info["contact_steps"] > 0
        assert 0 < info["peak_normalized_deformation"] < 1


@pytest.mark.parametrize("parameters", [{"max_branches": 5, "layout": "mixed_canopy"}, {"layout": "missing"}])
def test_invalid_mixed_scene_settings_are_rejected(parameters) -> None:
    with pytest.raises(ValueError):
        generate_scene(np.random.default_rng(0), **parameters)
