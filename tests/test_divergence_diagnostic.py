"""Validate action-level diagnosis, exact replay checks and one-action interventions."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "divergence_diagnostic_test",
    ROOT / "docs/experiments/012_divergence_control/diagnose_divergence.py",
)
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


class ConstantNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.values = torch.nn.Parameter(torch.tensor([0., 0., 5., 0., 0., 0., 0., 0.]))
        self.batch_sizes = []

    def forward(self, observations):
        self.batch_sizes.append(len(observations))
        return self.values.unsqueeze(0).expand(len(observations), -1)


class TinyEnvironment:
    max_steps = 4

    def reset(self, *, seed):
        assert seed == 10001
        self.position = np.zeros(2, dtype=np.float32)
        self.index = 0
        self.action_history = []
        return self.position.copy(), {}

    def step(self, action):
        before = self.position.copy()
        self.position += [float(action == 2), float(action == 1)]
        self.index += 1
        self.action_history.append(action)
        info = {"success": False, "damage": False, "budget_exhausted": self.index == 4,
                "travel_distance": float(self.index), "peak_normalized_deformation": 0.,
                "distance_to_target": float(10 - self.position[0]),
                "executed_motion": self.position - before, "collision_blocked": False}
        return self.position.copy(), 1., False, self.index == 4, info


def make_fixture():
    network = ConstantNetwork()
    agent = SimpleNamespace(
        learner=SimpleNamespace(online=network, update_count=0,
                                optimizer=torch.optim.Adam(network.parameters())),
        rng=np.random.default_rng(0), encoder=lambda value: value.copy(),
    )

    def act(observation, epsilon):
        assert epsilon == 0
        with torch.no_grad():
            return int(network(torch.from_numpy(observation).unsqueeze(0)).argmax(dim=1).item())

    agent.act = act
    observations = np.array([[0, 0], [1, 0], [1, 1], [2, 1]], dtype=np.float32)
    actions = np.array([2, 1, 2, 2])
    expected = {
        "training_scene_only": True, "seed": 10001, "return": 4., "steps": 4,
        "success": False, "damage": False, "budget_exhausted": True,
        "travel_distance": 4., "peak_deformation": 0., "distance_to_target": 6.,
        "stationary_steps": 0, "blocked_steps": 0,
        "first_reference_action_divergence": {"step": 2, "predicted_action": 2, "demonstrated_action": 1},
        "actions": [2, 2, 2, 2], "robot_path": [[float(i), 0.] for i in range(5)],
    }
    return agent, TinyEnvironment(), observations, actions, expected


def test_signed_margin_and_hinge_shortfall_keep_direction():
    result = diagnostic.q_metrics([4., 3.5, 0., 0., 0., 0., 0., 0.], 1)
    assert result["chosen_action"] == 0
    assert result["signed_demonstration_margin"] == -0.5
    assert result["top1_minus_top2"] == 0.5
    assert result["hinge_margin_shortfall"] == pytest.approx(1.3)
    assert not result["demonstration_is_greedy"]
    good = diagnostic.q_metrics([4., 3., 0., 0., 0., 0., 0., 0.], 0)
    assert good["signed_demonstration_margin"] == 1.
    assert good["hinge_margin_shortfall"] == 0.


def test_action_tie_is_not_reported_as_a_positive_margin():
    result = diagnostic.q_metrics([1., 1., 0., 0., 0., 0., 0., 0.], 1)
    assert result["chosen_action"] == 0
    assert result["signed_demonstration_margin"] == 0.
    assert not result["demonstration_is_greedy"]


def test_first_mismatch_is_not_assigned_an_outcome_label():
    assert diagnostic.first_divergence([2, 6], [2, 1]) == {
        "step": 2, "predicted_action": 6, "demonstrated_action": 1,
    }
    assert diagnostic.first_divergence([2], [2]) is None
    assert diagnostic.first_divergence([2, 1], [2])["demonstrated_action"] is None


def test_batch_of_one_matches_policy_and_rejects_batched_input():
    agent, _, observations, _, _ = make_fixture()
    before = diagnostic.parameter_hash(agent)
    scores = diagnostic.single_state_q(agent, observations[0])
    assert int(scores.argmax()) == agent.act(observations[0], epsilon=0.)
    assert agent.learner.online.batch_sizes == [1, 1]
    assert diagnostic.parameter_hash(agent) == before
    with pytest.raises(ValueError, match="exactly one"):
        diagnostic.single_state_q(agent, observations)


def test_greedy_trace_must_reproduce_all_stored_actions_positions_and_metrics():
    agent, env, observations, actions, expected = make_fixture()
    counted = []
    trace = diagnostic.trace_policy(agent, env, observations, actions, expected, progress=counted.append)
    assert trace["rollout"] == expected
    assert len(counted) == 4
    assert trace["forced_action_count"] == 0
    assert trace["original_greedy_exactly_reproduced"]
    assert all(len(row["q_values"]) == 8 for row in trace["records"])


def test_intervention_forces_only_first_divergence_then_resumes_policy():
    agent, env, observations, actions, expected = make_fixture()
    counted = []
    before = diagnostic.parameter_hash(agent)
    trace = diagnostic.trace_policy(agent, env, observations, actions, expected,
                                    intervene=True, progress=counted.append)
    assert trace["rollout"]["actions"] == [2, 1, 2, 2]
    assert [row["policy_action"] for row in trace["records"]] == [2, 2, 2, 2]
    assert [row["forced_reference_action"] for row in trace["records"]] == [False, True, False, False]
    assert len(counted) == trace["rollout"]["steps"] == 4
    assert trace["forced_action_count"] == 1
    assert trace["optimizer_updates"] == 0
    assert diagnostic.parameter_hash(agent) == before
    assert agent.learner.optimizer.state == {}


def test_bad_reset_fails_before_any_environment_action():
    agent, env, observations, actions, expected = make_fixture()
    observations[0, 0] = 9.
    with pytest.raises(ValueError, match="reset does not match"):
        diagnostic.trace_policy(agent, env, observations, actions, expected)
    assert env.action_history == []


def test_common_state_mismatch_is_detected_at_divergent_action():
    agent, env, observations, actions, expected = make_fixture()
    observations[1, 0] = 9.
    counted = []
    with pytest.raises(RuntimeError, match="common prefix state"):
        diagnostic.trace_policy(agent, env, observations, actions, expected, progress=counted.append)
    assert len(counted) == 1
    assert env.action_history == [2]


def test_replay_rejects_changed_original_actions():
    agent, env, observations, actions, expected = make_fixture()
    expected["actions"][0] = 6
    with pytest.raises(RuntimeError, match="policy action differs"):
        diagnostic.trace_policy(agent, env, observations, actions, expected)
    assert env.action_history == []


def test_motion_analysis_distinguishes_stationary_and_position_returns():
    path = [[0, 0], [1, 0], [0, 0], [0, 0], [0, 0]]
    result = diagnostic.motion_characteristics(path, [
        {"stationary": False}, {"stationary": False}, {"stationary": True}, {"stationary": True},
    ])
    assert result["longest_stationary_run"] == 2
    assert result["two_step_position_returns"] == [2]
    assert result["maximum_position_visits"] == 4
    assert result["repeated_position_visits"] == 3


def test_label_transition_is_distinct_from_an_exact_conflicting_state():
    _, _, observations, actions, _ = make_fixture()
    context = diagnostic.label_context(observations, actions, 1)
    assert context["label_transition_from_previous"]
    assert context["previous_action"] == 2
    assert context["nearest_different_label_l2_distance"] == 1.


def test_input_hash_mismatch_aborts(tmp_path):
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"altered")
    with pytest.raises(ValueError, match="hash mismatch"):
        diagnostic.require_hash(path, "0" * 64)
