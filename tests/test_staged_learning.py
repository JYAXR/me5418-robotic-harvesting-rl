"""Check supervised equivalence and a clean, reproducible transition to TD."""

import importlib.util
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from me5418_harvesting.agents.dqn import DQNConfig
from me5418_harvesting.agents.dqn_learning import DQNUpdater, TransitionBatch
from me5418_harvesting.agents.staged_learning import begin_td_phase, supervised_update

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fixed_scene_staged_reference",
    ROOT / "docs/experiments/005_fixed_scene_fit/run_fixed_scene_fit.py",
)
reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reference)


def make_learner_and_batch():
    torch.set_num_threads(1)
    torch.manual_seed(13)
    learner = DQNUpdater(
        5, 3, DQNConfig(hidden_sizes=(8, 8), batch_size=4,
                       replay_capacity=16, target_update_interval=200),
    )
    batch = TransitionBatch(
        torch.randn(4, 5), torch.tensor([0, 2, 1, 2]), torch.tensor([0.0, -0.5, 1.0, 10.0]),
        torch.randn(4, 5), torch.tensor([False, False, False, True]),
        torch.zeros(4, dtype=torch.bool),
    )
    return learner, batch


def assert_state_equal(actual, expected):
    """Compare nested model or optimizer states without tolerance."""
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_state_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for first, second in zip(actual, expected):
            assert_state_equal(first, second)
    else:
        assert actual == expected


@pytest.mark.parametrize("objective", ["margin_only", "cross_entropy"])
def test_supervised_updates_exactly_match_the_previous_experiment(objective):
    learner, batch = make_learner_and_batch()
    expected = deepcopy(learner)
    original_target = deepcopy(learner.target.state_dict())
    for _ in range(5):
        old_stats = reference.fitting_update(
            SimpleNamespace(learner=expected), batch, objective, 0.8,
        )
        new_stats = supervised_update(learner, batch, objective=objective)
        assert new_stats == old_stats
        assert_state_equal(learner.online.state_dict(), expected.online.state_dict())
        assert_state_equal(learner.optimizer.state_dict(), expected.optimizer.state_dict())
    assert learner.update_count == expected.update_count == 5
    assert_state_equal(learner.target.state_dict(), original_target)


@pytest.mark.parametrize("objective", ["margin_only", "cross_entropy"])
def test_supervision_ignores_rewards_next_states_and_the_target(objective):
    learner, batch = make_learner_and_batch()
    poisoned = deepcopy(learner)
    poisoned_batch = replace(
        batch, rewards=torch.full_like(batch.rewards, float("nan")),
        next_observations=torch.full_like(batch.next_observations, float("nan")),
        terminated=~batch.terminated, truncated=~batch.truncated,
    )
    with torch.no_grad():
        for parameter in poisoned.target.parameters():
            parameter.fill_(float("nan"))
    for _ in range(3):
        assert supervised_update(learner, batch, objective=objective) == supervised_update(
            poisoned, poisoned_batch, objective=objective,
        )
    assert_state_equal(learner.online.state_dict(), poisoned.online.state_dict())
    assert all(torch.isnan(parameter).all() for parameter in poisoned.target.parameters())


def test_phase_boundary_preserves_weights_and_resets_optimizer_target_and_clock():
    learner, batch = make_learner_and_batch()
    for _ in range(3):
        supervised_update(learner, batch)
    old_optimizer = learner.optimizer
    assert old_optimizer.state
    original_weights = deepcopy(learner.online.state_dict())
    assert any(not torch.equal(value, learner.target.state_dict()[name])
               for name, value in original_weights.items())
    learner.target.requires_grad_(True)
    learner.target.train()

    record = begin_td_phase(learner)

    assert record == {
        "pretraining_updates": 3, "phase_updates": 0,
        "optimizer_reset": True, "target_synchronized": True,
    }
    assert learner.optimizer is not old_optimizer
    assert not learner.optimizer.state
    assert learner.optimizer.param_groups[0]["lr"] == learner.config.learning_rate
    assert learner.update_count == 0
    assert_state_equal(learner.online.state_dict(), original_weights)
    assert_state_equal(learner.target.state_dict(), original_weights)
    assert not learner.target.training
    assert all(not parameter.requires_grad for parameter in learner.target.parameters())
    assert all(parameter.grad is None for parameter in learner.online.parameters())


def test_first_td_target_sync_uses_the_new_phase_clock():
    learner, batch = make_learner_and_batch()
    for _ in range(3):
        supervised_update(learner, batch)
    begin_td_phase(learner)
    initial_target = deepcopy(learner.target.state_dict())
    for index in range(1, 201):
        stats = learner.update(batch)
        assert stats.target_synchronized == (index == 200)
        if index == 199:
            assert_state_equal(learner.target.state_dict(), initial_target)
    assert learner.update_count == 200
    assert_state_equal(learner.target.state_dict(), learner.online.state_dict())
    assert all(not parameter.requires_grad for parameter in learner.target.parameters())


@pytest.mark.parametrize("arguments", [
    {"objective": "td"}, {"objective": "unknown"}, {"margin": 0.0},
    {"margin": -1.0}, {"margin": float("nan")}, {"margin": float("inf")},
    {"margin": True}, {"objective": "cross_entropy", "margin": -0.8},
])
def test_invalid_supervision_settings_fail_before_mutation(arguments):
    learner, batch = make_learner_and_batch()
    supervised_update(learner, batch)
    learner.online.eval()
    original_weights = deepcopy(learner.online.state_dict())
    original_optimizer = deepcopy(learner.optimizer.state_dict())
    original_gradients = [parameter.grad.clone() for parameter in learner.online.parameters()]
    with pytest.raises(ValueError):
        supervised_update(learner, batch, **arguments)
    assert not learner.online.training
    assert learner.update_count == 1
    assert_state_equal(learner.online.state_dict(), original_weights)
    assert_state_equal(learner.optimizer.state_dict(), original_optimizer)
    for parameter, original in zip(learner.online.parameters(), original_gradients):
        torch.testing.assert_close(parameter.grad, original, rtol=0, atol=0)
