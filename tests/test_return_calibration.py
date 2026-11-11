"""Check complete-return targets and separation of calibration from bootstrapping."""

from copy import deepcopy
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from me5418_harvesting.agents.dqn import DQNConfig
from me5418_harvesting.agents.dqn_learning import DQNUpdater, TransitionBatch
from me5418_harvesting.agents.return_calibration import (
    calibration_update,
    discounted_demonstration_returns,
)


def example_batch():
    states = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    return TransitionBatch(
        states[:-1].clone(), torch.tensor([0, 1, 0]), torch.tensor([1.0, 2.0, 4.0]),
        states[1:].clone(), torch.tensor([False, False, True]), torch.zeros(3, dtype=torch.bool),
    )


def example_learner():
    torch.set_num_threads(1)
    torch.manual_seed(19)
    return DQNUpdater(2, 2, DQNConfig(hidden_sizes=(8, 8), learning_rate=0.01,
                                    batch_size=3, replay_capacity=8, target_update_interval=2))


def assert_nested_equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_nested_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for first, second in zip(actual, expected):
            assert_nested_equal(first, second)
    else:
        assert actual == expected


@pytest.mark.parametrize("gamma, expected", [(0.0, [1.0, 2.0, 4.0]),
                                            (0.5, [3.0, 4.0, 4.0]),
                                            (1.0, [7.0, 6.0, 4.0])])
def test_returns_include_current_and_terminal_rewards_once(gamma, expected):
    batch = example_batch()
    batch = replace(batch, rewards=batch.rewards.to(torch.float64).requires_grad_(True))
    returns = discounted_demonstration_returns(batch, gamma)
    torch.testing.assert_close(returns, torch.tensor(expected, dtype=torch.float64), rtol=0, atol=0)
    assert returns.dtype == batch.rewards.dtype
    assert returns.device == batch.rewards.device
    assert not returns.requires_grad
    assert returns.grad_fn is None
    assert batch.rewards.grad is None


def test_single_terminal_transition_returns_its_recorded_reward():
    batch = example_batch()
    single = TransitionBatch(*(getattr(batch, name)[-1:] for name in batch.__dataclass_fields__))
    assert discounted_demonstration_returns(single, 0.99).tolist() == [4.0]


@pytest.mark.parametrize("gamma", [True, -0.1, 1.1, float("nan"), float("inf"), "0.9"])
def test_reject_invalid_discounts(gamma):
    with pytest.raises(ValueError, match="Gamma"):
        discounted_demonstration_returns(example_batch(), gamma)


@pytest.mark.parametrize("case", ["partial", "multiple_episodes", "truncation", "shuffled",
                                  "nonfinite_reward", "wrong_flag_dtype", "wrong_state_count",
                                  "nonfinite_state", "empty", "integer_rewards"])
def test_reject_incomplete_or_misordered_trajectories(case):
    batch = example_batch()
    if case == "partial":
        batch = replace(batch, terminated=torch.zeros(3, dtype=torch.bool))
    elif case == "multiple_episodes":
        batch = replace(batch, terminated=torch.tensor([True, False, True]))
    elif case == "truncation":
        batch = replace(batch, truncated=torch.tensor([False, True, False]))
    elif case == "shuffled":
        order = torch.tensor([1, 0, 2])
        batch = TransitionBatch(*(getattr(batch, name)[order] for name in batch.__dataclass_fields__))
    elif case == "nonfinite_reward":
        batch = replace(batch, rewards=torch.tensor([1.0, float("nan"), 4.0]))
    elif case == "wrong_flag_dtype":
        batch = replace(batch, terminated=batch.terminated.long())
    elif case == "wrong_state_count":
        batch = replace(batch, observations=batch.observations[:2])
    elif case == "nonfinite_state":
        batch = replace(batch, observations=torch.full_like(batch.observations, float("inf")))
    elif case == "empty":
        batch = TransitionBatch(*(getattr(batch, name)[:0] for name in batch.__dataclass_fields__))
    else:
        batch = replace(batch, rewards=batch.rewards.long())
    with pytest.raises(ValueError):
        discounted_demonstration_returns(batch, 0.99)


def test_calibration_ignores_target_network_rewards_and_next_states():
    learner = example_learner()
    poisoned = deepcopy(learner)
    batch = example_batch()
    targets = discounted_demonstration_returns(batch, 0.5)
    altered = replace(
        batch, rewards=torch.full_like(batch.rewards, float("nan")),
        next_observations=torch.full_like(batch.next_observations, float("nan")),
        terminated=~batch.terminated, truncated=~batch.truncated,
    )
    original_target = deepcopy(learner.target.state_dict())
    with torch.no_grad():
        for parameter in poisoned.target.parameters():
            parameter.fill_(float("nan"))
    for _ in range(5):
        assert calibration_update(learner, batch, targets) == calibration_update(
            poisoned, altered, targets,
        )
        assert_nested_equal(learner.online.state_dict(), poisoned.online.state_dict())
        assert_nested_equal(learner.optimizer.state_dict(), poisoned.optimizer.state_dict())
    assert learner.update_count == poisoned.update_count == 5
    assert_nested_equal(learner.target.state_dict(), original_target)
    assert all(torch.isnan(parameter).all() for parameter in poisoned.target.parameters())


def test_calibration_uses_detached_fixed_returns_and_reduces_value_error():
    learner = example_learner()
    batch = example_batch()
    targets = discounted_demonstration_returns(batch, 0.5).requires_grad_(True)
    with torch.no_grad():
        chosen = learner.online(batch.observations).gather(1, batch.actions[:, None]).squeeze(1)
        initial_error = torch.mean((chosen - targets) ** 2).item()
    for _ in range(200):
        stats = calibration_update(learner, batch, targets)
        assert not stats["target_synchronized"]
    with torch.no_grad():
        values = learner.online(batch.observations)
        final_error = torch.mean((values.gather(1, batch.actions[:, None]).squeeze(1) - targets) ** 2)
    assert final_error < initial_error * 0.02
    assert torch.equal(values.argmax(dim=1), batch.actions)
    assert targets.grad is None
    assert learner.update_count == 200


def test_zero_margin_weight_matches_manual_huber_and_adam_step():
    learner = example_learner()
    manual = deepcopy(learner)
    batch = example_batch()
    targets = discounted_demonstration_returns(batch, 0.5)
    manual.optimizer.zero_grad(set_to_none=True)
    chosen = manual.online(batch.observations).gather(1, batch.actions[:, None]).squeeze(1)
    expected_loss = torch.nn.functional.smooth_l1_loss(chosen, targets)
    expected_loss.backward()
    torch.nn.utils.clip_grad_norm_(manual.online.parameters(), manual.config.gradient_clip_norm,
                                  error_if_nonfinite=True)
    manual.optimizer.step()
    stats = calibration_update(learner, batch, targets, demonstration_weight=0.0)
    assert stats["loss_before"] == expected_loss.item()
    assert stats["demonstration_loss_before"] == stats["demonstration_loss_after"] == 0.0
    assert_nested_equal(learner.online.state_dict(), manual.online.state_dict())
    assert_nested_equal(learner.optimizer.state_dict(), manual.optimizer.state_dict())


@pytest.mark.parametrize("case", ["shape", "integer", "nonfinite", "device", "weight", "margin"])
def test_invalid_calibration_settings_leave_optimizer_weights_and_counter_unchanged(case):
    learner = example_learner()
    batch = example_batch()
    targets = discounted_demonstration_returns(batch, 0.5)
    calibration_update(learner, batch, targets)
    old_parameters = deepcopy(learner.online.state_dict())
    old_optimizer = deepcopy(learner.optimizer.state_dict())
    old_counter = learner.update_count
    kwargs = {}
    if case == "shape":
        targets = targets[:, None]
    elif case == "integer":
        targets = targets.long()
    elif case == "nonfinite":
        targets = torch.full_like(targets, float("nan"))
    elif case == "device":
        targets = torch.empty(3, device="meta")
    elif case == "weight":
        kwargs["demonstration_weight"] = -1.0
    else:
        kwargs["demonstration_margin"] = 0.0
    with pytest.raises(ValueError):
        calibration_update(learner, batch, targets, **kwargs)
    assert_nested_equal(learner.online.state_dict(), old_parameters)
    assert_nested_equal(learner.optimizer.state_dict(), old_optimizer)
    assert learner.update_count == old_counter
