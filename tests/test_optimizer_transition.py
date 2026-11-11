"""Verify that optimizer retention changes only Adam state at the TD boundary."""

from copy import deepcopy

import pytest

torch = pytest.importorskip("torch")

from me5418_harvesting.agents.dqn import DQNConfig
from me5418_harvesting.agents.dqn_learning import DQNUpdater, TransitionBatch
from me5418_harvesting.agents.optimizer_transition import (
    begin_td_with_optimizer,
    optimizer_state_summary,
)
from me5418_harvesting.agents.return_calibration import calibration_update


def fixture():
    torch.set_num_threads(1)
    torch.manual_seed(31)
    learner = DQNUpdater(5, 3, DQNConfig(hidden_sizes=(8, 8), target_update_interval=2))
    batch = TransitionBatch(
        torch.randn(4, 5), torch.tensor([0, 2, 1, 2]), torch.tensor([0.0, -0.5, 1.0, 10.0]),
        torch.randn(4, 5), torch.tensor([False, False, False, True]),
        torch.zeros(4, dtype=torch.bool),
    )
    for _ in range(3):
        calibration_update(learner, batch, torch.tensor([8.0, 8.5, 9.0, 10.0]))
    return learner, batch


def equal_state(actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            equal_state(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for first, second in zip(actual, expected):
            equal_state(first, second)
    else:
        assert actual == expected


def raw_optimizer_snapshot(optimizer):
    """Inspect deliberately malformed optimizers without calling state_dict()."""
    return {
        "state": {id(parameter): deepcopy(state) for parameter, state in optimizer.state.items()},
        "groups": [{key: [id(parameter) for parameter in value] if key == "params"
                    else deepcopy(value) for key, value in group.items()}
                   for group in optimizer.param_groups],
    }


@pytest.mark.parametrize("preserve", [False, True])
def test_transition_preserves_weights_and_has_identical_target_clock(preserve):
    learner, _ = fixture()
    weights = deepcopy(learner.online.state_dict())
    old_optimizer = learner.optimizer
    old_state = deepcopy(old_optimizer.state_dict())
    before = optimizer_state_summary(learner)
    learner.target.requires_grad_(True)
    learner.target.train()
    for parameter in learner.target.parameters():
        parameter.grad = torch.ones_like(parameter)
    record = begin_td_with_optimizer(learner, preserve_optimizer=preserve)
    assert record["previous_phase_updates"] == 3
    assert record["phase_updates"] == learner.update_count == 0
    assert record["optimizer_reset"] == (not preserve)
    assert record["target_synchronized"]
    assert record["optimizer_before"] == before
    assert record["optimizer_after"]["settings_sha256"] == before["settings_sha256"]
    equal_state(learner.online.state_dict(), weights)
    equal_state(learner.target.state_dict(), weights)
    assert not learner.target.training
    assert all(not parameter.requires_grad for parameter in learner.target.parameters())
    assert all(parameter.grad is None for parameter in learner.online.parameters())
    assert all(parameter.grad is None for parameter in learner.target.parameters())
    if preserve:
        assert learner.optimizer is old_optimizer
        equal_state(learner.optimizer.state_dict(), old_state)
        assert record["optimizer_after"] == before
    else:
        assert learner.optimizer is not old_optimizer
        assert not learner.optimizer.state
        assert record["optimizer_after"]["step_min"] is None
        assert record["optimizer_after"]["sha256"] != before["sha256"]


def test_identical_td_gradients_use_different_bias_correction_steps():
    preserved, batch = fixture()
    reset = deepcopy(preserved)
    begin_td_with_optimizer(preserved, preserve_optimizer=True)
    begin_td_with_optimizer(reset, preserve_optimizer=False)
    initial_target = deepcopy(preserved.target.state_dict())
    first_preserved = preserved.update(batch, demonstration_weight=1.0)
    first_reset = reset.update(batch, demonstration_weight=1.0)
    assert first_preserved.loss_before == first_reset.loss_before
    assert first_preserved.gradient_norm == first_reset.gradient_norm
    assert first_preserved.parameter_delta_norm != first_reset.parameter_delta_norm
    assert not first_preserved.target_synchronized and not first_reset.target_synchronized
    assert optimizer_state_summary(preserved)["step_min"] == 4
    assert optimizer_state_summary(reset)["step_min"] == 1
    equal_state(preserved.target.state_dict(), initial_target)
    equal_state(reset.target.state_dict(), initial_target)
    assert preserved.update(batch).target_synchronized
    assert reset.update(batch).target_synchronized
    assert preserved.update_count == reset.update_count == 2


def test_reset_preserves_nondefault_settings_and_parameter_groups():
    learner, batch = fixture()
    parameters = list(learner.online.parameters())
    learner.optimizer = torch.optim.Adam([
        {"params": parameters[:2], "lr": 7e-5, "betas": (0.7, 0.95)},
        {"params": parameters[2:], "lr": 2e-4, "weight_decay": 0.01, "amsgrad": True},
    ], eps=1e-6, foreach=False)
    learner.update(batch)
    before = optimizer_state_summary(learner)
    record = begin_td_with_optimizer(learner, preserve_optimizer=False)
    assert record["optimizer_after"]["settings_sha256"] == before["settings_sha256"]
    assert record["optimizer_after"]["parameter_groups"] == before["parameter_groups"]
    assert not learner.optimizer.state
    learner.update(batch)
    assert optimizer_state_summary(learner)["step_min"] == 1


def test_hash_ignores_object_identity_but_detects_moments_steps_and_settings():
    learner, _ = fixture()
    cloned = deepcopy(learner)
    reference = optimizer_state_summary(learner)
    assert optimizer_state_summary(cloned) == reference
    first = next(iter(cloned.optimizer.state.values()))
    first["exp_avg"].flatten()[0] += 0.01
    changed = optimizer_state_summary(cloned)
    assert changed["sha256"] != reference["sha256"]
    assert changed["settings_sha256"] == reference["settings_sha256"]
    cloned = deepcopy(learner)
    next(iter(cloned.optimizer.state.values()))["step"] += 1
    assert optimizer_state_summary(cloned)["sha256"] != reference["sha256"]
    cloned = deepcopy(learner)
    cloned.optimizer.param_groups[0]["lr"] *= 0.5
    assert optimizer_state_summary(cloned)["settings_sha256"] != reference["settings_sha256"]


@pytest.mark.parametrize("corruption", [
    "empty", "partial", "nan_moment", "negative_second_moment", "nan_step", "fractional_step",
    "wrong_shape", "foreign_parameter", "duplicate_parameter", "invalid_clock", "nan_weights",
])
def test_invalid_state_is_rejected_before_any_transition_mutation(corruption):
    learner, _ = fixture()
    state = next(iter(learner.optimizer.state.values()))
    if corruption == "empty":
        learner.optimizer.state.clear()
    elif corruption == "partial":
        del state["exp_avg"]
    elif corruption == "nan_moment":
        state["exp_avg"].flatten()[0] = float("nan")
    elif corruption == "negative_second_moment":
        state["exp_avg_sq"].flatten()[0] = -1
    elif corruption == "nan_step":
        state["step"].fill_(float("nan"))
    elif corruption == "fractional_step":
        state["step"].fill_(1.5)
    elif corruption == "wrong_shape":
        state["exp_avg"] = torch.zeros(1)
    elif corruption == "foreign_parameter":
        learner.optimizer.param_groups[0]["params"][0] = torch.nn.Parameter(torch.zeros(1))
    elif corruption == "duplicate_parameter":
        learner.optimizer.param_groups[0]["params"][0] = learner.optimizer.param_groups[0]["params"][1]
    elif corruption == "invalid_clock":
        learner.update_count = -1
    elif corruption == "nan_weights":
        with torch.no_grad():
            next(learner.online.parameters()).flatten()[0] = float("nan")
    before_weights = deepcopy(learner.online.state_dict())
    before_target = deepcopy(learner.target.state_dict())
    before_optimizer = raw_optimizer_snapshot(learner.optimizer)
    before_count = learner.update_count
    optimizer = learner.optimizer
    gradients = [None if p.grad is None else p.grad.clone() for p in learner.online.parameters()]
    with pytest.raises(ValueError):
        begin_td_with_optimizer(learner, preserve_optimizer=True)
    assert learner.optimizer is optimizer
    assert learner.update_count == before_count
    equal_state(learner.online.state_dict(), before_weights)
    equal_state(learner.target.state_dict(), before_target)
    equal_state(raw_optimizer_snapshot(learner.optimizer), before_optimizer)
    equal_state([p.grad for p in learner.online.parameters()], gradients)


@pytest.mark.parametrize("setting", [None, 0, 1, "preserve"])
def test_nonboolean_selection_is_rejected(setting):
    learner, _ = fixture()
    before = optimizer_state_summary(learner)
    with pytest.raises(TypeError, match="boolean"):
        begin_td_with_optimizer(learner, preserve_optimizer=setting)
    assert optimizer_state_summary(learner) == before
    assert learner.update_count == 3
