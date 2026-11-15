"""Check actual return boundaries, staged objectives and optimizer continuity."""

import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "test030", ROOT / "docs/experiments/030_expanded_stages/run_stages.py")
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)


@pytest.fixture(scope="module")
def inputs():
    torch.set_num_threads(1)
    return r.inputs()


def test_actual_returns_have_separate_terminal_boundaries(inputs):
    _, batch, returns, _, metadata, error = inputs
    assert len(metadata["episodes"]) == 33 and len(returns) == 1440
    assert error < 0.0002
    for episode in metadata["episodes"]:
        start, end = episode["start"], episode["stop"] - 1
        assert returns[end] == batch.rewards[end]
        assert batch.terminated[end] and not batch.terminated[start:end].any()
        assert float(returns[start]) == episode["initial_discounted_return"]
    assert not set(r.TEST_SEEDS) & (set(metadata["seeds"]) | set(range(280000, 280100))
                                  | set(range(390000, 390100)))
    assert len(r.TEST_SEEDS) == 100


@pytest.mark.parametrize("phase", ["supervised", "calibration"])
def test_independent_objective_and_complete_adam_state(inputs, phase):
    source, batch, returns, recovery, _, _ = inputs
    pairs = [r.restore(source) for _ in range(2)]
    try:
        for agent, _ in pairs:
            r.begin_td_with_optimizer(agent.learner, preserve_optimizer=False)
        target_before = {k: v.clone() for k, v in pairs[0][0].learner.target.state_dict().items()}
        actual = r.update(pairs[0][0], phase, batch, returns, recovery)
        expected = r.update(pairs[1][0], phase, batch, returns, recovery, independent=True)
        r.equal(actual, expected, "Independently calculated objective and update")
        r.equal(r.snapshot(pairs[0][0], source["environment_config"]),
                r.snapshot(pairs[1][0], source["environment_config"]), "Complete staged state")
        r.equal(target_before, pairs[0][0].learner.target.state_dict(), "No bootstrapped target update")
        before = r.snapshot(pairs[0][0], source["environment_config"])
        transition = r.begin_td_with_optimizer(pairs[0][0].learner, preserve_optimizer=True)
        after = r.snapshot(pairs[0][0], source["environment_config"])
        r.equal(before["optimizer"], after["optimizer"], "Retained moments and bias-correction clock")
        r.equal(before["online"], after["target"], "Boundary target synchronized")
        assert transition["optimizer_before"]["sha256"] == transition["optimizer_after"]["sha256"]
    finally:
        for _, env in pairs:
            env.close()


@pytest.mark.parametrize("phase", ["supervised", "calibration"])
def test_no_next_state_reward_or_target_dependency(inputs, phase):
    source, batch, returns, recovery, _, _ = inputs
    pairs = [r.restore(source) for _ in range(2)]
    try:
        changed = replace(batch, rewards=torch.full_like(batch.rewards, -999),
                          next_observations=torch.full_like(batch.next_observations, 999))
        for parameter in pairs[1][0].learner.target.parameters():
            parameter.data.fill_(999)
        a = r.update(pairs[0][0], phase, batch, returns, recovery)
        b = r.update(pairs[1][0], phase, changed, returns, recovery)
        r.equal(a, b, "Pre-TD stage uses no bootstrapping or mutable reward target")
        r.equal(pairs[0][0].learner.online.state_dict(), pairs[1][0].learner.online.state_dict(),
                "Unchanged online update")
        r.equal(pairs[0][0].learner.optimizer.state_dict(), pairs[1][0].learner.optimizer.state_dict(),
                "Unchanged optimizer update")
    finally:
        for _, env in pairs:
            env.close()


def test_invalid_return_target_rejected_before_mutation(inputs):
    source, batch, returns, recovery, _, _ = inputs
    agent, env = r.restore(source)
    try:
        before = r.snapshot(agent, source["environment_config"])
        bad = returns.clone()
        bad[0] = float("nan")
        with pytest.raises(ValueError, match="finite"):
            r.core.stage_update(agent.learner, batch, recovery, bad)
        r.equal(before, r.snapshot(agent, source["environment_config"]), "Rejected update has no effect")
    finally:
        env.close()
