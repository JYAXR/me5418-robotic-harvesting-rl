"""Verify isolated local weighting with retained Adam across target synchronization."""
import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


r = load("local022_test", ROOT / "docs/experiments/022_half_td_local_supervision/run_local_supervision.py")
base = load("local022_fixture", ROOT / "tests/test_td_weight_experiment.py")


def fixture():
    agent, env, config, original, _ = base.fixture()
    batch = base.TransitionBatch(**{name: getattr(original, name).repeat((39, 1) if name in
        ("observations", "next_observations") else (39,)) for name in original.__dataclass_fields__})
    return agent, env, config, batch


@pytest.mark.parametrize("weight", [1.0, 2.0])
def test_exact_independent_updates_and_full_state_across_sync(weight):
    agent, env, config, batch = fixture()
    other = deepcopy(agent)
    try:
        for _ in range(3):
            expected = r.independent_update(other.learner, batch, weight)
            actual = r.weighted_td_update(agent.learner, batch, weight)
            assert actual == expected
            r.equal(r.snapshot(agent, config), r.snapshot(other, config), "independent full-state equality")
            assert all(torch.equal(a.grad, b.grad) for a, b in zip(agent.learner.online.parameters(), other.learner.online.parameters()))
    finally:
        env.close()


def test_weight_one_exactly_matches_historical_half_td():
    agent, env, config, batch = fixture()
    other = deepcopy(agent)
    try:
        for _ in range(3):
            expected = r.r.weighted_td_update(other.learner, batch)
            actual = r.weighted_td_update(agent.learner, batch, 1.0)
            actual.pop("selected_weight")
            assert actual == expected
            r.equal(r.snapshot(agent, config), r.snapshot(other, config), "historical control equality")
    finally:
        env.close()


@pytest.mark.parametrize("weight", [True, 0, -1, 1.5, float("nan")])
def test_invalid_weight_rejected_before_mutation(weight):
    agent, env, config, batch = fixture()
    before = r.snapshot(agent, config)
    try:
        with pytest.raises(ValueError):
            r.weighted_td_update(agent.learner, batch, weight)
        r.equal(r.snapshot(agent, config), before, "unchanged invalid-input state")
    finally:
        env.close()
