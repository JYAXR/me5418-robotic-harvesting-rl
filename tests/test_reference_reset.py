"""Verify opt-in real-dynamics route certification at episode reset."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from me5418_harvesting import HarvestingEnv
from me5418_harvesting.evaluation import scene_feasibility


def test_reset_keeps_all_reference_checks_and_retries(monkeypatch):
    results = iter([
        {"success": False, "reason": "planner_found_no_route"},
        {"success": True, "reason": "success"},
    ])
    monkeypatch.setattr(scene_feasibility, "verify_reference_route", lambda env: next(results))
    env = HarvestingEnv({"verify_reference_route": True, "reference_max_attempts": 2})
    observation, info = env.reset(seed=0)
    assert info["reference_route_attempts"] == 2
    assert not info["reference_route_checks"][0]["success"]
    assert info["reference_route"]["success"]
    assert env.step_count == 0 and observation[-1] == 1
    env.close()


def test_reference_retry_exhaustion_does_not_leave_a_usable_scene(monkeypatch):
    calls = []

    def fail(env):
        calls.append(1)
        return {"success": False, "reason": "planner_found_no_route"}

    monkeypatch.setattr(scene_feasibility, "verify_reference_route", fail)
    env = HarvestingEnv({"verify_reference_route": True, "reference_max_attempts": 2})
    with pytest.raises(ValueError, match="after 2 candidates.*does not prove"):
        env.reset(seed=0)
    assert len(calls) == 2
    assert env.scene is None
    env.close()


def test_certification_does_not_change_agent_initial_state_or_random_stream():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/environment_multiroute.yaml").read_text())
    checked = HarvestingEnv(config)
    plain = HarvestingEnv({**config, "verify_reference_route": False})
    try:
        first, info = checked.reset(seed=0)
        second, _ = plain.reset(seed=0)
        assert info["reference_route"]["success"]
        assert info["reference_route"]["steps"] > 0
        assert checked.step_count == 0 and checked.total_travel == 0
        assert len(checked.trajectory) == 1
        np.testing.assert_array_equal(first, second)
        assert checked.np_random.bit_generator.state == plain.np_random.bit_generator.state
        for branch in checked.scene.branches:
            np.testing.assert_array_equal(branch.current_nodes, branch.rest_nodes)
    finally:
        checked.close()
        plain.close()


@pytest.mark.parametrize("config", [
    {"verify_reference_route": "yes"}, {"reference_max_attempts": 0},
])
def test_invalid_reference_configuration(config):
    with pytest.raises((TypeError, ValueError)):
        HarvestingEnv(config)
