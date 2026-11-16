"""Reject unequal-state comparisons and stop at the first action divergence."""

import importlib.util
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("own005tests", Path(__file__).with_name("run_diagnosis.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def row(step, action):
    q = [0.0] * 8
    q[action] = 1.25
    obs = np.array([step, 0], dtype=np.float32)
    return {"step": step, "action": action, "q_values": q, "position": [0, 0],
            "encoded_observation_sha256": m.frozen.encoded_hash(obs), "raw_observation": obs.tolist(),
            "rng_before": {"state": step}, "raw_next_observation": [step + 1, 0],
            "rng_after": {"state": step + 1}, "reward": 0, "terminated": False,
            "truncated": False, "info": {"step": step}}


def test_first_policy_fork_and_gap_ignore_post_fork_alignment():
    current = [row(1, 0), row(2, 2), row(3, 3)]
    baseline = [row(1, 0), row(2, 4), row(3, 5)]
    baseline[2]["raw_observation"] = [999, 999]
    actual = m.first_policy_fork(current, baseline)
    assert actual["step"] == 2 and actual["common_prefix_steps"] == 1
    assert actual["current_chosen_minus_comparison_q"] == 1.25
    assert actual["baseline_chosen_minus_current_action_q"] == 1.25


def test_equal_actions_do_not_hide_changed_input():
    current = [row(1, 0)]
    baseline = deepcopy(current)
    baseline[0]["rng_before"] = {"state": 99}
    with pytest.raises(ValueError, match="Common policy input"):
        m.first_policy_fork(current, baseline)


def test_reference_fork_has_explicit_gap_orientation():
    actual = m.first_reference_fork([row(1, 0), row(2, 2)],
                                  np.array([[1, 0], [2, 0]], dtype=np.float32), np.array([0, 4]))
    assert actual["step"] == 2 and actual["comparison_action"] == 4
    assert actual["current_chosen_minus_comparison_q"] == 1.25


def test_reference_state_mismatch_is_rejected():
    with pytest.raises(AssertionError, match="Reference state mismatch"):
        m.first_reference_fork([row(1, 0)], np.array([[99, 0]], dtype=np.float32), np.array([4]))


def test_identical_policy_trajectories_have_no_fork():
    rows = [row(1, 0), row(2, 2)]
    assert m.first_policy_fork(rows, deepcopy(rows)) is None
