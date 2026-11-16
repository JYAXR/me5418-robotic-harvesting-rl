"""Reject hidden interventions and changed inputs in paired action-rescue traces."""

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("own_rescue_audit_tests", Path(__file__).with_name("run_rescue.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def pair(seed):
    runner.configure(seed, Path("unused"))
    fork, policy, _ = runner.CASES[seed]
    q = [0.0] * 8
    q[policy] = 1.0
    baseline = [{"step": step, "position": [0.1, 0.2], "encoded_observation_sha256": "same-state",
                 "q_values": q.copy(), "q_top_gap": 1.0, "policy_action": policy,
                 "action": policy, "forced_action": False, "scripted_reference": False}
                for step in range(1, fork + 2)]
    corrected = deepcopy(baseline)
    corrected[fork - 1].update(action=runner.REPLACEMENT, forced_action=True)
    return baseline, corrected


@pytest.mark.parametrize("seed", runner.CASES)
def test_exactly_one_correction_is_accepted(seed):
    baseline, corrected = pair(seed)
    result = runner.legacy.audit_pair(baseline, corrected)
    assert result["corrected_overrides"] == 1
    assert result["identical_prefix_steps"] == runner.CASES[seed][0] - 1


@pytest.mark.parametrize("seed", runner.CASES)
def test_extra_later_correction_is_rejected(seed):
    baseline, corrected = pair(seed)
    corrected[-1].update(action=runner.REPLACEMENT, forced_action=True)
    with pytest.raises(AssertionError):
        runner.legacy.audit_pair(baseline, corrected)


@pytest.mark.parametrize("seed", runner.CASES)
def test_changed_fork_q_values_are_rejected(seed):
    baseline, corrected = pair(seed)
    corrected[runner.CASES[seed][0] - 1]["q_values"][7] = 0.5
    with pytest.raises(ValueError, match="Identical fork input and policy"):
        runner.legacy.audit_pair(baseline, corrected)


@pytest.mark.parametrize("seed", runner.CASES)
def test_unmarked_action_override_is_rejected(seed):
    baseline, corrected = pair(seed)
    corrected[-1]["action"] = runner.REPLACEMENT
    with pytest.raises(AssertionError):
        runner.legacy.audit_pair(baseline, corrected)
