"""Ensure the two-arm experiment preserves the first correction and only adds one."""

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("second_rescue_tests", Path(__file__).with_name("run_rescue.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def pair(seed):
    runner.configure(seed, Path("unused"))
    first, first_action, second, second_action = runner.CASES[seed]
    q = [0.0] * 8
    q[5] = 1.0
    control = [{"step": step, "position": [0.1, 0.2], "encoded_observation_sha256": "same-state",
                "q_values": q.copy(), "q_top_gap": 1.0, "policy_action": 5,
                "action": first_action if step == first else 5,
                "forced_action": step == first, "scripted_reference": False}
               for step in range(1, second + 2)]
    treatment = deepcopy(control)
    treatment[second - 1].update(action=second_action, forced_action=True)
    return control, treatment


@pytest.mark.parametrize("seed", runner.CASES)
def test_one_fixed_plus_one_additional_correction(seed):
    control, treatment = pair(seed)
    result = runner.audit_pair(seed, control, treatment)
    assert result["override_counts"] == {"baseline": 1, "corrected": 2}
    for mode, rows in (("baseline", control), ("corrected", treatment)):
        for row in rows:
            assert runner.legacy.selected_action(row["policy_action"], row["step"], mode) == (
                row["action"], row["forced_action"])


@pytest.mark.parametrize("seed", runner.CASES)
def test_missing_shared_correction_is_rejected(seed):
    control, treatment = pair(seed)
    for rows in (control, treatment):
        rows[runner.CASES[seed][0] - 1].update(action=5, forced_action=False)
    with pytest.raises(AssertionError, match="Unexpected intervention"):
        runner.audit_pair(seed, control, treatment)


@pytest.mark.parametrize("seed", runner.CASES)
def test_extra_later_override_is_rejected(seed):
    control, treatment = pair(seed)
    treatment[-1].update(action=6, forced_action=True)
    with pytest.raises(AssertionError, match="Unexpected intervention"):
        runner.audit_pair(seed, control, treatment)


@pytest.mark.parametrize("seed", runner.CASES)
def test_changed_fork_input_is_rejected(seed):
    control, treatment = pair(seed)
    treatment[runner.CASES[seed][2] - 1]["encoded_observation_sha256"] = "different-state"
    with pytest.raises(ValueError, match="Identical fork input and policy"):
        runner.audit_pair(seed, control, treatment)
