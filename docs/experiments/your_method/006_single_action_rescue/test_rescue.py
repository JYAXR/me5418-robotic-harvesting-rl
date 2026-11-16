"""Check intervention scope and reject mismatched paired states."""

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("own006tests", Path(__file__).with_name("run_rescue.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def pair(seed):
    step, replacement = m.CASES[seed]
    policy = (replacement + 1) % 8
    rows = []
    for index in range(1, step + 3):
        q = [0.0] * 8
        q[policy] = 2.0
        rows.append({"step": index, "position": [index, 0], "raw_observation": [index],
                     "encoded_observation_sha256": str(index), "rng_before": {"state": index},
                     "q_values": q, "q_top_gap": 2.0, "policy_action": policy,
                     "action": policy, "action_name": m.frozen.ACTION_NAMES[policy], "forced_action": False})
    corrected = deepcopy(rows)
    corrected[step - 1].update(action=replacement, action_name=m.frozen.ACTION_NAMES[replacement], forced_action=True)
    return rows, corrected


@pytest.mark.parametrize("seed", m.CASES)
def test_only_declared_step_changes(seed):
    baseline, corrected = pair(seed)
    result = m.audit_pair(seed, baseline, corrected)
    assert result["corrected_overrides"] == 1
    assert result["policy_minus_replacement_q"] == 2.0
    assert m.selected_action(seed, m.CASES[seed][0], 3, False) == (3, False)


def test_extra_later_override_is_rejected():
    baseline, corrected = pair(290021)
    corrected[-1]["forced_action"] = True
    with pytest.raises(AssertionError, match="Unexpected action replacement"):
        m.audit_pair(290021, baseline, corrected)


def test_changed_fork_rng_is_rejected():
    baseline, corrected = pair(290022)
    corrected[8]["rng_before"] = {"state": 999}
    with pytest.raises(ValueError, match="Identical intervention input"):
        m.audit_pair(290022, baseline, corrected)


def test_early_termination_cannot_silently_skip_override():
    baseline, corrected = pair(290017)
    with pytest.raises(AssertionError, match="Intervention was not reached"):
        m.audit_pair(290017, baseline, corrected[:4])
