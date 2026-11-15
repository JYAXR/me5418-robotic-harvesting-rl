"""Audit the off-reference single-step correction without simulator work."""
import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("step25_test", ROOT / "docs/experiments/024_step25_rescue/run_step25_rescue.py")
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def pair():
    rows = [{"step": i, "position": [i, 0], "next_position": [i + 1, 0],
             "encoded_observation_sha256": str(i), "q_values": [0.] * 8,
             "policy_action": 6, "executed_action": 6, "forced_action": False,
             "reward": 1.} for i in range(1, 27)]
    rows[21]["policy_action"] = rows[21]["executed_action"] = 0
    control = {"records": rows}
    corrected = deepcopy(control)
    corrected["records"][24].update(executed_action=4, forced_action=True)
    return control, corrected


def test_exact_action_selection():
    for intervene in (False, True):
        for step in range(1, 201):
            for policy in range(8):
                assert m.corrected_action(policy, step, intervene) == (4 if intervene and step == 25 else policy)


def test_valid_pair_preserves_step22():
    control, corrected = pair()
    assert m.paired_fork(control, corrected)["identical_prefix"]
    assert corrected["records"][21]["executed_action"] == 0


@pytest.mark.parametrize("index,key,value", [
    (0, "reward", 2.), (21, "executed_action", 4),
    (24, "q_values", [1.] * 8), (24, "encoded_observation_sha256", "changed"),
    (24, "executed_action", 0), (24, "forced_action", False),
    (25, "forced_action", True), (25, "executed_action", 4),
])
def test_rejects_extra_changes(index, key, value):
    control, corrected = pair()
    corrected["records"][index][key] = value
    with pytest.raises(ValueError):
        m.paired_fork(control, corrected)


def test_rejects_forced_control():
    control, corrected = pair()
    control["records"][25]["forced_action"] = True
    with pytest.raises(ValueError):
        m.paired_fork(control, corrected)


def test_required_helper_interfaces():
    assert callable(m.diagnostic.array_hash)
    assert callable(m.diagnostic.first_divergence)
    assert callable(m.v.multi.diagnostic.fit.fixed_scene_rollout)
    assert callable(m.v.multi.diagnostic.fit.episode_metrics)
