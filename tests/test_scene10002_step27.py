"""Guard the conditional intervention against changed prefixes or extra overrides."""

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "step27_test", ROOT / "docs/experiments/032_scene10002_step27/run_step27.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def synthetic_pair():
    rows = []
    for step in range(1, 33):
        policy = 6 if step == 27 else 0
        q = [0.0] * 8
        q[policy] = 2.0
        rows.append({"step": step, "position": [float(step), 0.0],
                     "encoded_observation_sha256": str(step), "q_values": q,
                     "q_top_gap": 2.0, "policy_action": policy,
                     "action": 4 if step == 23 else policy,
                     "forced_action": step == 23, "scripted_reference": False, "reward": 1.0})
    corrected = deepcopy(rows)
    corrected[26].update(action=0, forced_action=True)
    return rows, corrected


def test_action_selection_keeps_first_correction_and_changes_only_step27():
    for mode in ("baseline", "corrected"):
        for step in range(1, 201):
            for policy in range(8):
                expected = 4 if step == 23 else (0 if mode == "corrected" and step == 27 else policy)
                forced = step == 23 or (mode == "corrected" and step == 27)
                assert m.selected_action(policy, step, mode) == (expected, forced)
    with pytest.raises(AssertionError, match="Unknown"):
        m.selected_action(0, 27, "unknown")


def test_valid_pair_has_one_additional_override():
    result = m.audit_pair(*synthetic_pair())
    assert result["identical_prefix_steps"] == 26
    assert result["override_counts"] == {"baseline": 1, "corrected": 2}


@pytest.mark.parametrize("side,index,key,value", [
    (1, 0, "reward", 2.0), (1, 22, "action", 0),
    (1, 22, "forced_action", False), (1, 25, "action", 4),
    (1, 26, "position", [999.0, 0.0]),
    (1, 26, "encoded_observation_sha256", "wrong"),
    (1, 26, "q_values", [3.0] * 8), (1, 26, "action", 6),
    (1, 26, "forced_action", False), (1, 30, "forced_action", True),
    (1, 30, "action", 4), (1, 30, "policy_action", 1),
    (0, 26, "forced_action", True), (0, 26, "action", 0),
    (1, 30, "step", 27), (1, 30, "scripted_reference", True),
])
def test_rejects_extra_changes(side, index, key, value):
    pair = synthetic_pair()
    pair[side][index][key] = value
    with pytest.raises((AssertionError, ValueError)):
        m.audit_pair(*pair)


def test_rejects_removing_shared_step23_correction_from_both_arms():
    pair = synthetic_pair()
    for rows in pair:
        rows[22].update(action=0, forced_action=False)
    with pytest.raises(AssertionError):
        m.audit_pair(*pair)


@pytest.mark.historical_replay
def test_frozen_inputs_and_optimizer_guard():
    m.torch.set_num_threads(1)
    source, _, history, result, fork = m.inputs()
    assert len(history) == 200 and result["forced_actions"] == 1
    assert fork["step"] == 27 and fork["policy_action"] == 6 and fork["reference_action"] == 0
    assert fork["policy_minus_reference"] == 0.03324127197265625
    agent, env = m.load_frozen(source)
    try:
        with pytest.raises(RuntimeError, match="attempt 032"):
            agent.learner.optimizer.step()
        m.finish_state(agent, source)
    finally:
        env.close()
