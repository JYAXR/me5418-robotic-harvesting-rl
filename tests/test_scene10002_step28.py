"""Guard the conditional intervention against changed prefixes or extra overrides."""

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "step28_test", ROOT / "docs/experiments/033_scene10002_step28/run_step28.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def synthetic_pair():
    rows = []
    for step in range(1, 33):
        policy = 6 if step in (27, 28) else 0
        q = [0.0] * 8
        q[policy] = 2.0
        rows.append({"step": step, "position": [float(step), 0.0],
                     "encoded_observation_sha256": str(step), "q_values": q,
                     "q_top_gap": 2.0, "policy_action": policy,
                     "action": 4 if step == 23 else (0 if step == 27 else policy),
                     "forced_action": step in (23, 27), "scripted_reference": False, "reward": 1.0})
    corrected = deepcopy(rows)
    corrected[27].update(action=4, forced_action=True)
    return rows, corrected


def test_action_selection_keeps_first_correction_and_changes_only_step28():
    for mode in ("baseline", "corrected"):
        for step in range(1, 201):
            for policy in range(8):
                expected = 0 if step == 27 else (4 if step == 23 or (mode == "corrected" and step == 28) else policy)
                forced = step in (23, 27) or (mode == "corrected" and step == 28)
                assert m.selected_action(policy, step, mode) == (expected, forced)
    with pytest.raises(AssertionError, match="Unknown"):
        m.selected_action(0, 27, "unknown")


def test_valid_pair_has_one_additional_override():
    result = m.audit_pair(*synthetic_pair())
    assert result["identical_prefix_steps"] == 27
    assert result["override_counts"] == {"baseline": 2, "corrected": 3}


@pytest.mark.parametrize("side,index,key,value", [
    (1, 0, "reward", 2.0), (1, 22, "action", 0),
    (1, 22, "forced_action", False), (1, 26, "action", 6),
    (1, 26, "forced_action", False), (1, 27, "position", [999.0, 0.0]),
    (1, 27, "encoded_observation_sha256", "wrong"),
    (1, 27, "q_values", [3.0] * 8), (1, 27, "action", 6),
    (1, 27, "forced_action", False), (1, 30, "forced_action", True),
    (1, 30, "action", 4), (1, 30, "policy_action", 1),
    (0, 27, "forced_action", True), (0, 27, "action", 4),
    (1, 30, "step", 28), (1, 30, "scripted_reference", True),
])
def test_rejects_extra_changes(side, index, key, value):
    pair = synthetic_pair()
    pair[side][index][key] = value
    with pytest.raises((AssertionError, ValueError)):
        m.audit_pair(*pair)


@pytest.mark.parametrize("index,action", [(22, 0), (26, 6)])
def test_rejects_removing_shared_correction_from_both_arms(index, action):
    pair = synthetic_pair()
    for rows in pair:
        rows[index].update(action=action, forced_action=False)
    with pytest.raises(AssertionError):
        m.audit_pair(*pair)


@pytest.mark.historical_replay
def test_frozen_inputs_and_optimizer_guard():
    m.torch.set_num_threads(1)
    source, _, history, result, fork = m.inputs()
    assert len(history) == 58 and result["forced_actions"] == 2
    assert fork["step"] == 28 and fork["policy_action"] == 6 and fork["reference_action"] == 4
    assert fork["policy_minus_reference"] == 0.46216583251953125
    agent, env = m.load_frozen(source)
    try:
        with pytest.raises(RuntimeError, match="attempt 033"):
            agent.learner.optimizer.step()
        m.finish_state(agent, source)
    finally:
        env.close()
