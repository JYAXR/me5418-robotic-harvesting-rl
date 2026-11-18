"""Reject interventions that could invalidate the frozen first-fork comparison."""

import importlib.util
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "rescue031_test", ROOT / "docs/experiments/031_scene10002_rescue/run_rescue.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def synthetic_pair():
    rows = [{"step": i, "position": [float(i), 0.0],
             "encoded_observation_sha256": str(i), "q_values": [2.0, 0, 0, 0, 1.9, 0, 0, 0],
             "q_top_gap": 0.1, "policy_action": 0, "action": 0,
             "forced_action": False, "scripted_reference": False, "reward": 1.0}
            for i in range(1, 30)]
    corrected = deepcopy(rows)
    corrected[22].update(action=4, forced_action=True)
    return rows, corrected


def test_pair_allows_only_one_correction():
    baseline, corrected = synthetic_pair()
    result = m.audit_pair(baseline, corrected)
    assert result["identical_prefix_steps"] == 22
    assert result["corrected_overrides"] == 1


@pytest.mark.parametrize("side,index,key,value", [
    (1, 0, "reward", 2.0), (1, 21, "action", 4),
    (1, 22, "position", [999.0, 0.0]),
    (1, 22, "encoded_observation_sha256", "wrong"),
    (1, 22, "q_values", [3.0] * 8), (1, 22, "action", 0),
    (1, 22, "forced_action", False), (1, 25, "forced_action", True),
    (1, 25, "action", 4), (1, 25, "policy_action", 1),
    (0, 25, "forced_action", True),
    (1, 25, "step", 23), (1, 25, "scripted_reference", True),
])
def test_rejects_extra_changes(side, index, key, value):
    pair = synthetic_pair()
    pair[side][index][key] = value
    with pytest.raises((AssertionError, ValueError)):
        m.audit_pair(*pair)


def test_first_fork_requires_identical_state_but_ignores_later_trajectory():
    observations = np.arange(6, dtype=np.float32).reshape(3, 2)
    reference = {"observations": observations, "actions": np.array([0, 4, 4])}
    rows = [{"step": i + 1, "encoded_observation_sha256": m.frozen.encoded_hash(obs),
             "action": 0, "q_values": [2.0, 0, 0, 0, 1.5, 0, 0, 0]}
            for i, obs in enumerate(observations)]
    rows[2]["encoded_observation_sha256"] = "expected divergence after the fork"
    result = m.first_fork(rows, reference)
    assert result["step"] == 2
    assert result["policy_minus_reference"] == 0.5
    rows[1]["encoded_observation_sha256"] = "invalid fork input"
    with pytest.raises(AssertionError, match="State differs"):
        m.first_fork(rows, reference)


@pytest.mark.historical_replay
def test_full_snapshot_matches_policy_and_rejects_optimizer_steps():
    # Match the source run's reduction order for exact optimizer-summary norms.
    m.torch.set_num_threads(1)
    source, _, _, _, fork = m.inputs()
    agent, env = m.load_frozen(source)
    try:
        assert fork["step"] == 23 and fork["policy_action"] == 0
        assert fork["reference_action"] == 4
        assert fork["policy_minus_reference"] == 0.0416259765625
        with pytest.raises(RuntimeError, match="forbidden"):
            agent.learner.optimizer.step()
        m.finish_state(agent, source)
    finally:
        env.close()
