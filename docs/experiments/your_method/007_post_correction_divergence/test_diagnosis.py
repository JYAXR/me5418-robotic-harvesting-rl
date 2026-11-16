"""Verify that a shared executed correction does not hide the next fork."""

import importlib.util
from copy import deepcopy
from pathlib import Path

spec = importlib.util.spec_from_file_location("own007tests", Path(__file__).with_name("run_diagnosis.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_fixed_non_greedy_step_is_followed_by_same_state_fork():
    current = []
    for step in range(1, 8):
        action = 5 if step == 5 else 0
        q = [0.0] * 8
        q[0] = 2.0
        current.append({"step": step, "action": action, "q_values": q,
                        "position": [step, 0], "raw_observation": [step],
                        "encoded_observation_sha256": str(step), "rng_before": step,
                        "raw_next_observation": [step + 1], "rng_after": step + 1,
                        "reward": 0.0, "terminated": False, "truncated": False, "info": {}})
    baseline = deepcopy(current)
    baseline[4]["q_values"][5] = 3.0
    baseline[5]["action"] = 2
    baseline[5]["q_values"][2] = 4.0
    baseline[6]["raw_observation"] = [999]
    fork = m.prior.first_policy_fork(current, baseline)
    assert fork["step"] == 6 and fork["common_prefix_steps"] == 5
    assert fork["current_action"] == 0 and fork["comparison_action"] == 2
    assert fork["current_chosen_minus_comparison_q"] == 2.0
    assert fork["baseline_chosen_minus_current_action_q"] == 2.0
