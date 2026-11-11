"""Exercise the independent fixed-step intervention audit with synthetic traces."""

import importlib.util
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "rescue_verification_test",
    ROOT / "docs/experiments/015_single_action_rescue/verify_single_action_rescue.py",
)
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


class FixedNetwork(torch.nn.Module):
    def forward(self, inputs):
        values = torch.tensor([1., 0., 0., 0., 2., 0., 0., 0.])
        return values.expand(len(inputs), -1)


def fixtures():
    observations = np.zeros((22, 341), dtype=np.float32)
    observations[:, 0] = np.arange(22)
    observations[21, :2] = [20., 1.]
    actions = np.array([4] * 20 + [0, 4])
    values = [1., 0., 0., 0., 2., 0., 0., 0.]

    def make_trace(intervene):
        path = [[float(index), 0.] for index in range(23)]
        if intervene:
            path[21:] = [[20., 1.], [21., 1.]]
        rows = []
        executed = [4] * 22
        if intervene:
            executed[20] = 0
        for index in range(22):
            rows.append({
                "step": index + 1, "position": path[index], "next_position": path[index + 1],
                "encoded_observation_sha256": verifier.array_hash(observations[index]),
                "q_values": values.copy(), "policy_action": 4, "policy_action_name": "north_large",
                "executed_action": executed[index],
                "executed_action_name": verifier.ACTION_NAMES[executed[index]],
                "forced_reference_action": intervene and index == 20,
                "reward": 1., "terminated": index == 21, "truncated": False,
                "stationary": False, "collision_blocked": False, "executed_motion": [1., 0.],
                "distance_to_target": 0. if intervene and index >= 20 else 5.,
                "peak_normalized_deformation": 0.,
            })
        rollout = {
            "training_scene_only": True, "seed": 10001, "steps": 22, "actions": executed,
            "robot_path": path, "return": 22., "success": intervene, "damage": False,
            "budget_exhausted": not intervene, "stationary_steps": 0, "blocked_steps": 0,
            "distance_to_target": rows[-1]["distance_to_target"], "peak_deformation": 0.,
            "first_reference_action_divergence": verifier.first_divergence(executed, actions),
        }
        return {
            "mode": "single_reference_action_then_neural" if intervene else "greedy_reproduction",
            "diagnostic_only": True, "optimizer_updates": 0, "forced_action_count": int(intervene),
            "intervention_step": 21 if intervene else None, "original_greedy_exactly_reproduced": not intervene,
            "records": rows, "rollout": rollout,
        }

    greedy, corrected = make_trace(False), make_trace(True)
    paired = {"step": 21, "prefix_records_equal_before_intervention": True,
              "decision_state_and_q_equal": True, "policy_action": 4,
              "greedy_executed_action": 4, "counterfactual_executed_action": 0,
              "signed_demonstration_gap": -1.}
    return greedy, corrected, paired, observations, actions, FixedNetwork()


def test_readonly_audit_accepts_one_corrected_action_and_resumed_greedy_decisions():
    greedy, corrected, paired, observations, actions, network = fixtures()
    original = deepcopy(greedy)
    for trace, intervene in ((greedy, False), (corrected, True)):
        result = verifier.verify_trace(trace, greedy["rollout"], observations, actions, network,
                                       intervene=intervene)
        assert result["prefix_q_vectors_recomputed"] == 21
        assert result["forced_actions"] == int(intervene)
    assert verifier.verify_pair(greedy, corrected, paired) == paired
    assert greedy == original


@pytest.mark.parametrize("mutation", [
    "zero_based_step", "wrong_action", "second_override", "postfork_teacher_forcing",
    "wrong_q_argmax", "wrong_prefix_q", "wrong_state_hash", "wrong_count",
    "wrong_reward", "missing_end", "early_end", "wrong_next_reference_state",
])
def test_rejects_invalid_intervention_or_inference_evidence(mutation):
    greedy, trace, _, observations, actions, network = fixtures()
    if mutation == "zero_based_step":
        trace["intervention_step"] = 20
    elif mutation == "wrong_action":
        trace["records"][20]["executed_action"] = 1
    elif mutation == "second_override":
        trace["records"][21]["forced_reference_action"] = True
    elif mutation == "postfork_teacher_forcing":
        trace["records"][21]["executed_action"] = 0
    elif mutation == "wrong_q_argmax":
        trace["records"][21]["q_values"][0] = 3.
    elif mutation == "wrong_prefix_q":
        trace["records"][5]["q_values"][4] = 2.25
    elif mutation == "wrong_state_hash":
        trace["records"][20]["encoded_observation_sha256"] = "changed"
    elif mutation == "wrong_count":
        trace["rollout"]["steps"] += 1
    elif mutation == "wrong_reward":
        trace["rollout"]["return"] += 1
    elif mutation == "missing_end":
        trace["records"][-1]["terminated"] = False
    elif mutation == "early_end":
        trace["records"][20]["terminated"] = True
    else:
        trace["records"][21]["encoded_observation_sha256"] = "changed"
    with pytest.raises(ValueError, match="Verification failed"):
        verifier.verify_trace(trace, greedy["rollout"], observations, actions, network, intervene=True)


def test_changed_original_greedy_outcome_is_rejected():
    greedy, _, _, observations, actions, network = fixtures()
    expected = deepcopy(greedy["rollout"])
    expected["return"] = 99.
    with pytest.raises(ValueError, match="unmodified greedy"):
        verifier.verify_trace(greedy, expected, observations, actions, network, intervene=False)


@pytest.mark.parametrize("mutation", ["prefix", "fork_q", "gap"])
def test_paired_check_rejects_changes_before_the_single_intervention(mutation):
    greedy, corrected, paired, _, _, _ = fixtures()
    if mutation == "prefix":
        corrected["records"][19]["reward"] = 2.
    elif mutation == "fork_q":
        corrected["records"][20]["q_values"][4] += 1.
    else:
        paired["signed_demonstration_gap"] = -.5
    with pytest.raises(ValueError, match="Verification failed"):
        verifier.verify_pair(greedy, corrected, paired)


def test_recorded_full_state_check_is_bound_to_the_original_adam_step():
    snapshot = {"parameter_sha256": "same", "optimizer_summary": {"step_min": 22009, "step_max": 22009}}
    trace = {"parameter_sha256": "same", "full_training_state_checks": {
        "exact_snapshot_at_restore": True, "exact_snapshot_after_rollout": True,
        "absolute_td_update": 2009, "adam_parameter_step": 22009,
        "parameter_sha256": "same", "optimizer_state": snapshot["optimizer_summary"],
    }}
    verifier.verify_state_claim(trace, snapshot, 2009)
    trace["full_training_state_checks"]["adam_parameter_step"] = 0
    with pytest.raises(ValueError, match="full-state"):
        verifier.verify_state_claim(trace, snapshot, 2009)
