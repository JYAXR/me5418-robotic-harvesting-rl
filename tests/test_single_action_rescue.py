"""Guard a fixed one-action diagnostic against broader interventions and learning."""

import importlib.util
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from me5418_harvesting.agents.dqn import DQNConfig
from me5418_harvesting.agents.dqn_agent import DQNAgent
from me5418_harvesting.agents.dqn_learning import TransitionBatch
from me5418_harvesting.envs.harvesting_env import HarvestingEnv

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "single_action_rescue_tests",
    ROOT / "docs/experiments/015_single_action_rescue/run_single_action_rescue.py",
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def contract():
    actions = np.zeros(41, dtype=np.int64)
    expected = {
        "seed": 10001, "training_scene_only": True, "success": False, "damage": False,
        "budget_exhausted": True, "steps": 200,
        "actions": [0] * 20 + [4] * 180,
        "first_reference_action_divergence": {"step": 21, "predicted_action": 4, "demonstrated_action": 0},
    }
    return expected, actions


@pytest.mark.parametrize("mutation", ("step", "original_action", "forced_action", "stored_label", "success"))
def test_fixed_intervention_rejects_any_changed_condition(mutation):
    expected, actions = contract()
    runner.validate_intervention(expected, actions)
    if mutation == "step":
        expected["actions"][19] = 4
    elif mutation == "original_action":
        expected["actions"][20] = 6
    elif mutation == "forced_action":
        actions[20] = 1
    elif mutation == "stored_label":
        expected["first_reference_action_divergence"]["step"] = 22
    else:
        expected["success"] = True
    with pytest.raises(ValueError, match="Verification failed"):
        runner.validate_intervention(expected, actions)


def synthetic_snapshot():
    torch.set_num_threads(1)
    environment = {"max_steps": 3}
    env = HarvestingEnv(environment)
    agent = DQNAgent(env, DQNConfig(hidden_sizes=(8, 8), learning_rate=1e-4,
                                   target_update_interval=200), seed=47)
    batch = TransitionBatch(
        observations=torch.zeros(2, env.observation_size), actions=torch.tensor([0, 4]),
        rewards=torch.tensor([1., -1.]), next_observations=torch.zeros(2, env.observation_size),
        terminated=torch.ones(2, dtype=torch.bool), truncated=torch.zeros(2, dtype=torch.bool),
    )
    agent.learner.update(batch)
    agent.learner.update_count = 2009
    for state in agent.learner.optimizer.state.values():
        state["step"].fill_(22009)
    snapshot = runner.adam.training_snapshot(agent, environment)
    env.close()
    return snapshot


def test_each_arm_restores_full_state_and_runs_exactly_once(monkeypatch):
    snapshot = synthetic_snapshot()
    original = deepcopy(snapshot)
    expected, actions = contract()
    calls, agents = [], []

    def fake_trace(agent, env, observations, labels, stored, *, intervene, progress):
        calls.append(intervene)
        agents.append(agent)
        runner.require_equal(runner.adam.training_snapshot(agent, snapshot["environment_config"]), snapshot,
                             "independent initial state")
        agent.learner.online.eval()
        return {"forced_action_count": int(intervene)}

    monkeypatch.setattr(runner.diagnostic, "trace_policy", fake_trace)
    for intervene in (False, True):
        trace = runner.frozen_trace(snapshot, np.zeros((41, 341)), actions, expected, intervene=intervene)
        assert trace["full_training_state_checks"]["exact_snapshot_after_rollout"]
        assert trace["full_training_state_checks"]["absolute_td_update"] == 2009
        assert trace["full_training_state_checks"]["adam_parameter_step"] == 22009
    assert calls == [False, True]
    assert agents[0] is not agents[1]
    runner.require_equal(snapshot, original, "source snapshot stays immutable")


@pytest.mark.parametrize("mutation", ("online", "target", "adam", "counter", "agent_rng", "torch_rng"))
def test_trace_guard_rejects_changes_to_any_training_state(monkeypatch, mutation):
    snapshot = synthetic_snapshot()
    expected, actions = contract()

    def corrupt(agent, env, observations, labels, stored, *, intervene, progress):
        if mutation in ("online", "target"):
            with torch.no_grad():
                next(getattr(agent.learner, mutation).parameters()).add_(1)
        elif mutation == "adam":
            next(iter(agent.learner.optimizer.state.values()))["exp_avg"].add_(1)
        elif mutation == "counter":
            agent.learner.update_count += 1
        elif mutation == "agent_rng":
            agent.rng.random()
        else:
            torch.rand(1)
        return {"forced_action_count": 1}

    monkeypatch.setattr(runner.diagnostic, "trace_policy", corrupt)
    with pytest.raises(ValueError, match="complete full training state"):
        runner.frozen_trace(snapshot, np.zeros((41, 341)), actions, expected, intervene=True)


def paired_traces():
    left = []
    for step in range(1, 24):
        policy = 0 if step < 21 else 4 if step == 21 else 6
        q = [0.] * 8
        q[policy] = 2.
        left.append({
            "step": step, "position": [float(step), 0.],
            "encoded_observation_sha256": str(step), "q_values": q,
            "policy_action": policy, "executed_action": policy, "forced_reference_action": False,
        })
    right = deepcopy(left)
    right[20]["executed_action"] = 0
    right[20]["forced_reference_action"] = True
    return ({"records": left, "forced_action_count": 0},
            {"records": right, "forced_action_count": 1})


def test_pair_accepts_exactly_one_override_and_later_neural_actions():
    greedy, counterfactual = paired_traces()
    result = runner.paired_fork(greedy, counterfactual)
    assert result["step"] == 21
    assert result["signed_demonstration_gap"] == -2.
    assert result["decision_state_and_q_equal"]
    assert counterfactual["records"][21]["executed_action"] == 6


@pytest.mark.parametrize("mutation", ("prefix", "q", "second_override", "missing_override", "action"))
def test_pair_rejects_extra_changes_or_missing_correction(mutation):
    greedy, counterfactual = paired_traces()
    if mutation == "prefix":
        counterfactual["records"][19]["position"] = [-1., 0.]
    elif mutation == "q":
        counterfactual["records"][20]["q_values"][0] += 1
    elif mutation == "second_override":
        counterfactual["records"][21]["executed_action"] = 0
        counterfactual["records"][21]["forced_reference_action"] = True
    elif mutation == "missing_override":
        counterfactual["forced_action_count"] = 0
    else:
        counterfactual["records"][20]["executed_action"] = 1
    with pytest.raises(ValueError, match="Verification failed"):
        runner.paired_fork(greedy, counterfactual)


def test_invalid_fixed_contract_aborts_before_restore(monkeypatch):
    expected, actions = contract()
    expected["actions"][1] = 4

    def forbidden(*args):
        pytest.fail("A changed intervention must fail before restoring any model.")

    monkeypatch.setattr(runner.previous, "independent_restore", forbidden)
    with pytest.raises(ValueError, match="only permitted intervention"):
        runner.frozen_trace({}, np.zeros((41, 341)), actions, expected, intervene=True)


def test_existing_output_is_rejected_before_input_read(tmp_path, monkeypatch):
    def forbidden(*args):
        pytest.fail("Existing output must be rejected before reading source data.")

    monkeypatch.setattr(runner, "load_inputs", forbidden)
    with pytest.raises(FileExistsError, match="overwrite"):
        runner.run(tmp_path / "reference", tmp_path / "archive", tmp_path)
