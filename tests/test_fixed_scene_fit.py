"""Protect the controls and evidence in the fixed-scene fitting diagnostic."""

import importlib.util
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from me5418_harvesting.agents.dqn import DQNConfig
from me5418_harvesting.agents.dqn_agent import DQNAgent
from me5418_harvesting.agents.dqn_learning import TransitionBatch
from me5418_harvesting.agents.replay_buffer import ReplayBuffer
from me5418_harvesting.envs.harvesting_env import HarvestingEnv

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fixed_scene_diagnostic",
    ROOT / "docs/experiments/005_fixed_scene_fit/run_fixed_scene_fit.py",
)
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


def make_fixture():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    env = HarvestingEnv({"max_steps": 4})
    config = DQNConfig(hidden_sizes=(8, 8), batch_size=4, replay_capacity=32,
                       target_update_interval=2)
    agent = DQNAgent(env, config, seed=7)
    replay = ReplayBuffer(32)
    observation, _ = env.reset(seed=3)
    for _ in range(4):
        next_observation, reward, terminated, truncated, _ = env.step(1)
        replay.add(agent.encoder(observation), 1, reward,
                   agent.encoder(next_observation), terminated, truncated)
        observation = next_observation
        if terminated or truncated:
            break
    return env, agent, replay, diagnostic.complete_batch(replay)


def test_complete_batch_retains_order_continuity_and_terminal_transition():
    env, _, replay, batch = make_fixture()
    try:
        assert len(batch.actions) == len(replay) == 4
        torch.testing.assert_close(batch.next_observations[:-1], batch.observations[1:])
        assert batch.terminated.tolist() == [False, False, False, True]
        assert not batch.truncated.any()
        assert batch.actions.tolist() == [row[1] for row in replay.records]
        np.testing.assert_array_equal(batch.observations[0], replay.records[0][0])
        with pytest.raises(ValueError, match="complete successful"):
            diagnostic.complete_batch(ReplayBuffer(4))
    finally:
        env.close()


def test_fit_metrics_count_all_states_and_orient_confusion_rows_as_labels():
    scores = torch.tensor([[3.0, 1.0, 0.0], [0.0, 2.0, 1.0], [5.0, 0.0, 0.0]])
    agent = SimpleNamespace(learner=SimpleNamespace(online=torch.nn.Identity()), action_count=3)
    batch = TransitionBatch(scores, torch.tensor([0, 2, 0]), torch.zeros(3), scores.clone(),
                            torch.zeros(3, dtype=torch.bool), torch.zeros(3, dtype=torch.bool))
    result = diagnostic.fit_metrics(agent, batch, margin=0.8)
    assert result["correct"] == 2
    assert result["action_agreement"] == pytest.approx(2 / 3)
    assert result["prediction_counts"] == [2, 1, 0]
    assert result["confusion_true_rows_predicted_columns"] == [[2, 0, 0], [0, 0, 0], [0, 1, 0]]
    assert result["margin_loss"] == pytest.approx(1.8 / 3)


@pytest.mark.parametrize("objective", ["cross_entropy", "margin_only"])
def test_supervised_controls_ignore_rewards_next_states_and_target_network(objective):
    env, agent, _, batch = make_fixture()
    try:
        altered = deepcopy(agent)
        changed_batch = replace(
            batch, rewards=torch.full_like(batch.rewards, float("nan")),
            next_observations=torch.full_like(batch.next_observations, float("nan")),
            terminated=~batch.terminated,
        )
        with torch.no_grad():
            for parameter in altered.learner.target.parameters():
                parameter.fill_(float("nan"))
        original_hash = diagnostic.parameter_hash(agent)
        frozen_target = deepcopy(agent.learner.target.state_dict())
        for _ in range(3):
            expected = diagnostic.fitting_update(agent, batch, objective, 0.8)
            actual = diagnostic.fitting_update(altered, changed_batch, objective, 0.8)
            assert actual == expected
        assert diagnostic.parameter_hash(agent) != original_hash
        assert diagnostic.parameter_hash(agent) == diagnostic.parameter_hash(altered)
        assert agent.learner.update_count == 3
        assert not actual["target_synchronized"]
        for name, value in agent.learner.target.state_dict().items():
            torch.testing.assert_close(value, frozen_target[name], rtol=0, atol=0)
    finally:
        env.close()


def test_td_control_reacts_to_rewards_and_rejects_unknown_objective_without_mutation():
    env, agent, _, batch = make_fixture()
    try:
        altered = deepcopy(agent)
        unchanged_hash = diagnostic.parameter_hash(agent)
        with pytest.raises(ValueError, match="Unknown fitting objective"):
            diagnostic.fitting_update(agent, batch, "unregistered", 0.8)
        assert diagnostic.parameter_hash(agent) == unchanged_hash
        assert agent.learner.update_count == 0
        diagnostic.fitting_update(
            agent, replace(batch, rewards=torch.full_like(batch.rewards, 100.0)), "td_margin", 0.8,
        )
        diagnostic.fitting_update(
            altered, replace(batch, rewards=torch.full_like(batch.rewards, -100.0)), "td_margin", 0.8,
        )
        assert diagnostic.parameter_hash(agent) != diagnostic.parameter_hash(altered)
    finally:
        env.close()


def test_greedy_rollout_reports_first_divergence_without_reference_fallback(monkeypatch):
    env, agent, _, batch = make_fixture()
    try:
        with torch.no_grad():
            for parameter in agent.learner.online.parameters():
                parameter.zero_()
            agent.learner.online[-1].bias[0] = 1.0

        def forbidden_reference(*args, **kwargs):
            raise AssertionError("A greedy diagnostic rollout must not call the reference policy.")

        monkeypatch.setattr(diagnostic, "seed_replay_from_reference", forbidden_reference)
        before = diagnostic.parameter_hash(agent)
        result = diagnostic.fixed_scene_rollout(agent, env, 3, batch)
        assert result["actions"] == [0] * result["steps"]
        assert result["first_reference_action_divergence"] == {
            "step": 1, "predicted_action": 0, "demonstrated_action": 1,
        }
        assert len(result["robot_path"]) == result["steps"] + 1
        assert diagnostic.parameter_hash(agent) == before
        assert agent.learner.update_count == 0
        altered = replace(batch, observations=batch.observations + 1)
        with pytest.raises(ValueError, match="does not match"):
            diagnostic.fixed_scene_rollout(agent, env, 3, altered)
    finally:
        env.close()


def test_failed_collection_retains_failure_record_and_refuses_overwrite(tmp_path, monkeypatch):
    def failed_collection(*args, **kwargs):
        record = {"successful_episodes": 0, "accepted_transitions": 0}
        kwargs["progress"](record)
        return record

    monkeypatch.setattr(diagnostic, "seed_replay_from_reference", failed_collection)
    output = tmp_path / "failed_diagnostic"
    config = DQNConfig(hidden_sizes=(8, 8), batch_size=4, replay_capacity=32)
    with pytest.raises(RuntimeError, match="successful demonstration"):
        diagnostic.run({"max_steps": 4}, config, output, updates=1, checkpoints=(0, 1))
    record = json.loads((output / "run_status.json").read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["error_type"] == "RuntimeError"
    original_evidence = (output / "run_status.json").read_bytes()
    with pytest.raises(FileExistsError):
        diagnostic.run({"max_steps": 4}, config, output, updates=1, checkpoints=(0, 1))
    assert (output / "run_status.json").read_bytes() == original_evidence
