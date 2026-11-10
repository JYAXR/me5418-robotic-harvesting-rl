"""Check demonstration margins, unchanged TD defaults, and pretraining-only supervision."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch import nn
from torch.nn import functional as F

from me5418_harvesting.agents import dqn_learning
from me5418_harvesting.agents.dqn import DQNConfig
from me5418_harvesting.agents.dqn_agent import DQNAgent
from me5418_harvesting.agents.dqn_learning import (
    DQNUpdater,
    TransitionBatch,
    demonstration_margin_loss,
)
from me5418_harvesting.agents.pretraining import pretrain_on_replay
from me5418_harvesting.agents.replay_buffer import ReplayBuffer
from me5418_harvesting.agents.training import TrainingConfig, read_mapping, train
from me5418_harvesting.envs.harvesting_env import HarvestingEnv

ROOT = Path(__file__).resolve().parents[1]


def test_margin_loss_has_expected_value_and_pushes_toward_the_expert_action():
    values = torch.tensor([[1.0, 2.0, 0.0], [3.0, 0.0, 1.0]], requires_grad=True)
    actions = torch.tensor([0, 0])
    loss = demonstration_margin_loss(values, actions, margin=0.8)
    assert loss.item() == pytest.approx(0.9)
    loss.backward()
    torch.testing.assert_close(values.grad, torch.tensor([[-0.5, 0.5, 0.0], [0.0, 0.0, 0.0]]))


def test_satisfied_margin_has_zero_loss_and_no_gradient():
    values = torch.tensor([[1.0, 0.0], [0.2, 1.5]], requires_grad=True)
    loss = demonstration_margin_loss(values, torch.tensor([0, 1]), margin=0.8)
    assert loss.item() == 0
    loss.backward()
    torch.testing.assert_close(values.grad, torch.zeros_like(values))


@pytest.mark.parametrize("margin", [0.0, -0.1, float("nan"), float("inf"), True])
def test_margin_loss_rejects_invalid_margin(margin):
    with pytest.raises((TypeError, ValueError)):
        demonstration_margin_loss(torch.zeros(1, 2), torch.tensor([0]), margin)


@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf"), True, "1"])
def test_training_rejects_invalid_supervision_weights(weight):
    with pytest.raises((TypeError, ValueError)):
        TrainingConfig(demonstration_episodes=1, demonstration_pretrain_updates=1,
                       demonstration_pretrain_weight=weight)


@pytest.mark.parametrize("margin", [0.0, -0.1, float("nan"), float("inf"), True])
def test_training_rejects_invalid_supervision_margins(margin):
    with pytest.raises((TypeError, ValueError)):
        TrainingConfig(demonstration_episodes=1, demonstration_pretrain_updates=1,
                       demonstration_pretrain_margin=margin)


def test_supervision_requires_an_enabled_pretraining_stage():
    with pytest.raises(ValueError, match="[Pp]retrain"):
        TrainingConfig(demonstration_episodes=1, demonstration_pretrain_weight=1.0)
    assert TrainingConfig().demonstration_pretrain_weight == 0.0


def transition_batch():
    return TransitionBatch(
        observations=torch.tensor([[0.0, 1.0, 0.5], [1.0, 0.0, 0.1],
                                   [0.3, 0.2, 0.4], [0.8, 0.1, 0.9]]),
        actions=torch.tensor([0, 1, 1, 0]),
        rewards=torch.tensor([0.3, -1.0, 2.0, 0.8]),
        next_observations=torch.tensor([[0.1, 0.9, 0.5], [0.9, 0.1, 0.2],
                                        [0.4, 0.2, 0.5], [0.7, 0.2, 0.8]]),
        terminated=torch.tensor([False, True, False, True]),
        truncated=torch.tensor([False, False, True, False]),
    )


@pytest.mark.parametrize("explicit_zero", [False, True])
def test_zero_supervision_is_bitwise_identical_to_original_td_updates(monkeypatch, explicit_zero):
    torch.set_num_threads(1)
    torch.manual_seed(14)
    config = DQNConfig(hidden_sizes=(6, 5), batch_size=4, learning_rate=0.001,
                       target_update_interval=2)
    learner = DQNUpdater(3, 2, config)
    reference_online = deepcopy(learner.online)
    reference_target = deepcopy(learner.target)
    reference_optimizer = torch.optim.Adam(reference_online.parameters(), lr=config.learning_rate)
    batch = transition_batch()

    def unexpected_margin(*args, **kwargs):
        raise AssertionError("Disabled supervision must not evaluate its loss.")

    monkeypatch.setattr(dqn_learning, "demonstration_margin_loss", unexpected_margin)
    for step in range(1, 4):
        reference_optimizer.zero_grad(set_to_none=True)
        chosen = reference_online(batch.observations).gather(1, batch.actions[:, None]).squeeze(1)
        with torch.no_grad():
            next_values = reference_target(batch.next_observations).max(dim=1).values
            targets = batch.rewards + config.gamma * torch.where(
                batch.terminated, torch.zeros_like(next_values), next_values,
            )
        original_loss = F.smooth_l1_loss(chosen, targets)
        original_loss.backward()
        nn.utils.clip_grad_norm_(reference_online.parameters(), config.gradient_clip_norm,
                                 error_if_nonfinite=True)
        reference_optimizer.step()
        with torch.no_grad():
            after = reference_online(batch.observations).gather(
                1, batch.actions[:, None],
            ).squeeze(1)
            original_loss_after = F.smooth_l1_loss(after, targets)
        if step % config.target_update_interval == 0:
            reference_target.load_state_dict(reference_online.state_dict())
        stats = learner.update(batch, **({"demonstration_weight": 0.0} if explicit_zero else {}))
        assert stats.loss_before == original_loss.item()
        assert stats.loss_after == original_loss_after.item()
        assert stats.demonstration_loss_before == stats.demonstration_loss_after == 0
        assert stats.demonstration_action_accuracy is None
        assert stats.target_synchronized == (step % 2 == 0)
        for actual, expected in zip(learner.online.parameters(), reference_online.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual, expected in zip(learner.target.parameters(), reference_target.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual_state = learner.optimizer.state_dict()["state"]
        expected_state = reference_optimizer.state_dict()["state"]
        assert actual_state.keys() == expected_state.keys()
        for index in actual_state:
            for key in actual_state[index]:
                torch.testing.assert_close(actual_state[index][key], expected_state[index][key],
                                           rtol=0, atol=0)


def test_positive_supervision_improves_expert_gap_and_keeps_fixed_td_targets():
    torch.set_num_threads(1)
    config = DQNConfig(hidden_sizes=(4, 4), learning_rate=0.01,
                       target_update_interval=2, batch_size=2)
    learner = DQNUpdater(2, 2, config)
    for network in (learner.online, learner.target):
        for parameter in network.parameters():
            nn.init.zeros_(parameter)
    batch = TransitionBatch(torch.zeros(2, 2), torch.ones(2, dtype=torch.long),
                            torch.zeros(2), torch.zeros(2, 2),
                            torch.tensor([False, True]), torch.tensor([True, False]))
    target_before = deepcopy(learner.target.state_dict())
    stats = learner.update(batch, demonstration_weight=2.0, demonstration_margin=0.8)
    assert stats.td_loss_before == 0
    assert stats.demonstration_loss_before == pytest.approx(0.8)
    assert stats.loss_before == pytest.approx(1.6)
    assert stats.demonstration_action_accuracy == 0
    with torch.no_grad():
        values_after = learner.online(batch.observations)
        assert torch.all(values_after[:, 1] > values_after[:, 0])
        expected_td_after = F.smooth_l1_loss(values_after[:, 1], torch.zeros(2)).item()
        expected_margin_after = (values_after[:, 0] + 0.8 - values_after[:, 1]).mean().item()
    assert stats.td_loss_after == pytest.approx(expected_td_after)
    assert stats.demonstration_loss_after == pytest.approx(expected_margin_after)
    assert stats.loss_after == pytest.approx(expected_td_after + 2 * expected_margin_after)
    assert not stats.target_synchronized
    for name, value in learner.target.state_dict().items():
        torch.testing.assert_close(value, target_before[name], rtol=0, atol=0)
    assert all(not parameter.requires_grad and parameter.grad is None
               for parameter in learner.target.parameters())
    assert learner.update(batch, demonstration_weight=2.0).target_synchronized


def test_supervised_pretraining_logs_components_without_consuming_exploration(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(19)
    env = HarvestingEnv({"max_steps": 4})
    network = DQNConfig(hidden_sizes=(8, 8), batch_size=4, replay_capacity=32,
                        target_update_interval=2)
    agent = DQNAgent(env, network, seed=7)
    replay = ReplayBuffer(32)
    observation, _ = env.reset(seed=3)
    for index in range(4):
        next_observation, reward, terminated, truncated, _ = env.step(index)
        replay.add(agent.encoder(observation), index, reward, agent.encoder(next_observation),
                   terminated, truncated)
        observation = next_observation
    replay_before = deepcopy(replay.records)
    exploration_before = json.dumps(agent.rng.bit_generator.state)
    torch_rng_before = torch.get_rng_state().clone()
    try:
        result = pretrain_on_replay(agent, replay, updates=3, seed=21, output=tmp_path,
                                   demonstration_weight=1.5, demonstration_margin=0.7)
        assert result["completed_updates"] == 3 and result["environment_steps"] == 0
        assert result["demonstration_weight"] == 1.5
        assert result["demonstration_margin"] == 0.7
        assert json.dumps(agent.rng.bit_generator.state) == exploration_before
        assert torch.equal(torch.get_rng_state(), torch_rng_before)
        for actual, expected in zip(replay.records, replay_before):
            for value, before in zip(actual, expected):
                np.testing.assert_array_equal(value, before)
        rows = [json.loads(line) for line in
                (tmp_path / "pretraining_metrics.jsonl").read_text().splitlines()]
        assert len(rows) == 3
        for row in rows:
            assert row["loss_before"] == pytest.approx(
                row["td_loss_before"] + 1.5 * row["demonstration_loss_before"], rel=1e-6,
            )
            assert 0 <= row["demonstration_action_accuracy"] <= 1
    finally:
        env.close()


def test_training_supervises_only_demonstration_pretraining_updates(tmp_path, monkeypatch):
    calls = []
    original_update = DQNUpdater.update

    def record_supervision(learner, batch, **kwargs):
        calls.append((kwargs.get("demonstration_weight", 0.0),
                      kwargs.get("demonstration_margin", 0.8)))
        return original_update(learner, batch, **kwargs)

    monkeypatch.setattr(DQNUpdater, "update", record_supervision)
    settings = TrainingConfig(
        episodes=1, learning_starts=4, train_frequency=4, evaluation_episodes=1,
        epsilon_start=1.0, epsilon_end=1.0, validation_seed=100000,
        demonstration_episodes=1, demonstration_seed=10000,
        demonstration_pretrain_updates=3, demonstration_pretrain_weight=1.75,
        demonstration_pretrain_margin=0.6,
    )
    network = DQNConfig(hidden_sizes=(8, 8), batch_size=4, replay_capacity=256,
                        target_update_interval=2)
    result = train(read_mapping(ROOT / "configs/environment_easy.yaml"), network,
                   settings, tmp_path / "run")
    assert calls[:3] == [(1.75, 0.6)] * 3
    assert len(calls) > 3
    assert all(weight == 0.0 for weight, _ in calls[3:])
    assert result["pretraining_updates"] == 3
    assert len(calls[3:]) == result["online_updates"]
    assert result["updates"] == len(calls)
    assert result["pretraining"]["demonstration_weight"] == 1.75
