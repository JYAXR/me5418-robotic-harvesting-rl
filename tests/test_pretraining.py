"""Check isolated demonstration updates, phase accounting, and retained failure evidence."""

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from me5418_harvesting.agents import training
from me5418_harvesting.agents.dqn import DQNConfig
from me5418_harvesting.agents.dqn_agent import DQNAgent
from me5418_harvesting.agents.pretraining import pretrain_on_replay
from me5418_harvesting.agents.replay_buffer import ReplayBuffer
from me5418_harvesting.agents.training import TrainingConfig, read_mapping, train
from me5418_harvesting.envs.harvesting_env import HarvestingEnv

ROOT = Path(__file__).resolve().parents[1]


def create_agent_and_replay():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    env = HarvestingEnv({"max_steps": 4})
    network = DQNConfig(hidden_sizes=(8, 8), batch_size=4, replay_capacity=32,
                        target_update_interval=2)
    agent = DQNAgent(env, network, seed=7)
    replay = ReplayBuffer(network.replay_capacity)
    observation, _ = env.reset(seed=3)
    for index in range(8):
        action = index % 4
        next_observation, reward, terminated, truncated, _ = env.step(action)
        replay.add(agent.encoder(observation), action, reward,
                   agent.encoder(next_observation), terminated, truncated)
        observation = next_observation
        if terminated or truncated:
            observation, _ = env.reset(seed=4)
    return env, agent, replay


@pytest.mark.parametrize("updates", [-1, 1.5, True])
def test_pretraining_update_count_requires_nonnegative_integer(updates):
    with pytest.raises(ValueError, match="nonnegative integer"):
        TrainingConfig(demonstration_episodes=1, demonstration_pretrain_updates=updates)


def test_positive_pretraining_requires_demonstration_collection():
    with pytest.raises(ValueError, match="demonstration"):
        TrainingConfig(demonstration_pretrain_updates=1)
    assert TrainingConfig().demonstration_pretrain_updates == 0


def test_pretraining_changes_weights_without_consuming_exploration_or_mutating_replay(tmp_path):
    env, agent, replay = create_agent_and_replay()
    original_replay = deepcopy(replay.records)
    replay_position = replay.position
    exploration_rng = json.dumps(agent.rng.bit_generator.state)
    torch_rng = torch.get_rng_state().clone()
    weights = {name: value.clone() for name, value in agent.learner.online.state_dict().items()}
    observation = env._observation().copy()
    environment_steps = env.step_count
    try:
        result = pretrain_on_replay(agent, replay, updates=6, seed=21, output=tmp_path)
        assert result["status"] == "completed"
        assert result["requested_updates"] == result["completed_updates"] == 6
        assert result["input_transitions"] == len(replay) == 8
        assert result["batch_size"] == 4
        assert result["environment_steps"] == 0
        assert result["target_synchronizations"] == 3
        assert agent.learner.update_count == 6
        assert any(not torch.equal(value, weights[name])
                   for name, value in agent.learner.online.state_dict().items())
        for online, target in zip(agent.learner.online.parameters(), agent.learner.target.parameters()):
            torch.testing.assert_close(online, target, rtol=0, atol=0)
        assert json.dumps(agent.rng.bit_generator.state) == exploration_rng
        assert torch.equal(torch.get_rng_state(), torch_rng)
        assert replay.position == replay_position
        for actual, expected in zip(replay.records, original_replay):
            for value, before in zip(actual, expected):
                np.testing.assert_array_equal(value, before)
        np.testing.assert_array_equal(env._observation(), observation)
        assert env.step_count == environment_steps
        records = [json.loads(line) for line in
                   (tmp_path / "pretraining_metrics.jsonl").read_text().splitlines()]
        assert [row["pretraining_update"] for row in records] == list(range(1, 7))
        assert [row["total_updates"] for row in records] == list(range(1, 7))
        assert result["first_loss"] == records[0]["loss_before"]
        assert result["last_loss"] == records[-1]["loss_before"]
        assert json.loads((tmp_path / "pretraining.json").read_text()) == result
    finally:
        env.close()


def test_pretraining_is_reproducible_from_the_same_weights_replay_and_seed(tmp_path):
    first_env, first, replay = create_agent_and_replay()
    second_env, second, second_replay = create_agent_and_replay()
    first_output, second_output = tmp_path / "first", tmp_path / "second"
    first_output.mkdir()
    second_output.mkdir()
    try:
        left = pretrain_on_replay(first, replay, updates=5, seed=13, output=first_output)
        right = pretrain_on_replay(second, second_replay, updates=5, seed=13, output=second_output)
        assert left["first_loss"] == right["first_loss"]
        assert left["last_loss"] == right["last_loss"]
        for a, b in zip(first.learner.online.parameters(), second.learner.online.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert (first_output / "pretraining_metrics.jsonl").read_bytes() == (
            second_output / "pretraining_metrics.jsonl"
        ).read_bytes()
    finally:
        first_env.close()
        second_env.close()


@pytest.mark.parametrize("error, status", [
    (RuntimeError("Injected pretraining failure."), "failed"),
    (KeyboardInterrupt("Injected pretraining interruption."), "interrupted"),
])
def test_pretraining_preserves_completed_updates_on_failure(tmp_path, monkeypatch, error, status):
    env, agent, replay = create_agent_and_replay()
    original_update = agent.learner.update

    def fail_on_third_update(batch):
        if agent.learner.update_count == 2:
            raise error
        return original_update(batch)

    monkeypatch.setattr(agent.learner, "update", fail_on_third_update)
    try:
        with pytest.raises(type(error)) as raised:
            pretrain_on_replay(agent, replay, updates=6, seed=21, output=tmp_path)
        assert raised.value is error
        partial = json.loads((tmp_path / "pretraining.json").read_text())
        assert partial["status"] == status
        assert partial["requested_updates"] == 6
        assert partial["completed_updates"] == agent.learner.update_count == 2
        assert partial["environment_steps"] == 0
        records = [json.loads(line) for line in
                   (tmp_path / "pretraining_metrics.jsonl").read_text().splitlines()]
        assert len(records) == 2
        assert partial["last_loss"] == records[-1]["loss_before"]
    finally:
        env.close()


def test_pretraining_rejects_insufficient_replay_without_updating_weights(tmp_path):
    env, agent, _ = create_agent_and_replay()
    replay = ReplayBuffer(8)
    try:
        with pytest.raises(ValueError, match="[Bb]atch|replay|[Rr]eplay"):
            pretrain_on_replay(agent, replay, updates=2, seed=21, output=tmp_path)
        assert agent.learner.update_count == 0
    finally:
        env.close()


def test_pretraining_and_online_training_have_separate_counts_and_checkpoints(tmp_path, monkeypatch):
    environment = read_mapping(ROOT / "configs/environment_easy.yaml")
    network = DQNConfig(hidden_sizes=(8, 8), batch_size=4, replay_capacity=256,
                        target_update_interval=2)
    settings = TrainingConfig(
        episodes=1, learning_starts=4, train_frequency=2,
        epsilon_start=1.0, epsilon_end=1.0, evaluation_episodes=1,
        demonstration_episodes=1, demonstration_seed=10000,
        validation_seed=100000, evaluate_before_training=False,
    )
    sample_states = []
    original_sample = ReplayBuffer.sample

    def record_sampler_state(self, size, rng):
        sample_states.append(json.dumps(rng.bit_generator.state))
        return original_sample(self, size, rng)

    monkeypatch.setattr(ReplayBuffer, "sample", record_sampler_state)
    baseline = train(environment, network, settings, tmp_path / "no_pretraining")
    baseline_sample_states = sample_states.copy()
    sample_states.clear()
    output = tmp_path / "pretrained"
    result = train(environment, network, replace(settings, demonstration_pretrain_updates=6), output)
    # Fully random exploration makes the online action sequence independent of weights.
    assert result["environment_steps"] == baseline["environment_steps"]
    assert sample_states[6:] == baseline_sample_states
    assert result["online_updates"] == baseline["updates"]
    assert result["pretraining_updates"] == 6
    assert result["updates"] == result["online_updates"] + 6
    assert result["replay_size"] == baseline["replay_size"]
    assert result["total_collection_steps"] == baseline["total_collection_steps"]
    assert (output / "initial_validation.json").is_file()
    pretrained_validation = json.loads((output / "pretraining_validation.json").read_text())
    assert pretrained_validation == result["pretraining_validation"]
    assert pretrained_validation["seed"] == result["initial_validation"]["seed"] == 100000
    records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["epsilon"] == 1.0
    assert records[0]["pretraining_updates"] == 6
    assert records[0]["online_updates"] == baseline["updates"]
    assert records[0]["episode_updates"] == baseline["updates"]
    _, pretrained_env, pretrained = DQNAgent.load(output / "pretrained.pt")
    _, best_env, best = DQNAgent.load(output / "best.pt")
    try:
        assert pretrained["phase"] == "pretraining"
        assert pretrained["episode"] == pretrained["environment_steps"] == 0
        assert pretrained["updates"] == pretrained["pretraining_updates"] == 6
        assert pretrained["online_updates"] == 0
        assert pretrained["validation"] == pretrained_validation
        assert best["episode"] == 1
        assert best["updates"] == result["updates"]
    finally:
        pretrained_env.close()
        best_env.close()


def test_disabled_pretraining_keeps_default_training_identical(tmp_path, monkeypatch):
    def unexpected_pretraining(*args, **kwargs):
        raise AssertionError("Disabled pretraining must not run or consume random samples.")

    monkeypatch.setattr(training, "pretrain_on_replay", unexpected_pretraining)
    network = DQNConfig(hidden_sizes=(8, 8), batch_size=2, replay_capacity=16,
                        target_update_interval=2)
    settings = TrainingConfig(episodes=1, learning_starts=2, train_frequency=1,
                              evaluation_episodes=1)
    first_dir, second_dir = tmp_path / "default", tmp_path / "explicit_zero"
    first = train({"max_steps": 3}, network, settings, first_dir)
    second = train({"max_steps": 3}, network,
                   replace(settings, demonstration_pretrain_updates=0), second_dir)
    assert first["updates"] == first["online_updates"] == second["updates"] == 2
    assert first["pretraining_updates"] == second["pretraining_updates"] == 0
    assert first["validation"] == second["validation"]
    assert first["last_loss"] == second["last_loss"]
    for path in (first_dir, second_dir):
        assert not list(path.glob("pretrain*"))
        assert not (path / "initial_validation.json").exists()
    first_agent, first_env, _ = DQNAgent.load(first_dir / "latest.pt")
    second_agent, second_env, _ = DQNAgent.load(second_dir / "latest.pt")
    try:
        for left, right in zip(first_agent.learner.online.parameters(),
                               second_agent.learner.online.parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    finally:
        first_env.close()
        second_env.close()
