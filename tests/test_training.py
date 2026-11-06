"""Regression checks for replay, preprocessing, training, and policy evaluation."""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from me5418_harvesting.agents.dqn import DQNConfig
from me5418_harvesting.agents.dqn_agent import DQNAgent, ObservationEncoder
from me5418_harvesting.agents.replay_buffer import ReplayBuffer
from me5418_harvesting.agents.training import TrainingConfig, evaluate, read_mapping, train
from me5418_harvesting.envs.harvesting_env import HarvestingEnv

ROOT = Path(__file__).resolve().parents[1]


def test_replay_copies_wraps_and_keeps_end_flags():
    replay = ReplayBuffer(2)
    observation = np.zeros(3, dtype=np.float32)
    replay.add(observation, 0, 1, observation + 1, False, True)
    observation[:] = 99
    replay.add(observation, 1, 2, observation + 1, True, False)
    batch = replay.sample(2, np.random.default_rng(0))
    external = batch.actions == 0
    assert torch.all(batch.observations[external] == 0)
    assert batch.truncated[external].all()
    assert not batch.terminated[external].any()
    replay.add(observation, 2, 3, observation + 1, False, False)
    assert len(replay) == 2
    assert set(replay.sample(2, np.random.default_rng(0)).actions.tolist()) == {1, 2}
    with pytest.raises(ValueError):
        replay.sample(3, np.random.default_rng(0))


@pytest.mark.parametrize("name", [
    "environment.yaml", "environment_complex.yaml", "environment_easy.yaml",
])
def test_encoder_preserves_masks_and_budget_and_scales_stiffness(name):
    env = HarvestingEnv(read_mapping(ROOT / "configs" / name))
    try:
        raw, _ = env.reset(seed=0)
        before = raw.copy()
        encoded = ObservationEncoder(env)(raw)
        np.testing.assert_array_equal(raw, before)
        assert encoded[-1] == raw[-1] == 1
        assert np.isfinite(encoded).all()
        width = 4 * env.nodes_per_branch + 4
        for start in range(4, len(raw) - 1, width):
            assert encoded[start + width - 1] == raw[start + width - 1]
            assert encoded[start + width - 3] == raw[start + width - 3]
            assert 0 <= encoded[start + width - 4] <= 1.01
            if raw[start + width - 1] == 0:
                assert np.all(encoded[start:start + width] == 0)
    finally:
        env.close()


def test_training_updates_across_episodes_and_reload_is_identical(tmp_path):
    torch.set_num_threads(1)
    config = TrainingConfig(
        episodes=3, learning_starts=4, train_frequency=1,
        evaluation_interval=2, evaluation_episodes=1, epsilon_decay_steps=8,
    )
    network = DQNConfig(hidden_sizes=(16, 16), batch_size=4, replay_capacity=20,
                        target_update_interval=2)
    environment = {"max_steps": 4}
    torch.manual_seed(config.seed)
    original_env = HarvestingEnv(environment)
    initial = DQNAgent(original_env, network, config.seed)
    summary = train(environment, network, config, tmp_path / "run")
    assert summary["updates"] == 9
    assert summary["replay_size"] == summary["environment_steps"] == 12
    records = [json.loads(line) for line in (tmp_path / "run/metrics.jsonl").read_text().splitlines()]
    assert [row["replay_size"] for row in records] == [4, 8, 12]
    assert all(row["budget_exhausted"] for row in records)
    assert records[-1]["validation"]["seed"] >= config.episodes
    assert (tmp_path / "run/best.pt").is_file()
    agent, env, metadata = DQNAgent.load(tmp_path / "run/latest.pt")
    try:
        assert metadata["updates"] == 9
        assert any(not torch.equal(a, b) for a, b in zip(
            initial.learner.online.parameters(), agent.learner.online.parameters(),
        ))
        before = {key: value.clone() for key, value in agent.learner.online.state_dict().items()}
        rng_before = json.dumps(agent.rng.bit_generator.state)
        result = evaluate(agent, env, episodes=2, seed=100)
        assert result == evaluate(agent, env, episodes=2, seed=100)
        assert json.dumps(agent.rng.bit_generator.state) == rng_before
        assert all(torch.equal(value, before[key])
                   for key, value in agent.learner.online.state_dict().items())
        agent.save(tmp_path / "copy.pt", environment, metadata)
        copied, copied_env, _ = DQNAgent.load(tmp_path / "copy.pt")
        try:
            observation, _ = env.reset(seed=9)
            assert agent.act(observation) == copied.act(observation)
            with torch.no_grad():
                encoded = torch.from_numpy(agent.encoder(observation))
                torch.testing.assert_close(agent.learner.online(encoded),
                                           copied.learner.online(encoded), rtol=0, atol=0)
        finally:
            copied_env.close()
        with pytest.raises(FileExistsError):
            train(environment, network, config, tmp_path / "run")
    finally:
        env.close()
        original_env.close()


def test_epsilon_and_invalid_training_settings():
    config = TrainingConfig(epsilon_decay_steps=10)
    assert config.epsilon(0) == 1
    assert config.epsilon(5) == pytest.approx(0.525)
    assert config.epsilon(100) == pytest.approx(0.05)
    for values in ({"episodes": 0}, {"train_frequency": 0}, {"seed": -1},
                   {"epsilon_end": 1.1}, {"epsilon_decay_steps": 0}):
        with pytest.raises(ValueError):
            TrainingConfig(**values)
