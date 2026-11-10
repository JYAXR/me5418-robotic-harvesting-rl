"""Check reference data integrity, split isolation, and optional replay initialization."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from me5418_harvesting.agents import training
from me5418_harvesting.agents.demonstrations import seed_replay_from_reference
from me5418_harvesting.agents.dqn import DQNConfig
from me5418_harvesting.agents.dqn_agent import DQNAgent, ObservationEncoder
from me5418_harvesting.agents.replay_buffer import ReplayBuffer
from me5418_harvesting.agents.training import TrainingConfig, read_mapping, train
from me5418_harvesting.envs.harvesting_env import HarvestingEnv
from me5418_harvesting.planners.collision_free import collision_free_actions

ROOT = Path(__file__).resolve().parents[1]


def easy_config():
    return read_mapping(ROOT / "configs/environment_easy.yaml")


def test_successful_demonstration_matches_actual_steps_and_preserves_agent():
    torch.set_num_threads(1)
    config = easy_config()
    original_config = deepcopy(config)
    env = HarvestingEnv(config)
    replay = ReplayBuffer(100)
    agent = DQNAgent(env, DQNConfig(hidden_sizes=(8, 8)), seed=41)
    weights = {key: value.clone() for key, value in agent.learner.online.state_dict().items()}
    exploration_rng = json.dumps(agent.rng.bit_generator.state)
    torch_rng = torch.get_rng_state().clone()
    try:
        result = seed_replay_from_reference(config, agent.encoder, replay, episodes=1, seed=3)
        assert result["successful_episodes"] == 1
        assert result["accepted_transitions"] == result["collection_environment_steps"] == len(replay)
        assert result["optimizer_updates"] == agent.learner.update_count == 0
        assert result["records"][0]["reason"] == "success"
        assert config == original_config
        assert json.dumps(agent.rng.bit_generator.state) == exploration_rng
        assert torch.equal(torch.get_rng_state(), torch_rng)
        for key, value in agent.learner.online.state_dict().items():
            assert torch.equal(value, weights[key])

        observation, _ = env.reset(seed=3)
        planned = collision_free_actions(env, include_current_shape=True)
        total_reward = 0.0
        for index, stored in enumerate(replay.records):
            encoded, action, reward, encoded_next, terminated, truncated = stored
            assert action == planned[index]
            np.testing.assert_array_equal(encoded, agent.encoder(observation))
            next_observation, actual_reward, actual_terminated, actual_truncated, info = env.step(action)
            np.testing.assert_array_equal(encoded_next, agent.encoder(next_observation))
            assert reward == actual_reward
            assert (terminated, truncated) == (actual_terminated, actual_truncated)
            assert terminated == (index == len(replay) - 1)
            assert not truncated
            observation = next_observation
            total_reward += reward
        assert info["success"] and not info["damage"]
        assert replay.records[-1][3][-1] < 1.0
        assert result["records"][0]["return"] == total_reward
    finally:
        env.close()


def test_unsuccessful_budget_limited_routes_are_recorded_but_not_inserted():
    config = {**easy_config(), "max_steps": 1}
    env = HarvestingEnv(config)
    replay = ReplayBuffer(4)
    try:
        result = seed_replay_from_reference(config, ObservationEncoder(env), replay,
                                            episodes=2, seed=3)
        assert len(replay) == result["accepted_transitions"] == result["successful_episodes"] == 0
        assert result["requested_episodes"] == result["collection_environment_steps"] == 2
        assert [row["seed"] for row in result["records"]] == [3, 4]
        for row in result["records"]:
            assert row["steps"] == 1 and row["budget_exhausted"]
            assert row["reason"] == "budget_exhausted"
            assert not row["success"] and row["accepted_transitions"] == 0
    finally:
        env.close()


def test_insufficient_capacity_is_rejected_without_changing_existing_replay():
    config = easy_config()
    env = HarvestingEnv(config)
    replay = ReplayBuffer(config["max_steps"])
    replay.add(np.zeros(env.observation_size), 2, 1.5,
               np.ones(env.observation_size), False, False)
    original = deepcopy(replay.records)
    original_position = replay.position
    try:
        with pytest.raises(ValueError, match="capacity"):
            seed_replay_from_reference(config, ObservationEncoder(env), replay, episodes=1, seed=3)
        assert len(replay) == 1 and replay.position == original_position
        for current, previous in zip(replay.records[0], original[0]):
            np.testing.assert_array_equal(current, previous)
    finally:
        env.close()


def test_interrupted_collection_reports_partial_steps_without_inserting_partial_route(monkeypatch):
    config = easy_config()
    env = HarvestingEnv(config)
    replay = ReplayBuffer(200)
    snapshots = []
    original_step = HarvestingEnv.step

    def interrupt_after_first_step(self, action):
        if self.step_count == 1:
            raise KeyboardInterrupt("Injected demonstration interruption.")
        return original_step(self, action)

    monkeypatch.setattr(HarvestingEnv, "step", interrupt_after_first_step)
    try:
        with pytest.raises(KeyboardInterrupt, match="demonstration interruption"):
            seed_replay_from_reference(
                config, ObservationEncoder(env), replay, episodes=2, seed=3,
                progress=lambda result: snapshots.append(deepcopy(result)),
            )
        assert snapshots[0]["records"][0]["reason"] == "collecting"
        partial = snapshots[-1]
        assert partial["requested_episodes"] == 2
        assert partial["collection_environment_steps"] == 1
        assert partial["accepted_transitions"] == len(replay) == 0
        assert partial["successful_episodes"] == 0
        assert len(partial["records"]) == 1
        assert partial["records"][0]["reason"] == "collection_interrupted"
        assert partial["records"][0]["error_type"] == "KeyboardInterrupt"
    finally:
        env.close()


@pytest.mark.parametrize("field", ["demonstration_episodes", "demonstration_seed"])
@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_demonstration_settings_require_nonnegative_integers(field, value):
    with pytest.raises(ValueError, match="nonnegative integer"):
        TrainingConfig(**{field: value})


@pytest.mark.parametrize("seed", [9, 10, 19])
def test_demonstration_online_seed_overlap_is_rejected(seed):
    with pytest.raises(ValueError, match="online training seed ranges"):
        TrainingConfig(seed=10, episodes=10, evaluation_episodes=3,
                       demonstration_episodes=2, demonstration_seed=seed, validation_seed=100)


@pytest.mark.parametrize("seed", [99, 100, 102])
def test_demonstration_explicit_validation_overlap_is_rejected(seed):
    with pytest.raises(ValueError, match="validation seed ranges"):
        TrainingConfig(seed=10, episodes=10, evaluation_episodes=3,
                       demonstration_episodes=2, demonstration_seed=seed, validation_seed=100)


@pytest.mark.parametrize("seed", [20, 22])
def test_demonstration_derived_validation_overlap_is_rejected(seed):
    with pytest.raises(ValueError, match="validation seed ranges"):
        TrainingConfig(seed=10, episodes=10, evaluation_episodes=3,
                       demonstration_episodes=2, demonstration_seed=seed)


@pytest.mark.parametrize("seed", [8, 23, 10000])
def test_demonstration_disjoint_seed_ranges_are_accepted(seed):
    TrainingConfig(seed=10, episodes=10, evaluation_episodes=3,
                   demonstration_episodes=2, demonstration_seed=seed)


def test_demonstrations_are_disabled_by_default(tmp_path, monkeypatch):
    def unexpected_collection(*args, **kwargs):
        raise AssertionError("Disabled demonstrations must not call the reference collector.")

    monkeypatch.setattr(training, "seed_replay_from_reference", unexpected_collection)
    config = TrainingConfig(episodes=1, learning_starts=2, train_frequency=1,
                            evaluation_episodes=1, demonstration_seed=0)
    assert config.demonstration_episodes == 0
    network = DQNConfig(hidden_sizes=(8, 8), batch_size=2, replay_capacity=16)
    output = tmp_path / "disabled"
    result = train({"max_steps": 3}, network, config, output)
    assert result["environment_steps"] == result["replay_size"] == 3
    assert result["updates"] == 2
    assert "demonstrations" not in result
    assert "total_collection_steps" not in result
    assert not (output / "demonstrations.json").exists()


def test_training_with_demonstrations_keeps_online_and_collection_accounting_separate(tmp_path):
    config = TrainingConfig(episodes=1, learning_starts=4, train_frequency=2,
                            evaluation_episodes=1, demonstration_episodes=1,
                            demonstration_seed=10000, validation_seed=100000)
    network = DQNConfig(hidden_sizes=(8, 8), batch_size=4, replay_capacity=256,
                        target_update_interval=5)
    output = tmp_path / "seeded"
    result = train(easy_config(), network, config, output)
    demonstrations = json.loads((output / "demonstrations.json").read_text())
    assert demonstrations["successful_episodes"] == 1
    assert demonstrations["optimizer_updates"] == 0
    assert result["updates"] > 0
    assert result["total_collection_steps"] == (
        result["environment_steps"] + demonstrations["collection_environment_steps"]
    )
    assert result["replay_size"] == (
        result["environment_steps"] + demonstrations["accepted_transitions"]
    )
    assert result["validation"]["seed"] == 100000
    assert json.loads((output / "run_status.json").read_text())["status"] == "completed"
    loaded, env, metadata = DQNAgent.load(output / "latest.pt")
    try:
        observation, _ = env.reset(seed=100000)
        assert env.action_space.contains(loaded.act(observation))
        assert metadata["updates"] == result["updates"]
    finally:
        env.close()
