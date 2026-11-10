"""Check fixed validation splits, initial-policy comparison, and stalled motion metrics."""

import json
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from me5418_harvesting.agents.dqn import DQNConfig
from me5418_harvesting.agents.dqn_agent import DQNAgent
from me5418_harvesting.agents.training import TrainingConfig, evaluate, train
from me5418_harvesting.envs.harvesting_env import HarvestingEnv


class LeftWallPolicy:
    def act(self, observation):
        return 7


def test_evaluation_counts_stationary_motion_at_workspace_boundary():
    env = HarvestingEnv({"max_steps": 10})
    try:
        result = evaluate(LeftWallPolicy(), env, episodes=2, seed=100)
        assert result["mean_stationary_fraction"] == pytest.approx(0.8)
        # Workspace clipping is stationary motion, not a branch collision.
        assert result["mean_blocked_fraction"] == 0
        for row in result["records"]:
            assert row["stationary_steps"] == 8
            assert row["blocked_steps"] == 0
            assert row["budget_exhausted"]
            assert row["distance_to_target"] > 0.8
        assert result["mean_distance_to_target"] == pytest.approx(
            np.mean([row["distance_to_target"] for row in result["records"]])
        )
    finally:
        env.close()


def test_initial_validation_is_saved_without_changing_training_updates(tmp_path):
    torch.set_num_threads(1)
    network = DQNConfig(
        hidden_sizes=(16, 16), batch_size=2, replay_capacity=16, target_update_interval=2,
    )
    settings = TrainingConfig(
        episodes=2, learning_starts=2, train_frequency=1, evaluation_interval=2,
        evaluation_episodes=1, validation_seed=100, seed=0,
    )
    environment = {"max_steps": 4}
    before_run = train(environment, network, settings, tmp_path / "without_initial")
    after_run = train(environment, network, replace(settings, evaluate_before_training=True),
                      tmp_path / "with_initial")
    initial = json.loads((tmp_path / "with_initial/initial_validation.json").read_text())
    assert after_run["initial_validation"] == initial
    assert "initial_validation" not in before_run
    assert initial["seed"] == after_run["validation"]["seed"] == 100
    assert before_run["updates"] == after_run["updates"] == 7
    assert before_run["validation"] == after_run["validation"]
    assert before_run["last_loss"] == after_run["last_loss"]
    records = [json.loads(line) for line in
               (tmp_path / "with_initial/metrics.jsonl").read_text().splitlines()]
    assert len(records) == settings.episodes
    for row in records:
        assert 0 <= row["stationary_steps"] <= row["steps"]
        assert 0 <= row["blocked_steps"] <= row["steps"]
        assert np.isfinite(row["distance_to_target"])
    first, first_env, _ = DQNAgent.load(tmp_path / "without_initial/latest.pt")
    second, second_env, _ = DQNAgent.load(tmp_path / "with_initial/latest.pt")
    try:
        for left, right in zip(first.learner.online.parameters(), second.learner.online.parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    finally:
        first_env.close()
        second_env.close()


@pytest.mark.parametrize("validation_seed", [9, 10, 11, 19])
def test_overlapping_validation_ranges_are_rejected(validation_seed):
    with pytest.raises(ValueError, match="must not overlap"):
        TrainingConfig(seed=10, episodes=10, evaluation_episodes=2,
                       validation_seed=validation_seed)


@pytest.mark.parametrize("validation_seed", [0, 8, 20, 100000, None])
def test_disjoint_validation_ranges_are_accepted(validation_seed):
    TrainingConfig(seed=10, episodes=10, evaluation_episodes=2, validation_seed=validation_seed)


@pytest.mark.parametrize("settings", [
    {"validation_seed": -1}, {"validation_seed": 1.5}, {"validation_seed": True},
    {"evaluate_before_training": "yes"},
])
def test_invalid_initial_validation_settings(settings):
    with pytest.raises((ValueError, TypeError)):
        TrainingConfig(**settings)
