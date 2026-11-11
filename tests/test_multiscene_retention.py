"""Exercise multi-scene phases, episode eligibility and failure accounting."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from me5418_harvesting.agents.dqn import DQNConfig
from me5418_harvesting.agents.dqn_agent import DQNAgent
from me5418_harvesting.agents.multiscene_demonstrations import build_multiscene_batch
from me5418_harvesting.agents.replay_buffer import ReplayBuffer
from me5418_harvesting.envs.harvesting_env import HarvestingEnv

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "multiscene_retention_test", ROOT / "docs/experiments/011_multiscene_retention/run_multiscene_retention.py",
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def fixture():
    torch.set_num_threads(1)
    torch.manual_seed(31)
    environment = {"max_steps": 3}
    env = HarvestingEnv(environment)
    config = DQNConfig(hidden_sizes=(8, 8), batch_size=3, replay_capacity=16,
                       target_update_interval=2)
    agent = DQNAgent(env, config, seed=31)
    trajectories = []
    for action, seed in enumerate((10000, 10001)):
        replay = ReplayBuffer(16)
        observation, _ = env.reset(seed=seed)
        for _ in range(3):
            following, reward, terminated, truncated, _ = env.step(action)
            replay.add(agent.encoder(observation), action, reward, agent.encoder(following),
                       terminated, truncated)
            observation = following
        trajectories.append(runner.fit.complete_batch(replay))
    batch, returns, episodes = build_multiscene_batch(trajectories, [10000, 10001], config.gamma)
    return agent, env, environment, batch, returns, episodes


def fake_rollout(agent, env, seed, batch):
    return {"training_scene_only": True, "seed": seed, "success": True,
            "damage": False, "budget_exhausted": False, "steps": len(batch.actions)}


def simple_observation(step, successful):
    scenes = [{"seed": seed, "rollout": {"success": seed in successful, "damage": False,
                                         "budget_exhausted": seed not in successful, "steps": 3}}
              for seed in (10000, 10001, 10002)]
    return {"update": step, "scenes": scenes, "aggregate": runner.aggregate_scenes(scenes)}


def test_retention_distinguishes_acquired_scene_from_later_recovery():
    initial = simple_observation(0, {10001})
    observations = [initial, simple_observation(1, {10000, 10001, 10002}),
                    simple_observation(10, {10000, 10001, 10002})]
    metrics = runner.retention_metrics(observations, initial)
    assert metrics["eligible_seeds"] == [10001]
    assert not metrics["all_scene_retention_eligible"]
    assert not metrics["all_scenes_retained_at_all_sampled_checkpoints"]
    assert metrics["acquired_subset_retained_at_all_sampled_checkpoints"]
    assert metrics["per_scene"]["10001"]["retained_at_all_sampled_checkpoints"]
    assert not metrics["per_scene"]["10000"]["retention_eligible"]
    assert metrics["per_scene"]["10000"]["failed_sampled_updates"] == [0]
    assert metrics["per_scene"]["10001"]["first_ten_updates_observation_count"] == 2


def test_no_acquired_scene_or_empty_input_cannot_claim_retention():
    initial = simple_observation(0, set())
    result = runner.retention_metrics([initial, simple_observation(1, {10000, 10001, 10002})], initial)
    assert not result["all_scene_retention_eligible"]
    assert not result["acquired_subset_retained_at_all_sampled_checkpoints"]
    empty = runner.retention_metrics([], {"scenes": []})
    assert not empty["all_scene_retention_eligible"]
    assert not empty["all_scenes_retained_at_all_sampled_checkpoints"]


def test_pretraining_and_calibration_have_separate_acquisition_boundaries():
    supervised = simple_observation(20000, {10000, 10001})
    calibration = [simple_observation(0, {10000, 10001}), simple_observation(20000, {10001, 10002})]
    pre_to_cal = runner.retention_metrics(calibration, supervised)
    td = runner.retention_metrics([simple_observation(0, {10001, 10002}),
                                   simple_observation(1, {10001, 10002})], calibration[-1])
    assert pre_to_cal["eligible_seeds"] == [10000, 10001]
    assert pre_to_cal["per_scene"]["10000"]["first_sampled_failure_update"] == 20000
    assert td["eligible_seeds"] == [10001, 10002]
    assert td["acquired_subset_retained_at_all_sampled_checkpoints"]
    assert not td["all_scene_retention_eligible"]


def test_real_three_stage_updates_and_both_td_snapshot_forks(tmp_path, monkeypatch):
    agent, env, environment, batch, returns, episodes = fixture()
    monkeypatch.setattr(runner.fit, "fixed_scene_rollout", fake_rollout)
    try:
        pretrained = runner.execute_phase(agent, env, environment, batch, returns, episodes,
                                           "pretraining", 2, (0, 1, 2), tmp_path, 0)
        assert pretrained["optimizer_updates"] == 2
        assert pretrained["target_synchronizations"] == 0
        assert pretrained["retention"] is None
        runner.start_phase(agent, False, pretrained["final_parameter_sha256"], 2,
                            tmp_path / "calibration_boundary.json")
        calibrated = runner.execute_phase(agent, env, environment, batch, returns, episodes,
                                           "calibration", 3, (0, 1, 3), tmp_path, 2, pretrained)
        assert calibrated["optimizer_updates"] == 3
        assert calibrated["total_updates_on_branch"] == 5
        assert calibrated["retention"]["all_scene_retention_eligible"]
        payload = torch.load(tmp_path / calibrated["training_snapshot_file"], weights_only=True)
        summaries = {}
        for branch in runner.ARMS:
            (tmp_path / branch).mkdir()
            fork, fork_env = runner.adam.restore_training_snapshot(payload)
            try:
                runner.start_phase(fork, branch == "preserve", calibrated["final_parameter_sha256"],
                                    5, tmp_path / branch / "td_boundary.json")
                result = runner.execute_phase(fork, fork_env, environment, batch, returns, episodes,
                                               f"{branch}/td", 2, (0, 1, 2), tmp_path, 5, calibrated)
                summaries[branch] = result
                assert result["target_synchronizations"] == 1
                assert result["optimizer_updates"] == 2
                assert result["total_updates_on_branch"] == 7
                assert result["retention"]["all_scenes_retained_at_all_sampled_checkpoints"]
                assert result["evaluation_environment_steps"] == 18
                final = torch.load(tmp_path / result["training_snapshot_file"], weights_only=True)
                assert final["optimizer_summary"]["step_min"] == (5 if branch == "preserve" else 2)
                loaded, loaded_env, metadata = DQNAgent.load(tmp_path / result["directory"] / "update_2.pt")
                try:
                    assert runner.fit.parameter_hash(loaded) == result["final_parameter_sha256"]
                    assert metadata["scene_seeds"] == [10000, 10001]
                    assert runner.calibration.return_fit(loaded, batch, returns) == result["final_observation"]["return_fit"]
                finally:
                    loaded_env.close()
            finally:
                fork_env.close()
        assert summaries["reset"]["initial_parameter_sha256"] == summaries["preserve"]["initial_parameter_sha256"]
        assert summaries["reset"]["final_parameter_sha256"] != summaries["preserve"]["final_parameter_sha256"]
        accounting = runner.write_accounting(tmp_path)
        assert accounting["optimizer_updates_total"] == 9
        assert accounting["td_optimizer_updates"] == 4
        assert accounting["evaluation_environment_steps"] == 72
        assert accounting["inherited_optimizer_updates"] == 0
    finally:
        env.close()


@pytest.mark.parametrize("field", ["fit", "return_fit", "scenes", "aggregate"])
def test_boundary_behavior_mismatch_stops_before_training(tmp_path, monkeypatch, field):
    agent, env, environment, batch, returns, episodes = fixture()
    monkeypatch.setattr(runner.fit, "fixed_scene_rollout", fake_rollout)
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    expected = {"final_observation": runner.observe(agent, env, environment, batch, returns,
                                                     episodes, "reference", 0, 0, reference_dir)}
    if field == "scenes":
        expected["final_observation"][field][0]["rollout"]["success"] = False
    else:
        expected["final_observation"][field]["unexpected"] = True
    initial_hash = runner.fit.parameter_hash(agent)
    try:
        with pytest.raises(RuntimeError, match="per-scene behavior"):
            runner.execute_phase(agent, env, environment, batch, returns, episodes, "calibration",
                                  2, (0, 1, 2), tmp_path, 0, expected)
        assert agent.learner.update_count == 0
        assert runner.fit.parameter_hash(agent) == initial_hash
        assert not agent.learner.optimizer.state
        assert (tmp_path / "calibration/updates.jsonl").read_text() == ""
    finally:
        env.close()


def test_collection_keeps_failed_declared_seed_without_replacement(tmp_path, monkeypatch):
    agent, env, environment, batch, _, episodes = fixture()
    attempted = []

    def collector(environment, encoder, replay, count, seed, progress):
        attempted.append(seed)
        success = seed == 10000
        if success:
            start, stop = episodes[0]["start"], episodes[0]["stop"]
            for index in range(start, stop):
                replay.add(batch.observations[index], int(batch.actions[index]), float(batch.rewards[index]),
                           batch.next_observations[index], bool(batch.terminated[index]),
                           bool(batch.truncated[index]))
        record = {"successful_episodes": int(success), "collection_environment_steps": 3 if success else 1,
                  "accepted_transitions": 3 if success else 0,
                  "records": [{"seed": seed, "success": success, "damage": not success}]}
        progress(record)
        return record

    monkeypatch.setattr(runner, "seed_replay_from_reference", collector)
    try:
        with pytest.raises(RuntimeError, match="10001"):
            runner.collect_dataset(environment, agent, tmp_path, seeds=(10000, 10001, 10002))
        assert attempted == [10000, 10001]
        record = json.loads((tmp_path / "demonstrations.json").read_text())
        assert record["requested_seeds"] == [10000, 10001, 10002]
        assert record["successful_episodes"] == 1
        assert record["collection_environment_steps"] == 4
        assert not record["collections"]["10001"]["records"][0]["success"]
        assert not (tmp_path / "demonstration.npz").exists()
    finally:
        env.close()


def test_pooled_return_error_is_computed_from_all_transitions(tmp_path, monkeypatch):
    agent, env, environment, batch, returns, episodes = fixture()
    monkeypatch.setattr(runner.fit, "fixed_scene_rollout", fake_rollout)
    try:
        row = runner.observe(agent, env, environment, batch, returns, episodes, "fixture", 0, 0, tmp_path)
        pooled = np.asarray(row["return_fit"]["chosen_values"], dtype=np.float64) - returns.numpy()
        assert row["return_fit"]["rmse"] == pytest.approx(float(np.sqrt(np.mean(pooled ** 2))))
        assert row["aggregate"]["scene_count"] == 2
        assert row["aggregate"]["evaluation_environment_steps"] == 6
        assert row["fit"]["transitions"] == len(batch.actions)
    finally:
        env.close()
