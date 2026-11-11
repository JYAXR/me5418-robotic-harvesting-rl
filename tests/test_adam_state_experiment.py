"""Check independent optimizer forks, retained bias correction and comparison gates."""

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from me5418_harvesting.agents.dqn import DQNConfig
from me5418_harvesting.agents.dqn_agent import DQNAgent
from me5418_harvesting.agents.replay_buffer import ReplayBuffer
from me5418_harvesting.envs.harvesting_env import HarvestingEnv

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "adam_state_experiment", ROOT / "docs/experiments/010_adam_state/run_adam_state.py",
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def make_fixture():
    torch.set_num_threads(1)
    torch.manual_seed(31)
    environment = {"max_steps": 3}
    env = HarvestingEnv(environment)
    config = DQNConfig(hidden_sizes=(8, 8), batch_size=3, replay_capacity=16,
                       target_update_interval=2)
    agent = DQNAgent(env, config, seed=31)
    replay = ReplayBuffer(16)
    observation, _ = env.reset(seed=runner.staged.SCENE_SEED)
    for _ in range(3):
        following, reward, terminated, truncated, _ = env.step(0)
        replay.add(agent.encoder(observation), 0, reward, agent.encoder(following),
                   terminated, truncated)
        observation = following
    batch = runner.fit.complete_batch(replay)
    returns, _ = runner.calibration.compute_targets(batch, config.gamma)
    for _ in range(3):
        runner.calibration.calibration_update(agent.learner, batch, returns)
    return agent, env, environment, batch, returns


def successful_rollout(agent, env, seed, batch):
    return {"training_scene_only": True, "seed": seed, "success": True,
            "damage": False, "steps": 3}


def expected_start(agent, env, batch, returns):
    return {"optimizer_updates": agent.learner.update_count,
            "final_fit": runner.fit.fit_metrics(agent, batch, 0.8),
            "final_rollout": successful_rollout(agent, env, runner.staged.SCENE_SEED, batch),
            "final_return_fit": runner.calibration.return_fit(agent, batch, returns)}


def test_serialized_snapshot_has_independent_adam_state_and_weights(tmp_path):
    agent, env, environment, batch, _ = make_fixture()
    snapshot = runner.training_snapshot(agent, environment)
    path = tmp_path / "snapshot.pt"
    torch.save(snapshot, path)
    restored, restored_env = runner.restore_training_snapshot(
        torch.load(path, weights_only=True),
    )
    other, other_env = runner.restore_training_snapshot(snapshot)
    try:
        before = runner.optimizer_state_summary(agent.learner)
        assert runner.optimizer_state_summary(restored.learner) == before
        assert runner.optimizer_state_summary(other.learner) == before
        assert runner.fit.parameter_hash(restored) == runner.fit.parameter_hash(agent)
        for original, first, second in zip(agent.learner.optimizer.state.values(),
                                           restored.learner.optimizer.state.values(),
                                           other.learner.optimizer.state.values()):
            for name in ("step", "exp_avg", "exp_avg_sq"):
                assert len({original[name].data_ptr(), first[name].data_ptr(),
                            second[name].data_ptr()}) == 3
        restored.learner.update(batch, demonstration_weight=1.0)
        assert runner.optimizer_state_summary(agent.learner) == before
        assert runner.optimizer_state_summary(other.learner) == before
        assert snapshot["optimizer_summary"] == before
        assert runner.fit.parameter_hash(other) == snapshot["parameter_sha256"]
    finally:
        env.close()
        restored_env.close()
        other_env.close()


def test_both_td_branches_share_weights_but_only_preserve_keeps_adam_steps(tmp_path, monkeypatch):
    agent, env, environment, batch, returns = make_fixture()
    monkeypatch.setattr(runner.fit, "fixed_scene_rollout", successful_rollout)
    snapshot = runner.training_snapshot(agent, environment)
    expected = expected_start(agent, env, batch, returns)
    states = {}
    try:
        for branch in runner.ARMS:
            (tmp_path / branch).mkdir()
            fork, fork_env = runner.restore_training_snapshot(snapshot)
            try:
                boundary = runner.start_td(fork, branch, snapshot, "test-snapshot", tmp_path)
                assert fork.learner.update_count == 0
                assert runner.fit.parameter_hash(fork) == snapshot["parameter_sha256"]
                assert boundary["optimizer_before"] == snapshot["optimizer_summary"]
                assert (boundary["optimizer_after"]["settings_sha256"]
                        == snapshot["optimizer_summary"]["settings_sha256"])
                result = runner.execute_td(fork, fork_env, environment, batch, returns,
                                           branch, tmp_path, expected, updates=2, checkpoints=(0, 1, 2))
                states[branch] = result
                assert result["optimizer_updates"] == 2
                assert result["target_synchronizations"] == 1
                assert result["total_updates_on_branch"] == runner.INHERITED_UPDATES + 5
                assert result["evaluation_environment_steps"] == 9
                final = torch.load(tmp_path / result["training_snapshot_file"], weights_only=True)
                assert final["optimizer_summary"] == result["final_optimizer_state"]
                assert final["update_count"] == 2
                if branch == "preserve":
                    assert boundary["optimizer_after"] == snapshot["optimizer_summary"]
                    assert result["final_optimizer_state"]["step_min"] == 5
                else:
                    assert boundary["optimizer_after"]["initialized_parameter_count"] == 0
                    assert result["final_optimizer_state"]["step_min"] == 2
            finally:
                fork_env.close()
        assert states["reset"]["initial_parameter_sha256"] == states["preserve"]["initial_parameter_sha256"]
        assert states["reset"]["final_parameter_sha256"] != states["preserve"]["final_parameter_sha256"]
        accounting = runner.write_partial_accounting(tmp_path)
        assert accounting["optimizer_updates_total"] == 4
        assert accounting["td_optimizer_updates"] == 4
        assert accounting["evaluation_environment_steps"] == 18
    finally:
        env.close()


@pytest.mark.parametrize("field", ["final_fit", "final_rollout", "final_return_fit"])
def test_td_behavior_mismatch_stops_before_first_update(tmp_path, monkeypatch, field):
    agent, env, environment, batch, returns = make_fixture()
    monkeypatch.setattr(runner.fit, "fixed_scene_rollout", successful_rollout)
    expected = expected_start(agent, env, batch, returns)
    expected[field]["unexpected"] = True
    snapshot = runner.training_snapshot(agent, environment)
    (tmp_path / "preserve").mkdir()
    runner.start_td(agent, "preserve", snapshot, "test-snapshot", tmp_path)
    before = runner.optimizer_state_summary(agent.learner)
    try:
        with pytest.raises(RuntimeError, match="starting behavior"):
            runner.execute_td(agent, env, environment, batch, returns, "preserve", tmp_path,
                              expected, updates=2, checkpoints=(0, 1, 2))
        assert agent.learner.update_count == 0
        assert runner.optimizer_state_summary(agent.learner) == before
        assert (tmp_path / "preserve/td/updates.jsonl").read_text() == ""
    finally:
        env.close()


def test_calibration_reproduction_failure_is_saved_before_raising(tmp_path):
    current, previous = tmp_path / "current", tmp_path / "previous"
    for directory in (current, previous):
        directory.mkdir()
        runner.fit.save_json(directory / "observations.json", [{"update": 0}])
        (directory / "updates.jsonl").write_text('{"update": 1, "loss": 1.0}\n')
    reference = {"final_parameter_sha256": "expected", "final_fit": {}, "final_rollout": {},
                 "final_return_fit": {}, "optimizer_updates": 1}
    output = tmp_path / "reproduction.json"
    result = {**reference, "final_parameter_sha256": "changed"}
    with pytest.raises(RuntimeError, match="TD was not started"):
        runner.compare_reproduction(result, reference, current, previous, output)
    recorded = runner.read_json(output)
    assert not recorded["final_parameters_equal"]
    assert recorded["all_update_records_equal"]


def test_reset_comparison_ignores_extra_diagnostics_but_rejects_changed_metrics(tmp_path):
    current, previous = tmp_path / "current", tmp_path / "previous"
    row = {"update": 0, "fit": {}, "rollout": {}, "value_metrics": {}, "return_fit": {},
           "total_updates_on_branch": 40000}
    result = {"final_parameter_sha256": "weights", "target_synchronizations": 1}
    for directory in (current, previous):
        directory.mkdir()
        (directory / "updates.jsonl").write_text('{"update": 1, "loss": 1.0}\n')
        runner.fit.save_json(directory / "summary.json", result)
    runner.fit.save_json(previous / "observations.json", [row])
    runner.fit.save_json(current / "observations.json", [{**row, "optimizer_state": {"step": 3}},
                                                         {**row, "update": 1}])
    output = tmp_path / "reset_reproduction.json"
    checks = runner.compare_reset(result, current, previous, output)
    assert checks["all_shared_observations_equal"]
    (current / "updates.jsonl").write_text('{"update": 1, "loss": 2.0}\n')
    with pytest.raises(RuntimeError, match="reset control"):
        runner.compare_reset(result, current, previous, output)
    assert not runner.read_json(output)["all_update_records_equal"]
