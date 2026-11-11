"""Test one-scalar interventions, exact baseline gates and inherited-update accounting."""

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from me5418_harvesting.agents.dqn import DQNConfig
from me5418_harvesting.agents.dqn_agent import DQNAgent
from me5418_harvesting.agents.multiscene_demonstrations import build_multiscene_batch
from me5418_harvesting.agents.replay_buffer import ReplayBuffer
from me5418_harvesting.envs.harvesting_env import HarvestingEnv

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "divergence_control_test", ROOT / "docs/experiments/012_divergence_control/run_control.py",
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def fixture():
    torch.set_num_threads(1)
    torch.manual_seed(38)
    environment = {"max_steps": 3}
    env = HarvestingEnv(environment)
    config = DQNConfig(hidden_sizes=(8, 8), learning_rate=3e-4, batch_size=3,
                       replay_capacity=16, target_update_interval=2)
    agent = DQNAgent(env, config, seed=38)
    trajectories = []
    for action, seed in enumerate((10000, 10001, 10002)):
        replay = ReplayBuffer(16)
        observation, _ = env.reset(seed=seed)
        for _ in range(3):
            following, reward, terminated, truncated, _ = env.step(action)
            replay.add(agent.encoder(observation), action, reward, agent.encoder(following),
                       terminated, truncated)
            observation = following
        trajectories.append(runner.fit.complete_batch(replay))
    batch, returns, episodes = build_multiscene_batch(trajectories, runner.SEEDS, config.gamma)
    for _ in range(2):
        agent.learner.update(batch, demonstration_weight=1.0)
    return agent, env, environment, batch, returns, episodes


def fake_rollout(agent, env, seed, batch):
    return {"training_scene_only": True, "seed": seed, "success": True,
            "damage": False, "budget_exhausted": False, "steps": len(batch.actions)}


@pytest.mark.parametrize("factor,value", [("learning_rate", 1e-4), ("demonstration_weight", 2.0)])
def test_factor_preserves_weights_targets_rng_and_every_adam_tensor(tmp_path, factor, value):
    agent, env, environment, _, _, _ = fixture()
    try:
        runner.multi.start_phase(agent, True, runner.fit.parameter_hash(agent), 2, tmp_path / "boundary.json")
        before = runner.adam.training_snapshot(agent, environment)
        result = runner.apply_factor(agent, factor, value)
        after = runner.adam.training_snapshot(agent, environment)
        assert runner.exact_equal(before["optimizer"]["state"], after["optimizer"]["state"])
        for key in ("online", "target", "agent_rng_state", "torch_rng_state"):
            assert runner.exact_equal(before[key], after[key])
        assert agent.learner.update_count == 0
        assert result["settings_after"] == {**runner.BASELINE, factor: value}
        assert result["optimizer_after"]["step_min"] == 2
        if factor == "learning_rate":
            assert agent.learner.config.learning_rate == value
            assert agent.learner.optimizer.defaults["lr"] == value
            assert all(group["lr"] == value for group in agent.learner.optimizer.param_groups)
            assert result["optimizer_before"]["settings_sha256"] != result["optimizer_after"]["settings_sha256"]
        else:
            assert runner.exact_equal(before, after)
            assert result["optimizer_before"] == result["optimizer_after"]
    finally:
        env.close()


@pytest.mark.parametrize("factor,value", [
    ("unknown", 2), ("demonstration_weight", 1.0), ("learning_rate", 3e-4),
    ("learning_rate", 0), ("learning_rate", -1), ("demonstration_weight", float("nan")),
    ("demonstration_weight", float("inf")), ("learning_rate", True),
])
def test_invalid_factor_rejected_before_any_mutation(tmp_path, factor, value):
    agent, env, environment, _, _, _ = fixture()
    try:
        runner.multi.start_phase(agent, True, runner.fit.parameter_hash(agent), 2, tmp_path / "boundary.json")
        before = runner.adam.training_snapshot(agent, environment)
        with pytest.raises(ValueError):
            runner.apply_factor(agent, factor, value)
        assert runner.exact_equal(before, runner.adam.training_snapshot(agent, environment))
    finally:
        env.close()


def test_factor_rejected_before_td_boundary_without_mutation():
    agent, env, environment, _, _, _ = fixture()
    try:
        before = runner.adam.training_snapshot(agent, environment)
        with pytest.raises(ValueError, match="boundary"):
            runner.apply_factor(agent, "learning_rate", 1e-4)
        assert runner.exact_equal(before, runner.adam.training_snapshot(agent, environment))
    finally:
        env.close()


def test_same_snapshot_forks_reproduce_every_log_observation_and_full_state(tmp_path, monkeypatch):
    agent, env, environment, batch, returns, episodes = fixture()
    monkeypatch.setattr(runner.fit, "fixed_scene_rollout", fake_rollout)
    snapshot = runner.adam.training_snapshot(agent, environment)
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    expected = runner.multi.observe(agent, env, environment, batch, returns, episodes,
                                    "calibration", 2, 0, initial_dir)
    env.close()
    try:
        summaries = []
        for arm in ("reference", "baseline", "treatment"):
            fork, fork_env = runner.adam.restore_training_snapshot(snapshot)
            try:
                runner.multi.start_phase(fork, True, snapshot["parameter_sha256"], 2,
                                         tmp_path / f"{arm}_boundary.json")
                if arm == "treatment":
                    runner.apply_factor(fork, "demonstration_weight", 2)
                reference_updates = runner.read_updates(tmp_path / "reference/td/updates.jsonl") if arm == "baseline" else None
                reference_observations = runner.read_json(tmp_path / "reference/td/observations.json") if arm == "baseline" else None
                summary = runner.execute_td(
                    fork, fork_env, environment, batch, returns, episodes, arm, tmp_path,
                    expected, 2 if arm == "treatment" else 1, updates=2, checkpoints=(0, 1, 2),
                    reference_updates=reference_updates, reference_observations=reference_observations,
                )
                summaries.append(summary)
                assert summary["optimizer_updates"] == 2
                assert summary["total_updates_on_branch"] == 40002
                assert summary["target_synchronizations"] == 1
                assert summary["retention"]["eligible_seeds"] == list(runner.SEEDS)
                assert summary["retention"]["all_scenes_retained_at_all_sampled_checkpoints"]
                for step in (0, 1, 2):
                    full = torch.load(tmp_path / arm / "td" / f"training_state_{step}.pt", weights_only=True)
                    assert full["optimizer_summary"]["step_min"] == 2 + step
                    assert full["update_count"] == step
                if arm == "baseline":
                    actual = torch.load(tmp_path / "baseline/td/training_state.pt", weights_only=True)
                    reference = torch.load(tmp_path / "reference/td/training_state.pt", weights_only=True)
                    assert runner.exact_equal(actual, reference)
            finally:
                fork_env.close()
        assert summaries[0]["final_parameter_sha256"] == summaries[1]["final_parameter_sha256"]
        assert summaries[1]["final_parameter_sha256"] != summaries[2]["final_parameter_sha256"]
        accounting = runner.write_accounting(tmp_path)
        assert accounting["optimizer_updates_total"] == 6
        assert accounting["inherited_optimizer_updates"] == 40000
        assert accounting["inherited_supervised_updates"] == 20000
        assert accounting["inherited_calibration_updates"] == 20000
        assert accounting["demonstration_collection_steps"] == 0
        assert accounting["scheduled_evaluation_episodes"] == 27
    finally:
        env.close()


@pytest.mark.parametrize("kind", ["update", "observation"])
def test_exact_baseline_gate_records_discrepancy_and_blocks_treatment(tmp_path, kind):
    with pytest.raises(RuntimeError, match="treatment is blocked"):
        runner.compare_record({"value": 1}, {"value": 1.0000000001}, tmp_path, kind, 1)
    record = runner.read_json(tmp_path / "baseline_equivalence.json")
    assert record["passed"] is False
    assert record["treatment_started"] is False
    assert record["kind"] == kind
    assert record["actual"] != record["expected"]


def test_boundary_behavior_mismatch_stops_before_td_update(tmp_path, monkeypatch):
    agent, env, environment, batch, returns, episodes = fixture()
    monkeypatch.setattr(runner.fit, "fixed_scene_rollout", fake_rollout)
    try:
        runner.multi.start_phase(agent, True, runner.fit.parameter_hash(agent), 2, tmp_path / "boundary.json")
        reference_dir = tmp_path / "initial"
        reference_dir.mkdir()
        expected = runner.multi.observe(agent, env, environment, batch, returns, episodes,
                                        "initial", 0, 40000, reference_dir)
        expected = deepcopy(expected)
        expected["scenes"][1]["rollout"]["success"] = False
        before = runner.adam.training_snapshot(agent, environment)
        with pytest.raises(RuntimeError, match="per-scene behavior"):
            runner.execute_td(agent, env, environment, batch, returns, episodes, "baseline", tmp_path,
                               expected, 1, updates=2, checkpoints=(0, 1, 2))
        assert runner.exact_equal(before, runner.adam.training_snapshot(agent, environment))
        assert (tmp_path / "baseline/td/updates.jsonl").read_text() == ""
        assert not (tmp_path / "treatment").exists()
    finally:
        env.close()
