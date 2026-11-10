"""Check phase handoff guards, sampled retention outcomes and checkpoint reloads."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from me5418_harvesting.agents.dqn import DQNConfig
from me5418_harvesting.agents.dqn_agent import DQNAgent
from me5418_harvesting.agents.replay_buffer import ReplayBuffer
from me5418_harvesting.agents.staged_learning import begin_td_phase
from me5418_harvesting.envs.harvesting_env import HarvestingEnv

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "staged_retention_runner",
    ROOT / "docs/experiments/007_staged_retention/run_staged_retention.py",
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def make_fixture():
    torch.set_num_threads(1)
    torch.manual_seed(19)
    environment = {"max_steps": 3}
    env = HarvestingEnv(environment)
    config = DQNConfig(hidden_sizes=(8, 8), batch_size=3, replay_capacity=16,
                       target_update_interval=2)
    agent = DQNAgent(env, config, seed=19)
    replay = ReplayBuffer(16)
    observation, _ = env.reset(seed=runner.SCENE_SEED)
    for _ in range(3):
        next_observation, reward, terminated, truncated, _ = env.step(0)
        replay.add(agent.encoder(observation), 0, reward, agent.encoder(next_observation),
                   terminated, truncated)
        observation = next_observation
        if terminated or truncated:
            break
    begin_td_phase(agent.learner)
    return agent, env, environment, runner.fit.complete_batch(replay)


def controlled_rollout(agent, env, seed, batch):
    """Supply known outcomes so the test isolates phase accounting from policy quality."""
    return {
        "training_scene_only": True, "seed": seed,
        "success": agent.learner.update_count != 2,
        "damage": agent.learner.update_count == 2, "steps": 3,
    }


def test_phase_preserves_start_and_records_failure_even_after_recovery(tmp_path, monkeypatch):
    agent, env, environment, batch = make_fixture()
    monkeypatch.setattr(runner.fit, "fixed_scene_rollout", controlled_rollout)
    expected = {
        "final_fit": runner.fit.fit_metrics(agent, batch, runner.MARGIN),
        "final_rollout": controlled_rollout(agent, env, runner.SCENE_SEED, batch),
    }
    initial_weights = deepcopy(agent.learner.online.state_dict())
    try:
        result, observations = runner.execute_phase(
            agent, env, environment, batch, "td_only", 4, (0, 1, 2, 4), tmp_path,
            pretraining_updates=20, expected_start=expected,
        )
        assert observations[0]["fit"] == expected["final_fit"]
        assert observations[0]["rollout"] == expected["final_rollout"]
        assert [row["total_updates_on_branch"] for row in observations] == [20, 21, 22, 24]
        assert result["optimizer_updates"] == agent.learner.update_count == 4
        assert result["target_synchronizations"] == 2
        assert result["evaluation_environment_steps"] == 12
        assert result["first_sampled_failure_update"] == 2
        assert not result["success_at_every_sampled_checkpoint"]
        assert result["final_rollout"]["success"]

        initial, initial_env, initial_metadata = DQNAgent.load(tmp_path / "td_only/update_0.pt")
        final, final_env, final_metadata = DQNAgent.load(tmp_path / "td_only/update_4.pt")
        try:
            for name, value in initial_weights.items():
                torch.testing.assert_close(initial.learner.online.state_dict()[name], value,
                                           rtol=0, atol=0)
            assert initial_metadata["fit"] == expected["final_fit"]
            assert initial_metadata["total_updates_on_branch"] == 20
            assert final_metadata["updates"] == 4
            assert final_metadata["pretraining_updates"] == 20
            assert final_metadata["total_updates_on_branch"] == 24
            assert runner.fit.parameter_hash(final) == result["final_parameter_sha256"]
            assert runner.fit.fit_metrics(final, batch, runner.MARGIN) == result["final_fit"]
            assert final_metadata["fit"] == result["final_fit"]
        finally:
            initial_env.close()
            final_env.close()
    finally:
        env.close()


@pytest.mark.parametrize("changed_field", ["final_fit", "final_rollout"])
def test_changed_start_aborts_before_any_optimizer_update(tmp_path, monkeypatch, changed_field):
    agent, env, environment, batch = make_fixture()
    monkeypatch.setattr(runner.fit, "fixed_scene_rollout", controlled_rollout)
    expected = {
        "final_fit": runner.fit.fit_metrics(agent, batch, runner.MARGIN),
        "final_rollout": controlled_rollout(agent, env, runner.SCENE_SEED, batch),
    }
    if changed_field == "final_fit":
        expected[changed_field]["correct"] += 1
    else:
        expected[changed_field]["success"] = False
    original_hash = runner.fit.parameter_hash(agent)

    def forbidden_update(*args, **kwargs):
        raise AssertionError("An optimizer update ran before checking the phase-zero policy.")

    monkeypatch.setattr(runner, "update_agent", forbidden_update)
    try:
        with pytest.raises(RuntimeError, match="phase-zero policy"):
            runner.execute_phase(
                agent, env, environment, batch, "td_only", 2, (0, 1, 2), tmp_path,
                pretraining_updates=20, expected_start=expected,
            )
        assert agent.learner.update_count == 0
        assert not agent.learner.optimizer.state
        assert runner.fit.parameter_hash(agent) == original_hash
        retained = json.loads((tmp_path / "td_only/observations.json").read_text())
        assert len(retained) == 1 and retained[0]["update"] == 0
        assert (tmp_path / "td_only/updates.jsonl").read_text() == ""
    finally:
        env.close()
