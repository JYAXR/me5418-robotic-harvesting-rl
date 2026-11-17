"""Check valid recovery labels, frozen evaluation and supervised-only updates."""

import importlib.util
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from me5418_harvesting.agents.dqn import DQNConfig
from me5418_harvesting.agents.dqn_agent import DQNAgent, ObservationEncoder
from me5418_harvesting.agents.training import evaluate
from me5418_harvesting.envs.branch_physics import Branch
from me5418_harvesting.envs.harvesting_env import HarvestingEnv
from me5418_harvesting.envs.scene_generator import Scene

spec = importlib.util.spec_from_file_location("batch_tests", Path(__file__).with_name("run_batch.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.mark.parametrize("route,success,budget", [([6], True, 5), ([3], False, 5), ([2, 2], False, 1)])
def test_only_successful_recovery_labels_are_accepted(monkeypatch, route, success, budget):
    env = HarvestingEnv({"max_steps": budget, "target_radius": 0.01})
    nodes = np.array([[0.8, y] for y in np.linspace(0.0, 1.0, 5)], dtype=np.float32)
    branch = Branch(nodes.copy(), nodes.copy(), 1e6, 0.1)
    env.scene = Scene(np.array([0.1, 0.5], dtype=np.float32), np.array([0.16, 0.5], dtype=np.float32), [branch])
    env.np_random = np.random.default_rng(20)
    env._initialize_episode()
    before = m.scene_state(env)
    monkeypatch.setattr(m, "collision_free_actions", lambda *args, **kwargs: route)
    result, rows, x, y = m.recovery_probe(env, ObservationEncoder(env))
    assert m.scene_state(env) == before
    assert result["success"] == success
    assert len(y) == (len(rows) if success else 0)
    assert len(x) == len(y) == result["accepted_samples"]
    assert len(rows) <= budget
    env.close()


def test_planner_failure_is_recorded_without_labels(monkeypatch):
    env = HarvestingEnv()
    env.reset(seed=20)

    def fail(*args, **kwargs):
        raise m.NoCollisionFreeRoute("Synthetic no-route case")

    monkeypatch.setattr(m, "collision_free_actions", fail)
    result, rows, x, y = m.recovery_probe(env, ObservationEncoder(env))
    assert result["reason"] == "planner_found_no_route"
    assert rows == x == y == []
    env.close()


def test_evaluation_matches_production_evaluator_and_preserves_policy(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "HERE", tmp_path)
    m.worker_init()
    config = yaml.safe_load((m.ROOT / "configs/environment_easy.yaml").read_text())
    config.update(max_steps=6, verify_reference_route=False)
    env = HarvestingEnv(config)
    agent = DQNAgent(env, DQNConfig(), seed=0)
    model = tmp_path / "model.pt"
    agent.save(model, config, {})
    expected_agent, expected_env, _ = DQNAgent.load(model)
    expected = evaluate(expected_agent, expected_env, 1, 20)["records"][0]
    actual = m.evaluation_task(str(model), 20, str(tmp_path))
    assert actual["metrics"] == {k: v for k, v in expected.items() if k != "seed"}
    assert actual["frozen_state_unchanged"] and actual["action_overrides"] == 0
    expected_env.close()
    env.close()


def test_supervised_update_changes_online_only_and_resume_is_exact():
    torch.set_num_threads(1)
    torch.manual_seed(81)
    env = HarvestingEnv()
    agent = DQNAgent(env, DQNConfig(learning_rate=m.LR), seed=0)
    agent.learner.update_count = 14000
    target = deepcopy(agent.learner.target.state_dict())
    online = deepcopy(agent.learner.online.state_dict())
    x = torch.randn(8, env.observation_size)
    y = torch.arange(8)
    m.supervised_step(agent, x, y)
    assert agent.learner.update_count == 14000
    assert m._digest(target) == m._digest(agent.learner.target.state_dict())
    assert m._digest(online) != m._digest(agent.learner.online.state_dict())
    other = DQNAgent(env, DQNConfig(learning_rate=m.LR), seed=0)
    other.learner.online.load_state_dict(agent.learner.online.state_dict())
    other.learner.optimizer.load_state_dict(deepcopy(agent.learner.optimizer.state_dict()))
    m.supervised_step(agent, x, y)
    m.supervised_step(other, x, y)
    assert m._digest(agent.learner.online.state_dict()) == m._digest(other.learner.online.state_dict())
    assert m._digest(agent.learner.optimizer.state_dict()) == m._digest(other.learner.optimizer.state_dict())
    env.close()


def test_validation_collection_is_rejected_before_model_loading():
    with pytest.raises(AssertionError, match="training split"):
        m.collect_task("nonexistent.pt", m.VALIDATION[0], "unused")


def test_early_stop_requires_retention_and_both_thresholds():
    summary = {"original_three": {"successes": 3}, "training": {"successes": 27},
               "validation": {"successes": 9}}
    assert m.ready_to_stop(summary, 6)
    assert not m.ready_to_stop(summary, 7)
    summary["original_three"]["successes"] = 2
    assert not m.ready_to_stop(summary, 0)
