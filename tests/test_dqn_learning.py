"""Verify learning semantics and the executable network demonstration."""

import json
import subprocess
import sys
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from me5418_harvesting import HarvestingEnv
from me5418_harvesting.agents.dqn import DQNConfig, build_q_network
from me5418_harvesting.agents.dqn_learning import (
    DQNUpdater,
    TransitionBatch,
    collect_transitions,
    td_loss,
)

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "scripts/run_network_demo.py"


@pytest.mark.parametrize(
    ("environment_config", "dimension"),
    [({}, 125), ({"max_branches": 14, "scene": {"layout": "mixed_canopy"}}, 341)],
)
def test_q_network_accepts_real_observations_and_backpropagates(environment_config, dimension):
    env = HarvestingEnv(environment_config)
    experience = collect_transitions(env, count=4, seed=3)
    torch.manual_seed(3)
    online = build_q_network(env.observation_space.shape[0], env.action_space.n, DQNConfig())
    output = online(experience.observations)
    assert output.shape == (4, 8)
    assert experience.observations.shape == (4, dimension)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    for parameter in online.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    env.close()


def test_td_targets_distinguish_termination_from_truncation_and_detach_target():
    online = nn.Linear(2, 2)
    target = nn.Linear(2, 2)
    with torch.no_grad():
        online.weight.zero_()
        online.bias.copy_(torch.tensor([1.0, 10.0]))
        target.weight.zero_()
        target.bias.copy_(torch.tensor([2.0, 4.0]))
    batch = TransitionBatch(
        observations=torch.zeros(3, 2),
        actions=torch.tensor([0, 1, 0]),
        rewards=torch.ones(3),
        next_observations=torch.zeros(3, 2),
        terminated=torch.tensor([True, False, False]),
        truncated=torch.tensor([False, True, False]),
    )
    loss, chosen_q, targets = td_loss(online, target, batch, gamma=0.5)
    torch.testing.assert_close(chosen_q, torch.tensor([1.0, 10.0, 1.0]))
    torch.testing.assert_close(targets, torch.tensor([1.0, 3.0, 3.0]))
    assert loss.item() == pytest.approx(8.0 / 3.0)
    assert not targets.requires_grad
    loss.backward()
    assert online.bias.grad is not None
    assert all(parameter.grad is None for parameter in target.parameters())


def test_collector_preserves_final_observation_and_seed_across_episode_resets():
    env = HarvestingEnv({"max_steps": 1})
    first = collect_transitions(env, count=4, seed=7)
    repeated = collect_transitions(env, count=4, seed=7)
    for field in fields(TransitionBatch):
        torch.testing.assert_close(getattr(first, field.name), getattr(repeated, field.name))
    assert first.truncated.all()
    assert not first.terminated.any()
    assert first.actions.dtype == torch.long
    assert first.observations.dtype == torch.float32
    assert first.observations[:, -1].tolist() == [1.0] * 4
    assert first.next_observations[:, -1].tolist() == [0.0] * 4
    # A reset observation has a full budget, so it cannot replace a final next state.
    assert not torch.equal(first.next_observations[0], first.observations[1])
    env.close()


def test_update_clips_gradients_changes_online_weights_and_syncs_only_on_schedule():
    torch.manual_seed(0)
    config = DQNConfig(
        hidden_sizes=(16, 16), batch_size=4, target_update_interval=2, gradient_clip_norm=1e-3
    )
    env = HarvestingEnv()
    batch = collect_transitions(env, count=4, seed=0)
    learner = DQNUpdater(125, 8, config)
    before = [parameter.detach().clone() for parameter in learner.target.parameters()]
    stats = learner.update(batch)
    assert stats.parameter_delta_norm > 0
    assert stats.gradient_norm > config.gradient_clip_norm
    assert np.isfinite(stats.loss_before) and np.isfinite(stats.loss_after)
    norm = torch.linalg.vector_norm(
        torch.cat([parameter.grad.flatten() for parameter in learner.online.parameters()])
    )
    assert norm.item() <= config.gradient_clip_norm * 1.01
    assert not stats.target_synchronized
    assert not learner.target.training
    for old, parameter in zip(before, learner.target.parameters()):
        torch.testing.assert_close(old, parameter)
        assert not parameter.requires_grad
        assert parameter.grad is None
    assert learner.update(batch).target_synchronized
    for online, target in zip(learner.online.parameters(), learner.target.parameters()):
        torch.testing.assert_close(online, target)
        assert target.grad is None
    env.close()


def test_nonfinite_loss_is_rejected_before_changing_parameters():
    env = HarvestingEnv()
    batch = collect_transitions(env, count=4, seed=0)
    learner = DQNUpdater(125, 8, DQNConfig(hidden_sizes=(16, 16)))
    before = [parameter.detach().clone() for parameter in learner.online.parameters()]
    with pytest.raises(FloatingPointError, match="not finite"):
        learner.update(replace(batch, rewards=torch.full((4,), float("nan"))))
    for old, parameter in zip(before, learner.online.parameters()):
        torch.testing.assert_close(old, parameter)
    assert learner.update_count == 0
    env.close()


@pytest.mark.parametrize("size", [0, 5])
def test_sampling_rejects_empty_or_oversized_minibatches(size):
    env = HarvestingEnv()
    experience = collect_transitions(env, count=4, seed=0)
    with pytest.raises(ValueError, match="Batch size"):
        experience.sample(size, np.random.default_rng(0))
    env.close()


@pytest.mark.parametrize("scene", ["environment.yaml", "environment_complex.yaml"])
def test_cli_runs_from_another_directory_and_demonstrates_an_executed_network_action(scene, tmp_path):
    command = [
        sys.executable, str(DEMO), "--batch-size", "4", "--transitions", "8", "--updates", "2",
    ]
    if scene == "environment_complex.yaml":
        command.extend(["--config", str(ROOT / "configs" / scene)])
    first = subprocess.run(
        command, check=False, cwd=tmp_path, capture_output=True, text=True, timeout=60
    )
    assert first.returncode == 0, first.stderr
    result = json.loads(first.stdout)
    dimension = 341 if scene == "environment_complex.yaml" else 125
    assert result["architecture"] == [dimension, 256, 256, 8]
    assert result["forward"]["input_shape"] == [4, dimension]
    assert result["forward"]["output_shape"] == [4, 8]
    assert len(result["forward"]["sample_input"]) == dimension
    assert not result["target_has_gradients"]
    for update in result["updates"]:
        assert update["gradient_norm"] > 0
        assert update["parameter_delta_norm"] > 0
        assert np.isfinite(update["loss_before"]) and np.isfinite(update["loss_after"])
    controlled = result["network_controlled_step"]
    assert controlled["action"] == int(np.argmax(controlled["q_values"]))
    np.testing.assert_allclose(
        np.subtract(controlled["position_after"], controlled["position_before"]),
        controlled["executed_motion"], atol=1e-7,
    )
    assert controlled["next_observation_shape"] == [dimension]
    repeated = subprocess.run(
        command, check=False, cwd=tmp_path, capture_output=True, text=True, timeout=60
    )
    assert repeated.returncode == 0, repeated.stderr
    assert json.loads(repeated.stdout) == result
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--updates", "0"], "number of updates must be positive"),
        (["--batch-size", "0"], "must be positive integers"),
        (["--batch-size", "4", "--transitions", "3"], "at least batch_size"),
    ],
)
def test_cli_rejects_invalid_demo_settings(arguments, message, tmp_path):
    result = subprocess.run(
        [sys.executable, str(DEMO), *arguments],
        check=False, cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 2
    assert message in result.stderr
    assert "Traceback" not in result.stderr
