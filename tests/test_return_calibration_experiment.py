"""Check calibration phase evidence, handoffs and honest retention eligibility."""

import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from me5418_harvesting.agents.dqn import DQNConfig
from me5418_harvesting.agents.dqn_agent import DQNAgent
from me5418_harvesting.agents.dqn_learning import TransitionBatch
from me5418_harvesting.agents.replay_buffer import ReplayBuffer
from me5418_harvesting.envs.harvesting_env import HarvestingEnv

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "return_calibration_experiment", ROOT / "docs/experiments/009_return_calibration/run_return_calibration.py",
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def make_fixture():
    torch.set_num_threads(1)
    torch.manual_seed(29)
    environment = {"max_steps": 3}
    env = HarvestingEnv(environment)
    config = DQNConfig(hidden_sizes=(8, 8), batch_size=3, replay_capacity=16,
                       target_update_interval=2)
    agent = DQNAgent(env, config, seed=29)
    replay = ReplayBuffer(16)
    observation, _ = env.reset(seed=runner.staged.SCENE_SEED)
    for _ in range(3):
        next_observation, reward, terminated, truncated, _ = env.step(0)
        replay.add(agent.encoder(observation), 0, reward, agent.encoder(next_observation),
                   terminated, truncated)
        observation = next_observation
    batch = runner.staged.fit.complete_batch(replay)
    returns, _ = runner.compute_targets(batch, config.gamma)
    return agent, env, environment, batch, returns


def successful_rollout(agent, env, seed, batch):
    return {"training_scene_only": True, "seed": seed, "success": True,
            "damage": False, "steps": 3}


def test_return_fit_metrics_match_known_chosen_scores():
    values = torch.tensor([[4.0, 8.0], [5.0, 3.0], [1.0, 6.0]])
    batch = TransitionBatch(values, torch.tensor([0, 1, 0]), torch.zeros(3), values,
                            torch.tensor([False, False, True]), torch.zeros(3, dtype=torch.bool))
    agent = SimpleNamespace(learner=SimpleNamespace(online=torch.nn.Identity()))
    result = runner.return_fit(agent, batch, torch.tensor([3.0, 5.0, 1.0]))
    assert result["rmse"] == pytest.approx((5 / 3) ** 0.5)
    assert result["mae"] == 1.0
    assert result["max_abs_error"] == 2.0
    assert result["bias"] == pytest.approx(-1 / 3)
    assert result["chosen_values"] == [4.0, 3.0, 1.0]


def test_actual_calibration_and_td_phases_preserve_history_and_reload(tmp_path, monkeypatch):
    agent, env, environment, batch, returns = make_fixture()
    monkeypatch.setattr(runner.staged.fit, "fixed_scene_rollout", successful_rollout)
    expected = {"final_fit": runner.staged.fit.fit_metrics(agent, batch, runner.MARGIN),
                "final_rollout": successful_rollout(agent, env, runner.staged.SCENE_SEED, batch)}
    branch = "calibrated"
    (tmp_path / branch).mkdir()
    agent.learner.update_count = runner.INHERITED_UPDATES
    initial_hash = runner.staged.fit.parameter_hash(agent)
    try:
        runner.start_phase(agent, branch, "calibration", tmp_path, initial_hash, 0)
        calibration = runner.execute_phase(
            agent, env, environment, batch, returns, branch, "calibration", 3, (0, 1, 3),
            tmp_path, runner.INHERITED_UPDATES, expected,
        )
        assert calibration["optimizer_updates"] == 3
        assert calibration["target_synchronizations"] == 0
        assert calibration["total_updates_on_branch"] == runner.INHERITED_UPDATES + 3
        assert calibration["initial_parameter_sha256"] == initial_hash
        assert calibration["final_parameter_sha256"] != initial_hash
        before_boundary = runner.staged.fit.parameter_hash(agent)
        runner.start_phase(agent, branch, "td", tmp_path, before_boundary, 3)
        assert runner.staged.fit.parameter_hash(agent) == before_boundary
        assert not agent.learner.optimizer.state
        assert agent.learner.update_count == 0
        td = runner.execute_phase(
            agent, env, environment, batch, returns, branch, "td", 2, (0, 1, 2),
            tmp_path, runner.INHERITED_UPDATES + 3, calibration,
        )
        assert td["optimizer_updates"] == 2
        assert td["target_synchronizations"] == 1
        assert td["total_updates_on_branch"] == runner.INHERITED_UPDATES + 5
        boundary = json.loads((tmp_path / branch / "td_boundary.json").read_text())
        assert boundary["helper_record"]["pretraining_updates"] == 3
        assert boundary["updates_on_branch_before_phase"] == runner.INHERITED_UPDATES + 3
        loaded, loaded_env, metadata = DQNAgent.load(tmp_path / branch / "td/update_2.pt")
        try:
            assert runner.staged.fit.parameter_hash(loaded) == td["final_parameter_sha256"]
            assert runner.return_fit(loaded, batch, returns) == td["final_return_fit"]
            assert metadata["total_updates_on_branch"] == runner.INHERITED_UPDATES + 5
            assert metadata["fit"] == td["final_fit"]
        finally:
            loaded_env.close()
    finally:
        env.close()


@pytest.mark.parametrize("changed_field", ["final_fit", "final_rollout"])
def test_mismatched_phase_zero_stops_before_updates(tmp_path, monkeypatch, changed_field):
    agent, env, environment, batch, returns = make_fixture()
    monkeypatch.setattr(runner.staged.fit, "fixed_scene_rollout", successful_rollout)
    expected = {"final_fit": runner.staged.fit.fit_metrics(agent, batch, runner.MARGIN),
                "final_rollout": successful_rollout(agent, env, runner.staged.SCENE_SEED, batch)}
    if changed_field == "final_fit":
        expected[changed_field]["correct"] += 1
    else:
        expected[changed_field]["success"] = False
    initial_hash = runner.staged.fit.parameter_hash(agent)
    try:
        with pytest.raises(RuntimeError, match="starting behavior"):
            runner.execute_phase(
                agent, env, environment, batch, returns, "calibrated", "calibration",
                2, (0, 1, 2), tmp_path, runner.INHERITED_UPDATES, expected,
            )
        assert agent.learner.update_count == 0
        assert not agent.learner.optimizer.state
        assert runner.staged.fit.parameter_hash(agent) == initial_hash
        assert (tmp_path / "calibrated/calibration/updates.jsonl").read_text() == ""
    finally:
        env.close()


def test_downstream_recovery_is_not_called_retention_when_calibration_failed(tmp_path, monkeypatch):
    agent, env, environment, batch, _ = make_fixture()
    reference_path = tmp_path / "reference"
    (reference_path / "pretraining").mkdir(parents=True)
    monkeypatch.setattr(runner, "CALIBRATION_UPDATES", 2)
    monkeypatch.setattr(runner, "TD_UPDATES", 2)
    monkeypatch.setattr(runner, "CALIBRATION_CHECKPOINTS", (0, 1, 2))
    monkeypatch.setattr(runner, "TD_CHECKPOINTS", (0, 1, 2))
    current_phase = {"branch": None, "phase": None}

    def controlled_rollout(policy, current_env, seed, current_batch):
        result = successful_rollout(policy, current_env, seed, current_batch)
        if current_phase["branch"] == "calibrated":
            failed = (current_phase["phase"] == "calibration" and policy.learner.update_count == 2
                      or current_phase["phase"] == "td" and policy.learner.update_count == 0)
            result["success"] = not failed
        return result

    original_phase = runner.execute_phase

    def track_phase(*args, **kwargs):
        current_phase.update(branch=args[5], phase=args[6])
        return original_phase(*args, **kwargs)

    monkeypatch.setattr(runner, "execute_phase", track_phase)
    monkeypatch.setattr(runner.staged.fit, "fixed_scene_rollout", controlled_rollout)
    fit = runner.staged.fit.fit_metrics(agent, batch, runner.MARGIN)
    reference = {
        "protocol": {"initialization_seed": 29},
        "pretraining": {"final_fit": fit,
                        "final_rollout": successful_rollout(agent, env, runner.staged.SCENE_SEED, batch),
                        "final_parameter_sha256": runner.staged.fit.parameter_hash(agent)},
    }
    settings = {"environment": environment, "dqn": json.loads(json.dumps(asdict(agent.learner.config)))}
    checkpoint = reference_path / "pretraining/update_20000.pt"
    agent.save(checkpoint, environment, {"objective": "pretraining",
                                        "updates": runner.INHERITED_UPDATES, "fit": fit})
    np.savez_compressed(reference_path / "demonstration.npz", **{
        name: getattr(batch, name).numpy() for name in batch.__dataclass_fields__
    })
    dataset = {"sha256": runner.staged.fit.file_hash(reference_path / "demonstration.npz")}
    for name in ("run_status.json", "summary.json", "settings.json", "provenance.json",
                 "pretraining_reproduction.json", "dataset.json", "pretraining/updates.jsonl"):
        (reference_path / name).write_text("{}\n")
    monkeypatch.setattr(runner.reference_module, "load_reference", lambda _: (
        reference, settings, {"git_head": "test-fixture", "source_sha256": {}},
        dataset, batch, checkpoint,
    ))
    try:
        result = runner.run(reference_path, tmp_path / "run")
        failed_calibration = result["arms"]["calibrated"]
        assert not failed_calibration["calibration"]["final_rollout"]["success"]
        assert failed_calibration["td"]["final_rollout"]["success"]
        assert not failed_calibration["retention_eligible"]
        assert not failed_calibration["retained_at_all_sampled_td_checkpoints"]
        assert (failed_calibration["td_interpretation"]
                == "downstream_behavior_after_unsuccessful_calibration")
        assert result["arms"]["continued_margin"]["retention_eligible"]
        assert result["optimizer_updates_total"] == 8
        assert result["td_optimizer_updates"] == 4
        assert result["inherited_pretraining_updates"] == runner.INHERITED_UPDATES
    finally:
        env.close()
