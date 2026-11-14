"""Validate isolated TD scaling, complete-state continuity and certification accounting."""

import importlib.util
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from me5418_harvesting.agents.dqn import DQNConfig
from me5418_harvesting.agents.dqn_agent import DQNAgent
from me5418_harvesting.agents.dqn_learning import TransitionBatch
from me5418_harvesting.envs.harvesting_env import HarvestingEnv

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "weight019_test", ROOT / "docs/experiments/019_td_weight/run_td_weight.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def fixture(counter=2199):
    torch.set_num_threads(1)
    torch.manual_seed(193)
    environment = {"max_steps": 3}
    env = HarvestingEnv(environment)
    agent = DQNAgent(env, DQNConfig(hidden_sizes=(8, 8), learning_rate=1e-4,
                                   batch_size=3, target_update_interval=200), seed=193)
    batch = TransitionBatch(
        observations=torch.randn(3, env.observation_size), actions=torch.tensor([0, 1, 2]),
        rewards=torch.tensor([0.2, -0.1, 1.0]), next_observations=torch.randn(3, env.observation_size),
        terminated=torch.tensor([False, False, True]), truncated=torch.tensor([False, True, False]))
    for _ in range(2):
        agent.learner.update(batch, demonstration_weight=1)
    agent.learner.update_count = counter
    return agent, env, environment, batch, torch.tensor([1.3, 2.2, 1.0])


def independent_step(learner, batch):
    """Construct scaled TD and original hinge from their definitions, without runner losses."""
    learner.online.train()
    learner.optimizer.zero_grad(set_to_none=True)
    chosen = learner.online(batch.observations).gather(1, batch.actions[:, None]).squeeze(1)
    with torch.no_grad():
        future = learner.target(batch.next_observations).max(dim=1).values
        targets = batch.rewards + learner.config.gamma * torch.where(batch.terminated, 0.0, future)
    td = torch.nn.functional.smooth_l1_loss(chosen, targets)
    q = learner.online(batch.observations)
    offset = torch.full_like(q, 0.8)
    offset[torch.arange(len(q)), batch.actions] = 0.0
    hinge = ((q + offset).max(dim=1).values - q.gather(1, batch.actions[:, None]).squeeze(1)).mean()
    total = 0.5 * td + 1.0 * hinge
    total.backward()
    norm = torch.nn.utils.clip_grad_norm_(learner.online.parameters(), learner.config.gradient_clip_norm)
    gradients = [parameter.grad.detach().clone() for parameter in learner.online.parameters()]
    learner.optimizer.step()
    learner.update_count += 1
    if learner.update_count % learner.config.target_update_interval == 0:
        learner.target.load_state_dict(learner.online.state_dict())
    return float(td.detach()), float(hinge.detach()), float(total.detach()), float(norm), gradients, targets


def test_coefficient_one_exactly_reproduces_original_state_and_logs_across_target_sync():
    source, env, environment, batch, _ = fixture()
    original = deepcopy(source)
    try:
        for step in (2200, 2201, 2202):
            expected = asdict(original.learner.update(batch, demonstration_weight=1.0, demonstration_margin=0.8))
            actual = runner.weighted_td_update(source.learner, batch, 1.0)
            assert actual == {**expected, "td_weight": 1.0}
            assert actual["target_synchronized"] == (step == 2200)
            assert runner.exact_equal(runner.adam.training_snapshot(original, environment),
                                      runner.adam.training_snapshot(source, environment))
            assert all(torch.equal(a.grad, b.grad) for a, b in zip(
                original.learner.online.parameters(), source.learner.online.parameters(), strict=True))
    finally:
        env.close()


def test_half_td_matches_independent_gradients_optimizer_and_frozen_targets():
    source, env, environment, batch, _ = fixture()
    other = deepcopy(source)
    initial_steps = [float(state["step"]) for state in source.learner.optimizer.state.values()]
    initial_rng = torch.get_rng_state().clone()
    try:
        for offset, step in enumerate((2200, 2201), 1):
            td, margin, total, norm, gradients, targets = independent_step(other.learner, batch)
            actual = runner.weighted_td_update(source.learner, batch)
            assert actual["td_loss_before"] == td
            assert actual["demonstration_loss_before"] == margin
            assert actual["loss_before"] == total
            assert actual["gradient_norm"] == norm
            assert actual["td_weight"] == 0.5
            assert actual["loss_before"] == pytest.approx(0.5 * td + margin, abs=2e-7)
            assert all(torch.equal(p.grad, g) for p, g in zip(source.learner.online.parameters(), gradients, strict=True))
            assert runner.exact_equal(runner.adam.training_snapshot(source, environment),
                                      runner.adam.training_snapshot(other, environment))
            assert [float(state["step"]) for state in source.learner.optimizer.state.values()] == [
                value + offset for value in initial_steps]
            with torch.no_grad():
                chosen = source.learner.online(batch.observations).gather(1, batch.actions[:, None]).squeeze(1)
                assert actual["td_loss_after"] == float(torch.nn.functional.smooth_l1_loss(chosen, targets))
            assert actual["loss_after"] == pytest.approx(
                0.5 * actual["td_loss_after"] + actual["demonstration_loss_after"], abs=2e-7)
            assert actual["target_synchronized"] == (step == 2200)
            assert all(p.grad is None for p in source.learner.target.parameters())
            assert not any("return" in key or "anchor" in key for key in actual)
        assert torch.equal(torch.get_rng_state(), initial_rng)
        assert all(group["lr"] == 0.0001 for group in source.learner.optimizer.param_groups)
    finally:
        env.close()


def test_reducing_td_weight_is_not_equivalent_to_halving_the_common_learning_rate():
    source, env, _, batch, _ = fixture()
    slower = deepcopy(source.learner)
    try:
        for group in slower.optimizer.param_groups:
            group["lr"] *= 0.5
        runner.weighted_td_update(source.learner, batch)
        slower.update(batch, demonstration_weight=1)
        assert any(not torch.equal(a, b) for a, b in zip(source.learner.online.parameters(), slower.online.parameters()))
        assert any(not torch.equal(a["exp_avg"], b["exp_avg"])
                   for a, b in zip(source.learner.optimizer.state.values(), slower.optimizer.state.values()))
    finally:
        env.close()


@pytest.mark.parametrize("weight", [-1, 0, 0.25, 2, float("nan"), float("inf"), True, "0.5"])
def test_undeclared_weight_rejected_before_state_modes_or_gradients_change(weight):
    agent, env, environment, batch, _ = fixture()
    agent.learner.online.eval()
    before = runner.adam.training_snapshot(agent, environment)
    gradients = [p.grad.clone() for p in agent.learner.online.parameters()]
    try:
        with pytest.raises(ValueError, match="TD weight"):
            runner.weighted_td_update(agent.learner, batch, weight)
        assert runner.exact_equal(runner.adam.training_snapshot(agent, environment), before)
        assert not agent.learner.online.training
        assert all(torch.equal(p.grad, g) for p, g in zip(agent.learner.online.parameters(), gradients, strict=True))
    finally:
        env.close()


def test_initial_q_gate_records_all_inputs_without_changing_source_rng_or_modes():
    agent, env, environment, batch, _ = fixture(2000)
    agent.learner.online.eval()
    before = runner.adam.training_snapshot(agent, environment)
    try:
        record = runner.initial_q_equivalence(agent, batch, before)
        assert record["passed"] and record["shape"] == [3, 8] and record["rows"] == 3
        assert record["maximum_absolute_difference"] == record["new_optimizer_updates"] == record["environment_actions"] == 0
        assert runner.exact_equal(runner.adam.training_snapshot(agent, environment), before)
        assert not agent.learner.online.training
        with torch.no_grad():
            next(agent.learner.online.parameters()).add_(0.01)
        with pytest.raises(ValueError, match="initial restored Q"):
            runner.initial_q_equivalence(agent, batch, before)
    finally:
        env.close()


class FakeResetEnvironment:
    verify_reference_route = True

    def __init__(self):
        self.calls = []
        self.answer = (object(), {"seed": 7, "reference_route_checks": [
            {"steps": 9, "success": False}, {"steps": 3, "success": True}]})

    def reset(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.answer


def test_reset_recorder_preserves_return_identity_arguments_and_counts_rejected_checks(tmp_path):
    env = FakeResetEnvironment()
    recorder = runner.ResetCertificationRecorder(env)
    result = recorder.observe(lambda: env.reset(seed=7, options={"example": True}), 2200)
    assert result is env.answer
    assert env.calls == [((), {"seed": 7, "options": {"example": True}})]
    assert "reset" not in env.__dict__ and env.reset.__func__ is FakeResetEnvironment.reset
    record = recorder.save(tmp_path)
    assert record["completed_reset_count"] == 1 and record["certification_attempts"] == 2
    assert record["certification_environment_steps"] == 12 and record["accounting_complete"]
    assert record["records"][0]["reference_route_checks"] == env.answer[1]["reference_route_checks"]
    assert record["records"][0]["reference_route_checks"] is not env.answer[1]["reference_route_checks"]


def test_reset_recorder_restores_instance_method_even_if_observer_fails():
    env = FakeResetEnvironment()
    custom = lambda **kwargs: env.answer
    env.reset = custom
    recorder = runner.ResetCertificationRecorder(env)

    def broken():
        env.reset(seed=7)
        raise RuntimeError("Observer failed after reset.")

    with pytest.raises(RuntimeError, match="Observer failed"):
        recorder.observe(broken, 2200)
    assert env.reset is custom and len(recorder.records) == 1


def test_failed_reset_marks_unknown_partial_certification_instead_of_claiming_zero(tmp_path):
    env = FakeResetEnvironment()

    def failed(**kwargs):
        raise ValueError("Certification failed before returning info.")

    env.reset = failed
    recorder = runner.ResetCertificationRecorder(env)
    with pytest.raises(ValueError, match="Certification failed"):
        recorder.observe(lambda: env.reset(seed=7), 2200)
    record = recorder.save(tmp_path)
    assert not record["accounting_complete"]
    assert record["incomplete_resets"][0]["certification_steps_unknown"]
    assert env.reset is failed and not record["records"]


def observation(agent, step):
    return {"update": step, "total_updates_on_branch": 40000 + step,
            "optimizer_state": runner.extension.control.optimizer_state_summary(agent.learner),
            "scenes": [{"seed": 10001, "rollout": {"success": True, "steps": 3}}],
            "aggregate": {"evaluation_environment_steps": 3}}


def fake_observer(agent, env, environment, batch, returns, episodes, phase, step, prior, directory):
    assert phase == "extension/td" and prior == 40000
    for episode in episodes:
        env.reset(seed=episode["seed"])
    agent.save(directory / f"update_{step}.pt", environment, {"objective": phase, "updates": step})
    return observation(agent, step)


def test_treatment_clock_target_sources_logging_and_new_work_accounting(tmp_path, monkeypatch):
    agent, env, environment, batch, returns = fixture(2199)
    direct = deepcopy(agent)
    expected = [runner.weighted_td_update(direct.learner, batch) for _ in range(2)]
    monkeypatch.setattr(runner.multi, "observe", fake_observer)
    baseline = {"optimizer_updates": 8000, "scheduled_evaluation_episodes": 87, "evaluation_environment_steps": 3711,
                "new_optimizer_updates": 0, "new_evaluation_episodes": 0, "new_environment_steps": 0}
    try:
        phase = runner.execute_treatment(
            agent, env, environment, batch, returns, [{"seed": 10001, "start": 0, "stop": 3}],
            tmp_path, observation(agent, 2199), start=2199, final=2201,
            checkpoints=(2199, 2201), reference_steps=(1, 2))
        assert runner.read_updates(tmp_path / "treatment/td/updates.jsonl") == [
            {"update": step, "extension_update": index + 1, **stats}
            for index, (step, stats) in enumerate(zip((2200, 2201), expected, strict=True))]
        assert phase["target_source_updates"] == [2200] and phase["target_synchronizations"] == 1
        assert phase["optimizer_updates"] == 2 and phase["td_weight"] == 0.5
        assert phase["return_anchor_weight"] == 0
        assert (tmp_path / "treatment/td/target_source_state_2200.pt").is_file()
        assert not (tmp_path / "treatment/td/update_2200.pt").exists()
        assert not (tmp_path / "baseline").exists()
        saved = torch.load(tmp_path / "treatment/td/training_state.pt", weights_only=True)
        assert runner.exact_equal(saved, runner.adam.training_snapshot(direct, environment))
        accounting = runner.write_accounting(tmp_path, baseline)
        assert accounting["optimizer_updates_total"] == 2 and accounting["absolute_td_update"] == 2201
        assert accounting["scheduled_evaluation_episodes"] == accounting["environment_resets"] == 2
        assert accounting["evaluation_environment_steps"] == accounting["total_simulator_environment_steps"] == 6
        assert accounting["reset_certification_environment_steps"] == 0 and accounting["reset_accounting_complete"]
        assert accounting["sentinel_forward_passes"] == 6
        assert accounting["reused_historical_control"] == baseline
    finally:
        env.close()


def test_initial_observation_mismatch_stops_before_training_and_keeps_evaluation_accounting(tmp_path, monkeypatch):
    agent, env, environment, batch, returns = fixture(2000)
    before = runner.adam.training_snapshot(agent, environment)
    expected = observation(agent, 2000)
    expected["aggregate"]["evaluation_environment_steps"] = 999
    monkeypatch.setattr(runner.multi, "observe", fake_observer)
    try:
        with pytest.raises(ValueError, match="exact source behavior"):
            runner.execute_treatment(agent, env, environment, batch, returns,
                                     [{"seed": 10001, "start": 0, "stop": 3}], tmp_path, expected,
                                     final=2001, checkpoints=(2000, 2001), reference_steps=(1,))
        assert runner.exact_equal(runner.adam.training_snapshot(agent, environment), before)
        assert (tmp_path / "treatment/td/updates.jsonl").read_text() == ""
        recorded = json.loads((tmp_path / "reset_certification.json").read_text())
        assert recorded["completed_reset_count"] == 1
    finally:
        env.close()


def test_protocol_declares_only_td_coefficient_change_and_no_mc():
    protocol = runner.make_protocol({"helper_sha256": {"helper.py": "abc"}})
    assert protocol["attempt"] == 19 and protocol["td_weight"] == 0.5 and protocol["baseline_td_weight"] == 1
    assert protocol["return_anchor_weight"] == protocol["baseline_return_anchor_weight"] == 0
    assert not protocol["mc_loss_constructed"] and protocol["returns_used_only_for_evaluation"]
    assert protocol["learning_rate"] == 0.0001 and protocol["demonstration_weight"] == 1
    assert protocol["effective_batch_size"] == protocol["normalization_denominator"] == 117
    assert len(protocol["td_checkpoints"]) == 29 and len(protocol["sentinel_reference_steps"]) == 7
    assert protocol["preserve_adam_moments_and_parameter_steps"] and not protocol["synchronize_target_at_resume"]
    assert protocol["reset_certification_preserved"] and not protocol["early_stopping"]
