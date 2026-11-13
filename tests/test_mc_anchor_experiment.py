"""Validate the isolated return objective, complete state continuity and source gates."""

import importlib.util
import json
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
    "anchor017_test", ROOT / "docs/experiments/017_mc_anchor/run_mc_anchor.py",
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def fixture(counter=2199):
    torch.set_num_threads(1)
    torch.manual_seed(73)
    environment = {"max_steps": 3}
    env = HarvestingEnv(environment)
    agent = DQNAgent(env, DQNConfig(hidden_sizes=(8, 8), learning_rate=1e-4,
                                   batch_size=3, target_update_interval=200), seed=73)
    batch = TransitionBatch(
        observations=torch.randn(3, env.observation_size), actions=torch.tensor([0, 1, 2]),
        rewards=torch.tensor([0.2, -0.1, 1.0]), next_observations=torch.randn(3, env.observation_size),
        terminated=torch.tensor([False, False, True]), truncated=torch.zeros(3, dtype=torch.bool),
    )
    for _ in range(2):
        agent.learner.update(batch, demonstration_weight=1)
    agent.learner.update_count = counter
    return agent, env, environment, batch, torch.tensor([1.3, 2.2, 1.0])


def independent_step(learner, batch, returns):
    """Construct the same objective from its definition, without production loss helpers."""
    learner.online.train()
    learner.optimizer.zero_grad(set_to_none=True)
    chosen = learner.online(batch.observations).gather(1, batch.actions[:, None]).squeeze(1)
    with torch.no_grad():
        future = learner.target(batch.next_observations).max(dim=1).values
        target = batch.rewards + learner.config.gamma * torch.where(batch.terminated, 0.0, future)
    td = torch.nn.functional.smooth_l1_loss(chosen, target)
    q = learner.online(batch.observations)
    offset = torch.full_like(q, 0.8)
    offset[torch.arange(len(q)), batch.actions] = 0.0
    margin = ((q + offset).max(dim=1).values - q.gather(1, batch.actions[:, None]).squeeze(1)).mean()
    mc = torch.nn.functional.smooth_l1_loss(chosen, returns.detach())
    total = td + margin + mc
    total.backward()
    preclip = torch.nn.utils.clip_grad_norm_(learner.online.parameters(), learner.config.gradient_clip_norm)
    gradients = [parameter.grad.detach().clone() for parameter in learner.online.parameters()]
    learner.optimizer.step()
    learner.update_count += 1
    if learner.update_count % learner.config.target_update_interval == 0:
        learner.target.load_state_dict(learner.online.state_dict())
    return float(td.detach()), float(margin.detach()), float(mc.detach()), float(preclip), gradients, target


def test_zero_anchor_matches_original_state_and_logs_across_target_sync():
    source, env, environment, batch, returns = fixture()
    original, original_env = runner.extension.restore_continuation(runner.adam.training_snapshot(source, environment))
    anchored, anchored_env = runner.extension.restore_continuation(runner.adam.training_snapshot(source, environment))
    try:
        for step in (2200, 2201, 2202):
            expected = asdict(original.learner.update(batch, demonstration_weight=1, demonstration_margin=0.8))
            actual = runner.anchored_update(anchored.learner, batch, returns, 0)
            assert {key: actual[key] for key in expected} == expected
            assert actual["return_anchor_weight"] == 0
            assert actual["target_synchronized"] == (step == 2200)
            assert runner.exact_equal(runner.adam.training_snapshot(original, environment),
                                      runner.adam.training_snapshot(anchored, environment))
    finally:
        env.close()
        original_env.close()
        anchored_env.close()


def test_anchor_matches_independent_gradients_states_and_frozen_target_diagnostics():
    source, env, environment, batch, returns = fixture()
    returns = returns.clone().requires_grad_()
    other, other_env = runner.extension.restore_continuation(runner.adam.training_snapshot(source, environment))
    try:
        for step in (2200, 2201):
            td, margin, mc, norm, gradients, targets = independent_step(other.learner, batch, returns)
            actual = runner.anchored_update(source.learner, batch, returns)
            assert actual["td_loss_before"] == td
            assert actual["demonstration_loss_before"] == margin
            assert actual["return_loss_before"] == mc
            assert actual["loss_before"] == pytest.approx(td + margin + mc, abs=2e-7)
            assert actual["gradient_norm"] == norm
            assert all(torch.equal(p.grad, g) for p, g in zip(source.learner.online.parameters(), gradients, strict=True))
            assert runner.exact_equal(runner.adam.training_snapshot(source, environment),
                                      runner.adam.training_snapshot(other, environment))
            diff = targets.double() - returns.detach().double()
            assert actual["td_target_return_rmse"] == float(diff.square().mean().sqrt())
            assert actual["td_target_return_mean_difference"] == float(diff.mean())
            assert actual["td_target_return_max_abs_difference"] == float(diff.abs().max())
            with torch.no_grad():
                chosen_after = source.learner.online(batch.observations).gather(1, batch.actions[:, None]).squeeze(1)
                assert actual["td_loss_after"] == float(torch.nn.functional.smooth_l1_loss(chosen_after, targets))
                assert actual["return_loss_after"] == float(torch.nn.functional.smooth_l1_loss(chosen_after, returns.detach()))
            assert actual["target_synchronized"] == (step == 2200)
            assert returns.grad is None
    finally:
        env.close()
        other_env.close()


@pytest.mark.parametrize("weight", [-1, float("nan"), float("inf"), True, "1"])
def test_invalid_weight_rejected_before_state_modes_or_gradients_change(weight):
    agent, env, environment, batch, returns = fixture()
    agent.learner.online.eval()
    before = runner.adam.training_snapshot(agent, environment)
    gradients = [p.grad.clone() for p in agent.learner.online.parameters()]
    try:
        with pytest.raises(ValueError, match="weight"):
            runner.anchored_update(agent.learner, batch, returns, weight)
        assert runner.exact_equal(runner.adam.training_snapshot(agent, environment), before)
        assert not agent.learner.online.training
        assert all(torch.equal(p.grad, g) for p, g in zip(agent.learner.online.parameters(), gradients, strict=True))
    finally:
        env.close()


@pytest.mark.parametrize("kind", ["empty", "matrix", "short", "integer", "double", "nan", "infinite", "list"])
def test_invalid_return_targets_rejected_before_mutation(kind):
    agent, env, environment, batch, returns = fixture()
    invalid = {"empty": torch.empty(0), "matrix": returns[:, None], "short": returns[:2],
               "integer": returns.long(), "double": returns.double(), "nan": torch.full_like(returns, float("nan")),
               "infinite": torch.full_like(returns, float("inf")), "list": returns.tolist()}[kind]
    before = runner.adam.training_snapshot(agent, environment)
    try:
        with pytest.raises(ValueError, match="Fixed returns"):
            runner.anchored_update(agent.learner, batch, invalid)
        assert runner.exact_equal(runner.adam.training_snapshot(agent, environment), before)
    finally:
        env.close()


def observation(agent, step):
    return {"update": step, "total_updates_on_branch": 40000 + step,
            "optimizer_state": runner.extension.control.optimizer_state_summary(agent.learner),
            "scenes": [{"seed": seed, "rollout": {"success": True, "steps": 3}} for seed in (10000, 10001, 10002)],
            "aggregate": {"evaluation_environment_steps": 9}}


def fake_observer(agent, env, environment, batch, returns, episodes, phase, step, prior, directory):
    assert phase == "extension/td" and prior == 40000
    agent.save(directory / f"update_{step}.pt", environment, {"objective": phase, "updates": step})
    return observation(agent, step)


def test_treatment_logging_absolute_clock_target_source_and_new_work_accounting(tmp_path, monkeypatch):
    agent, env, environment, batch, returns = fixture(2199)
    direct, direct_env = runner.extension.restore_continuation(runner.adam.training_snapshot(agent, environment))
    expected = [runner.anchored_update(direct.learner, batch, returns) for _ in range(2)]
    monkeypatch.setattr(runner.multi, "observe", fake_observer)
    baseline = {"optimizer_updates": 8000, "scheduled_evaluation_episodes": 87, "evaluation_environment_steps": 3711,
                "new_optimizer_updates": 0, "new_evaluation_episodes": 0, "new_environment_steps": 0}
    try:
        phase = runner.execute_treatment(
            agent, env, environment, batch, returns, [{"seed": 10001, "start": 0, "stop": 3}],
            tmp_path, observation(agent, 2199), start=2199, final=2201,
            checkpoints=(2199, 2201), reference_steps=(1, 2),
        )
        rows = runner.read_updates(tmp_path / "treatment/td/updates.jsonl")
        assert rows == [{"update": step, "extension_update": i + 1, **metrics}
                        for i, (step, metrics) in enumerate(zip((2200, 2201), expected, strict=True))]
        assert phase["target_source_updates"] == [2200]
        assert phase["target_synchronizations"] == 1
        assert phase["optimizer_updates"] == 2
        assert (tmp_path / "treatment/td/target_source_state_2200.pt").is_file()
        assert not (tmp_path / "treatment/td/update_2200.pt").exists()
        assert not (tmp_path / "baseline").exists()
        saved = torch.load(tmp_path / "treatment/td/training_state.pt", weights_only=True)
        assert runner.exact_equal(saved, runner.adam.training_snapshot(direct, environment))
        accounting = runner.write_accounting(tmp_path, baseline)
        assert accounting["optimizer_updates_total"] == 2
        assert accounting["scheduled_evaluation_episodes"] == 6
        assert accounting["evaluation_environment_steps"] == 18
        assert accounting["reused_historical_control"] == baseline
    finally:
        env.close()
        direct_env.close()


def test_initial_policy_mismatch_stops_before_any_training(tmp_path, monkeypatch):
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
    finally:
        env.close()


def archive_fixture(tmp_path, monkeypatch):
    reference, archive = tmp_path / "raw", tmp_path / "evidence"
    reference.mkdir()
    archive.mkdir()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "START", 2)
    monkeypatch.setattr(runner, "END", 4)
    files = {}
    for index in range(30):
        name = f"evidence_{index}.json"
        content = json.dumps({"value": index}).encode()
        (archive / name).write_bytes(content)
        (reference / name).write_bytes(content)
        files[name] = {"sha256": runner.fit.file_hash(archive / name), "bytes": len(content)}
    coverage = {}
    for arm in ("baseline", "treatment"):
        directory = reference / arm / "td"
        directory.mkdir(parents=True)
        for name in ("updates.jsonl", "supervision.jsonl", "sentinel_q.jsonl"):
            rows = [{"update": step} for step in range(2 if name == "sentinel_q.jsonl" else 3, 5)]
            path = directory / name
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            coverage[f"{arm}/td/{name}"] = {"raw_sha256": runner.fit.file_hash(path), "raw_records": len(rows)}
    manifest = {"status": "completed", "source_run": str(reference), "files": files, "raw_log_coverage": coverage}
    (archive / "archive_manifest.json").write_text(json.dumps(manifest))
    committed = (archive / "archive_manifest.json").read_bytes()
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *args, **kwargs: committed)
    return reference, archive


@pytest.mark.parametrize("location", ["archive", "raw", "log", "manifest"])
def test_committed_archive_gate_rejects_evidence_and_log_tampering(tmp_path, monkeypatch, location):
    reference, archive = archive_fixture(tmp_path, monkeypatch)
    result = runner.verify_archive(reference, archive)
    assert result["committed_archive_verified"]
    assert len(result["archived_files_sha256"]) == 30
    target = {"archive": archive / "evidence_0.json", "raw": reference / "evidence_0.json",
              "log": reference / "baseline/td/updates.jsonl", "manifest": archive / "archive_manifest.json"}[location]
    if location == "manifest":
        data = json.loads(target.read_text())
        data["status"] = "failed"
        target.write_text(json.dumps(data))
    else:
        target.write_text("{}")
    with pytest.raises(ValueError):
        runner.verify_archive(reference, archive)


def test_declared_schedule_and_historical_control_do_not_add_a_second_main_run():
    protocol = runner.make_protocol({"helper_sha256": {}})
    assert protocol["historical_control_reused_not_rerun"]
    assert protocol["new_treatment_optimizer_updates"] == 8000
    assert protocol["all_sample_weights"] == protocol["demonstration_weight"] == protocol["return_anchor_weight"] == 1
    assert protocol["normalization_denominator"] == 117
    assert protocol["total_loss_divided_by_three"] is False
    assert protocol["td_targets_mixed_with_returns"] is False
    assert len(protocol["td_checkpoints"]) == 29
    model_count = 2 * len(protocol["td_checkpoints"]) + len(protocol["target_source_updates"]) + 1
    assert model_count == 67


def test_existing_output_is_rejected_before_loading_inputs(tmp_path, monkeypatch):
    def forbidden(*args):
        raise AssertionError("Existing output must be rejected before source loading.")

    monkeypatch.setattr(runner, "load_inputs", forbidden)
    with pytest.raises(FileExistsError):
        runner.run(tmp_path, tmp_path, tmp_path)


def test_changing_fixed_returns_changes_the_training_update():
    source, env, environment, batch, returns = fixture()
    other, other_env = runner.extension.restore_continuation(runner.adam.training_snapshot(source, environment))
    try:
        first = runner.anchored_update(source.learner, batch, returns)
        second = runner.anchored_update(other.learner, batch, returns - 10)
        assert first["td_loss_before"] == second["td_loss_before"]
        assert first["demonstration_loss_before"] == second["demonstration_loss_before"]
        assert first["return_loss_before"] != second["return_loss_before"]
        assert not runner.exact_equal(source.learner.online.state_dict(), other.learner.online.state_dict())
    finally:
        env.close()
        other_env.close()
