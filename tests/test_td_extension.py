"""Check exact continuation, absolute target clocks and read-only sentinel probes."""

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
    "td_extension_test", ROOT / "docs/experiments/014_td_extension/run_td_extension.py",
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def synthetic_fixture(counter=2199):
    """Build initialized Adam state from tensors without collecting any environment data."""
    torch.set_num_threads(1)
    torch.manual_seed(47)
    environment = {"max_steps": 3}
    env = HarvestingEnv(environment)
    agent = DQNAgent(env, DQNConfig(hidden_sizes=(8, 8), learning_rate=1e-4,
                                   batch_size=3, target_update_interval=200), seed=47)
    width = env.observation_size
    batch = TransitionBatch(
        observations=torch.randn(3, width), actions=torch.tensor([0, 1, 2]),
        rewards=torch.tensor([0.2, -0.1, 1.0]), next_observations=torch.randn(3, width),
        terminated=torch.tensor([False, False, True]), truncated=torch.zeros(3, dtype=torch.bool),
    )
    for _ in range(2):
        agent.learner.update(batch, demonstration_weight=1.0)
    agent.learner.update_count = counter
    episodes = [{"seed": 10001, "start": 0, "stop": 3}]
    return agent, env, environment, batch, episodes


def observation(agent, step):
    return {
        "update": step, "total_updates_on_branch": runner.PRIOR_PHASE_UPDATES + step,
        "optimizer_state": runner.control.optimizer_state_summary(agent.learner),
        "scenes": [{"seed": seed, "rollout": {"success": True, "steps": 3}}
                   for seed in (10000, 10001, 10002)],
        "aggregate": {"evaluation_environment_steps": 9},
    }


def fake_observer(agent, env, environment, batch, returns, episodes, phase, step, prior, directory):
    assert phase == "extension/td"
    assert prior == 40000
    agent.save(directory / f"update_{step}.pt", environment, {"objective": phase, "updates": step})
    return observation(agent, step)


def test_fixed_schedule_and_target_sources_cover_absolute_clock():
    assert runner.CHECKPOINTS == (
        2000, *range(2001, 2011), 2200, 2201, *range(2500, 10001, 500))
    assert len(runner.CHECKPOINTS) == 29
    assert len(runner.CHECKPOINTS) * 3 == 87
    assert runner.TARGET_SOURCE_UPDATES == tuple(range(2400, 9401, 1000))
    assert runner.EXTENSION_UPDATES == 8000
    assert runner.INHERITED_UPDATES == 42000
    for step in runner.CHECKPOINTS:
        assert step - step % 200 in set(runner.CHECKPOINTS) | set(runner.TARGET_SOURCE_UPDATES)


def test_restore_keeps_unsynchronized_target_clock_and_independent_adam_storage():
    agent, env, environment, batch, _ = synthetic_fixture()
    snapshot = runner.adam.training_snapshot(agent, environment)
    unchanged = deepcopy(snapshot)
    restored, restored_env = runner.restore_continuation(snapshot)
    try:
        assert restored.learner.update_count == 2199
        assert not runner.exact_equal(snapshot["online"], snapshot["target"])
        assert runner.exact_equal(runner.adam.training_snapshot(restored, environment), snapshot)
        for state, source in zip(restored.learner.optimizer.state.values(), snapshot["optimizer"]["state"].values()):
            for name in ("step", "exp_avg", "exp_avg_sq"):
                assert state[name].data_ptr() != source[name].data_ptr()
        first = restored.learner.update(batch, demonstration_weight=1.0)
        assert restored.learner.update_count == 2200
        assert first.target_synchronized
        assert runner.exact_equal(restored.learner.online.state_dict(), restored.learner.target.state_dict())
        assert not restored.learner.update(batch, demonstration_weight=1.0).target_synchronized
        assert runner.exact_equal(snapshot, unchanged)
        assert runner.exact_equal(runner.adam.training_snapshot(agent, environment), unchanged)
    finally:
        env.close()
        restored_env.close()


def test_extension_matches_uninterrupted_updates_across_target_sync(tmp_path, monkeypatch):
    agent, env, environment, batch, episodes = synthetic_fixture()
    snapshot = runner.adam.training_snapshot(agent, environment)
    expected = observation(agent, 2199)
    direct, direct_env = runner.restore_continuation(snapshot)
    expected_updates = [asdict(direct.learner.update(batch, demonstration_weight=1.0)) for _ in range(2)]
    expected_final = runner.adam.training_snapshot(direct, environment)
    monkeypatch.setattr(runner.multi, "observe", fake_observer)
    try:
        result = runner.execute_extension(
            agent, env, environment, batch, torch.zeros(3), episodes, tmp_path, expected,
            start=2199, final=2201, checkpoints=(2199, 2200, 2201), reference_steps=(1, 2),
        )
        records = runner.read_updates(tmp_path / "td/updates.jsonl")
        assert records == [{"update": step, "extension_update": index + 1, **metrics}
                           for index, (step, metrics) in enumerate(zip((2200, 2201), expected_updates))]
        assert runner.exact_equal(torch.load(tmp_path / "td/training_state.pt", weights_only=True), expected_final)
        assert result["optimizer_updates"] == 2
        assert result["initial_td_update"] == 2199
        assert result["final_td_update"] == 2201
        assert result["target_synchronizations"] == 1
        assert result["total_updates_on_branch"] == 42201
        assert result["final_observation"]["optimizer_state"]["step_max"] == 4
        assert result["evaluation_environment_steps"] == 27
        probes = runner.read_updates(tmp_path / "td/sentinel_q.jsonl")
        assert [row["update"] for row in probes] == [2199, 2200, 2201]
        assert [row["extension_update"] for row in probes] == [0, 1, 2]
        assert runner.read_json(tmp_path / "initial_equivalence.json")["passed"]
    finally:
        env.close()
        direct_env.close()


def test_missing_target_generation_is_saved_without_an_extra_rollout(tmp_path, monkeypatch):
    agent, env, environment, batch, episodes = synthetic_fixture(2398)
    calls = []

    def record_observer(*args):
        calls.append(args[7])
        return fake_observer(*args)

    monkeypatch.setattr(runner.multi, "observe", record_observer)
    try:
        result = runner.execute_extension(
            agent, env, environment, batch, torch.zeros(3), episodes, tmp_path, observation(agent, 2398),
            start=2398, final=2401, checkpoints=(2398, 2399, 2401), reference_steps=(1,),
        )
        assert calls == [2398, 2399, 2401]
        assert result["target_source_updates"] == [2400]
        source = torch.load(tmp_path / "td/target_source_state_2400.pt", weights_only=True)
        final = torch.load(tmp_path / "td/training_state_2401.pt", weights_only=True)
        assert source["update_count"] == 2400
        assert runner.exact_equal(source["online"], final["target"])
        assert not (tmp_path / "td/update_2400.pt").exists()
    finally:
        env.close()


def test_initial_behavior_gate_stops_before_any_new_update(tmp_path, monkeypatch):
    agent, env, environment, batch, episodes = synthetic_fixture()
    before = runner.adam.training_snapshot(agent, environment)
    expected = observation(agent, 2199)
    expected["aggregate"]["evaluation_environment_steps"] = 99
    monkeypatch.setattr(runner.multi, "observe", fake_observer)
    try:
        with pytest.raises(RuntimeError, match="previous final observation"):
            runner.execute_extension(
                agent, env, environment, batch, torch.zeros(3), episodes, tmp_path, expected,
                start=2199, final=2200, checkpoints=(2199, 2200), reference_steps=(1,),
            )
        assert runner.exact_equal(runner.adam.training_snapshot(agent, environment), before)
        assert (tmp_path / "td/updates.jsonl").read_text() == ""
        assert not runner.read_json(tmp_path / "initial_equivalence.json")["passed"]
    finally:
        env.close()


def test_sentinel_preserves_rng_models_optimizer_and_mixed_module_modes():
    agent, env, environment, batch, episodes = synthetic_fixture()
    agent.learner.online.train()
    list(agent.learner.online.modules())[-1].eval()
    modes = [module.training for module in agent.learner.online.modules()]
    before = runner.adam.training_snapshot(agent, environment)
    try:
        result = runner.sentinel_record(agent, batch, episodes, 2200, reference_steps=(1, 2, 3))
        assert runner.exact_equal(runner.adam.training_snapshot(agent, environment), before)
        assert [module.training for module in agent.learner.online.modules()] == modes
        assert result["update"] == 2200
        assert result["extension_update"] == 200
        assert result["scene_seed"] == 10001
        for index, row in enumerate(result["states"]):
            q = torch.tensor(row["q_values"], dtype=torch.float32)
            action = row["demonstrated_action"]
            assert row["global_transition_index"] == index
            assert row["greedy_action"] == int(q.argmax())
            assert row["signed_demonstration_gap"] == float(q[action]) - float(torch.cat((q[:action], q[action + 1:])).max())
    finally:
        env.close()


def test_invalid_sentinel_step_restores_model_modes():
    agent, env, _, batch, episodes = synthetic_fixture()
    modes = [module.training for module in agent.learner.online.modules()]
    try:
        with pytest.raises(ValueError, match="outside"):
            runner.sentinel_record(agent, batch, episodes, 2000, reference_steps=(4,))
        assert [module.training for module in agent.learner.online.modules()] == modes
    finally:
        env.close()


def test_accounting_keeps_inherited_and_new_updates_separate(tmp_path):
    (tmp_path / "td").mkdir()
    (tmp_path / "td/updates.jsonl").write_text('{"update":2001}\n{"update":2002}\n')
    (tmp_path / "td/sentinel_q.jsonl").write_text('{}\n{}\n{}\n')
    runner.fit.save_json(tmp_path / "td/observations.json", [
        {"aggregate": {"evaluation_environment_steps": 117}, "scenes": [{}, {}, {}]},
        {"aggregate": {"evaluation_environment_steps": 119}, "scenes": [{}, {}, {}]},
    ])
    result = runner.write_accounting(tmp_path)
    assert result["optimizer_updates_total"] == 2
    assert result["inherited_optimizer_updates"] == 42000
    assert result["total_updates_on_branch"] == 42002
    assert result["absolute_td_update"] == 2002
    assert result["evaluation_environment_steps"] == 236
    assert result["scheduled_evaluation_episodes"] == 6
    assert result["sentinel_records"] == 3
    assert result["sentinel_forward_passes"] == 12
    assert result["sentinel_environment_steps"] == 0


def test_existing_destination_is_rejected_before_reading_inputs(tmp_path, monkeypatch):
    def unexpected(*args):
        pytest.fail("Existing output must be rejected before input verification")

    monkeypatch.setattr(runner, "load_inputs", unexpected)
    with pytest.raises(FileExistsError, match="overwrite"):
        runner.run(tmp_path, tmp_path / "reference", tmp_path / "archive")


@pytest.mark.parametrize("change", [
    "td_clock", "adam_clock", "learning_rate", "optimizer_rate", "target_interval",
    "supervision_weight", "branch_count", "failed_scene", "parameter_hash",
])
def test_resume_contract_rejects_changed_state_or_settings(change):
    agent, env, environment, _, _ = synthetic_fixture(2000)
    try:
        for state in agent.learner.optimizer.state.values():
            state["step"].fill_(22000)
        snapshot = runner.adam.training_snapshot(agent, environment)
    finally:
        env.close()
    settings = {"environment": environment, "dqn": {**snapshot["dqn_config"], "learning_rate": 3e-4}}
    phase = {
        "optimizer_updates": 2000, "total_updates_on_branch": 42000,
        "learning_rate": 1e-4, "demonstration_weight": 1.0,
        "final_parameter_sha256": snapshot["parameter_sha256"],
        "final_observation": {"optimizer_state": deepcopy(snapshot["optimizer_summary"]),
                              "scenes": [{"rollout": {"success": True}}] * 3},
    }
    runner.validate_resume_snapshot(snapshot, settings, phase)
    if change == "td_clock":
        snapshot["update_count"] = 0
    elif change == "adam_clock":
        snapshot["optimizer_summary"]["step_min"] = 0
    elif change == "learning_rate":
        snapshot["dqn_config"]["learning_rate"] = 3e-4
    elif change == "optimizer_rate":
        snapshot["optimizer"]["param_groups"][0]["lr"] = 3e-4
    elif change == "target_interval":
        snapshot["dqn_config"]["target_update_interval"] = 100
    elif change == "supervision_weight":
        phase["demonstration_weight"] = 2.0
    elif change == "branch_count":
        phase["total_updates_on_branch"] = 22000
    elif change == "failed_scene":
        phase["final_observation"]["scenes"][0]["rollout"]["success"] = False
    elif change == "parameter_hash":
        phase["final_parameter_sha256"] = "changed"
    with pytest.raises(ValueError, match="exact successful attempt-013"):
        runner.validate_resume_snapshot(snapshot, settings, phase)


def test_archive_gate_accepts_git_newlines_but_rejects_changed_evidence(tmp_path, monkeypatch):
    archive, reference = tmp_path / "archive", tmp_path / "reference"
    archive.mkdir()
    reference.mkdir()
    for directory in (archive, reference):
        (directory / "settings.json").write_bytes(b'{"setting":1}\r\n')
    manifest = {"status": "completed", "files": {
        "settings.json": {"sha256": runner.fit.file_hash(archive / "settings.json")}},
        "update_log_coverage": {}}
    runner.fit.save_json(archive / "archive_manifest.json", manifest)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *args, **kwargs: json.dumps(manifest).encode())
    assert runner.verify_archive(reference, archive)["committed_archive_verified"]
    (reference / "settings.json").write_bytes(b'{"setting":2}\r\n')
    with pytest.raises(ValueError, match="reference evidence"):
        runner.verify_archive(reference, archive)


def test_archive_gate_rejects_uncommitted_manifest(tmp_path, monkeypatch):
    runner.fit.save_json(tmp_path / "archive_manifest.json", {"status": "completed"})
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *args, **kwargs: b'{}')
    with pytest.raises(ValueError, match="committed evidence"):
        runner.verify_archive(tmp_path, tmp_path)
