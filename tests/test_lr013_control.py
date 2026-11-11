"""Test the fixed rate-only launch and exact preservation of resumed Adam tensors."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from me5418_harvesting.agents.dqn import DQNConfig
from me5418_harvesting.agents.dqn_agent import DQNAgent
from me5418_harvesting.agents.dqn_learning import TransitionBatch
from me5418_harvesting.envs.harvesting_env import HarvestingEnv

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "lr013_control_test",
    ROOT / "docs/experiments/013_td_learning_rate/run_learning_rate_control.py",
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def protocol():
    value = runner.control.read_json(runner.PRIOR_SETTINGS)["training"]
    value = deepcopy(value)
    value.update(factor="learning_rate", baseline_value=3e-4, treatment_value=1e-4,
                 treatment_settings={"learning_rate": 1e-4, "demonstration_weight": 1.0})
    return value


def test_wrapper_records_identity_before_launch_and_dispatches_only_fixed_rate(tmp_path, monkeypatch):
    captured = []
    identity = {"base_runner_script": runner.BASE_RUNNER.relative_to(ROOT).as_posix(),
                "base_runner_sha256": protocol()["script_sha256"]}
    monkeypatch.setattr(runner, "verify_base_runner", lambda: identity)

    def recorder(directory, environment, dqn, declaration):
        captured.append(deepcopy(declaration))

    def control_run(output, reference, archive, factor, value):
        assert (output, reference, archive) == tuple(
            (tmp_path / name).resolve() for name in ("output", "reference", "archive"))
        assert (factor, value) == ("learning_rate", 1e-4)
        declaration = protocol()
        runner.control.begin_run(output, {}, {}, declaration)
        assert captured[0] == declaration
        return {"protocol": declaration}

    monkeypatch.setattr(runner.control, "begin_run", recorder)
    monkeypatch.setattr(runner.control, "run", control_run)
    result = runner.run(tmp_path / "output", tmp_path / "reference", tmp_path / "archive")
    assert result["protocol"]["attempt"] == 13
    assert result["protocol"]["entrypoint_script"] == (
        "docs/experiments/013_td_learning_rate/run_learning_rate_control.py")
    assert result["protocol"]["entrypoint_sha256"] == runner.control.fit.file_hash(Path(runner.__file__))
    assert result["protocol"]["script_sha256"] == result["protocol"]["base_runner_sha256"]
    assert runner.control.begin_run is recorder


def test_wrapper_restores_metadata_hook_after_failure(tmp_path, monkeypatch):
    original = runner.control.begin_run
    monkeypatch.setattr(runner, "verify_base_runner", lambda: {"base_runner_sha256": "unused"})

    def fail(*args):
        assert runner.control.begin_run is not original
        raise RuntimeError("Synthetic preflight failure")

    monkeypatch.setattr(runner.control, "run", fail)
    with pytest.raises(RuntimeError, match="Synthetic preflight failure"):
        runner.run(tmp_path / "new", tmp_path / "reference", tmp_path / "archive")
    assert runner.control.begin_run is original


def test_existing_output_is_rejected_before_provenance_or_training(tmp_path, monkeypatch):
    def unexpected():
        pytest.fail("Provenance should not run for an existing destination")

    monkeypatch.setattr(runner, "verify_base_runner", unexpected)
    with pytest.raises(FileExistsError, match="overwrite"):
        runner.run(tmp_path, tmp_path / "reference", tmp_path / "archive")


@pytest.mark.parametrize("key,value", [
    ("factor", "demonstration_weight"), ("treatment_value", 3e-5),
    ("baseline_settings", {"learning_rate": 3e-4, "demonstration_weight": 2}),
    ("treatment_settings", {"learning_rate": 1e-4, "demonstration_weight": 2}),
    ("preserve_adam_moments_and_parameter_steps", False), ("td_updates", 2001),
    ("td_checkpoints", [0, 2000]), ("scene_seeds", [10000]),
    ("effective_batch_size", 64), ("early_stopping", True),
])
def test_predeclared_constraints_reject_other_changes_without_mutating_protocol(key, value):
    declaration = protocol()
    digest = declaration["script_sha256"]
    declaration[key] = value
    before = deepcopy(declaration)
    with pytest.raises(ValueError, match=key):
        runner.validate_fixed_protocol(declaration, digest)
    assert declaration == before


def test_source_anchor_allows_git_newlines_but_rejects_changed_training_code(tmp_path, monkeypatch):
    script = tmp_path / "base.py"
    script.write_bytes(b"original implementation\r\n")
    settings = tmp_path / "settings.json"
    record = {"training": {"script_sha256": runner.control.fit.file_hash(script)}}
    settings.write_text(json.dumps(record, indent=2), encoding="utf-8")
    committed = json.dumps(record, separators=(",", ":")).encode()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "BASE_RUNNER", script)
    monkeypatch.setattr(runner, "PRIOR_SETTINGS", settings)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *args, **kwargs: committed)
    identity = runner.verify_base_runner()
    assert identity["base_runner_sha256"] == record["training"]["script_sha256"]
    script.write_bytes(b"changed implementation\r\n")
    with pytest.raises(ValueError, match="implementation has changed"):
        runner.verify_base_runner()


def test_source_anchor_rejects_uncommitted_settings(tmp_path, monkeypatch):
    settings = tmp_path / "settings.json"
    settings.write_text('{"training": {"script_sha256": "modified"}}', encoding="utf-8")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "PRIOR_SETTINGS", settings)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *args, **kwargs: b'{}')
    with pytest.raises(ValueError, match="committed content"):
        runner.verify_base_runner()


def test_reduced_rate_retains_moments_and_scales_first_step_with_identical_gradient(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(39)
    environment = {"max_steps": 3}
    env = HarvestingEnv(environment)
    agent = DQNAgent(env, DQNConfig(hidden_sizes=(8, 8), learning_rate=3e-4,
                                   batch_size=3, target_update_interval=200), seed=39)
    width = env.observation_size
    batch = TransitionBatch(
        observations=torch.randn(3, width), actions=torch.tensor([0, 1, 2]),
        rewards=torch.tensor([0.2, -0.1, 1.0]), next_observations=torch.randn(3, width),
        terminated=torch.tensor([False, False, True]), truncated=torch.zeros(3, dtype=torch.bool),
    )
    try:
        for _ in range(2):
            agent.learner.update(batch, demonstration_weight=1.0)
        snapshot = runner.control.adam.training_snapshot(agent, environment)
    finally:
        env.close()
    forks, environments, results = [], [], []
    try:
        for label in ("baseline", "treatment"):
            fork, fork_env = runner.control.adam.restore_training_snapshot(snapshot)
            forks.append(fork)
            environments.append(fork_env)
            runner.control.multi.start_phase(fork, True, snapshot["parameter_sha256"], 2,
                                              tmp_path / f"{label}.json")
            before = runner.control.adam.training_snapshot(fork, environment)
            if label == "treatment":
                runner.control.apply_factor(fork, "learning_rate", runner.TREATMENT_LR)
            after = runner.control.adam.training_snapshot(fork, environment)
            assert runner.control.exact_equal(before["optimizer"]["state"], after["optimizer"]["state"])
            for field in ("online", "target", "agent_rng_state", "torch_rng_state"):
                assert runner.control.exact_equal(before[field], after[field])
            results.append(fork.learner.update(batch, demonstration_weight=1.0))
        assert results[0].gradient_norm == results[1].gradient_norm
        assert results[0].loss_before == results[1].loss_before
        assert results[1].parameter_delta_norm == pytest.approx(results[0].parameter_delta_norm / 3,
                                                               rel=3e-4, abs=2e-8)
        assert runner.control.exact_equal(forks[0].learner.optimizer.state_dict()["state"],
                                          forks[1].learner.optimizer.state_dict()["state"])
        assert all(group["lr"] == 1e-4 for group in forks[1].learner.optimizer.param_groups)
        assert forks[1].learner.optimizer.defaults["lr"] == 1e-4
        assert forks[1].learner.config.learning_rate == 1e-4
    finally:
        for environment in environments:
            environment.close()
