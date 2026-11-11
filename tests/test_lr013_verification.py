"""Reject undeclared controls and hidden state changes before learning-rate verification."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from me5418_harvesting.agents.dqn import DQNConfig
from me5418_harvesting.agents.dqn_agent import DQNAgent
from me5418_harvesting.envs.harvesting_env import HarvestingEnv

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "lr013_verification_test", ROOT / "docs/experiments/013_td_learning_rate/verify_learning_rate_control.py",
)
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


@pytest.fixture
def protocol(tmp_path, monkeypatch):
    monkeypatch.setattr(verifier, "ROOT", tmp_path)
    entrypoint, runner = tmp_path / "entry.py", tmp_path / "base.py"
    entrypoint.write_text("# Fixed learning-rate entrypoint.\n")
    runner.write_text("# Immutable control runner.\n")
    record = {
        "attempt": 13, "factor": "learning_rate", "baseline_value": 0.0003,
        "treatment_value": 0.0001,
        "baseline_settings": {"learning_rate": 0.0003, "demonstration_weight": 1.0},
        "treatment_settings": {"learning_rate": 0.0001, "demonstration_weight": 1.0},
        "arms": ["baseline", "treatment"], "scene_seeds": [10000, 10001, 10002],
        "td_updates": 2000, "td_checkpoints": [*range(11), 20, 50, 100, 200, 500, 1000, 2000],
        "margin": 0.8, "effective_batch_size": 117, "full_batch_weighting": "equal_per_transition",
        "inherited_optimizer_updates": 40000, "shared_start": "attempt_011_calibration_training_state",
        "td_boundary": "synchronized_targets_zero_phase_clock_before_scalar_intervention",
        "baseline_gate": "exact_all_updates_observations_and_final_state",
        "online_training_steps": 0, "held_out_scenes_accessed": 0,
        "entrypoint_script": "entry.py", "base_runner_script": "base.py",
        "entrypoint_sha256": verifier.file_hash(entrypoint),
        "base_runner_sha256": verifier.file_hash(runner), "script_sha256": verifier.file_hash(runner),
        "preserve_adam_moments_and_parameter_steps": True, "early_stopping": False,
    }
    return record, entrypoint, runner


@pytest.mark.parametrize("mutation", [
    "weight", "rate", "factor", "budget", "schedule", "scenes", "reset_adam", "entrypoint", "runner",
])
def test_fixed_protocol_rejects_undeclared_comparisons(protocol, mutation):
    record, entrypoint, runner = protocol
    assert verifier.verify_protocol(record, entrypoint, runner)["attempt"] == 13
    if mutation == "weight":
        record["treatment_settings"]["demonstration_weight"] = 2.0
    elif mutation == "rate":
        record["treatment_value"] = record["treatment_settings"]["learning_rate"] = 0.00003
    elif mutation == "factor":
        record["factor"] = "demonstration_weight"
    elif mutation == "budget":
        record["td_updates"] = 1000
    elif mutation == "schedule":
        record["td_checkpoints"].remove(50)
    elif mutation == "scenes":
        record["scene_seeds"][1] = 10003
    elif mutation == "reset_adam":
        record["preserve_adam_moments_and_parameter_steps"] = False
    else:
        (entrypoint if mutation == "entrypoint" else runner).write_text("# Changed after launch.\n")
    with pytest.raises(ValueError, match="Verification failed"):
        verifier.verify_protocol(record, entrypoint, runner)


@pytest.fixture
def initialized_snapshot():
    """Create explicit synthetic Adam tensors without performing a training update."""
    torch.set_num_threads(1)
    env = HarvestingEnv({"max_steps": 3})
    agent = DQNAgent(env, DQNConfig(hidden_sizes=(8, 8)), seed=17)
    for parameter in agent.learner.online.parameters():
        agent.learner.optimizer.state[parameter] = {
            "step": torch.tensor(20000.0),
            "exp_avg": torch.full_like(parameter, 0.01),
            "exp_avg_sq": torch.full_like(parameter, 0.04),
        }
    agent.learner.update_count = 20000
    snapshot = verifier.base.multi.diagnostic.adam.training_snapshot(agent, {"max_steps": 3})
    settings = {"environment": snapshot["environment_config"],
                "dqn": json.loads(json.dumps(snapshot["dqn_config"]))}
    try:
        yield snapshot, settings
    finally:
        env.close()


def make_boundary(snapshot, treatment):
    payload = deepcopy(snapshot)
    payload["update_count"] = 0
    payload["target"] = deepcopy(snapshot["online"])
    if treatment:
        payload["dqn_config"]["learning_rate"] = 0.0001
        for group in payload["optimizer"]["param_groups"]:
            group["lr"] = 0.0001
    agent, env = verifier.base.audit.independent_restore(payload)
    try:
        payload["optimizer_summary"] = verifier.optimizer_state_summary(agent.learner)
    finally:
        env.close()
    return payload


@pytest.mark.parametrize("treatment", [False, True])
def test_lr_boundary_keeps_moments_steps_weights_and_rng(initialized_snapshot, treatment):
    calibrated, settings = initialized_snapshot
    payload = make_boundary(calibrated, treatment)
    result = verifier.verify_boundary_payload(payload, calibrated, settings, treatment)
    assert result["learning_rate"] == (0.0001 if treatment else 0.0003)
    assert result["adam_parameter_step"] == 20000
    assert result["optimizer_updates_performed"] == result["environment_steps"] == result["environment_resets"] == 0


@pytest.mark.parametrize("mutation", ["first_moment", "second_moment", "step", "group_rate", "weights"])
def test_lr_boundary_rejects_undeclared_state_changes(initialized_snapshot, mutation):
    calibrated, settings = initialized_snapshot
    payload = make_boundary(calibrated, True)
    if mutation in ("first_moment", "second_moment", "step"):
        key = {"first_moment": "exp_avg", "second_moment": "exp_avg_sq", "step": "step"}[mutation]
        next(iter(payload["optimizer"]["state"].values()))[key].add_(1)
    elif mutation == "group_rate":
        payload["optimizer"]["param_groups"][0]["lr"] = 0.0003
    else:
        next(iter(payload["online"].values())).add_(1)
    with pytest.raises(ValueError, match="Verification failed"):
        verifier.verify_boundary_payload(payload, calibrated, settings, True)


@pytest.mark.parametrize("existing", ["outer", "base"])
def test_existing_evidence_blocks_replay_before_preflight(tmp_path, monkeypatch, existing):
    output = tmp_path / "reload_checks.json"
    selected = output if existing == "outer" else tmp_path / "reload_checks_base.json"
    selected.write_text("{}")

    def unexpected(_):
        raise AssertionError("Preflight must not run when evidence already exists.")

    monkeypatch.setattr(verifier, "preflight", unexpected)
    with pytest.raises(FileExistsError, match="overwrite"):
        verifier.verify(tmp_path, output)
