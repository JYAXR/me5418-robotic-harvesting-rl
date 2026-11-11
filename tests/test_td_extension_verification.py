"""Reject hidden state changes and misleading continuation diagnostic records."""

import importlib.util
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
    "td_extension_verification_test", ROOT / "docs/experiments/014_td_extension/verify_td_extension.py",
)
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


@pytest.fixture
def snapshot():
    torch.set_num_threads(1)
    torch.manual_seed(24)
    environment = {"max_steps": 3}
    env = HarvestingEnv(environment)
    agent = DQNAgent(env, DQNConfig(hidden_sizes=(8, 8), batch_size=3,
                                    learning_rate=0.0001, target_update_interval=200), seed=24)
    batch = TransitionBatch(
        observations=torch.randn(3, env.observation_size), actions=torch.tensor([0, 1, 2]),
        rewards=torch.tensor([1.0, 2.0, 3.0]), next_observations=torch.randn(3, env.observation_size),
        terminated=torch.tensor([False, False, True]), truncated=torch.tensor([False, False, False]),
    )
    agent.learner.update(batch, demonstration_weight=1.0)
    agent.learner.update_count = 2073
    payload = verifier.multi.diagnostic.adam.training_snapshot(agent, environment)
    assert any(not torch.equal(payload["online"][name], tensor) for name, tensor in payload["target"].items())
    try:
        yield payload
    finally:
        env.close()


def test_resume_preserves_off_period_target_clock_and_both_rng_states(snapshot):
    result = verifier.verify_resume_payload(deepcopy(snapshot), snapshot)
    assert result["full_snapshot_matches_exactly"]
    assert not result["target_resynchronized"]
    assert not result["counter_reset"]
    agent, env = verifier.independent_restore(snapshot)
    try:
        assert agent.learner.update_count == 2073
        assert agent.learner.optimizer.defaults["lr"] == 0.0001
        verifier.require_equal(torch.get_rng_state(), snapshot["torch_rng_state"], "test torch RNG")
        verifier.require_equal(agent.rng.bit_generator.state, snapshot["agent_rng_state"], "test agent RNG")
    finally:
        env.close()


@pytest.mark.parametrize("mutation", [
    "target_resync", "counter", "first_moment", "second_moment", "adam_step", "learning_rate",
    "online", "torch_rng", "agent_rng",
])
def test_resume_audit_rejects_each_hidden_boundary_change(snapshot, mutation):
    changed = deepcopy(snapshot)
    if mutation == "target_resync":
        changed["target"] = deepcopy(changed["online"])
    elif mutation == "counter":
        changed["update_count"] = 0
    elif mutation in {"first_moment", "second_moment", "adam_step"}:
        key = {"first_moment": "exp_avg", "second_moment": "exp_avg_sq", "adam_step": "step"}[mutation]
        next(iter(changed["optimizer"]["state"].values()))[key].add_(1)
    elif mutation == "learning_rate":
        changed["optimizer"]["param_groups"][0]["lr"] = 0.0003
    elif mutation == "online":
        next(iter(changed["online"].values())).add_(1)
    elif mutation == "torch_rng":
        changed["torch_rng_state"][0] ^= 1
    else:
        changed["agent_rng_state"]["state"]["state"] += 1
    with pytest.raises(ValueError, match="Verification failed"):
        verifier.verify_resume_payload(changed, snapshot)


@pytest.fixture
def sentinel():
    batch = TransitionBatch(
        observations=torch.zeros(117, 3), actions=torch.tensor([index % 8 for index in range(117)]),
        rewards=torch.zeros(117), next_observations=torch.zeros(117, 3),
        terminated=torch.zeros(117, dtype=torch.bool), truncated=torch.zeros(117, dtype=torch.bool),
    )
    states = []
    for step, index in zip(verifier.SENTINEL_STEPS, verifier.SENTINEL_INDICES):
        values = [float(value) for value in range(8)]
        action = int(batch.actions[index])
        gap = values[action] - max(value for i, value in enumerate(values) if i != action)
        states.append({"reference_step": step, "global_transition_index": index,
                       "demonstrated_action": action, "greedy_action": 7,
                       "q_values": values, "signed_demonstration_gap": gap})
    return batch, {"update": 2001, "extension_update": 1, "scene_seed": 10001, "states": states}


def test_sentinel_keeps_negative_gap_as_value_diagnostic(sentinel):
    batch, row = sentinel
    verifier.verify_sentinel_row(row, batch, 2001)
    assert row["states"][0]["signed_demonstration_gap"] < 0
    assert "success" not in row


@pytest.mark.parametrize("mutation", ["clock", "index", "step", "greedy", "gap", "nonfinite", "missing"])
def test_sentinel_audit_rejects_corrupted_diagnostics(sentinel, mutation):
    batch, source = sentinel
    row = deepcopy(source)
    state = row["states"][0]
    if mutation == "clock":
        row["extension_update"] = 0
    elif mutation == "index":
        state["global_transition_index"] += 1
    elif mutation == "step":
        state["reference_step"] += 1
    elif mutation == "greedy":
        state["greedy_action"] = 0
    elif mutation == "gap":
        state["signed_demonstration_gap"] = 99
    elif mutation == "nonfinite":
        state["q_values"][0] = float("nan")
    else:
        row["states"].pop()
    with pytest.raises(ValueError, match="Verification failed"):
        verifier.verify_sentinel_row(row, batch, 2001)


def test_update_log_uses_absolute_target_clock():
    rows = [{"update": step, "extension_update": step - 2073,
             "target_synchronized": step % 200 == 0, "loss_before": 0.5}
            for step in range(2074, 2202)]
    assert verifier.verify_update_log(rows, initial=2073, final=2201) == 1
    changed = deepcopy(rows)
    changed[-2]["target_synchronized"] = False
    with pytest.raises(ValueError, match="synchronization"):
        verifier.verify_update_log(changed, initial=2073, final=2201)


def test_update_log_rejects_missing_update():
    rows = [{"update": 2001, "extension_update": 1, "target_synchronized": False, "loss_before": 0.5}]
    with pytest.raises(ValueError, match="complete continuation"):
        verifier.verify_update_log(rows, initial=2000, final=2002)


def test_independent_replay_matches_continuation_states_metrics_and_probes(snapshot, tmp_path):
    runner_spec = importlib.util.spec_from_file_location("extension_replay_fixture", verifier.RUNNER)
    runner = importlib.util.module_from_spec(runner_spec)
    runner_spec.loader.exec_module(runner)
    source = deepcopy(snapshot)
    source["update_count"] = 2000
    agent, env = runner.adam.restore_training_snapshot(source)
    directory = tmp_path / "td"
    directory.mkdir()
    batch = TransitionBatch(
        observations=torch.zeros(117, env.observation_size), actions=torch.arange(117) % 8,
        rewards=torch.ones(117), next_observations=torch.zeros(117, env.observation_size),
        terminated=torch.ones(117, dtype=torch.bool), truncated=torch.zeros(117, dtype=torch.bool),
    )
    episodes = [{"seed": 10001, "start": 35, "stop": 76}]
    records, sentinels = {}, {2000: runner.sentinel_record(agent, batch, episodes, 2000)}
    try:
        for step in range(2001, 2004):
            stats = agent.learner.update(batch, demonstration_weight=1.0, demonstration_margin=0.8)
            records[step] = {"update": step, "extension_update": step - 2000, **asdict(stats)}
            sentinels[step] = runner.sentinel_record(agent, batch, episodes, step)
            torch.save(runner.adam.training_snapshot(agent, source["environment_config"]),
                       directory / f"training_state_{step}.pt")
    finally:
        env.close()
    result = verifier.replay_continuation(source, tmp_path, batch, records, sentinels, end=2003)
    assert result["additional_optimizer_updates"] == 3
    assert result["additional_environment_steps"] == result["additional_environment_resets"] == 0
    assert result["full_snapshot_updates_compared"] == [2001, 2002, 2003]
    assert result["sentinel_rows_compared"] == 4
