"""Check independent return anchoring, exact Adam restoration and audit semantics."""

import importlib.util
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
    "mc_anchor_verifier_test", ROOT / "docs/experiments/017_mc_anchor/verify_mc_anchor.py")
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def fixture():
    torch.set_num_threads(1)
    env = HarvestingEnv({"max_steps": 3})
    agent = DQNAgent(env, DQNConfig(hidden_sizes=(8, 8), learning_rate=0.0001,
                                    batch_size=3, target_update_interval=200), seed=91)
    torch.manual_seed(91)
    batch = TransitionBatch(
        observations=torch.randn(117, env.observation_size), actions=torch.arange(117) % 8,
        rewards=torch.linspace(-0.3, 1, 117), next_observations=torch.randn(117, env.observation_size),
        terminated=torch.arange(117) % 4 == 0, truncated=torch.zeros(117, dtype=torch.bool))
    agent.learner.update(batch, demonstration_weight=1.0)
    agent.learner.update_count = 2199
    snapshot = verifier.multi.diagnostic.adam.training_snapshot(agent, {"max_steps": 3})
    env.close()
    return snapshot, batch, torch.linspace(2.0, 4.0, 117)


def test_zero_anchor_exactly_matches_original_update_including_sync_and_adam():
    snapshot, batch, returns = fixture()
    original, env_a = verifier.prior.independent_restore(snapshot)
    rebuilt, env_b = verifier.prior.independent_restore(snapshot)
    try:
        expected = asdict(original.learner.update(batch, demonstration_weight=1.0))
        actual = verifier.independent_update(rebuilt.learner, batch, returns, 0.0)
        assert {key: value for key, value in actual.items() if key not in verifier.EXTRA_FIELDS} == expected
        assert actual["target_synchronized"]
        assert rebuilt.learner.update_count == 2200
        for name, a, b in (
            ("online", original.learner.online.state_dict(), rebuilt.learner.online.state_dict()),
            ("target", original.learner.target.state_dict(), rebuilt.learner.target.state_dict()),
            ("Adam", original.learner.optimizer.state_dict(), rebuilt.learner.optimizer.state_dict()),
        ):
            verifier.require_equal(a, b, name)
    finally:
        env_a.close()
        env_b.close()


def test_fixed_return_anchor_adds_loss_without_changing_shared_td_target_or_margin():
    snapshot, batch, returns = fixture()
    base, env_a = verifier.prior.independent_restore(snapshot)
    treatment, env_b = verifier.prior.independent_restore(snapshot)
    try:
        a = verifier.independent_update(base.learner, batch, returns, 0.0)
        b = verifier.independent_update(treatment.learner, batch, returns, 1.0)
        for name in ("td_loss_before", "demonstration_loss_before", "return_loss_before",
                     "td_target_return_rmse", "td_target_return_mean_difference",
                     "td_target_return_max_abs_difference"):
            assert a[name] == b[name]
        assert b["loss_before"] == float(torch.tensor(a["loss_before"]) + torch.tensor(b["return_loss_before"]))
        assert b["loss_before"] > a["loss_before"]
        assert not torch.equal(next(base.learner.online.parameters()), next(treatment.learner.online.parameters()))
        assert b["target_synchronized"]
    finally:
        env_a.close()
        env_b.close()


def test_mc_targets_are_detached_and_remain_unchanged():
    snapshot, batch, returns = fixture()
    returns.requires_grad_(True)
    before = returns.detach().clone()
    agent, env = verifier.prior.independent_restore(snapshot)
    try:
        verifier.independent_update(agent.learner, batch, returns, 1.0)
        assert returns.grad is None
        assert torch.equal(returns.detach(), before)
    finally:
        env.close()


@pytest.mark.parametrize("bad_weight", (True, -1.0, 0.5, 2.0, float("nan"), float("inf")))
def test_independent_replay_rejects_undeclared_weight_before_state_mutation(bad_weight):
    snapshot, batch, returns = fixture()
    agent, env = verifier.prior.independent_restore(snapshot)
    try:
        with pytest.raises(ValueError, match="predeclared"):
            verifier.independent_update(agent.learner, batch, returns, bad_weight)
        verifier.require_equal(verifier.multi.diagnostic.adam.training_snapshot(agent, snapshot["environment_config"]),
                               snapshot, "unchanged rejected state")
    finally:
        env.close()


@pytest.mark.parametrize("mutation", ("nan", "short", "matrix", "integer", "double"))
def test_independent_replay_rejects_invalid_return_targets_before_mutation(mutation):
    snapshot, batch, returns = fixture()
    if mutation == "nan":
        returns[0] = float("nan")
    elif mutation == "short":
        returns = returns[:-1]
    elif mutation == "matrix":
        returns = returns[:, None]
    elif mutation == "integer":
        returns = returns.long()
    else:
        returns = returns.double()
    agent, env = verifier.prior.independent_restore(snapshot)
    try:
        with pytest.raises(ValueError, match="return targets"):
            verifier.independent_update(agent.learner, batch, returns, 1.0)
        verifier.require_equal(verifier.multi.diagnostic.adam.training_snapshot(agent, snapshot["environment_config"]),
                               snapshot, "unchanged rejected return state")
    finally:
        env.close()


def test_restore_preserves_distinct_target_adam_clock_and_both_rng_states():
    snapshot, _, _ = fixture()
    agent, env = verifier.prior.independent_restore(snapshot)
    try:
        actual = verifier.multi.diagnostic.adam.training_snapshot(agent, snapshot["environment_config"])
        verifier.require_equal(actual, snapshot, "complete independently restored snapshot")
        assert agent.learner.update_count == 2199
        assert any(not torch.equal(value, snapshot["target"][name]) for name, value in snapshot["online"].items())
    finally:
        env.close()


def test_independent_mc_math_matches_new_updater_across_synchronization():
    spec = importlib.util.spec_from_file_location("anchor017_runner_fixture", verifier.RUNNER)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    snapshot, batch, returns = fixture()
    core, env_a = verifier.prior.independent_restore(snapshot)
    independent, env_b = verifier.prior.independent_restore(snapshot)
    try:
        for _ in range(3):
            actual = runner.anchored_update(core.learner, batch, returns, anchor_weight=1.0)
            expected = verifier.independent_update(independent.learner, batch, returns, 1.0)
            assert actual == expected
            for name, a, b in (
                ("online", core.learner.online.state_dict(), independent.learner.online.state_dict()),
                ("target", core.learner.target.state_dict(), independent.learner.target.state_dict()),
                ("Adam", core.learner.optimizer.state_dict(), independent.learner.optimizer.state_dict()),
            ):
                verifier.require_equal(a, b, name)
        assert core.learner.update_count == independent.learner.update_count == 2202
    finally:
        env_a.close()
        env_b.close()


def loss_records():
    return [{
        "update": step, "extension_update": step - 2000, "target_synchronized": step % 200 == 0,
        "loss_before": 0.75, "loss_after": 0.75, "td_loss_before": 0.25, "td_loss_after": 0.25,
        "demonstration_loss_before": 0.25, "demonstration_loss_after": 0.25,
        "return_loss_before": 0.25, "return_loss_after": 0.25, "return_anchor_weight": 1.0,
        "parameter_delta_norm": 0.1, "gradient_norm": 1.0, "demonstration_action_accuracy": 1.0,
        "td_target_return_rmse": 0.5, "td_target_return_mean_difference": 0.0,
        "td_target_return_max_abs_difference": 1.0,
    } for step in range(2001, 10001)]


def test_loss_audit_accepts_sum_of_three_means_with_absolute_sync_clock():
    verifier.verify_loss_records(loss_records())


@pytest.mark.parametrize("mutation", ("divide_total_by_three", "change_weight", "missing_update",
                                      "target_bias_exceeds_rmse", "stopped_learning"))
def test_loss_audit_rejects_hidden_objective_accounting_or_diagnostic_changes(mutation):
    records = loss_records()
    if mutation == "divide_total_by_three":
        records[0]["loss_before"] /= 3
    elif mutation == "change_weight":
        records[0]["return_anchor_weight"] = 0.5
    elif mutation == "missing_update":
        records.pop()
    elif mutation == "target_bias_exceeds_rmse":
        records[0]["td_target_return_mean_difference"] = 0.6
    else:
        records[0]["parameter_delta_norm"] = 0.0
    with pytest.raises(ValueError, match="Verification failed"):
        verifier.verify_loss_records(records)


def protocol_fixture():
    spec = importlib.util.spec_from_file_location("anchor017_protocol_fixture", verifier.RUNNER)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    protocol = runner.make_protocol({"helper_sha256": {}})
    settings = {"dqn": {"learning_rate": 0.0001, "gamma": 0.99, "target_update_interval": 200,
                        "gradient_clip_norm": 10, "hidden_sizes": [256, 256]}}
    return protocol, settings


def test_declared_protocol_matches_independent_fixed_conditions():
    protocol, settings = protocol_fixture()
    verifier.verify_protocol(protocol, settings)


@pytest.mark.parametrize("key,value", (("all_sample_weights", 2.0), ("total_loss_divided_by_three", True),
                                      ("td_targets_mixed_with_returns", True), ("historical_control_reused_not_rerun", False)))
def test_protocol_rejects_additional_factors_or_false_control_identity(key, value):
    protocol, settings = protocol_fixture()
    protocol[key] = value
    with pytest.raises(ValueError, match="fixed return-anchor protocol"):
        verifier.verify_protocol(protocol, settings)


def accounting_fixture():
    rows = [{"aggregate": {"evaluation_environment_steps": 117}} for _ in range(29)]
    accounting = {
        "optimizer_updates_total": 8000, "inherited_shared_optimizer_updates": 42000,
        "absolute_td_update": 10000, "total_updates_on_treatment_branch": 50000,
        "scheduled_evaluation_episodes": 87, "evaluation_environment_steps": 3393,
        "sentinel_records": 8001, "sentinel_forward_passes": 56007, "sentinel_environment_steps": 0,
        "online_training_steps": 0, "demonstration_collection_steps": 0, "held_out_scenes_accessed": 0,
        "reused_historical_control": {"optimizer_updates": 8000, "scheduled_evaluation_episodes": 87,
                                      "evaluation_environment_steps": 3711, "new_optimizer_updates": 0,
                                      "new_evaluation_episodes": 0, "new_environment_steps": 0}}
    return accounting, rows


def test_accounting_keeps_reused_control_separate_from_new_work():
    accounting, rows = accounting_fixture()
    verifier.verify_accounting(accounting, rows)


@pytest.mark.parametrize("mutation", ("count_control_updates_again", "count_control_episodes_again", "wrong_action_count"))
def test_accounting_rejects_reused_evidence_reported_as_new_work(mutation):
    accounting, rows = accounting_fixture()
    if mutation == "count_control_updates_again":
        accounting["optimizer_updates_total"] += 8000
    elif mutation == "count_control_episodes_again":
        accounting["scheduled_evaluation_episodes"] += 87
    else:
        accounting["evaluation_environment_steps"] += 1
    with pytest.raises(ValueError, match="main versus reused historical accounting"):
        verifier.verify_accounting(accounting, rows)
