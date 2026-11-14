"""Exercise independent weighted-TD math and immutable full-state continuation."""

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
    "td_weight_verifier_test", ROOT / "docs/experiments/019_td_weight/verify_td_weight.py")
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)
RUN_SPEC = importlib.util.spec_from_file_location(
    "td_weight_runner_verification_test", ROOT / "docs/experiments/019_td_weight/run_td_weight.py")
runner = importlib.util.module_from_spec(RUN_SPEC)
RUN_SPEC.loader.exec_module(runner)


def fixture():
    torch.set_num_threads(1)
    torch.manual_seed(19)
    env = HarvestingEnv({"max_steps": 3})
    agent = DQNAgent(env, DQNConfig(hidden_sizes=(8, 8), learning_rate=0.0001,
                                    batch_size=3, target_update_interval=200), seed=19)
    batch = TransitionBatch(
        observations=torch.randn(117, env.observation_size), actions=torch.arange(117) % 8,
        rewards=torch.linspace(-0.3, 1, 117), next_observations=torch.randn(117, env.observation_size),
        terminated=torch.arange(117) % 4 == 0, truncated=torch.arange(117) % 4 == 1)
    agent.learner.update(batch, demonstration_weight=1.0)
    agent.learner.update_count = 2199
    snapshot = verifier.multi.diagnostic.adam.training_snapshot(agent, {"max_steps": 3})
    env.close()
    return snapshot, batch


def test_weight_one_exactly_matches_original_log_adam_and_target_synchronization():
    snapshot, batch = fixture()
    first, env_a = verifier.prior.independent_restore(snapshot)
    second, env_b = verifier.prior.independent_restore(snapshot)
    try:
        expected = asdict(first.learner.update(batch, demonstration_weight=1.0))
        actual = verifier.independent_update(second.learner, batch, 1.0)
        assert actual.pop("td_weight") == 1.0
        assert actual == expected
        assert actual["target_synchronized"] and second.learner.update_count == 2200
        for name in ("online", "target", "optimizer"):
            verifier.require_equal(getattr(first.learner, name).state_dict(),
                                   getattr(second.learner, name).state_dict(), name)
    finally:
        env_a.close()
        env_b.close()


def test_half_td_changes_only_the_objective_coefficient_at_a_common_state():
    snapshot, batch = fixture()
    first, env_a = verifier.prior.independent_restore(snapshot)
    second, env_b = verifier.prior.independent_restore(snapshot)
    try:
        control = verifier.independent_update(first.learner, batch, 1.0)
        treatment = verifier.independent_update(second.learner, batch, 0.5)
        assert control["td_loss_before"] == treatment["td_loss_before"]
        assert control["demonstration_loss_before"] == treatment["demonstration_loss_before"]
        assert treatment["loss_before"] == float(
            0.5 * torch.tensor(control["td_loss_before"]) + torch.tensor(control["demonstration_loss_before"]))
        assert not any("return" in name or name.startswith("mc_") for name in treatment)
        assert all(group["lr"] == 0.0001 for group in second.learner.optimizer.param_groups)
        assert {float(state["step"]) for state in second.learner.optimizer.state.values()} == {2.0}
        assert any(not torch.equal(a, b) for a, b in zip(first.learner.online.parameters(), second.learner.online.parameters()))
    finally:
        env_a.close()
        env_b.close()


@pytest.mark.parametrize("weight", (True, 0, -1, 0.25, 2, float("nan"), float("inf")))
def test_invalid_weight_is_rejected_before_modes_gradients_or_optimizer_change(weight):
    snapshot, batch = fixture()
    agent, env = verifier.prior.independent_restore(snapshot)
    try:
        before = deepcopy(agent.learner.optimizer.state_dict())
        rng = torch.get_rng_state().clone()
        modes = [module.training for module in agent.learner.online.modules()]
        with pytest.raises(ValueError, match="predeclared"):
            verifier.independent_update(agent.learner, batch, weight)
        verifier.require_equal(agent.learner.optimizer.state_dict(), before, "unchanged Adam")
        verifier.require_equal(verifier.multi.diagnostic.adam.training_snapshot(agent, snapshot["environment_config"]),
                               snapshot, "unchanged complete snapshot")
        assert modes == [module.training for module in agent.learner.online.modules()]
        assert torch.equal(torch.get_rng_state(), rng)
    finally:
        env.close()


def test_after_td_loss_uses_frozen_pre_update_targets_even_when_target_synchronizes():
    snapshot, batch = fixture()
    agent, env = verifier.prior.independent_restore(snapshot)
    try:
        with torch.no_grad():
            following = agent.learner.target(batch.next_observations).max(dim=1).values
            targets = batch.rewards + agent.learner.config.gamma * torch.where(
                batch.terminated, torch.zeros_like(following), following)
        result = verifier.independent_update(agent.learner, batch, 0.5)
        assert result["target_synchronized"]
        with torch.no_grad():
            values = agent.learner.online(batch.observations).gather(1, batch.actions[:, None]).squeeze(1)
            expected = torch.nn.functional.smooth_l1_loss(values, targets).item()
        assert result["td_loss_after"] == expected
        assert agent.learner.target(batch.next_observations).grad_fn is None
    finally:
        env.close()


def loss_records():
    return [{"update": step, "extension_update": step - 2000,
             "target_synchronized": step % 200 == 0, "loss_before": 0.75, "loss_after": 0.75,
             "td_loss_before": 0.5, "td_loss_after": 0.5,
             "demonstration_loss_before": 0.5, "demonstration_loss_after": 0.5,
             "td_weight": 0.5, "parameter_delta_norm": 0.01, "gradient_norm": 1,
             "demonstration_action_accuracy": 1.0} for step in range(2001, 10001)]


def test_complete_loss_log_preserves_unweighted_components_and_weighted_total():
    verifier.verify_loss_records(loss_records())


@pytest.mark.parametrize("mutation", ("margin_reweighted", "wrong_td_weight", "loss_averaged", "mc_added",
                                      "missing_update", "negative_displacement", "wrong_sync", "negative_term"))
def test_log_audit_rejects_extra_objective_or_hidden_continuation_change(mutation):
    records = loss_records()
    if mutation == "margin_reweighted":
        records[0]["loss_before"] = 1.25
    elif mutation == "wrong_td_weight":
        records[0]["td_weight"] = 1.0
    elif mutation == "loss_averaged":
        records[0]["loss_before"] /= 1.5
    elif mutation == "mc_added":
        records[0]["return_loss_before"] = 0.0
    elif mutation == "missing_update":
        records.pop()
    elif mutation == "negative_displacement":
        records[0]["parameter_delta_norm"] = -0.01
    elif mutation == "wrong_sync":
        records[0]["target_synchronized"] = True
    else:
        records[0]["td_loss_after"] = -0.5
    with pytest.raises(ValueError, match="Verification failed"):
        verifier.verify_loss_records(records)


@pytest.mark.parametrize("weight", (0.5, 1.0))
def test_independent_math_matches_runner_before_at_and_after_target_synchronization(weight):
    snapshot, batch = fixture()
    snapshot["update_count"] = 2198
    first, env_a = verifier.prior.independent_restore(snapshot)
    second, env_b = verifier.prior.independent_restore(snapshot)
    try:
        for step in (2199, 2200, 2201):
            actual = verifier.independent_update(first.learner, batch, weight)
            expected = runner.weighted_td_update(second.learner, batch, weight)
            assert actual == expected
            assert actual["target_synchronized"] is (step == 2200)
            verifier.require_equal(
                verifier.multi.diagnostic.adam.training_snapshot(first, snapshot["environment_config"]),
                verifier.multi.diagnostic.adam.training_snapshot(second, snapshot["environment_config"]),
                "independent complete state across target synchronization")
    finally:
        env_a.close()
        env_b.close()


class FakeEnvironment:
    verify_reference_route = True

    def __init__(self):
        self.calls = []
        self.result = (object(), {"seed": 10001, "reference_route_checks": [
            {"steps": 7, "success": False, "reason": "contact"},
            {"steps": 11, "success": True, "reason": "reached"}]})

    def reset(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


def test_reset_observer_preserves_identity_arguments_and_counts_rejected_candidates():
    env, records, incomplete = FakeEnvironment(), [], []
    options = {"unchanged": object()}
    with verifier.capture_resets(env, 2200, records, incomplete):
        result = env.reset(seed=10001, options=options)
    assert result is env.result
    assert env.calls == [((), {"seed": 10001, "options": options})]
    assert env.calls[0][1]["options"] is options
    assert "reset" not in env.__dict__ and not incomplete
    assert records == [{"update": 2200, "seed": 10001, "certification_enabled": True,
                        "certification_attempts": 2, "certification_environment_steps": 18,
                        "reference_route_checks": env.result[1]["reference_route_checks"]}]
    env.result[1]["reference_route_checks"][0]["steps"] = 99
    assert records[0]["reference_route_checks"][0]["steps"] == 7


def test_reset_observer_restores_existing_instance_attribute_after_callback_failure():
    env, records, incomplete = FakeEnvironment(), [], []
    original = env.reset
    env.reset = original
    with (pytest.raises(RuntimeError, match="after reset"),
          verifier.capture_resets(env, 2200, records, incomplete)):
        env.reset(seed=10001)
        raise RuntimeError("after reset")
    assert env.__dict__["reset"] is original
    assert len(env.calls) == len(records) == 1 and incomplete == []


def test_failed_reset_marks_unknown_physics_and_restores_method():
    env, records, incomplete = FakeEnvironment(), [], []

    def failing_reset(**kwargs):
        raise RuntimeError("incomplete certification")

    env.reset = failing_reset
    with (pytest.raises(RuntimeError, match="incomplete certification"),
          verifier.capture_resets(env, 2200, records, incomplete)):
        env.reset(seed=10001)
    assert env.reset is failing_reset and records == []
    assert incomplete == [{"update": 2200, "seed": 10001, "certification_steps_unknown": True}]


def certification_fixture():
    rows = [{"update": update, "scenes": [{"seed": seed} for seed in (10000, 10001, 10002)]}
            for update in (2000, 2200)]
    records = []
    for row in rows:
        for scene in row["scenes"]:
            info = deepcopy(FakeEnvironment().result[1])
            info["seed"] = scene["seed"]
            records.append(verifier.certification_record(info, scene["seed"], row["update"], True))
    return rows, {"records": records, "completed_reset_count": 6, "certification_attempts": 12,
                  "certification_environment_steps": 108, "incomplete_resets": [], "accounting_complete": True}


def test_certification_audit_sums_all_candidates_without_any_additional_reset():
    rows, record = certification_fixture()
    verifier.verify_certification(record, rows, True)


@pytest.mark.parametrize("mutation", ("missing_reset", "missing_candidate", "wrong_total", "wrong_seed",
                                      "changed_same_seed", "incomplete", "disabled"))
def test_certification_audit_rejects_missing_physics_or_changed_reset_semantics(mutation):
    rows, record = certification_fixture()
    if mutation == "missing_reset":
        record["records"].pop()
    elif mutation == "missing_candidate":
        record["records"][0]["reference_route_checks"].pop(0)
    elif mutation == "wrong_total":
        record["certification_environment_steps"] -= 1
    elif mutation == "wrong_seed":
        record["records"][0]["seed"] += 5
    elif mutation == "changed_same_seed":
        record["records"][3]["reference_route_checks"][0]["steps"] += 1
        record["records"][3]["certification_environment_steps"] += 1
        record["certification_environment_steps"] += 1
    elif mutation == "incomplete":
        record["incomplete_resets"].append({"update": 2200})
    else:
        record["records"][0]["certification_enabled"] = False
    with pytest.raises(ValueError, match="Verification failed"):
        verifier.verify_certification(record, rows, True)


def protocol_fixture():
    protocol = runner.make_protocol({"helper_sha256": {}})
    settings = {"dqn": {"learning_rate": 0.0001, "gamma": 0.99, "target_update_interval": 200,
                        "gradient_clip_norm": 10.0, "hidden_sizes": [256, 256]}}
    return protocol, settings


def test_predeclared_protocol_matches_frozen_runner():
    verifier.verify_protocol(*protocol_fixture())


@pytest.mark.parametrize("field,value", (("td_weight", 1), ("demonstration_weight", 0.5),
                                         ("learning_rate", 0.00005), ("return_anchor_weight", 0.5),
                                         ("reset_certification_preserved", False), ("early_stopping", True)))
def test_protocol_audit_rejects_confounded_objective_optimizer_or_simulator(field, value):
    protocol, settings = protocol_fixture()
    protocol[field] = value
    with pytest.raises(ValueError, match="Verification failed"):
        verifier.verify_protocol(protocol, settings)


def test_initial_q_audit_matches_all_ordered_rows_without_environment_interaction(monkeypatch):
    snapshot, batch = fixture()
    agent, env = verifier.prior.independent_restore(snapshot)
    try:
        record = runner.initial_q_equivalence(agent, batch, snapshot)
    finally:
        env.close()

    def forbidden(*args, **kwargs):
        raise AssertionError("Initial Q verification must not reset or step the environment.")

    monkeypatch.setattr(HarvestingEnv, "reset", forbidden)
    monkeypatch.setattr(HarvestingEnv, "step", forbidden)
    verifier.verify_initial_q(record, snapshot, batch)
    record["q_values"][40][2] += 0.01
    with pytest.raises(ValueError, match="Verification failed"):
        verifier.verify_initial_q(record, snapshot, batch)
