"""Check independent weighted math, preserved clocks and local-control audit rules."""

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
    "local_supervision_verifier_test", ROOT / "docs/experiments/016_local_supervision/verify_local_supervision.py")
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
    return snapshot, batch


def test_independent_weight_one_exactly_matches_original_update_and_target_clock():
    snapshot, batch = fixture()
    original, env_a = verifier.prior.independent_restore(snapshot)
    rebuilt, env_b = verifier.prior.independent_restore(snapshot)
    try:
        expected = asdict(original.learner.update(batch, demonstration_weight=1.0))
        actual, diagnostics = verifier.independent_update(rebuilt.learner, batch, 1.0)
        assert actual == expected
        assert actual["target_synchronized"]
        assert rebuilt.learner.update_count == 2200
        for name, a, b in (
            ("online", original.learner.online.state_dict(), rebuilt.learner.online.state_dict()),
            ("target", original.learner.target.state_dict(), rebuilt.learner.target.state_dict()),
            ("Adam", original.learner.optimizer.state_dict(), rebuilt.learner.optimizer.state_dict()),
        ):
            verifier.require_equal(a, b, name)
        assert diagnostics["weighted_margin_before"] == diagnostics["unweighted_margin_before"]
        assert diagnostics["normalization_denominator"] == 117
    finally:
        env_a.close()
        env_b.close()


def test_only_selected_supervision_gradient_is_doubled_with_fixed_denominator():
    q = torch.arange(117 * 8, dtype=torch.float32).reshape(117, 8).requires_grad_()
    actions = torch.zeros(117, dtype=torch.long)
    one, hinges = verifier.independent_supervision(q, actions, 1.0)
    gradient_one, = torch.autograd.grad(one, q, retain_graph=True)
    two, _ = verifier.independent_supervision(q, actions, 2.0)
    gradient_two, = torch.autograd.grad(two, q)
    expected = gradient_one.clone()
    expected[55] *= 2
    assert torch.equal(gradient_two, expected)
    assert not torch.equal(gradient_two[54], 2 * gradient_one[54])
    assert torch.allclose(two, one + hinges[55] / 117, atol=1e-6, rtol=0)
    wrongly_normalized = two * (117 / 118)
    assert not torch.isclose(two, wrongly_normalized)


def test_local_weight_does_not_change_preupdate_td_targets_or_td_loss():
    snapshot, batch = fixture()
    a, env_a = verifier.prior.independent_restore(snapshot)
    b, env_b = verifier.prior.independent_restore(snapshot)
    try:
        base, _ = verifier.independent_update(a.learner, batch, 1.0)
        weighted, diagnostics = verifier.independent_update(b.learner, batch, 2.0)
        assert weighted["td_loss_before"] == base["td_loss_before"]
        assert diagnostics["selected_sample_weight"] == 2
        assert weighted["demonstration_loss_before"] > base["demonstration_loss_before"]
        assert a.learner.update_count == b.learner.update_count == 2200
        assert weighted["target_synchronized"]
        assert not torch.equal(next(a.learner.online.parameters()), next(b.learner.online.parameters()))
    finally:
        env_a.close()
        env_b.close()


@pytest.mark.parametrize("weight", (0.0, 1.5, 3.0, True))
def test_independent_objective_rejects_undeclared_weights(weight):
    with pytest.raises(ValueError, match="predeclared"):
        verifier.independent_supervision(torch.zeros(117, 8), torch.zeros(117, dtype=torch.long), weight)


def sentinel():
    _, batch = fixture()
    states = []
    for step, index in zip(verifier.SENTINEL_STEPS, verifier.SENTINEL_INDICES, strict=True):
        q = [float(i) for i in range(8)]
        action = int(batch.actions[index])
        states.append({
            "reference_step": step, "global_transition_index": index, "demonstrated_action": action,
            "q_values": q, "greedy_action": 7,
            "signed_demonstration_gap": q[action] - max(value for i, value in enumerate(q) if i != action),
        })
    return batch, {"update": 2009, "extension_update": 9, "scene_seed": 10001, "states": states}


def test_neighbor_probe_includes_both_selected_state_and_adjacent_decisions():
    batch, row = sentinel()
    verifier.verify_sentinel(row, batch, 2009)
    assert [state["reference_step"] for state in row["states"]] == [6, 18, 20, 21, 22, 23, 24]
    assert row["states"][3]["global_transition_index"] == 55


@pytest.mark.parametrize("mutation", ("wrong_index", "wrong_step", "wrong_gap", "missing_neighbor"))
def test_neighbor_probe_rejects_wrong_selected_state_or_missing_coverage(mutation):
    batch, row = sentinel()
    if mutation == "wrong_index":
        row["states"][3]["global_transition_index"] = 54
    elif mutation == "wrong_step":
        row["states"][3]["reference_step"] = 20
    elif mutation == "wrong_gap":
        row["states"][3]["signed_demonstration_gap"] = 12
    else:
        row["states"].pop()
    with pytest.raises(ValueError, match="Verification failed"):
        verifier.verify_sentinel(row, batch, 2009)


def supervision_logs(weight=2.0):
    row = {
        "update": 2001, "extension_update": 1, "selected_sample_weight": weight,
        "normalization_denominator": 117,
        "unweighted_margin_before": 1.0, "unweighted_margin_after": 0.5,
        "selected_hinge_before": 2.0, "selected_hinge_after": 1.0,
        "weighted_margin_before": 1.0 + (weight - 1) * 2 / 117,
        "weighted_margin_after": 0.5 + (weight - 1) / 117,
    }
    update = {"update": 2001, "extension_update": 1,
              "demonstration_loss_before": row["weighted_margin_before"],
              "demonstration_loss_after": row["weighted_margin_after"]}
    return row, update


def test_supervision_log_accepts_fixed_mean_and_distinguishes_raw_hinge():
    row, update = supervision_logs()
    verifier.verify_supervision([row], [update], 2.0)


@pytest.mark.parametrize("mutation", ("normalize_by_weight_sum", "unweighted_logged_as_weighted", "wrong_weight", "clock"))
def test_supervision_log_rejects_hidden_objective_or_counter_changes(mutation):
    row, update = supervision_logs()
    if mutation == "normalize_by_weight_sum":
        row["normalization_denominator"] = 118
    elif mutation == "unweighted_logged_as_weighted":
        update["demonstration_loss_before"] = row["unweighted_margin_before"]
    elif mutation == "wrong_weight":
        row["selected_sample_weight"] = 1
    else:
        row["extension_update"] = 0
    with pytest.raises(ValueError, match="Verification failed"):
        verifier.verify_supervision([row], [update], 2.0)


def test_independent_restore_keeps_distinct_target_adam_and_both_rng_states():
    snapshot, _ = fixture()
    restored, env = verifier.prior.independent_restore(snapshot)
    try:
        after = verifier.multi.diagnostic.adam.training_snapshot(restored, snapshot["environment_config"])
        verifier.require_equal(after, snapshot, "full restored state")
        assert restored.learner.update_count == 2199
        assert any(not torch.equal(value, snapshot["target"][name]) for name, value in snapshot["online"].items())
    finally:
        env.close()


def test_independent_weighted_replay_matches_new_updater_across_target_sync():
    spec = importlib.util.spec_from_file_location("local016_weighted_update_fixture", verifier.RUNNER)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    snapshot, batch = fixture()
    core, env_a = verifier.prior.independent_restore(snapshot)
    independent, env_b = verifier.prior.independent_restore(snapshot)
    try:
        for _ in range(3):
            before = runner.supervision_probe(core, batch, selected_weight=2.0)
            actual = asdict(runner.weighted_update(core.learner, batch))
            after = runner.supervision_probe(core, batch, selected_weight=2.0)
            expected, details = verifier.independent_update(independent.learner, batch, 2.0)
            assert actual == expected
            for when, values in (("before", before), ("after", after)):
                for key, value in values.items():
                    assert details[f"{key}_{when}"] == value
            verifier.require_equal(core.learner.optimizer.state_dict(), independent.learner.optimizer.state_dict(), "Adam")
            verifier.require_equal(core.learner.online.state_dict(), independent.learner.online.state_dict(), "online")
            verifier.require_equal(core.learner.target.state_dict(), independent.learner.target.state_dict(), "target")
        assert core.learner.update_count == independent.learner.update_count == 2202
    finally:
        env_a.close()
        env_b.close()


@pytest.mark.parametrize("mutation", ("shift_selected_row", "change_other_row", "normalize_by_118", "weight_td"))
def test_recorded_weight_table_rejects_extra_factor_changes(mutation):
    selected = {"global_transition_index": 55, "encoded_observation_sha256": "fixed"}
    table = {"selected_transition": selected, "normalization_denominator": 117,
             "arms": {"baseline": [1.] * 117, "treatment": [2. if i == 55 else 1. for i in range(117)]},
             "td_weights_changed": False}
    verifier.verify_weight_table(table, selected)
    if mutation == "shift_selected_row":
        table["arms"]["treatment"][55], table["arms"]["treatment"][54] = 1., 2.
    elif mutation == "change_other_row":
        table["arms"]["treatment"][56] = 2.
    elif mutation == "normalize_by_118":
        table["normalization_denominator"] = 118
    else:
        table["td_weights_changed"] = True
    with pytest.raises(ValueError, match="only one supervised coefficient"):
        verifier.verify_weight_table(table, selected)
