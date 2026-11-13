"""Check independent gradient, Adam, probe and replay-accounting verification."""

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from me5418_harvesting.agents.dqn import DQNConfig
from me5418_harvesting.agents.dqn_learning import DQNUpdater, TransitionBatch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "gradient018_verifier_test", ROOT / "docs/experiments/018_gradient_diagnostics/verify_gradient_diagnostics.py")
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def fixture():
    torch.set_num_threads(1)
    torch.manual_seed(718)
    learner = DQNUpdater(5, 3, DQNConfig(hidden_sizes=(8, 8), learning_rate=0.0001,
                                         gradient_clip_norm=0.1, target_update_interval=200))
    batch = TransitionBatch(torch.randn(117, 5), torch.arange(117) % 3,
                            torch.linspace(-1, 2, 117), torch.randn(117, 5),
                            torch.arange(117) % 5 == 0, torch.zeros(117, dtype=torch.bool))
    returns = torch.linspace(2, 4, 117)
    learner.update(batch, demonstration_weight=1.0)
    learner.update_count = 2199
    probe = {"name": "synthetic", "observation": batch.observations[17:18],
             "demonstrated_action": int(batch.actions[17])}
    return learner, batch, returns, probe


def state(learner):
    return deepcopy({"online": learner.online.state_dict(), "target": learner.target.state_dict(),
                     "optimizer": learner.optimizer.state_dict(), "counter": learner.update_count,
                     "torch_rng": torch.get_rng_state()})


@pytest.mark.parametrize("weight", (0.0, 1.0))
def test_independent_component_objective_and_clipping_match_actual_update(weight):
    learner, batch, returns, _ = fixture()
    before = state(learner)
    components = verifier.independent_components(learner, batch, returns, weight)
    actual, stats = verifier.independent_branch(learner, batch, returns, weight)
    assert components["losses"]["total"] == stats["loss_before"]
    assert components["gradient_norm"] == stats["gradient_norm"]
    assert components["clip_coefficient"] < 1
    raw_sum = components["gradients"]["td"] + components["gradients"]["margin"]
    if weight:
        raw_sum = raw_sum + components["gradients"]["mc"]
    assert torch.allclose(raw_sum, components["total_gradient"], rtol=1e-5, atol=1e-7)
    assert actual.update_count == 2200 and stats["target_synchronized"]
    verifier.require_equal(actual.target.state_dict(), actual.online.state_dict(), "synchronized target")
    verifier.require_equal(state(learner), before, "diagnostics preserve original state")


@pytest.mark.parametrize("weight", (0.0, 1.0))
def test_adam_reconstruction_includes_old_momentum_and_recorded_roundoff(weight):
    learner, batch, returns, _ = fixture()
    components = verifier.independent_components(learner, batch, returns, weight)
    actual, _ = verifier.independent_branch(learner, batch, returns, weight)
    parts = verifier.independent_adam_parts(learner, actual, components, weight)
    assert verifier.norm(parts["old_first_moment"]) > 0
    assert verifier.norm(parts["numerical_remainder"]) < 2e-7
    assert verifier.norm(parts["numerical_remainder"]) / verifier.norm(parts["actual_delta"]) < 0.001
    if not weight:
        assert torch.count_nonzero(parts["current_mc"]) == 0
    verifier.require_equal(parts["reconstructed_delta"] + parts["numerical_remainder"],
                           parts["actual_delta"], "remainder closes the displacement decomposition")


@pytest.mark.parametrize("branch", ("reset_adam", "zero_first_moment"))
def test_independent_adam_branches_preserve_targets_and_do_not_mutate_source(branch):
    learner, batch, returns, _ = fixture()
    before = state(learner)
    actual, _ = verifier.independent_branch(learner, batch, returns, 1.0)
    alternative, stats = verifier.independent_branch(learner, batch, returns, 1.0, branch)
    assert alternative.update_count == 2200 and stats["target_synchronized"]
    assert any(not torch.equal(a, b) for a, b in zip(actual.online.parameters(), alternative.online.parameters()))
    steps = [float(value["step"]) for value in alternative.optimizer.state.values()]
    assert set(steps) == ({1.0} if branch == "reset_adam" else {2.0})
    verifier.require_equal(state(learner), before, "counterfactual branches preserve original state")


def test_probe_derivative_matches_central_difference_with_fixed_competitor():
    learner, batch, _, probe = fixture()
    learner.online.double()
    probe["observation"] = probe["observation"].double()
    result = verifier.independent_probe(learner, probe)
    parameters = tuple(learner.online.parameters())
    original = torch.nn.utils.parameters_to_vector(parameters).detach().clone()
    direction = torch.linspace(-1, 1, len(original), dtype=torch.float64)
    direction /= torch.linalg.vector_norm(direction)
    estimates = []
    epsilon = 1e-6
    for sign in (-1, 1):
        torch.nn.utils.vector_to_parameters(original + sign * epsilon * direction, parameters)
        row = verifier.independent_after_probe(learner, probe, result["competitor"])
        estimates.append(row["fixed_gap"])
    derivative = (estimates[1] - estimates[0]) / (2 * epsilon)
    assert derivative == pytest.approx(float(result["gradient"] @ direction), abs=1e-9, rel=1e-6)
    torch.nn.utils.vector_to_parameters(original, parameters)
    assert batch.observations.dtype == torch.float32


@pytest.mark.parametrize("weight", (True, -1.0, 0.5, 2.0, float("nan")))
def test_independent_components_reject_undeclared_mc_objectives(weight):
    learner, batch, returns, _ = fixture()
    before = state(learner)
    with pytest.raises(ValueError, match="predeclared"):
        verifier.independent_components(learner, batch, returns, weight)
    verifier.require_equal(state(learner), before, "rejected objective leaves state unchanged")


def test_independent_adam_decomposition_rejects_unsupported_optimizer_semantics():
    learner, batch, returns, _ = fixture()
    components = verifier.independent_components(learner, batch, returns, 1.0)
    actual, _ = verifier.independent_branch(learner, batch, returns, 1.0)
    learner.optimizer.param_groups[0]["weight_decay"] = 0.01
    with pytest.raises(ValueError, match="ordinary historical Adam"):
        verifier.independent_adam_parts(learner, actual, components, 1.0)


@pytest.mark.parametrize("value", (float("nan"), float("inf"), -float("inf")))
def test_structural_audit_rejects_nonfinite_nested_diagnostics(value):
    with pytest.raises(ValueError, match="finite numeric"):
        verifier.finite_tree({"probe": [{"gap": value}]})


def test_zero_vectors_have_no_defined_cosine():
    assert verifier.cosine(torch.zeros(3), torch.ones(3)) is None
    assert verifier.cosine(torch.ones(3), -torch.ones(3)) == pytest.approx(-1)


@pytest.mark.parametrize("weight", (0.0, 1.0))
def test_full_independent_numerical_verifier_matches_diagnostic_core(weight):
    core = verifier.load_module("gradient018_core_numerical_test",
                                ROOT / "docs/experiments/018_gradient_diagnostics/gradient_core.py")
    learner, batch, returns, probe = fixture()
    record, expected = core.diagnose_update(learner, batch, returns, [probe], mc_weight=weight)
    actual, _ = verifier.verify_numeric_case(record, learner, batch, returns, [probe])
    for name in ("online", "target", "optimizer"):
        verifier.require_equal(getattr(actual, name).state_dict(), getattr(expected, name).state_dict(),
                               "independent complete numerical state")


@pytest.mark.parametrize("mutation", ("gradient_norm", "gradient_cosine", "adam_term", "probe_projection", "branch_q"))
def test_independent_numerics_detect_component_and_branch_tampering(mutation):
    core = verifier.load_module("gradient018_core_mutation_test",
                                ROOT / "docs/experiments/018_gradient_diagnostics/gradient_core.py")
    learner, batch, returns, probe = fixture()
    record, _ = core.diagnose_update(learner, batch, returns, [probe], mc_weight=1.0)
    if mutation == "gradient_norm":
        record["component_gradients"]["global"]["norms"]["td"] += 1
    elif mutation == "gradient_cosine":
        record["component_gradients"]["global"]["pairs"][0]["cosine"] += 0.1
    elif mutation == "adam_term":
        record["adam_decomposition"]["terms"]["old_first_moment"]["norm"] += 1
    elif mutation == "probe_projection":
        record["probes"][0]["negative_component_gradient_projections"]["margin"] += 1
    else:
        record["counterfactuals"]["reset_adam"]["probes"][0]["q_values"][0] += 1
    with pytest.raises(ValueError, match="Verification failed"):
        verifier.verify_numeric_case(record, learner, batch, returns, [probe])


def test_grouped_replay_uses_full_target_sources_and_costs_1395_per_arm():
    runner = verifier.load_module("gradient018_runner_group_test",
                                  ROOT / "docs/experiments/018_gradient_diagnostics/run_gradient_diagnostics.py")
    snapshots = {step: None for step in [2000, *range(2001, 2011), 2200, 2201,
                                        *range(2500, 10001, 500), *range(2400, 9401, 1000)]}
    expected = {2000: [2001], 2007: [2008], 2008: [2009], 2009: [2010],
                2010: [2018, 2152, 2158, 2161, 2200], 2200: [2201],
                2500: [3000], 4400: [4500], 8400: [8500], 9500: [10000]}
    assert runner.replay_groups(snapshots) == expected
    assert sum(max(cases) - start for start, cases in expected.items()) == 1395
    assert sum(max(cases) - start - len(cases) for start, cases in expected.items()) == 1381
    assert runner.source_before(snapshots, 8500) == 8400
    assert runner.source_before(snapshots, 2200) == 2010
    assert runner.source_before(snapshots, 2201) == 2200
    with pytest.raises(ValueError, match="No complete snapshot"):
        runner.source_before(snapshots, 2000)


def test_q_arithmetic_keeps_before_competitor_fixed_after_argmax_changes():
    initial = {"name": "probe", "demonstrated_action": 0, "fixed_competitor": 1,
               "q_values": [2.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0], "fixed_gap": 1.0,
               "greedy_action": 0}
    after = {"name": "probe", "demonstrated_action": 0, "fixed_competitor": 1,
             "q_values": [2.1, 1.1, 2.2, 0.0, 0.0, 0.0, 0.0, 0.0], "fixed_gap": 1.0,
             "greedy_action": 2, "best_other_after": 2, "competitor_changed": True,
             "greedy_gap": 2.1 - 2.2, "observed_fixed_gap_change": 0.0,
             "linear_gap_change": 0.05, "linear_gap_prediction": 1.05, "taylor_residual": -0.05}
    verifier.verify_q_arithmetic(initial)
    verifier.verify_q_arithmetic(after, initial)
    after["fixed_competitor"] = 2
    with pytest.raises(ValueError):
        verifier.verify_q_arithmetic(after, initial)


def structural_fixture(weight):
    core = verifier.load_module("gradient018_core_structural_test",
                                ROOT / "docs/experiments/018_gradient_diagnostics/gradient_core.py")
    torch.manual_seed(981)
    learner = DQNUpdater(5, 8, DQNConfig(hidden_sizes=(8, 8), learning_rate=0.0001,
                                         gradient_clip_norm=10, target_update_interval=200))
    batch = TransitionBatch(torch.randn(117, 5), torch.arange(117) % 8,
                            torch.linspace(-1, 2, 117), torch.randn(117, 5),
                            torch.arange(117) % 5 == 0, torch.zeros(117, dtype=torch.bool))
    learner.update(batch, demonstration_weight=1.0)
    learner.update_count = 2199
    for state in learner.optimizer.state.values():
        state["step"].fill_(22199)
    probes = [{"name": f"probe_{index}", "observation": batch.observations[index:index + 1],
               "demonstrated_action": int(batch.actions[index])} for index in range(9)]
    metadata = [{"name": row["name"], "demonstrated_action": row["demonstrated_action"]} for row in probes]
    record, _ = core.diagnose_update(learner, batch, torch.linspace(2, 4, 117), probes, mc_weight=weight)
    record.update(arm="mc_anchor" if weight else "baseline", pre_update_td=2199,
                  exact_historical_log_and_sentinel=True, exact_original_updater_state=True)
    return record, metadata


@pytest.mark.parametrize("weight", (0.0, 1.0))
def test_all_case_structural_audit_accepts_exact_clocks_nine_probes_and_alias_cost(weight):
    record, metadata = structural_fixture(weight)
    assert verifier.verify_case_structure(record, metadata) == (11 if weight else 10)


@pytest.mark.parametrize("mutation", ("missing_probe", "alias_charged", "stale_adam_clock", "wrong_target_sync",
                                      "loss_rescaled", "competitor_changed", "hidden_optimizer_step"))
def test_all_case_structural_audit_rejects_false_count_or_semantics(mutation):
    record, metadata = structural_fixture(0.0)
    if mutation == "missing_probe":
        record["counterfactuals"]["actual"]["probes"].pop()
    elif mutation == "alias_charged":
        record["counterfactuals"]["no_mc"]["optimizer_updates"] = 1
    elif mutation == "stale_adam_clock":
        record["counterfactuals"]["zero_first_moment"]["optimizer_steps_after"]["minimum"] = 1
    elif mutation == "wrong_target_sync":
        record["counterfactuals"]["actual"]["target_synchronized"] = False
    elif mutation == "loss_rescaled":
        record["counterfactuals"]["actual"]["loss_before"] /= 3
    elif mutation == "competitor_changed":
        row = record["counterfactuals"]["actual"]["probes"][0]
        row["competitor_changed"] = not row["competitor_changed"]
    else:
        record["independent_optimizer_steps"] += 1
    with pytest.raises(ValueError, match="Verification failed"):
        verifier.verify_case_structure(record, metadata)


def test_float32_weight_above_four_can_have_valid_rounding_residual_above_two_e_minus_seven():
    parameter = torch.tensor([-4.75], dtype=torch.float32)
    spacing = (parameter - torch.nextafter(parameter, torch.full_like(parameter, -torch.inf))).double()
    mathematical = -0.49 * spacing
    # The exact update is less than half one ULP, so storing the result changes no bits.
    stored = (parameter.double() + mathematical).float()
    assert torch.equal(stored, parameter)
    actual = stored.double() - parameter.double()
    residual = actual - mathematical
    assert float(residual.abs()) > 2e-7
    result = verifier.verify_rounding_envelope([stored], mathematical, residual, mathematical.abs())
    assert result["passed"]
    assert result["storage_half_spacing_max"] == float(spacing / 2)
    assert result["maximum_coordinate_fraction_of_envelope"] < 1


def test_scale_aware_envelope_rejects_error_beyond_weight_storage_and_adam_arithmetic():
    parameter = torch.tensor([-4.75], dtype=torch.float32)
    spacing = (parameter - torch.nextafter(parameter, torch.full_like(parameter, -torch.inf))).double()
    mathematical = -0.49 * spacing
    with pytest.raises(ValueError, match="scale-aware"):
        verifier.verify_rounding_envelope([parameter], mathematical, spacing, mathematical.abs())
