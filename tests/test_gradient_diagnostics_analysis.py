"""Test local ranking and Adam-summary semantics without model or environment work."""

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "gradient018_analysis_test", ROOT / "docs/experiments/018_gradient_diagnostics/analyze_gradient_diagnostics.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def case_fixture(arm="baseline", name="scene_10001_step_21"):
    initial, final, projection, mathematical = 0.2, -0.1, -0.25, -0.26
    before = {
        "name": name, "demonstrated_action": 1, "fixed_competitor": 2, "greedy_action": 1,
        "fixed_gap": initial, "negative_component_gradient_projections": {"td": 0.5, "margin": 1.5, "mc": -1.0},
        "weighted_negative_component_gradient_projections": {"td": 0.5, "margin": 1.5, "mc": 0.0},
        "negative_actual_gradient_projection": 2.0, "negative_clipped_gradient_projection": 1.0,
        "layer_projections": [], "adam_term_projections": {
            "old_first_moment": -0.3, "current_td": 0.01, "current_margin": 0.03,
            "current_mc": 0.0, "rounding_remainder": 0.0},
        "adam_mathematical_delta_projection": mathematical, "actual_delta_projection": projection,
    }
    after = {
        "name": name, "demonstrated_action": 1, "fixed_competitor": 2,
        "greedy_action": 2, "best_other_after": 2, "competitor_changed": False,
        "fixed_gap": final, "greedy_gap": final, "observed_fixed_gap_change": final - initial,
        "taylor_residual": final - (initial + projection),
    }
    actual = {
        "alias_of": None, "optimizer_updates": 1, "probes": [after],
        "gradient_norm": 20.0, "clip_coefficient": 0.5, "parameter_delta_norm_float64": 0.1,
        "optimizer_steps_before": {"minimum": 22000, "maximum": 22000},
        "optimizer_steps_after": {"minimum": 22001, "maximum": 22001},
    }
    branches = {name: deepcopy(actual) for name in analysis.BRANCHES}
    if arm == "baseline":
        branches["no_mc"] = {"alias_of": "actual", "optimizer_updates": 0}
    geometry = {"norms": {"td": 1.0, "margin": 2.0, "mc": 0.0}, "pairs": [
        {"first": "td", "second": "margin", "dot": -1.0, "cosine": -0.5},
        {"first": "td", "second": "mc", "dot": 0.0, "cosine": None},
        {"first": "margin", "second": "mc", "dot": 0.0, "cosine": None}]}
    case = {
        "arm": arm, "mc_weight": float(arm != "baseline"), "incoming_td_update": 2009,
        "pre_update_td": 2008, "source_td_update": 2008,
        "source_state_and_rng_unchanged": True, "exact_historical_log_and_sentinel": True,
        "exact_original_updater_state": True, "probes": [before], "counterfactuals": branches,
        "losses_before": {"td": 0.1, "margin": 0.2, "mc": 0.3},
        "component_gradients": {"global": geometry, "layers": []}, "adam_decomposition": {},
        "independent_optimizer_steps": 10 if arm == "baseline" else 11,
    }
    metadata = {name: {"demonstrated_action": 1}}
    return case, metadata


def test_adam_finite_step_can_oppose_naive_gradient_without_inventing_a_cause():
    case, metadata = case_fixture()
    result = analysis.summarize_case(case, metadata)
    row = result["probes"][0]
    assert row["naive_direction_sign"] == "positive"
    assert row["finite_adam_gap_change_sign"] == "negative"
    assert row["naive_vs_finite_adam_sign_mismatch"]
    assert row["naive_vs_adam_parameter_projection_sign_mismatch"]
    assert row["parameter_storage_and_arithmetic_projection_residual"] == pytest.approx(0.01)
    assert row["nonlinear_fixed_gap_residual"] == pytest.approx(-0.05)
    assert abs(row["decomposition_closure_residual"]) < 1e-15
    assert row["counterfactuals"]["actual"]["positive_to_negative_gap"]
    assert row["counterfactuals"]["actual"]["action_regressed_from_before"]


def test_global_gradient_opposition_does_not_imply_opposite_local_gap_directions():
    case, metadata = case_fixture()
    pairs = analysis.summarize_case(case, metadata)["global_alignment_vs_local_gap_directions"]
    assert pairs[0]["globally_opposed"]
    assert pairs[0]["local_opposite_direction_probe_count"] == 0
    assert pairs[0]["local_both_gap_increasing_count"] == 1
    assert pairs[1]["global_cosine"] is None and not pairs[1]["globally_opposed"]


def test_aliases_are_resolved_but_never_counted_as_additional_optimizer_updates():
    case, metadata = case_fixture()
    summary = analysis.summarize_case(case, metadata)
    counts = analysis.aggregate_cases([summary])["baseline"]
    assert counts["local_probe_update_pairs"] == 1
    assert counts["independent_optimizer_steps"] == 10
    assert counts["counterfactuals"]["no_mc"]["alias_case_count"] == 1
    assert summary["probes"][0]["counterfactuals"]["no_mc"]["after_greedy_gap"] == -0.1


def test_ties_are_separate_from_strict_gap_crossings_and_actual_argmax_recovery():
    case, metadata = case_fixture()
    before = case["probes"][0]
    before["fixed_gap"], before["greedy_action"] = 0.0, 0
    for branch in case["counterfactuals"].values():
        if branch.get("alias_of"):
            continue
        row = branch["probes"][0]
        row.update(fixed_gap=0.1, greedy_gap=0.1, greedy_action=1,
                   observed_fixed_gap_change=0.1, taylor_residual=0.35)
    row = analysis.summarize_case(case, metadata)["probes"][0]["counterfactuals"]["actual"]
    assert not row["negative_to_positive_gap"]
    assert row["action_recovered_from_before"]


def test_competitor_switch_uses_actual_max_other_gap_for_ranking():
    case, metadata = case_fixture()
    branch = case["counterfactuals"]["reset_adam"]["probes"][0]
    branch.update(fixed_gap=0.4, greedy_gap=-0.2, best_other_after=6,
                  competitor_changed=True, greedy_action=6, observed_fixed_gap_change=0.2)
    row = analysis.summarize_case(case, metadata)["probes"][0]["counterfactuals"]["reset_adam"]
    assert row["after_fixed_gap"] > 0 and row["after_greedy_gap"] < 0
    assert row["positive_to_negative_gap"] and not row["action_recovered_relative_to_actual"]


def test_arm_counts_remain_separate_and_recovery_is_local_only():
    baseline, metadata = case_fixture()
    treatment, _ = case_fixture("mc_anchor")
    after = treatment["counterfactuals"]["reset_adam"]["probes"][0]
    after.update(greedy_gap=0.5, fixed_gap=0.5, greedy_action=1, observed_fixed_gap_change=0.3)
    summaries = [analysis.summarize_case(case, metadata) for case in (baseline, treatment)]
    counts = analysis.aggregate_cases(summaries)
    assert counts["baseline"]["counterfactuals"]["reset_adam"]["action_recovered_relative_to_actual_count"] == 0
    assert counts["mc_anchor"]["counterfactuals"]["reset_adam"]["action_recovered_relative_to_actual_count"] == 1
    assert counts["baseline"]["independent_optimizer_steps"] == 10
    assert counts["mc_anchor"]["independent_optimizer_steps"] == 11


@pytest.mark.parametrize("fault", ["alias_cycle", "missing_branch", "wrong_clock", "missing_probe", "bad_residual", "nonfinite"])
def test_analysis_rejects_incomplete_or_inconsistent_case_evidence(fault):
    case, metadata = case_fixture()
    if fault == "alias_cycle":
        case["counterfactuals"]["actual"] = {"alias_of": "no_mc", "optimizer_updates": 0}
    elif fault == "missing_branch":
        case["counterfactuals"].pop("td_only")
    elif fault == "wrong_clock":
        case["pre_update_td"] = 2007
    elif fault == "missing_probe":
        case["counterfactuals"]["reset_adam"]["probes"] = []
    elif fault == "bad_residual":
        case["counterfactuals"]["actual"]["probes"][0]["taylor_residual"] = 0.8
    else:
        case["probes"][0]["fixed_gap"] = float("nan")
    with pytest.raises(ValueError):
        analysis.summarize_case(case, metadata)


def test_zero_projection_is_neither_positive_nor_a_sign_mismatch():
    case, metadata = case_fixture()
    case["probes"][0]["negative_actual_gradient_projection"] = 0.0
    summary = analysis.summarize_case(case, metadata)
    assert summary["probes"][0]["naive_direction_sign"] == "zero"
    counts = analysis.aggregate_cases([summary])["baseline"]
    assert counts["naive_vs_finite_adam_sign_mismatch_count"] == 0
    assert counts["zero_naive_or_finite_change_count"] == 1


def test_failure_context_keeps_earlier_divergence_distinct_from_later_probe():
    case, metadata = case_fixture("mc_anchor", "scene_10002_actual_step_40")
    summary = analysis.summarize_case(case, metadata)
    summary["incoming_td_update"] = 8500
    extra = deepcopy(summary["probes"][0])
    extra["name"] = "scene_10002_step_37"
    summary["probes"].append(extra)
    rollout = {"success": False, "steps": 200, "return": 7.5, "damage": False,
               "budget_exhausted": True, "first_reference_action_divergence": {"step": 37}}
    observations = {"mc_anchor": [{"update": 8500, "scenes": [{"seed": 10002, "rollout": rollout}]}]}
    failures, comparators = analysis.historical_contexts([summary], observations)
    assert len(failures) == 1 and comparators == []
    assert failures[0]["diagnostic_probe"]["name"] == "scene_10002_actual_step_40"
    assert failures[0]["earliest_reference_divergence_context"]["name"] == "scene_10002_step_37"
    assert "Neither observation alone proves a cause" in failures[0]["selection_note"]


def test_input_file_cap_prevents_unbounded_diagnostic_files(tmp_path):
    path = tmp_path / "oversized.json"
    with path.open("wb") as stream:
        stream.truncate(analysis.MAX_BYTES + 1)
    with pytest.raises(ValueError, match="8 MiB"):
        analysis.read_json(path)
