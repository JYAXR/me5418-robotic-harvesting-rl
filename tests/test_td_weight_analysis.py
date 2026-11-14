"""Check single-factor loss accounting, reuse semantics and complete Q-gap coverage."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analysis = load_module("td019_analysis_test", ROOT / "docs/experiments/019_td_weight/analyze_td_weight.py")
plotting = load_module("td019_plot_test", ROOT / "docs/experiments/019_td_weight/plot_td_weight.py")


def update_fixture(tmp_path, weight=0.5):
    row = {"update": 2001, "extension_update": 1,
           "loss_before": weight * 0.3 + 0.2, "loss_after": weight * 0.2 + 0.1,
           "td_loss_before": 0.3, "td_loss_after": 0.2,
           "demonstration_loss_before": 0.2, "demonstration_loss_after": 0.1,
           "gradient_norm": 12.0, "parameter_delta_norm": 0.001, "target_synchronized": False}
    if weight != 1:
        row["td_weight"] = weight
    path = tmp_path / "updates.jsonl"
    path.write_text(json.dumps(row), encoding="utf-8")
    return path, row


@pytest.mark.parametrize("weight", (0.5, 1.0))
def test_unweighted_components_are_retained_and_missing_mc_losses_are_not_zeroed(tmp_path, weight):
    _, row = update_fixture(tmp_path, weight)
    result = analysis.update_statistics(tmp_path, 10.0, weight, end=2001)
    assert result["first_update"] == row
    assert result["td_weight"] == weight and result["mc_weight"] == 0
    assert result["statistics"]["td_loss_before"]["mean"] == 0.3
    assert result["statistics"]["demonstration_loss_before"]["mean"] == 0.2
    assert result["historical_control_reused"] is (weight == 1)
    assert "return_loss_before" not in result["statistics"]
    assert "not replaced with zeros" in result["mc_log_availability"]
    assert result["gradient_norm_above_clip_count"] == 1
    assert result["nonzero_parameter_update_count"] == 1


@pytest.mark.parametrize("fault", ("normalize_total", "scale_margin", "missing_td_weight", "wrong_td_weight",
                                  "mc_loss", "mc_weight", "mc_target_bias", "nonfinite", "negative_component",
                                  "target_sync", "update_clock"))
def test_reject_objective_or_clock_drift(tmp_path, fault):
    path, row = update_fixture(tmp_path)
    if fault == "normalize_total":
        row["loss_before"] /= 1.5
    elif fault == "scale_margin":
        row["loss_after"] -= row["demonstration_loss_after"] / 2
    elif fault == "missing_td_weight":
        row.pop("td_weight")
    elif fault == "wrong_td_weight":
        row["td_weight"] = 1.0
    elif fault == "mc_loss":
        row["return_loss_before"] = 0.0
    elif fault == "mc_weight":
        row["return_anchor_weight"] = 0.0
    elif fault == "mc_target_bias":
        row["td_target_return_mean_difference"] = 0.0
    elif fault == "nonfinite":
        row["td_loss_before"] = float("nan")
    elif fault == "negative_component":
        row["td_loss_after"] = -0.2
    elif fault == "target_sync":
        row["target_synchronized"] = True
    else:
        row["extension_update"] = 2001
    path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError):
        analysis.update_statistics(tmp_path, 10.0, 0.5, end=2001)


def test_first_objective_changes_only_by_half_the_original_td_mean(tmp_path):
    _, control = update_fixture(tmp_path, 1.0)
    _, treatment = update_fixture(tmp_path, 0.5)
    result = analysis.first_objective_comparison(control, treatment)
    assert result["combined_loss_difference"] == pytest.approx(-0.15)
    assert result["expected_float64_difference"] == -0.15
    treatment["demonstration_loss_before"] += 0.001
    with pytest.raises(ValueError, match="share the original"):
        analysis.first_objective_comparison(control, treatment)


def retention_row(step):
    return {"update": step, "scenes": [
        {"seed": seed, "rollout": {"success": True, "steps": 41}} for seed in analysis.SEEDS]}


def test_historical_episodes_are_separate_from_new_main_work():
    control = [retention_row(step) for step in (2000, 2001)]
    treatment = deepcopy(control)
    control[-1]["scenes"][1]["rollout"]["success"] = False
    treatment[-1]["scenes"][0]["rollout"]["success"] = False
    result = analysis.compare_retention({"baseline": control, "treatment": treatment})
    assert result["new_episode_count"] == result["reused_historical_control_episode_count"] == 6
    assert result["total_compared_episode_records"] == 12
    assert result["treatment_rescued_scene_checks"][0]["seed"] == 10001
    assert result["treatment_regressed_scene_checks"][0]["seed"] == 10000


def phase_fixture():
    phase = {"optimizer_updates": 8000, "initial_td_update": 2000, "final_td_update": 10000,
             "target_synchronizations": 40, "learning_rate": 0.0001, "demonstration_weight": 1.0,
             "normalization_denominator": 117, "selected_sample_weight": 1.0, "td_weight": 0.5,
             "return_anchor_weight": 0.0, "all_sample_weights": 1.0,
             "checkpoint_sha256": dict.fromkeys(analysis.expected_model_names(), "digest")}
    summary = {"treatment": deepcopy(phase), "historical_control": {
        "return_anchor_weight": 0.0, "all_sample_weights": 1.0, "td_weight": 1.0,
        "new_optimizer_updates": 0, "new_evaluation_episodes": 0, "new_environment_steps": 0,
        "checkpoint_sha256": deepcopy(phase["checkpoint_sha256"])}}
    return phase, summary


def test_phase_roles_and_all_67_models_are_declared():
    phase, summary = phase_fixture()
    assert len(analysis.expected_model_names()) == 67
    analysis.validate_phase(phase, "baseline", summary)
    analysis.validate_phase(phase, "treatment", summary)


@pytest.mark.parametrize("fault", ("local_weight", "new_control_work", "td_weight", "mc_weight", "missing_model"))
def test_phase_validation_rejects_second_factors_and_recounted_history(fault):
    phase, summary = phase_fixture()
    arm = "baseline"
    if fault == "local_weight":
        phase["selected_sample_weight"] = 2
    elif fault == "new_control_work":
        summary["historical_control"]["new_optimizer_updates"] = 8000
    elif fault in ("td_weight", "mc_weight"):
        arm = "treatment"
        phase["td_weight" if fault == "td_weight" else "return_anchor_weight"] = 1.0
        summary["treatment"] = deepcopy(phase)
    else:
        phase["checkpoint_sha256"].pop("target_source_state_9400.pt")
        summary["historical_control"]["checkpoint_sha256"] = phase["checkpoint_sha256"]
    with pytest.raises(ValueError):
        analysis.validate_phase(phase, arm, summary)


@pytest.mark.parametrize("status", ("running", "failed", "interrupted"))
def test_incomplete_runs_stop_before_model_or_source_access(tmp_path, monkeypatch, status):
    (tmp_path / "run_status.json").write_text(json.dumps({"status": status}), encoding="utf-8")

    def forbidden(*args, **kwargs):
        pytest.fail("Incomplete runs must not access models or the source verifier.")

    monkeypatch.setattr(analysis, "import_file", forbidden)
    with pytest.raises(ValueError, match="completed run"):
        analysis.analyze(tmp_path)


def dense_fixture(tmp_path):
    values = np.zeros((24, 8), dtype=np.float32)
    values[:, 0] = 1.0
    rows = [{"update": update, "extension_update": update - 2000, "scene_seed": 10001,
             "states": [{"reference_step": step, "global_transition_index": step - 1,
                         "demonstrated_action": 0, "greedy_action": 0, "q_values": values[step - 1].tolist(),
                         "signed_demonstration_gap": 1.0} for step in analysis.FOCAL_STEPS]}
            for update in (2000, 2001)]
    path = tmp_path / "sentinel_q.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path, rows, values


def test_all_seven_dense_states_are_compared_and_plotted_including_step_six(tmp_path):
    path, _, values = dense_fixture(tmp_path)
    summary, dense = analysis.analyze_sentinels(
        path, np.zeros((24, 3), dtype=np.float32), np.zeros(24, dtype=np.int64), {"start": 0},
        {2000: values, 2001: values}, end=2001, checkpoints=(2000, 2001))
    assert summary["all_supervised_sample_coefficients"] == 1
    assert summary["reloaded_checkpoint_state_crosschecks"] == 14
    assert dense["updates"] == [2000, 2001]
    updates, gaps = plotting.dense_gaps(path, end=2001)
    assert updates == [2000, 2001]
    assert tuple(gaps) == analysis.FOCAL_STEPS and gaps[6] == [1.0, 1.0]


@pytest.mark.parametrize("fault", ("missing_step", "nonfinite", "duplicate_update", "wrong_scene", "reset_clock"))
def test_plotting_rejects_silent_probe_coverage_drift(tmp_path, fault):
    path, rows, _ = dense_fixture(tmp_path)
    if fault == "missing_step":
        rows[0]["states"].pop(0)
    elif fault == "nonfinite":
        rows[0]["states"][0]["signed_demonstration_gap"] = float("inf")
    elif fault == "duplicate_update":
        rows.append(deepcopy(rows[-1]))
    elif fault == "wrong_scene":
        rows[0]["scene_seed"] = 10002
    else:
        rows[0]["extension_update"] = 2000
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with pytest.raises(ValueError):
        plotting.dense_gaps(path, end=2001)


def certification_fixture():
    rows = [retention_row(2000)]
    rows[0]["aggregate"] = {"evaluation_environment_steps": 123}
    records = [{"update": 2000, "seed": seed, "certification_attempts": 1,
                "certification_environment_steps": 42, "reference_route_checks": [{"steps": 42}]}
               for seed in analysis.SEEDS]
    certification = {"records": records, "completed_reset_count": 3, "certification_attempts": 3,
                     "certification_environment_steps": 126, "accounting_complete": True, "incomplete_resets": []}
    accounting = {"environment_resets": 3, "reset_certification_attempts": 3,
                  "reset_certification_environment_steps": 126, "evaluation_environment_steps": 123,
                  "total_simulator_environment_steps": 249, "reset_accounting_complete": True,
                  "incomplete_resets": [], "reused_historical_control": {"evaluation_environment_steps": 3711}}
    return accounting, certification, rows


def test_reset_certification_is_not_counted_as_policy_actions_or_fabricated_for_history():
    accounting, certification, rows = certification_fixture()
    result = analysis.simulator_accounting(accounting, certification, rows)
    assert result["new_policy_evaluation_steps"] == 123
    assert result["new_reset_certification_steps"] == 126
    assert result["new_total_simulator_steps"] == 249
    assert result["historical_reset_certification_steps"] is None
    assert result["historical_policy_evaluation_steps"] == 3711


@pytest.mark.parametrize("fault", ("double_count", "missing_reset", "wrong_candidate_steps", "incomplete"))
def test_simulator_accounting_rejects_mixed_or_incomplete_work(fault):
    accounting, certification, rows = certification_fixture()
    if fault == "double_count":
        accounting["evaluation_environment_steps"] += 126
    elif fault == "missing_reset":
        certification["records"].pop()
    elif fault == "wrong_candidate_steps":
        certification["records"][0]["reference_route_checks"][0]["steps"] = 43
    else:
        certification["accounting_complete"] = False
    with pytest.raises(ValueError):
        analysis.simulator_accounting(accounting, certification, rows)
