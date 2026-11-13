"""Verify objective accounting and reuse semantics in the read-only MC-anchor analysis."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "anchor017_analysis_test", ROOT / "docs/experiments/017_mc_anchor/analyze_mc_anchor.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def update_fixture(tmp_path, anchored=True):
    row = {
        "update": 2001, "extension_update": 1, "loss_before": 0.8 if anchored else 0.5,
        "loss_after": 0.6 if anchored else 0.3, "td_loss_before": 0.3, "td_loss_after": 0.2,
        "demonstration_loss_before": 0.2, "demonstration_loss_after": 0.1,
        "gradient_norm": 12.0, "parameter_delta_norm": 0.001, "target_synchronized": False,
    }
    if anchored:
        row.update(return_loss_before=0.3, return_loss_after=0.3, return_anchor_weight=1.0,
                   td_target_return_rmse=0.4, td_target_return_mean_difference=-0.1,
                   td_target_return_max_abs_difference=0.8)
    path = tmp_path / "updates.jsonl"
    path.write_text(json.dumps(row), encoding="utf-8")
    return path, row


def test_historical_missing_mc_losses_are_not_fabricated_as_zeros(tmp_path):
    update_fixture(tmp_path, anchored=False)
    result = analysis.update_statistics(tmp_path, 10.0, 0.0, end=2001)
    assert result["historical_control_reused"] is True
    assert result["mc_weight"] == 0
    assert "return_loss_before" not in result["statistics"]
    assert "not replaced with zeros" in result["mc_log_availability"]
    assert result["nonzero_parameter_update_count"] == 1


def test_three_means_are_summed_and_signed_target_bias_stays_signed(tmp_path):
    _, row = update_fixture(tmp_path)
    result = analysis.update_statistics(tmp_path, 10.0, 1.0, end=2001)
    assert result["first_update"] == row
    assert result["normalization_denominator"] == 117
    assert result["statistics"]["return_loss_before"]["mean"] == 0.3
    assert result["statistics"]["td_target_return_mean_difference"]["mean"] == -0.1
    assert result["gradient_norm_above_clip_count"] == 1
    assert "old target generation" in result["td_target_return_gap_semantics"]


@pytest.mark.parametrize("fault", ["divide_total", "missing_component", "anchor_weight", "nonfinite",
                                  "negative_return_loss", "target_bias", "target_sync", "update_clock"])
def test_analysis_rejects_objective_or_clock_drift(tmp_path, fault):
    path, row = update_fixture(tmp_path)
    if fault == "divide_total":
        row["loss_before"] /= 3
    elif fault == "missing_component":
        row["loss_after"] -= row["demonstration_loss_after"]
    elif fault == "anchor_weight":
        row["return_anchor_weight"] = 2
    elif fault == "nonfinite":
        row["td_target_return_rmse"] = float("nan")
    elif fault == "negative_return_loss":
        row["return_loss_after"] = -0.3
    elif fault == "target_bias":
        row["td_target_return_mean_difference"] = 1.0
    elif fault == "target_sync":
        row["target_synchronized"] = True
    else:
        row["extension_update"] = 2001
    path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError):
        analysis.update_statistics(tmp_path, 10.0, 1.0, end=2001)


def retention_row(step):
    return {"update": step, "scenes": [
        {"seed": seed, "rollout": {"success": True, "steps": 41}} for seed in analysis.SEEDS]}


def test_reused_control_episodes_are_not_counted_as_new_computation():
    baseline = [retention_row(step) for step in (2000, 2001)]
    treatment = deepcopy(baseline)
    baseline[-1]["scenes"][1]["rollout"]["success"] = False
    treatment[-1]["scenes"][0]["rollout"]["success"] = False
    result = analysis.compare_retention({"baseline": baseline, "treatment": treatment})
    assert result["new_episode_count"] == 6
    assert result["reused_historical_control_episode_count"] == 6
    assert result["total_compared_episode_records"] == 12
    assert result["treatment_rescued_scene_checks"][0]["seed"] == 10001
    assert result["treatment_regressed_scene_checks"][0]["seed"] == 10000


def test_probe_labels_do_not_claim_local_reweighting(tmp_path):
    values = np.zeros((24, 8), dtype=np.float32)
    values[:, 0] = 1.0
    observations, actions = np.zeros((24, 3), dtype=np.float32), np.zeros(24, dtype=np.int64)
    rows = [{"update": step, "extension_update": step - 2000, "scene_seed": 10001,
             "states": [{"reference_step": point, "global_transition_index": point - 1,
                         "demonstrated_action": 0, "greedy_action": 0, "q_values": values[point - 1].tolist(),
                         "signed_demonstration_gap": 1.0} for point in analysis.FOCAL_STEPS]}
            for step in (2000, 2001)]
    path = tmp_path / "sentinel_q.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    summary, dense = analysis.analyze_sentinels(
        path, observations, actions, {"start": 0}, {2000: values, 2001: values},
        end=2001, checkpoints=(2000, 2001))
    assert summary["reference_states"]["21"]["role"] == "Previous first-divergence sentinel"
    assert summary["all_supervised_sample_coefficients"] == 1
    assert summary["reloaded_checkpoint_state_crosschecks"] == 14
    assert dense["updates"] == [2000, 2001]


def test_state_comparison_names_calibration_reference_without_calling_it_control():
    values = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    calibrated = np.array([[0.9, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    result = analysis.state_comparison(values, calibrated, np.zeros((1, 3), dtype=np.float32),
                                       np.array([0]), {"seed": 10001, "start": 0}, 1)
    assert "baseline_same_state" not in result
    assert result["calibration_reference_same_state"]["chosen_action"] == 1
    assert result["chosen_action"] == 0


def phase_fixture():
    phase = {"optimizer_updates": 8000, "initial_td_update": 2000, "final_td_update": 10000,
             "target_synchronizations": 40, "learning_rate": 0.0001, "demonstration_weight": 1.0,
             "normalization_denominator": 117, "selected_sample_weight": 1.0,
             "return_anchor_weight": 1.0, "all_sample_weights": 1.0,
             "checkpoint_sha256": dict.fromkeys(analysis.expected_model_names(), "digest")}
    summary = {"treatment": deepcopy(phase), "historical_control": {
        "return_anchor_weight": 0.0, "all_sample_weights": 1.0, "new_optimizer_updates": 0,
        "new_evaluation_episodes": 0, "new_environment_steps": 0,
        "checkpoint_sha256": deepcopy(phase["checkpoint_sha256"])}}
    return phase, summary


def test_phase_roles_and_model_count_are_explicit():
    phase, summary = phase_fixture()
    assert len(analysis.expected_model_names()) == 67
    analysis.validate_phase(phase, "baseline", summary)
    analysis.validate_phase(phase, "treatment", summary)


@pytest.mark.parametrize("fault", ["local_weight", "new_control_work", "anchor_weight", "missing_model"])
def test_phase_validation_prevents_second_factors_or_recounting_history(fault):
    phase, summary = phase_fixture()
    arm = "baseline"
    if fault == "local_weight":
        phase["selected_sample_weight"] = 2
    elif fault == "new_control_work":
        summary["historical_control"]["new_optimizer_updates"] = 8000
    elif fault == "anchor_weight":
        arm = "treatment"
        phase["return_anchor_weight"] = 0.5
        summary["treatment"] = deepcopy(phase)
    else:
        phase["checkpoint_sha256"].pop("target_source_state_9400.pt")
        summary["historical_control"]["checkpoint_sha256"] = phase["checkpoint_sha256"]
    with pytest.raises(ValueError):
        analysis.validate_phase(phase, arm, summary)


@pytest.mark.parametrize("status", ["running", "failed", "interrupted"])
def test_incomplete_analysis_stops_before_accessing_models_or_sources(tmp_path, monkeypatch, status):
    (tmp_path / "run_status.json").write_text(json.dumps({"status": status}), encoding="utf-8")

    def forbidden(*args, **kwargs):
        pytest.fail("Incomplete analysis must not access the verifier or source models.")

    monkeypatch.setattr(analysis, "import_file", forbidden)
    with pytest.raises(ValueError, match="completed run"):
        analysis.analyze(tmp_path)


def test_complete_dense_series_writer_preserves_rows_and_refuses_overwrite(tmp_path):
    result = {"arms": {arm: {} for arm in analysis.ARMS}}
    dense = {arm: {"updates": [2000, 2001], "reference_states": {"21": {
        "signed_demonstration_gaps": [0.1, -0.2], "greedy_actions": [0, 4]}}} for arm in analysis.ARMS}
    path = tmp_path / "q_analysis.json"
    analysis.write_outputs(path, result, dense)
    for arm in analysis.ARMS:
        artifact = tmp_path / f"q_analysis_{arm}_dense.json"
        assert json.loads(artifact.read_text()) == dense[arm]
        assert result["arms"][arm]["dense_series_sha256"] == analysis.file_hash(artifact)
    with pytest.raises(ValueError, match="overwrite"):
        analysis.write_outputs(path, result, dense)
