"""Keep local reference-state rankings separate from measured three-scene outcomes."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "local016_analysis_test", ROOT / "docs/experiments/016_local_supervision/analyze_local_supervision.py",
)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def test_compact_intervals_preserve_ranking_events_without_inventing_failed_episodes():
    result = analysis.series_summary(
        [2000, 2001, 2002, 2003, 2004], [1., -0.2, -0.1, 0., 0.5],
        [2, 6, 6, 0, 2], 2, (2000, 2004),
    )
    assert result["negative_gap_count_new_updates"] == 2
    assert result["greedy_disagreement_count_new_updates"] == 3
    assert result["greedy_disagreement_at_zero_gap_count"] == 1
    assert result["minimum_gap_update"] == 2001
    assert result["negative_gap_intervals"] == [[2001, 2002, 2, -0.2, 2001, []]]
    assert result["greedy_disagreement_intervals"] == [[2001, 2003, 3, -0.2, 2001, []]]
    assert result["negative_intervals_entirely_between_physical_checks"] == 1
    assert result["actual_failed_rollouts_inferred"] == 0


def test_resume_boundary_is_not_counted_as_a_new_negative_update():
    result = analysis.series_summary([2000, 2001], [-0.1, 1.], [6, 2], 2, (2000, 2001))
    assert result["negative_gap_count_including_resume_boundary"] == 1
    assert result["negative_gap_count_new_updates"] == 0
    assert result["negative_gap_intervals"][0][5] == [2000]


def sentinel_fixture(tmp_path):
    observations = np.zeros((24, 3), dtype=np.float32)
    actions = np.zeros(24, dtype=np.int64)
    values = np.zeros((24, 8), dtype=np.float32)
    values[:, 0] = 1.
    rows = [{
        "update": step, "extension_update": step - 2000, "scene_seed": 10001,
        "states": [{"reference_step": point, "global_transition_index": point - 1,
                    "demonstrated_action": 0, "greedy_action": 0,
                    "q_values": values[point - 1].tolist(), "signed_demonstration_gap": 1.}
                   for point in analysis.FOCAL_STEPS],
    } for step in (2000, 2001)]
    path = tmp_path / "sentinel_q.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path, rows, observations, actions, values


def test_seven_dense_states_crosscheck_models_and_retain_every_compact_value(tmp_path):
    path, _, observations, actions, values = sentinel_fixture(tmp_path)
    result, dense = analysis.analyze_sentinels(
        path, observations, actions, {"start": 0}, {2000: values, 2001: values},
        end=2001, checkpoints=(2000, 2001),
    )
    assert result["reloaded_checkpoint_state_crosschecks"] == 14
    assert result["reference_states"]["21"]["role"] == "Locally reweighted state"
    assert result["reference_states"]["20"]["role"] == "Adjacent decision"
    assert dense["reference_states"]["21"]["signed_demonstration_gaps"] == [1., 1.]
    assert "negative_gap_intervals" not in result["reference_states"]["21"]
    assert "negative_gap_intervals" in dense["reference_states"]["21"]


@pytest.mark.parametrize("fault", ["missing", "clock", "row", "gap", "argmax", "model", "neighbor"])
def test_dense_analysis_rejects_incomplete_or_changed_reference_evidence(tmp_path, fault):
    path, rows, observations, actions, values = sentinel_fixture(tmp_path)
    if fault == "missing":
        rows.pop()
    elif fault == "clock":
        rows[1]["extension_update"] = 2001
    elif fault == "model":
        values[20, 0] = 2.
    elif fault == "neighbor":
        rows[0]["states"].pop()
    else:
        key = {"row": "global_transition_index", "gap": "signed_demonstration_gap", "argmax": "greedy_action"}[fault]
        rows[0]["states"][3][key] += 1
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with pytest.raises(ValueError):
        analysis.analyze_sentinels(path, observations, actions, {"start": 0}, {2000: values},
                                   end=2001, checkpoints=(2000, 2001))


def paired_dense_fixture():
    base = {"updates": [2000, 2001, 2002], "reference_states": {
        str(step): {"demonstrated_action": 0, "signed_demonstration_gaps": [1., 1., 1.],
                    "greedy_actions": [0, 0, 0]} for step in analysis.FOCAL_STEPS}}
    result = {"baseline": deepcopy(base), "treatment": deepcopy(base)}
    result["baseline"]["reference_states"]["21"].update(
        signed_demonstration_gaps=[1., -0.1, -0.2], greedy_actions=[0, 4, 4])
    result["treatment"]["reference_states"]["22"].update(
        signed_demonstration_gaps=[1., -0.1, 1.], greedy_actions=[0, 4, 0])
    return result


def test_better_target_gap_does_not_hide_neighbor_action_regression():
    result = analysis.compare_dense(paired_dense_fixture())
    target = result["reference_states"]["21"]
    assert target["demonstration_action_improvement_new_update_count"] == 2
    assert target["negative_gap_count_baseline_new_updates"] == 2
    assert result["reference_states"]["22"]["demonstration_action_regression_new_update_count"] == 1
    assert result["higher_step21_gap_with_neighbor_action_regression_new_update_count"] == 1
    assert result["any_neighbor_action_regression_new_update_count"] == 1
    assert "not inferred physical" in result["interpretation"]


def test_paired_dense_clocks_must_match():
    pairs = paired_dense_fixture()
    pairs["treatment"]["updates"][-1] = 2003
    with pytest.raises(ValueError, match="clocks"):
        analysis.compare_dense(pairs)


def update_fixture(directory, weight=2):
    before, after, hinge_before, hinge_after = 0.2, 0.1, 0.7, 0.6
    weighted_before = before + (weight - 1) * hinge_before / 117
    weighted_after = after + (weight - 1) * hinge_after / 117
    metrics = {"update": 2001, "extension_update": 1, "loss_before": 0.3 + weighted_before,
               "loss_after": 0.2 + weighted_after, "td_loss_before": 0.3, "td_loss_after": 0.2,
               "demonstration_loss_before": weighted_before, "demonstration_loss_after": weighted_after,
               "gradient_norm": 12., "parameter_delta_norm": 0.001, "target_synchronized": False}
    supplemental = {"update": 2001, "extension_update": 1, "unweighted_margin_before": before,
                    "unweighted_margin_after": after, "weighted_margin_before": weighted_before,
                    "weighted_margin_after": weighted_after, "selected_hinge_before": hinge_before,
                    "selected_hinge_after": hinge_after, "selected_sample_weight": weight,
                    "normalization_denominator": 117}
    (directory / "updates.jsonl").write_text(json.dumps(metrics), encoding="utf-8")
    (directory / "supervision.jsonl").write_text(json.dumps(supplemental), encoding="utf-8")
    return metrics, supplemental


def test_weighted_and_unweighted_losses_are_recorded_with_fixed_denominator(tmp_path):
    _, supplemental = update_fixture(tmp_path)
    result = analysis.update_statistics(tmp_path, 10, 2, end=2001)
    assert result["gradient_norm_above_clip_count"] == 1
    assert result["nonzero_parameter_update_count"] == 1
    assert result["normalization_denominator"] == 117
    assert result["first_supervision_record"] == supplemental
    assert result["supervision_statistics"]["unweighted_margin_before"]["mean"] == 0.2


@pytest.mark.parametrize("fault", ["normalization", "coefficient", "weighted_loss", "clock", "target_sync"])
def test_update_analysis_rejects_changed_single_factor_or_clock(tmp_path, fault):
    metrics, supplemental = update_fixture(tmp_path)
    if fault == "normalization":
        supplemental["normalization_denominator"] = 118
    elif fault == "coefficient":
        supplemental["selected_sample_weight"] = 3
    elif fault == "weighted_loss":
        supplemental["weighted_margin_before"] /= 2
    elif fault == "clock":
        supplemental["update"] = 1
    else:
        metrics["target_synchronized"] = True
    (tmp_path / "updates.jsonl").write_text(json.dumps(metrics), encoding="utf-8")
    (tmp_path / "supervision.jsonl").write_text(json.dumps(supplemental), encoding="utf-8")
    with pytest.raises(ValueError):
        analysis.update_statistics(tmp_path, 10, 2, end=2001)


def retention_row(step):
    return {"update": step, "scenes": [
        {"seed": seed, "rollout": {"success": True, "steps": 41}} for seed in analysis.SEEDS]}


def test_retention_keeps_new_scene_failures_even_when_target_scene_is_rescued():
    baseline = [retention_row(step) for step in (2000, 2001)]
    treatment = deepcopy(baseline)
    baseline[-1]["scenes"][1]["rollout"]["success"] = False
    treatment[-1]["scenes"][0]["rollout"]["success"] = False
    result = analysis.compare_retention({"baseline": baseline, "treatment": treatment})
    assert result["new_episode_count"] == 12
    assert result["treatment_rescued_scene_checks"][0]["seed"] == 10001
    assert result["treatment_regressed_scene_checks"][0]["seed"] == 10000
    assert result["arms"]["treatment"]["all_scene_success_checkpoint_count"] == 1


@pytest.mark.parametrize("status", ["running", "failed", "interrupted"])
def test_analysis_refuses_incomplete_runs_before_model_or_source_access(tmp_path, monkeypatch, status):
    (tmp_path / "run_status.json").write_text(json.dumps({"status": status}), encoding="utf-8")

    def forbidden(*args, **kwargs):
        pytest.fail("Incomplete runs must not load source models or import verification.")

    monkeypatch.setattr(analysis, "import_file", forbidden)
    with pytest.raises(ValueError, match="completed run"):
        analysis.analyze(tmp_path)


def test_dense_artifacts_retain_all_rows_and_refuse_overwrite(tmp_path):
    result = {"arms": {"baseline": {}, "treatment": {}}}
    dense = paired_dense_fixture()
    output = tmp_path / "q_analysis.json"
    analysis.write_outputs(output, result, dense)
    for arm in analysis.ARMS:
        stored = json.loads((tmp_path / f"q_analysis_{arm}_dense.json").read_text())
        assert stored == dense[arm]
        assert result["arms"][arm]["dense_series_sha256"] == analysis.file_hash(
            tmp_path / f"q_analysis_{arm}_dense.json")
    with pytest.raises(ValueError, match="overwrite"):
        analysis.write_outputs(output, result, dense)
