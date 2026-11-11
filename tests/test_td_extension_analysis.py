"""Guard the distinction between dense reference rankings and physical retention."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "extension014_analysis_test", ROOT / "docs/experiments/014_td_extension/analyze_td_extension.py",
)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def test_reference_rank_intervals_do_not_become_rollout_failures():
    result = analysis.sentinel_series_summary(
        list(range(2000, 2007)), [1, -0.1, -0.2, 1, 0, -0.3, 1],
        [2, 6, 6, 2, 0, 6, 2], 2, (2000, 2003, 2006),
    )
    assert result["negative_gap_count_including_resume_boundary"] == 3
    assert result["negative_gap_count_new_updates"] == 3
    assert result["greedy_disagreement_at_zero_gap_count"] == 1
    assert result["minimum_gap_update"] == 2005
    assert result["first_negative_gap_update"] == 2001
    assert result["negative_intervals_entirely_between_physical_checks"] == 2
    first, last = result["negative_gap_intervals"]
    assert (first["start_update"], first["end_update"], first["update_count"]) == (2001, 2002, 2)
    assert first["preceding_physical_checkpoint"] == 2000
    assert first["following_physical_checkpoint"] == 2003
    assert last["scheduled_physical_checkpoints_within_interval"] == []
    assert all(not row["actual_failed_rollout_inferred"] for row in result["rank_transitions"])
    assert all(not row["actual_failed_rollout_inferred"] for row in result["negative_gap_intervals"])
    assert len(result["greedy_disagreement_intervals"]) == 2
    assert result["greedy_disagreement_intervals"][-1]["start_update"] == 2004


def test_initial_negative_control_is_counted_separately_from_new_updates():
    result = analysis.sentinel_series_summary([2000, 2001], [-0.1, 0.2], [6, 2], 2, (2000, 2001))
    assert result["negative_gap_count_including_resume_boundary"] == 1
    assert result["negative_gap_count_new_updates"] == 0
    assert result["negative_gap_intervals"][0]["scheduled_physical_checkpoints_within_interval"] == [2000]


def test_missing_update_breaks_contiguous_interval():
    result = analysis.intervals([1, 2, 4], [True, True, True], [-1, -2, -3], ())
    assert [(row["start_update"], row["end_update"]) for row in result] == [(1, 2), (4, 4)]


def sentinel_fixture(tmp_path):
    observations, actions = np.zeros((24, 3), dtype=np.float32), np.full(24, 2, dtype=int)
    values = np.zeros((24, 8), dtype=np.float32)
    values[:, 2] = 1
    rows = [{"update": update, "extension_update": update - 2000, "scene_seed": 10001,
             "states": [{"reference_step": step, "global_transition_index": step - 1,
                         "demonstrated_action": 2, "greedy_action": 2,
                         "q_values": values[step - 1].tolist(), "signed_demonstration_gap": 1.0}
                        for step in analysis.FOCAL_STEPS]} for update in (2000, 2001)]
    path = tmp_path / "sentinel.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path, rows, observations, actions, values


def test_dense_sentinels_recompute_raw_values_and_crosscheck_saved_models(tmp_path):
    path, _, observations, actions, values = sentinel_fixture(tmp_path)
    result = analysis.analyze_sentinels(path, observations, actions, {"start": 0}, {2000: values, 2001: values},
                                        end=2001, checkpoints=(2000, 2001))
    assert result["reloaded_checkpoint_state_crosschecks"] == 8
    assert result["raw_records"] == 2
    assert result["reference_states"]["24"]["role"].startswith("Known successful")


@pytest.mark.parametrize("fault", ["missing", "clock", "action", "gap", "dataset_index", "saved_model"])
def test_dense_sentinels_reject_missing_or_falsified_evidence(tmp_path, fault):
    path, rows, observations, actions, values = sentinel_fixture(tmp_path)
    if fault == "missing":
        rows.pop()
    elif fault == "clock":
        rows[1]["extension_update"] = 2001
    elif fault == "saved_model":
        values[5, 2] = 2
    else:
        key = {"action": "greedy_action", "gap": "signed_demonstration_gap",
               "dataset_index": "global_transition_index"}[fault]
        rows[0]["states"][0][key] += 1
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with pytest.raises(ValueError):
        analysis.analyze_sentinels(path, observations, actions, {"start": 0}, {2000: values}, end=2001)


def retention_row(update):
    return {"update": update, "scenes": [{"seed": seed, "rollout": {"success": True, "steps": length}}
                                         for seed, length in zip(analysis.SEEDS, (35, 41, 41), strict=True)],
            "aggregate": {"success_count": 3}}


def test_historical_evidence_is_reused_without_double_counting_boundary():
    historical = [retention_row(update) for update in analysis.HISTORICAL_CHECKPOINTS]
    continuation = [retention_row(update) for update in analysis.CHECKPOINTS]
    result = analysis.combined_retention(historical, continuation)
    assert result["historical_attempt_013"]["scene_check_count"] == 54
    assert result["new_attempt_014_evaluations"]["scene_check_count"] == 87
    assert result["combined_distinct_updates"]["scene_check_count"] == 138
    assert result["combined_distinct_updates"]["checkpoint_count"] == 46
    changed = deepcopy(continuation)
    changed[0]["scenes"][0]["rollout"]["success"] = False
    with pytest.raises(ValueError, match="repeated update-2000"):
        analysis.combined_retention(historical, changed)


def test_update_statistics_require_absolute_continuation_clock(tmp_path):
    path = tmp_path / "updates.jsonl"
    row = {"update": 2001, "extension_update": 1, "loss_before": 0.3, "loss_after": 0.2,
           "td_loss_before": 0.1, "td_loss_after": 0.1, "demonstration_loss_before": 0.2,
           "demonstration_loss_after": 0.1, "gradient_norm": 12, "parameter_delta_norm": 0.001}
    path.write_text(json.dumps(row), encoding="utf-8")
    result = analysis.update_statistics(path, 10, end=2001)
    assert result["gradient_norm_above_clip_count"] == 1
    assert result["nonzero_parameter_update_count"] == 1
    row["update"] = 1
    path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="absolute clock"):
        analysis.update_statistics(path, 10, end=2001)


@pytest.mark.parametrize("status", ["running", "failed", "interrupted"])
def test_unfinished_run_is_refused_before_importing_verifier_or_models(tmp_path, monkeypatch, status):
    (tmp_path / "run_status.json").write_text(json.dumps({"status": status}), encoding="utf-8")

    def forbidden(*args, **kwargs):
        raise AssertionError("An unfinished run must not inspect experiment inputs or models.")

    monkeypatch.setattr(analysis, "import_file", forbidden)
    monkeypatch.setattr(analysis.helper, "load_verified_agent", forbidden)
    with pytest.raises(ValueError, match="completed run"):
        analysis.analyze(tmp_path)
