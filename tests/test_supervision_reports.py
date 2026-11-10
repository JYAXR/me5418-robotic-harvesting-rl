"""Distinguish supervised pretraining losses and training-only action agreement."""

import importlib.util
import json
from pathlib import Path

import pytest
from matplotlib.figure import Figure
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "summarize_supervision", ROOT / "scripts/summarize_training.py",
)
SUMMARIZE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARIZE)


def validation():
    return {
        "success_rate": 0.0, "damage_rate": 0.0, "mean_distance_to_target": 0.7,
        "mean_stationary_fraction": 0.1, "mean_blocked_fraction": 0.2,
    }


@pytest.fixture
def supervised_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    rows = [{
        "episode": index, "return": -index, "success": False, "damage": False,
        "epsilon": 0.5, "distance_to_target": 0.7, "updates": index + 120,
        "validation": validation(),
    } for index in (1, 2)]
    pretraining = {
        "completed_updates": 120, "status": "completed", "demonstration_weight": 2.0,
        "demonstration_margin": 0.8, "method": "demonstration_dqn_td_and_margin_updates",
    }
    summary = {
        "episodes": 2, "environment_steps": 4, "updates": 122,
        "pretraining_updates": 120, "online_updates": 2,
        "pretraining": pretraining, "validation": validation(),
    }
    files = {
        "summary.json": summary, "settings.json": {"training": {"seed": 0}},
        "initial_validation.json": validation(), "pretraining_validation.json": validation(),
        "pretraining.json": pretraining,
    }
    for name, values in files.items():
        (run / name).write_text(json.dumps(values), encoding="utf-8")
    (run / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
    )
    losses = [{
        "pretraining_update": index + 1, "loss_before": 2.0 + 2.0 * index / 100,
        "td_loss_before": 2.0, "demonstration_loss_before": index / 100,
        "demonstration_action_accuracy": index / 119,
    } for index in range(120)]
    (run / "pretraining_metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in losses), encoding="utf-8",
    )
    return run


def test_summary_separates_weighted_losses_and_training_agreement(supervised_run, monkeypatch):
    figures = []
    original_save = Figure.savefig

    def capture(figure, *args, **kwargs):
        axis = figure.axes[4]
        figures.append({
            "axes": len(figure.axes), "title": axis.get_title(), "ylabel": axis.get_ylabel(),
            "lines": {line.get_label(): line.get_ydata().tolist() for line in axis.lines},
        })
        return original_save(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    raw_before = (supervised_run / "pretraining_metrics.jsonl").read_bytes()
    result = SUMMARIZE.summarize(supervised_run)
    diagnostics = result["pretraining_supervision"]
    assert diagnostics["weight"] == 2.0
    assert diagnostics["margin"] == 0.8
    assert diagnostics["mean_first_100_batch_accuracy"] == pytest.approx(49.5 / 119)
    assert diagnostics["mean_last_100_batch_accuracy"] == pytest.approx(69.5 / 119)
    assert diagnostics["batch_accuracy_window_updates"] == 100
    assert "not held-out task success" in diagnostics["batch_accuracy_note"]
    assert result["pretraining_validation"]["success_rate"] == 0.0
    assert result["optimizer_updates"] == 122
    assert len(figures) == 1 and figures[0]["axes"] == 6
    assert "50-update mean" in figures[0]["title"]
    assert "Huber" not in figures[0]["ylabel"]
    plotted = figures[0]["lines"]
    assert set(plotted) == {"TD", "Weighted supervision", "Total"}
    assert plotted["TD"][-1] == 2.0
    assert plotted["Weighted supervision"][-1] == pytest.approx(1.89)
    assert plotted["Total"][-1] == pytest.approx(3.89)
    assert plotted["Total"][0] == 2.0
    assert (supervised_run / "pretraining_metrics.jsonl").read_bytes() == raw_before
    assert json.loads((supervised_run / "analysis.json").read_text()) == result
    with Image.open(supervised_run / "learning_diagnostics.png") as picture:
        picture.verify()


def test_pure_td_run_keeps_existing_loss_label_and_analysis(supervised_run, monkeypatch):
    summary_path = supervised_run / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["pretraining"].pop("demonstration_weight")
    summary["pretraining"].pop("demonstration_margin")
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    labels = []
    monkeypatch.setattr(Figure, "savefig", lambda figure, *args, **kwargs:
                        labels.append(figure.axes[4].get_ylabel()))
    result = SUMMARIZE.summarize(supervised_run)
    assert "pretraining_supervision" not in result
    assert labels == ["Batch Huber loss"]


@pytest.mark.parametrize("accuracy", [None, -0.1, 1.1, float("nan")])
def test_invalid_training_accuracy_is_not_reported_as_valid(accuracy):
    with pytest.raises(ValueError, match="accuracy"):
        SUMMARIZE.supervision_diagnostics(
            {"demonstration_weight": 1.0, "demonstration_margin": 0.8},
            [{"demonstration_action_accuracy": accuracy}],
        )


def test_short_log_reports_available_window_and_requires_enabled_loss_evidence():
    values = {"demonstration_weight": 1.0, "demonstration_margin": 0.8}
    result = SUMMARIZE.supervision_diagnostics(values, [
        {"demonstration_action_accuracy": 0.25}, {"demonstration_action_accuracy": 0.75},
    ])
    assert result["batch_accuracy_window_updates"] == 2
    assert result["mean_first_100_batch_accuracy"] == result["mean_last_100_batch_accuracy"] == 0.5
    with pytest.raises(ValueError, match="per-update"):
        SUMMARIZE.supervision_diagnostics(values, [])
    assert SUMMARIZE.supervision_diagnostics({"demonstration_weight": 0.0}, []) is None
