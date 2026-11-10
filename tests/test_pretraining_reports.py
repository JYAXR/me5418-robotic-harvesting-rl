"""Preserve pretraining stages and update accounting in plots and evidence archives."""

import importlib.util
import json
from pathlib import Path

import pytest
from matplotlib.figure import Figure
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SUMMARIZE = load_script("summarize_training")
ARCHIVE = load_script("archive_training_attempt")


def validation(success, distance):
    return {
        "success_rate": success, "damage_rate": 0.0,
        "mean_distance_to_target": distance,
        "mean_stationary_fraction": 0.1, "mean_blocked_fraction": 0.2,
    }


@pytest.fixture
def run_dir(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    rows = [{
        "episode": index, "return": -index, "success": False, "damage": False,
        "epsilon": 0.5, "distance_to_target": 0.7, "updates": index + 2,
        "validation": validation(0.0, 0.7),
    } for index in (1, 2)]
    summary = {
        "episodes": 2, "environment_steps": 4, "updates": 4,
        "pretraining_updates": 2, "online_updates": 2,
        "pretraining": {"completed_updates": 2, "status": "completed"},
        "validation": rows[-1]["validation"],
    }
    files = {
        "summary.json": summary,
        "settings.json": {"training": {"seed": 0}},
        "initial_validation.json": validation(0.0, 0.9),
        "pretraining_validation.json": validation(1.0, 0.01),
        "pretraining.json": summary["pretraining"],
    }
    for name, content in files.items():
        (run / name).write_text(json.dumps(content), encoding="utf-8")
    (run / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
    )
    (run / "pretraining_metrics.jsonl").write_text(
        "".join(json.dumps({"pretraining_update": index, "loss_before": 1.0 / index}) + "\n"
                for index in (1, 2)), encoding="utf-8",
    )
    return run


def test_summary_distinguishes_pretraining_from_both_initial_and_online_policy(run_dir, monkeypatch):
    figures = []
    original_save = Figure.savefig

    def capture(figure, *args, **kwargs):
        figures.append({
            "axes": len(figure.axes),
            "labels": [tick.get_text() for tick in figure.axes[1].get_xticklabels()],
        })
        return original_save(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    result = SUMMARIZE.summarize(run_dir)
    assert result["optimizer_updates"] == 4
    assert result["pretraining_updates"] == result["online_updates"] == 2
    assert result["environment_steps"] == 4
    assert result["initial_validation"]["success_rate"] == 0
    assert result["pretraining_validation"]["success_rate"] == 1
    assert result["final_validation"]["success_rate"] == 0
    stages = result["validation_checkpoints"]
    assert [row["phase"] for row in stages] == ["initial", "pretraining", "online", "online"]
    assert [row["online_episode"] for row in stages] == [0, 0, 1, 2]
    assert [row["optimizer_updates"] for row in stages] == [0, 2, 3, 4]
    assert figures == [{"axes": 6, "labels": ["Initial", "Pretrained", "Ep 1", "Ep 2"]}]
    assert json.loads((run_dir / "analysis.json").read_text()) == result
    with Image.open(run_dir / "learning_diagnostics.png") as picture:
        picture.verify()


def test_old_run_summary_stays_compatible_without_pretraining_files(run_dir, monkeypatch):
    for path in run_dir.glob("pretraining*"):
        path.unlink()
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    for key in ("pretraining_updates", "online_updates", "pretraining"):
        summary.pop(key)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    axes_counts = []
    monkeypatch.setattr(Figure, "savefig", lambda figure, *args, **kwargs:
                        axes_counts.append(len(figure.axes)))
    result = SUMMARIZE.summarize(run_dir)
    assert axes_counts == [4]
    assert "pretraining_validation" not in result and "pretraining_updates" not in result
    assert result["optimizer_updates"] == 4
    assert result["initial_validation"]["mean_distance_to_target"] == 0.9


def test_inconsistent_update_totals_are_rejected(run_dir):
    path = run_dir / "summary.json"
    summary = json.loads(path.read_text())
    summary["online_updates"] = 3
    path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="sum to total"):
        SUMMARIZE.summarize(run_dir)
    assert not (run_dir / "analysis.json").exists()


def test_archive_includes_pretraining_evidence_but_excludes_pretrained_weights(run_dir, tmp_path):
    (run_dir / "pretrained.pt").write_bytes(b"checkpoint fixture")
    result = ARCHIVE.archive_attempt(run_dir, tmp_path / "evidence")
    for name in ("pretraining.json", "pretraining_metrics.jsonl", "pretraining_validation.json"):
        assert name in result["files"]
        assert (tmp_path / "evidence" / name).read_bytes() == (run_dir / name).read_bytes()
    assert "pretrained.pt" not in result["files"]
    assert not (tmp_path / "evidence/pretrained.pt").exists()
