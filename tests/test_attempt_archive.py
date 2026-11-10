"""Preserve experiment evidence without mistaking failures for completed runs."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/archive_training_attempt.py"
SPEC = importlib.util.spec_from_file_location("archive_training_attempt", SCRIPT)
ARCHIVE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ARCHIVE)


@pytest.fixture
def run_dir(tmp_path):
    directory = tmp_path / "run"
    directory.mkdir()
    (directory / "settings.json").write_text('{"training": {"seed": 0}}', encoding="utf-8")
    return directory


def test_archive_copies_evidence_and_hashes_without_models_or_console(run_dir, tmp_path):
    (run_dir / "summary.json").write_text('{"episodes": 1}', encoding="utf-8")
    (run_dir / "metrics.jsonl").write_text('{"episode": 1}\n', encoding="utf-8")
    (run_dir / "learning_diagnostics.png").write_bytes(b"image fixture")
    (run_dir / "best.pt").write_bytes(b"model fixture")
    (run_dir / "console.log").write_text("raw output", encoding="utf-8")
    output = tmp_path / "evidence"

    result = ARCHIVE.archive_attempt(run_dir, output)

    assert result["status"] == "completed"
    assert result["status_source"] == "summary.json"
    assert set(result["files"]) == {
        "settings.json", "summary.json", "metrics.jsonl", "learning_diagnostics.png",
    }
    assert {path.name for path in output.iterdir()} == set(result["files"]) | {
        "archive_manifest.json",
    }
    for name, information in result["files"].items():
        copied = (output / name).read_bytes()
        assert copied == (run_dir / name).read_bytes()
        assert information == {
            "sha256": hashlib.sha256(copied).hexdigest(), "bytes": len(copied),
        }
    assert "provenance.json" in result["missing_evidence_files"]
    assert json.loads((output / "archive_manifest.json").read_text()) == result


def test_log_without_completion_record_is_incomplete(run_dir, tmp_path):
    (run_dir / "metrics.jsonl").write_text('{"episode": 1}\n', encoding="utf-8")
    result = ARCHIVE.archive_attempt(run_dir, tmp_path / "evidence")
    assert result["status"] == "incomplete"
    assert "summary.json" in result["missing_evidence_files"]


@pytest.mark.parametrize("status", ["failed", "running", "interrupted", "completed"])
def test_explicit_status_takes_precedence_over_summary(run_dir, tmp_path, status):
    (run_dir / "run_status.json").write_text(json.dumps({"status": status}), encoding="utf-8")
    (run_dir / "summary.json").write_text('{"episodes": 1}', encoding="utf-8")
    result = ARCHIVE.archive_attempt(run_dir, tmp_path / "evidence")
    assert result["status"] == status
    assert result["status_source"] == "run_status.json"


@pytest.mark.parametrize("files", [[], ["summary.json"], ["settings.json"]])
def test_required_evidence_missing_does_not_create_archive(tmp_path, files):
    directory = tmp_path / "run"
    directory.mkdir()
    for name in files:
        (directory / name).write_text("{}", encoding="utf-8")
    output = tmp_path / "evidence"
    with pytest.raises(ValueError, match="requires"):
        ARCHIVE.archive_attempt(directory, output)
    assert not output.exists()


def test_existing_archive_is_never_overwritten(run_dir, tmp_path):
    (run_dir / "summary.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "evidence"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("original", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        ARCHIVE.archive_attempt(run_dir, output)
    assert sentinel.read_text() == "original"
    assert len(list(output.iterdir())) == 1


@pytest.mark.parametrize("state", [{}, {"status": ""}, {"status": None}, []])
def test_invalid_status_is_not_silently_reported_completed(run_dir, tmp_path, state):
    (run_dir / "run_status.json").write_text(json.dumps(state), encoding="utf-8")
    (run_dir / "summary.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "evidence"
    with pytest.raises(ValueError, match="status string"):
        ARCHIVE.archive_attempt(run_dir, output)
    assert not output.exists()
