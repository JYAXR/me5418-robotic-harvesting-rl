"""Keep provenance and explicit outcomes when training completes or raises."""

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pytest

pytest.importorskip("torch")

from me5418_harvesting.agents import training
from me5418_harvesting.agents.dqn import DQNConfig
from me5418_harvesting.agents.run_record import begin_run
from me5418_harvesting.agents.training import TrainingConfig

ROOT = Path(__file__).resolve().parents[1]


def test_begin_run_records_actual_source_hashes_and_initial_settings(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    environment = {"max_steps": 3}
    network = {"hidden_sizes": [8, 8]}
    settings = {"seed": 7, "episodes": 1}
    begin_run(output, environment, network, settings)

    provenance = json.loads((output / "provenance.json").read_text())
    relative = "src/me5418_harvesting/agents/replay_buffer.py"
    assert provenance["source_sha256"][relative] == hashlib.sha256(
        (ROOT / relative).read_bytes()
    ).hexdigest()
    assert provenance["python"] and provenance["platform"]
    assert provenance["packages"]["torch"] is not None
    assert "git_head" in provenance and "git_dirty" in provenance
    status = json.loads((output / "run_status.json").read_text())
    assert status["status"] == "running"
    assert status["started_at_utc"] == provenance["started_at_utc"]
    assert datetime.fromisoformat(status["started_at_utc"]).utcoffset().total_seconds() == 0
    assert json.loads((output / "settings.json").read_text()) == {
        "environment": environment, "dqn": network, "training": settings,
    }


@pytest.mark.parametrize("error, status", [
    (RuntimeError("Injected training failure."), "failed"),
    (KeyboardInterrupt("Injected user interruption."), "interrupted"),
])
def test_training_exception_preserves_provenance_and_partial_episode_log(
    tmp_path, monkeypatch, error, status,
):
    def fail_after_episode(environment, dqn, config, output):
        assert json.loads((output / "run_status.json").read_text())["status"] == "running"
        (output / "metrics.jsonl").write_text('{"episode": 1}\n', encoding="utf-8")
        raise error

    monkeypatch.setattr(training, "_train", fail_after_episode)
    output = tmp_path / "failed"
    with pytest.raises(type(error)) as raised:
        training.train({"max_steps": 3}, DQNConfig(), TrainingConfig(episodes=1), output)
    assert raised.value is error
    record = json.loads((output / "run_status.json").read_text())
    assert record["status"] == status
    assert record["error_type"] == type(error).__name__
    assert record["error_message"] == str(error)
    assert datetime.fromisoformat(record["finished_at_utc"]) >= datetime.fromisoformat(
        record["started_at_utc"]
    )
    assert (output / "provenance.json").is_file()
    assert json.loads((output / "settings.json").read_text())["training"]["episodes"] == 1
    assert (output / "metrics.jsonl").read_text() == '{"episode": 1}\n'
    assert not (output / "summary.json").exists()


def test_successful_training_returns_result_and_marks_completion(tmp_path, monkeypatch):
    expected = {"episodes": 1, "validation": {"success_rate": 0.0}}

    def complete(environment, dqn, config, output):
        (output / "summary.json").write_text(json.dumps(expected), encoding="utf-8")
        return expected

    monkeypatch.setattr(training, "_train", complete)
    output = tmp_path / "completed"
    assert training.train({}, DQNConfig(), TrainingConfig(episodes=1), output) is expected
    record = json.loads((output / "run_status.json").read_text())
    assert record["status"] == "completed"
    assert "finished_at_utc" in record and "error_type" not in record
    assert json.loads((output / "summary.json").read_text()) == expected


def test_provenance_initialization_failure_still_records_failed_attempt(tmp_path, monkeypatch):
    def unexpected_training(*args, **kwargs):
        raise AssertionError("Training must not start with unserializable settings.")

    monkeypatch.setattr(training, "_train", unexpected_training)
    output = tmp_path / "invalid_settings"
    with pytest.raises(TypeError, match="JSON serializable"):
        training.train({"invalid": object()}, DQNConfig(), TrainingConfig(episodes=1), output)
    record = json.loads((output / "run_status.json").read_text())
    assert record["status"] == "failed"
    assert record["error_type"] == "TypeError"
    assert "JSON serializable" in record["error_message"]
    assert "started_at_utc" in record and "finished_at_utc" in record
    assert not (output / "provenance.json").exists()
