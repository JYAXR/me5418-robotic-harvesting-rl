"""Protect held-out evaluation denominators, diagnostics and inference-only behavior."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("frozen028", ROOT / "docs/experiments/028_frozen_generalization/run_frozen_generalization.py")
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)


@pytest.mark.parametrize("success,damage,budget,blocked,tail,expected", [
    (True, False, True, 100, 30, "success"),
    (False, True, True, 100, 30, "damage"),
    (False, False, True, 100, 20, "budget_blocked_stall"),
    (False, False, True, 99, 20, "budget_stationary_stall"),
    (False, False, True, 100, 19, "budget_while_moving"),
    (False, False, False, 100, 20, "external_truncation"),
])
def test_terminal_precedence_and_stall_boundaries(success, damage, budget, blocked, tail, expected):
    assert r.failure_signature({"success": success, "damage": damage,
        "budget_exhausted": budget, "steps": 200, "blocked_steps": blocked}, tail) == expected


def test_reset_failures_are_not_removed_from_requested_denominator():
    rows = [{"status": "reset_error", "category": "reset_error", "policy_actions": 0,
             "certification": [{"steps": 7}]},
            {"status": "evaluated", "category": "success", "policy_actions": 35,
             "certification": [{"steps": 35}],
             "metrics": {"success": True, "steps": 35, "return": 107.0}}]
    s = r.summary(rows)
    assert s["success_rate_requested"] == 0.5 and s["success_rate_valid"] == 1.0
    assert s["certification_actions"] == 42 and s["policy_actions"] == 35
    assert s["categories"] == {"reset_error": 1, "success": 1}


def test_wilson_interval_has_nonzero_uncertainty_at_extremes():
    assert r.wilson(0, 0) is None
    assert r.wilson(0, 100)[1] == pytest.approx(0.0369934982)
    assert r.wilson(100, 100)[0] == pytest.approx(0.9630065018)
    assert r.wilson(50, 100) == pytest.approx([0.4038315304, 0.5961684696])


def test_certification_capture_restores_function_on_exception(monkeypatch):
    def source(env):
        return {"success": False, "steps": 17}
    monkeypatch.setattr(r.scene_feasibility, "verify_reference_route", source)
    records = []
    with pytest.raises(ValueError), r.certification_checks(records):
        assert r.scene_feasibility.verify_reference_route(None) == {"success": False, "steps": 17}
        raise ValueError("Synthetic reset failure.")
    assert records == [{"success": False, "steps": 17}]
    assert r.scene_feasibility.verify_reference_route is source


def test_instrumented_known_scene_matches_delivered_policy(tmp_path):
    torch.set_num_threads(1)
    agent, env = r.load_frozen()
    before = r.state(agent)
    try:
        with pytest.raises(RuntimeError, match="forbidden"):
            agent.learner.optimizer.step()
        result = r.rollout(agent, env, 10000, tmp_path / "trace.jsonl")
        expected = json.loads((ROOT / "models/step6_repair_027.json").read_text())["evaluation"]["records"][0]
        assert result["metrics"] == expected
        assert r.equal(before, r.state(agent))
        trace = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
        assert len(trace) == expected["steps"] == result["policy_actions"]
        assert all(row["action"] == max(range(8), key=lambda i: row["q_values"][i]) for row in trace)
        assert trace[-1]["info"]["success"]
    finally:
        env.close()
