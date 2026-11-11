"""Check that learning-rate diagnostics reject confounded or unfinished comparisons."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "lr013_analysis_test", ROOT / "docs/experiments/013_td_learning_rate/analyze_learning_rate_q.py",
)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def protocol():
    value = json.loads((ROOT / "docs/experiments/012_divergence_control/evidence/control/settings.json")
                       .read_text(encoding="utf-8"))["training"]
    value.update({
        "attempt": 13, "factor": "learning_rate", "baseline_value": 3e-4,
        "treatment_value": 1e-4, "entrypoint_script": analysis.ENTRYPOINT,
        "base_runner_script": analysis.BASE_RUNNER, "base_runner_sha256": value["script_sha256"],
        "treatment_settings": {"learning_rate": 1e-4, "demonstration_weight": 1.0},
    })
    return value


@pytest.mark.parametrize("field,value", [
    ("treatment_value", 3e-5), ("factor", "demonstration_weight"),
    ("preserve_adam_moments_and_parameter_steps", False), ("td_updates", 1000),
    ("treatment_settings", {"learning_rate": 1e-4, "demonstration_weight": 2.0}),
])
def test_protocol_rejects_another_factor_or_budget(field, value):
    declared = protocol()
    analysis.validate_protocol(declared)
    declared[field] = value
    with pytest.raises(ValueError):
        analysis.validate_protocol(declared)


@pytest.mark.parametrize("status", ["running", "failed"])
def test_unfinished_analysis_refused_before_input_or_model_access(tmp_path, monkeypatch, status):
    (tmp_path / "run_status.json").write_text(json.dumps({"status": status}), encoding="utf-8")

    def forbidden(*args, **kwargs):
        raise AssertionError("An unfinished run must not load training inputs or models.")

    monkeypatch.setattr(analysis.audit, "verify_inputs", forbidden)
    monkeypatch.setattr(analysis.helper, "load_verified_agent", forbidden)
    with pytest.raises(ValueError, match="completed run"):
        analysis.analyze(tmp_path)


def first_step_fixture():
    original = {"loss_before": 0.2, "td_loss_before": 0.1, "demonstration_loss_before": 0.1,
                "gradient_norm": 20.0, "parameter_delta_norm": 0.03}
    changed = {**original, "parameter_delta_norm": 0.01001}
    arms = {"baseline": {"update_statistics": {"first_update": original}},
            "treatment": {"update_statistics": {"first_update": changed}}}
    changes = {"baseline": torch.tensor([0.02, -0.01, 0.02]),
               "treatment": torch.tensor([0.00667, -0.00333, 0.00667])}
    return arms, changes


def test_first_step_reports_measured_ratio_instead_of_assuming_exact_scaling():
    arms, changes = first_step_fixture()
    result = analysis.first_update_comparison(arms, changes)
    assert result["learning_rate_ratio"] == pytest.approx(1 / 3)
    assert result["logged_parameter_change_norm_ratio"] == pytest.approx(0.01001 / 0.03)
    assert result["maximum_absolute_proportionality_residual"] > 0
    assert result["relative_l2_proportionality_residual"] > 0
    assert all(result["equal_initial_fields"].values())


def test_first_step_rejects_different_initial_gradient_or_objective():
    arms, changes = first_step_fixture()
    different = deepcopy(arms)
    different["treatment"]["update_statistics"]["first_update"]["gradient_norm"] += 1
    with pytest.raises(ValueError, match="initial objective and gradient norm"):
        analysis.first_update_comparison(different, changes)


def test_zero_parameter_change_is_not_reported_as_a_learning_rate_ratio():
    arms, changes = first_step_fixture()
    arms["baseline"]["update_statistics"]["first_update"]["parameter_delta_norm"] = 0
    with pytest.raises(ValueError, match="nonzero parameter change"):
        analysis.first_update_comparison(arms, changes)
