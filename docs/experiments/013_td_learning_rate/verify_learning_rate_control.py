"""Verify the predeclared learning-rate-only control and its preserved Adam state."""

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import torch

from me5418_harvesting.agents.optimizer_transition import optimizer_state_summary

ROOT = Path(__file__).resolve().parents[3]
BASE_VERIFIER = ROOT / "docs/experiments/012_divergence_control/verify_control.py"
BASE_RUNNER = ROOT / "docs/experiments/012_divergence_control/run_control.py"
ENTRYPOINT = Path(__file__).with_name("run_learning_rate_control.py")
PRIOR_SETTINGS = ROOT / "docs/experiments/012_divergence_control/evidence/control/settings.json"
SPEC = importlib.util.spec_from_file_location("lr013_independent_control", BASE_VERIFIER)
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)
read_json, file_hash, require = base.read_json, base.file_hash, base.require
CHECKPOINTS = [*range(11), 20, 50, 100, 200, 500, 1000, 2000]


def verify_protocol(protocol, entrypoint=ENTRYPOINT, base_runner=BASE_RUNNER):
    """Reject an undeclared experiment before any update or rollout is replayed."""
    expected = {
        "attempt": 13,
        "factor": "learning_rate",
        "baseline_value": 0.0003,
        "treatment_value": 0.0001,
        "baseline_settings": {"learning_rate": 0.0003, "demonstration_weight": 1.0},
        "treatment_settings": {"learning_rate": 0.0001, "demonstration_weight": 1.0},
        "arms": ["baseline", "treatment"],
        "scene_seeds": [10000, 10001, 10002],
        "td_updates": 2000,
        "td_checkpoints": CHECKPOINTS,
        "margin": 0.8,
        "effective_batch_size": 117,
        "full_batch_weighting": "equal_per_transition",
        "inherited_optimizer_updates": 40000,
        "shared_start": "attempt_011_calibration_training_state",
        "td_boundary": "synchronized_targets_zero_phase_clock_before_scalar_intervention",
        "baseline_gate": "exact_all_updates_observations_and_final_state",
        "online_training_steps": 0,
        "held_out_scenes_accessed": 0,
        "entrypoint_script": entrypoint.relative_to(ROOT).as_posix(),
        "base_runner_script": base_runner.relative_to(ROOT).as_posix(),
    }
    for name, value in expected.items():
        require(protocol.get(name) == value, f"attempt 013 declared {name}")
    require(protocol.get("preserve_adam_moments_and_parameter_steps") is True
            and protocol.get("early_stopping") is False,
            "preserved Adam and no early stopping")
    for name in ("baseline_settings", "treatment_settings"):
        require(type(protocol[name]["demonstration_weight"]) in (float, int)
                and not isinstance(protocol[name]["demonstration_weight"], bool),
                "numeric fixed supervision weight")
    require(protocol.get("entrypoint_sha256") == file_hash(entrypoint),
            "attempt 013 entrypoint provenance")
    require(protocol.get("base_runner_sha256") == protocol.get("script_sha256")
            == file_hash(base_runner), "immutable generic runner provenance")
    return {**expected, "entrypoint_sha256": file_hash(entrypoint),
            "base_runner_sha256": file_hash(base_runner),
            "preserve_adam_moments_and_parameter_steps": True,
            "early_stopping": False}


def verify_boundary_payload(payload, calibrated, settings, treatment):
    """Confirm that only the learning-rate settings differ before TD begins."""
    base.verify_initial_state(payload, calibrated, settings, "learning_rate", 0.0001, treatment)
    rate = 0.0001 if treatment else 0.0003
    for source, name in ((calibrated, "calibrated"), (payload, "TD boundary")):
        steps = [float(state["step"]) for state in source["optimizer"]["state"].values()]
        require(steps and all(step == 20000 for step in steps), f"{name} inherited Adam steps")
    agent, env = base.audit.independent_restore(payload)
    try:
        require(agent.learner.config.learning_rate == agent.learner.optimizer.defaults["lr"] == rate
                and all(group["lr"] == rate for group in agent.learner.optimizer.param_groups),
                "consistent configuration, Adam default and group learning rates")
        actual = optimizer_state_summary(agent.learner)
        require(actual == payload["optimizer_summary"], "independently restored boundary summary")
        require(agent.learner.update_count == 0, "zero TD phase clock")
        return {
            "learning_rate": rate, "demonstration_weight": 1.0,
            "exact_calibrated_weights_and_adam_tensors_preserved": True,
            "optimizer_settings_and_defaults_verified": True,
            "adam_parameter_step": 20000, "td_phase_updates": 0,
            "optimizer_summary": actual,
            "optimizer_updates_performed": 0, "environment_steps": 0, "environment_resets": 0,
        }
    finally:
        env.close()


def verify_prior_source_anchor(protocol):
    """Bind the reused loop to the previously committed experiment independently."""
    relative = PRIOR_SETTINGS.relative_to(ROOT).as_posix()
    require(protocol["prior_control_settings_file"] == relative, "prior source anchor location")
    committed = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)
    previous = read_json(PRIOR_SETTINGS)
    require(json.loads(committed) == previous
            and protocol["prior_control_settings_sha256"] == file_hash(PRIOR_SETTINGS)
            and protocol["committed_prior_settings_blob_sha256"] == hashlib.sha256(committed).hexdigest(),
            "committed attempt 012 settings and content identity")
    require(previous["training"]["script_sha256"] == protocol["script_sha256"] == file_hash(BASE_RUNNER),
            "unchanged previously executed generic training implementation")
    return {"prior_control_settings_file": relative,
            "prior_control_settings_sha256": file_hash(PRIOR_SETTINGS),
            "committed_prior_settings_blob_sha256": hashlib.sha256(committed).hexdigest()}


def preflight(run_dir):
    """Validate the fixed protocol and both initial states without running a policy."""
    torch.set_num_threads(1)
    require(read_json(run_dir / "run_status.json")["status"] == "completed", "completed attempt 013")
    settings, summary = read_json(run_dir / "settings.json"), read_json(run_dir / "summary.json")
    require(settings["training"] == summary["protocol"], "shared settings and summary protocol")
    checked = verify_protocol(settings["training"])
    anchor = verify_prior_source_anchor(settings["training"])
    manifest = read_json(run_dir / "input_manifest.json")
    reference = Path(manifest["reference_run"])
    calibration_path = reference / manifest["calibrated_snapshot_file"]
    require(file_hash(calibration_path) == manifest["calibrated_snapshot_sha256"],
            "calibrated snapshot file integrity")
    calibrated = torch.load(calibration_path, map_location="cpu", weights_only=True)
    boundaries = {}
    for branch in ("baseline", "treatment"):
        phase = summary["arms"][branch]
        require(phase["demonstration_weight"] == 1.0, f"{branch} fixed supervision weight")
        path = run_dir / branch / "td/training_state_0.pt"
        require(file_hash(path) == phase["checkpoint_sha256"][path.name], f"{branch} initial snapshot integrity")
        boundaries[branch] = verify_boundary_payload(
            torch.load(path, map_location="cpu", weights_only=True), calibrated, settings,
            branch == "treatment",
        )
    return {"protocol": checked, "prior_source_anchor": anchor, "boundaries": boundaries,
            "optimizer_updates_performed": 0, "environment_steps": 0, "environment_resets": 0}


def verify(run_dir, output):
    if output.suffix != ".json":
        raise ValueError("The verification output must be a .json file.")
    underlying_output = output.with_name(f"{output.stem}_base.json")
    if output.exists() or underlying_output.exists():
        raise FileExistsError("Refusing to overwrite either attempt 013 verification report.")
    strict = preflight(run_dir)
    # The immutable verifier performs exactly 20 updates and 18 policy reload episodes.
    # The extra preflight above performs neither; both reports describe the same replay.
    result = base.verify(run_dir, underlying_output)
    result = {
        **result, "attempt": 13, "strict_learning_rate_preflight": strict,
        "verification_wrapper_sha256": file_hash(Path(__file__)),
        "base_verifier_sha256": file_hash(BASE_VERIFIER),
        "base_verification_report": str(underlying_output.resolve()),
        "base_verification_report_sha256": file_hash(underlying_output),
        "accounting_note": "This report and the base report describe the same 20 verification updates and 18 episodes; do not add their counts together.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output.resolve()), "attempt": 13,
                      "additional_verification_optimizer_updates": result["additional_verification_optimizer_updates"],
                      "additional_verification_episodes": result["additional_verification_episodes"],
                      "additional_verification_environment_steps": result["additional_verification_environment_steps"]}),
          flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.run_dir.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
