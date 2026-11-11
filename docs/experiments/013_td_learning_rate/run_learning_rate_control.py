"""Run the predeclared learning-rate-only control with the immutable attempt-012 loop."""

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE_RUNNER = ROOT / "docs/experiments/012_divergence_control/run_control.py"
PRIOR_SETTINGS = ROOT / "docs/experiments/012_divergence_control/evidence/control/settings.json"
SPEC = importlib.util.spec_from_file_location("lr013_immutable_control", BASE_RUNNER)
control = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(control)

BASELINE_LR = 3e-4
TREATMENT_LR = 1e-4
CHECKPOINTS = (*range(11), 20, 50, 100, 200, 500, 1000, 2000)


def verify_base_runner():
    """Bind the reused training implementation to committed attempt-012 evidence."""
    relative = PRIOR_SETTINGS.relative_to(ROOT).as_posix()
    committed = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)
    previous = control.read_json(PRIOR_SETTINGS)
    if json.loads(committed) != previous:
        raise ValueError("Attempt-012 settings differ from their committed content.")
    expected = previous["training"]["script_sha256"]
    actual = control.fit.file_hash(BASE_RUNNER)
    if actual != expected:
        raise ValueError("The reused attempt-012 training implementation has changed.")
    return {
        "base_runner_script": BASE_RUNNER.relative_to(ROOT).as_posix(),
        "base_runner_sha256": actual,
        "prior_control_settings_file": relative,
        "prior_control_settings_sha256": control.fit.file_hash(PRIOR_SETTINGS),
        "committed_prior_settings_blob_sha256": hashlib.sha256(committed).hexdigest(),
    }


def validate_fixed_protocol(protocol, base_runner_sha256):
    """Reject any change beyond the one rate chosen before running the comparison."""
    expected = {
        "factor": "learning_rate", "baseline_value": BASELINE_LR,
        "treatment_value": TREATMENT_LR,
        "baseline_settings": {"learning_rate": BASELINE_LR, "demonstration_weight": 1.0},
        "treatment_settings": {"learning_rate": TREATMENT_LR, "demonstration_weight": 1.0},
        "scene_seeds": [10000, 10001, 10002], "arms": ["baseline", "treatment"],
        "td_updates": 2000, "td_checkpoints": list(CHECKPOINTS), "margin": 0.8,
        "inherited_optimizer_updates": 40000,
        "shared_start": "attempt_011_calibration_training_state",
        "preserve_adam_moments_and_parameter_steps": True,
        "td_boundary": "synchronized_targets_zero_phase_clock_before_scalar_intervention",
        "full_batch_weighting": "equal_per_transition", "effective_batch_size": 117,
        "early_stopping": False,
        "baseline_gate": "exact_all_updates_observations_and_final_state",
        "online_training_steps": 0, "held_out_scenes_accessed": 0,
        "script_sha256": base_runner_sha256,
    }
    for name, value in expected.items():
        if protocol.get(name) != value:
            raise ValueError(f"Attempt-013 predeclared protocol differs at {name}.")


def run(output, reference, archive):
    """Add launch identity without changing the shared training loop or its accounting."""
    output, reference, archive = output.resolve(), reference.resolve(), archive.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite the learning-rate experiment: {output}")
    identity = {
        **verify_base_runner(), "attempt": 13,
        "entrypoint_script": Path(__file__).relative_to(ROOT).as_posix(),
        "entrypoint_sha256": control.fit.file_hash(Path(__file__)),
    }
    original_begin = control.begin_run
    begin_count = 0

    def record_launch(directory, environment, dqn, protocol):
        nonlocal begin_count
        begin_count += 1
        if begin_count != 1:
            raise RuntimeError("The immutable control attempted more than one run launch.")
        validate_fixed_protocol(protocol, identity["base_runner_sha256"])
        protocol.update(identity)
        # The generic summary retains this same dictionary, so settings and summary agree.
        return original_begin(directory, environment, dqn, protocol)

    # The imported module is private to this entrypoint. Only metadata recording is
    # adapted; its model restoration, updates, evaluations and failure handling remain intact.
    control.begin_run = record_launch
    try:
        result = control.run(output, reference, archive, "learning_rate", TREATMENT_LR)
        if begin_count != 1:
            raise RuntimeError("The immutable control did not record its launch.")
        return result
    finally:
        control.begin_run = original_begin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.output_dir, args.reference_run, args.archive_dir)


if __name__ == "__main__":
    main()
