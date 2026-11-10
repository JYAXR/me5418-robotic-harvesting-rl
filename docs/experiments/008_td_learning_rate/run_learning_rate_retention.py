"""Test smaller TD adaptation rates from the same frozen supervised policy.

The pretrained weights and fixed demonstration data come from completed attempt
007. No pretraining, optimizer resume, exploration or new collection is performed.
"""

import argparse
import importlib.util
import json
import shutil
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from me5418_harvesting.agents.dqn_agent import DQNAgent
from me5418_harvesting.agents.dqn_learning import TransitionBatch
from me5418_harvesting.agents.run_record import begin_run, finish_run
from me5418_harvesting.agents.staged_learning import begin_td_phase

ROOT = Path(__file__).resolve().parents[3]
STAGED_RUNNER = ROOT / "docs/experiments/007_staged_retention/run_staged_retention.py"
SPEC = importlib.util.spec_from_file_location("staged_retention_reference", STAGED_RUNNER)
staged = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(staged)

RATES = (("lr_3e_5", 3e-5), ("lr_3e_6", 3e-6))
HISTORICAL_RATE = 3e-4
PHASE_UPDATES = 2000
PRETRAIN_UPDATES = 20000
CHECKPOINTS = (0, 1, 10, 50, 100, 200, 500, 1000, 2000)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_reference(reference_run):
    """Reject changed or incomplete reference inputs before any optimizer update."""
    status = read_json(reference_run / "run_status.json")
    if status["status"] != "completed":
        raise ValueError("The reference retention run must be completed.")
    summary = read_json(reference_run / "summary.json")
    settings = read_json(reference_run / "settings.json")
    provenance = read_json(reference_run / "provenance.json")
    reproduction = read_json(reference_run / "pretraining_reproduction.json")
    if (not reproduction or any(value is not True for value in reproduction.values())
            or summary["pretraining_reproduction"] != reproduction):
        raise ValueError("The reference must have completely reproduced supervised pretraining.")
    protocol = summary["protocol"]
    if (protocol != settings["training"] or protocol["pretraining_updates"] != PRETRAIN_UPDATES
            or protocol["phase_updates"] != PHASE_UPDATES
            or tuple(protocol["phase_checkpoints"]) != CHECKPOINTS
            or protocol["scene_seed"] != staged.SCENE_SEED
            or protocol["margin"] != staged.MARGIN
            or protocol["td_margin_weight"] != 1.0
            or settings["dqn"]["learning_rate"] != HISTORICAL_RATE):
        raise ValueError("The reference does not match the declared attempt-007 control settings.")
    for name, expected in provenance["source_sha256"].items():
        if staged.fit.file_hash(ROOT / name) != expected:
            raise ValueError(f"A reference production source has changed: {name}")
    if staged.fit.file_hash(STAGED_RUNNER) != protocol["script_sha256"]:
        raise ValueError("The reference staged runner has changed since its recorded run.")
    if staged.fit.file_hash(staged.FIT_RUNNER) != protocol["fit_runner_sha256"]:
        raise ValueError("The shared fitting and rollout helper has changed.")
    checkpoint = reference_run / "pretraining/update_20000.pt"
    expected_checkpoint = summary["pretraining"]["checkpoint_sha256"][checkpoint.name]
    if staged.fit.file_hash(checkpoint) != expected_checkpoint:
        raise ValueError("The frozen supervised checkpoint hash differs from the reference.")
    dataset_metadata = read_json(reference_run / "dataset.json")
    if staged.fit.file_hash(reference_run / "demonstration.npz") != dataset_metadata["sha256"]:
        raise ValueError("The frozen demonstration data hash differs from the reference.")
    with np.load(reference_run / "demonstration.npz", allow_pickle=False) as dataset:
        batch = TransitionBatch(**{
            name: torch.from_numpy(dataset[name]) for name in TransitionBatch.__dataclass_fields__
        })
    if len(batch.actions) != dataset_metadata["transitions"]:
        raise ValueError("The frozen demonstration count differs from the reference.")
    return summary, settings, provenance, dataset_metadata, batch, checkpoint


def run(reference_run, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite a previous adaptation run: {output}")
    reference, settings, provenance, dataset, batch, checkpoint = load_reference(reference_run)
    protocol = {
        "kind": "smaller_learning_rates_after_supervised_pretraining",
        "scene_seed": staged.SCENE_SEED, "objective": "td_margin",
        "learning_rates": dict(RATES), "historical_learning_rate": HISTORICAL_RATE,
        "phase_updates": PHASE_UPDATES, "phase_checkpoints": CHECKPOINTS,
        "pretraining_updates": PRETRAIN_UPDATES,
        "pretraining_updates_reused": PRETRAIN_UPDATES, "pretraining_updates_performed": 0,
        "margin": staged.MARGIN, "demonstration_weight": 1.0,
        "phase_boundary": "identical_frozen_weights_fresh_adam_synced_target_zero_phase_clock",
        "batch_mode": "same_frozen_single_demonstration", "early_stopping": False,
        "reference_run": str(reference_run.resolve()),
        "reference_git_head": provenance["git_head"],
        "reference_checkpoint_sha256": staged.fit.file_hash(checkpoint),
        "reference_dataset_sha256": dataset["sha256"],
        "script_sha256": staged.fit.file_hash(Path(__file__)),
        "staged_runner_sha256": staged.fit.file_hash(STAGED_RUNNER),
        "fit_runner_sha256": staged.fit.file_hash(staged.FIT_RUNNER),
        "online_training_steps": 0, "held_out_scenes_accessed": 0,
    }
    output.mkdir(parents=True, exist_ok=False)
    try:
        begin_run(output, settings["environment"], settings["dqn"], protocol)
        torch.set_num_threads(1)
        sources = (
            "run_status.json", "summary.json", "settings.json", "provenance.json",
            "pretraining_reproduction.json", "dataset.json", "demonstration.npz",
            "pretraining/update_20000.pt", "pretraining/updates.jsonl",
            "td_margin/observations.json", "td_margin/updates.jsonl",
        )
        manifest = {
            "reference_run": str(reference_run.resolve()),
            "reference_git_head": provenance["git_head"],
            "reference_files_sha256": {
                name: staged.fit.file_hash(reference_run / name) for name in sources
            },
            "reference_source_files_verified": len(provenance["source_sha256"]),
            "historical_baseline_reexecuted": False,
            "optimizer_state_reused": False, "weights_and_data_reused": True,
            "pretraining_optimizer_updates_performed": 0, "new_collection_steps": 0,
        }
        staged.fit.save_json(output / "input_manifest.json", manifest)
        shutil.copy2(reference_run / "demonstration.npz", output / "demonstration.npz")
        if staged.fit.file_hash(output / "demonstration.npz") != dataset["sha256"]:
            raise RuntimeError("The copied fixed dataset does not match its source.")
        staged.fit.save_json(output / "dataset.json", dataset)
        historical = {
            "historical_reference_only": True, "learning_rate": HISTORICAL_RATE,
            "summary": reference["arms"]["td_margin"],
            "observations": read_json(reference_run / "td_margin/observations.json"),
        }
        staged.fit.save_json(output / "historical_td_margin.json", historical)
        arms = {}
        for name, learning_rate in RATES:
            arm_root = output / name
            arm_root.mkdir()
            torch.manual_seed(reference["protocol"]["initialization_seed"])
            agent, env, metadata = DQNAgent.load(checkpoint)
            try:
                if (metadata["objective"] != "pretraining"
                        or metadata["updates"] != PRETRAIN_UPDATES
                        or metadata["fit"] != reference["pretraining"]["final_fit"]):
                    raise ValueError("The loaded checkpoint is not the expected supervised policy.")
                if (staged.fit.parameter_hash(agent)
                        != reference["pretraining"]["final_parameter_sha256"]):
                    raise ValueError("The loaded supervised parameter hash differs from the reference.")
                agent.learner.config = replace(agent.learner.config, learning_rate=learning_rate)
                agent.learner.update_count = PRETRAIN_UPDATES
                boundary = begin_td_phase(agent.learner)
                if (agent.learner.optimizer.param_groups[0]["lr"] != learning_rate
                        or agent.learner.optimizer.state or agent.learner.update_count != 0):
                    raise RuntimeError("The new phase did not receive its fresh requested optimizer.")
                initial_hash = staged.fit.parameter_hash(agent)
                if initial_hash != reference["pretraining"]["final_parameter_sha256"]:
                    raise RuntimeError("The adaptation boundary changed the pretrained policy.")
                target_matches = all(torch.equal(online, target) for online, target in zip(
                    agent.learner.online.parameters(), agent.learner.target.parameters(),
                ))
                if not target_matches:
                    raise RuntimeError("TD targets must start from the pretrained online weights.")
                staged.fit.save_json(arm_root / "settings.json", {
                    "environment": settings["environment"], "dqn": asdict(agent.learner.config),
                    "objective": "td_margin", "phase_updates": PHASE_UPDATES,
                    "phase_checkpoints": CHECKPOINTS,
                })
                staged.fit.save_json(arm_root / "phase_boundary.json", {
                    **boundary, "learning_rate": learning_rate,
                    "policy_parameter_sha256": initial_hash,
                    "target_equals_online": True, "optimizer_state_empty": True,
                })
                result, _ = staged.execute_phase(
                    agent, env, settings["environment"], batch, "td_margin", PHASE_UPDATES,
                    CHECKPOINTS, arm_root, PRETRAIN_UPDATES,
                    expected_start=reference["pretraining"],
                )
                if result["final_parameter_sha256"] == initial_hash:
                    raise RuntimeError("The adaptation phase did not change model parameters.")
                arms[name] = {
                    **result, "learning_rate": learning_rate,
                    "directory": f"{name}/td_margin", "weights_changed": True,
                }
            finally:
                env.close()
        summary = {
            "protocol": protocol, "arms": arms,
            "pretrained_policy": reference["pretraining"],
            "historical_td_margin": historical,
            "optimizer_updates_total": sum(arm["optimizer_updates"] for arm in arms.values()),
            "td_optimizer_updates": sum(arm["optimizer_updates"] for arm in arms.values()),
            "pretraining_updates_performed": 0, "pretraining_updates_reused": PRETRAIN_UPDATES,
            "demonstration_collection_steps": 0,
            "evaluation_environment_steps": sum(
                arm["evaluation_environment_steps"] for arm in arms.values()
            ),
            "online_training_steps": 0, "held_out_scenes_accessed": 0,
            "accounting_note": "New evaluation actions exclude reset-time feasibility probes.",
            "limitations": [
                "One known training scene and one fixed demonstrated route; generalization untested.",
                "The historical high-rate control is reused, not reexecuted or included in new costs.",
                "Smaller steps may delay forgetting; sampled retention does not prove convergence.",
                "TD uses fixed demonstration data, with no online replay or exploration.",
                "Checkpoint selection and early stopping are not used; all failures are retained.",
            ],
        }
        staged.fit.save_json(output / "summary.json", summary)
    except BaseException as exc:
        if (output / "run_status.json").exists():
            finish_run(output, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", exc)
        raise
    else:
        finish_run(output, "completed")
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path,
                        default=ROOT / "results/training/staged_retention_20261111")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.reference_run.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
