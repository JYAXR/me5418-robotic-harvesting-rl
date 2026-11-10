"""Verify the shared 2000-update prefix of two fixed-scene fitting diagnostics.

This read-only comparison does not execute a policy, update parameters, or audit
the later run beyond its shared prefix. The later run may still be running.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

OBJECTIVES = ("td_margin", "margin_only", "cross_entropy")
CHECKPOINTS = (0, 500, 1000, 2000)
PREFIX_UPDATES = 2000


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_equal(earlier, later, label):
    if earlier != later:
        raise ValueError(f"The compared runs differ in {label}.")


def canonical_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def update_prefix(path):
    """Read only complete records in the declared shared prefix."""
    rows = []
    with path.open(encoding="utf-8") as stream:
        for index in range(1, PREFIX_UPDATES + 1):
            line = stream.readline()
            if not line or not line.endswith("\n"):
                raise ValueError(f"The shared update prefix is not complete: {path}")
            row = json.loads(line)
            if row.get("update") != index:
                raise ValueError(f"Unexpected update sequence in {path} at update {index}.")
            rows.append(row)
    return rows


def observation_prefix(path):
    records = [row for row in read_json(path) if row["update"] <= PREFIX_UPDATES]
    require_equal([row["update"] for row in records], list(CHECKPOINTS), str(path))
    return records


def compare_parameters(earlier_path, later_path):
    """Compare tensor contents without constructing an agent or running inference."""
    earlier = torch.load(earlier_path, map_location="cpu", weights_only=True)["online"]
    later = torch.load(later_path, map_location="cpu", weights_only=True)["online"]
    require_equal(list(earlier), list(later), "checkpoint parameter names and order")
    digest = hashlib.sha256()
    for name, tensor in earlier.items():
        other = later[name]
        if tensor.dtype != other.dtype or tensor.shape != other.shape:
            raise ValueError(f"Checkpoint parameter layout differs: {name}")
        if not torch.equal(tensor, other):
            raise ValueError(f"Checkpoint parameter values differ: {name}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite checkpoint parameter: {name}")
        digest.update(tensor.detach().contiguous().numpy().tobytes())
    return {
        "parameter_values_match_exactly": True,
        "parameter_sha256": digest.hexdigest(),
        "earlier_checkpoint_sha256": sha256(earlier_path),
        "later_checkpoint_sha256": sha256(later_path),
    }


def compare(earlier_run, later_run, output):
    earlier_run, later_run, output = (
        path.resolve() for path in (earlier_run, later_run, output)
    )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite prefix evidence: {output}")
    require_equal(read_json(earlier_run / "run_status.json")["status"], "completed",
                  "earlier run completion status")
    earlier_settings = read_json(earlier_run / "settings.json")
    later_settings = read_json(later_run / "settings.json")
    for name in ("environment", "dqn"):
        require_equal(earlier_settings[name], later_settings[name], name)
    earlier_protocol = earlier_settings["training"].copy()
    later_protocol = later_settings["training"].copy()
    require_equal(earlier_protocol.pop("updates_per_arm"), PREFIX_UPDATES,
                  "earlier update budget")
    later_budget = later_protocol.pop("updates_per_arm")
    if later_budget <= PREFIX_UPDATES:
        raise ValueError("The later run must have a longer declared update budget.")
    require_equal(earlier_protocol.pop("evaluation_updates"), list(CHECKPOINTS),
                  "earlier evaluation checkpoints")
    later_checkpoints = later_protocol.pop("evaluation_updates")
    require_equal([step for step in later_checkpoints if step <= PREFIX_UPDATES],
                  list(CHECKPOINTS), "shared evaluation checkpoints")
    require_equal(earlier_protocol, later_protocol, "all other diagnostic settings")
    earlier_provenance = read_json(earlier_run / "provenance.json")
    later_provenance = read_json(later_run / "provenance.json")
    require_equal(earlier_provenance["source_sha256"], later_provenance["source_sha256"],
                  "recorded production source hashes")
    for name in ("demonstrations.json", "dataset.json"):
        require_equal(read_json(earlier_run / name), read_json(later_run / name), name)
    with (
        np.load(earlier_run / "demonstration.npz", allow_pickle=False) as earlier_data,
        np.load(later_run / "demonstration.npz", allow_pickle=False) as later_data,
    ):
        require_equal(earlier_data.files, later_data.files, "dataset array names")
        for name in earlier_data.files:
            first, second = earlier_data[name], later_data[name]
            if first.dtype != second.dtype or not np.array_equal(first, second):
                raise ValueError(f"The frozen dataset array differs: {name}")
    earlier_dataset_hash = sha256(earlier_run / "demonstration.npz")
    later_dataset_hash = sha256(later_run / "demonstration.npz")
    require_equal(earlier_dataset_hash, later_dataset_hash, "dataset file hashes")
    require_equal(earlier_dataset_hash, read_json(earlier_run / "dataset.json")["sha256"],
                  "recorded dataset hash")

    arms = {}
    for objective in OBJECTIVES:
        earlier_arm, later_arm = earlier_run / objective, later_run / objective
        updates = update_prefix(earlier_arm / "updates.jsonl")
        require_equal(updates, update_prefix(later_arm / "updates.jsonl"),
                      f"{objective} first {PREFIX_UPDATES} update records")
        observations = observation_prefix(earlier_arm / "observations.json")
        require_equal(observations, observation_prefix(later_arm / "observations.json"),
                      f"{objective} shared observation records")
        checkpoints = {
            str(step): compare_parameters(
                earlier_arm / f"update_{step}.pt", later_arm / f"update_{step}.pt",
            ) for step in CHECKPOINTS
        }
        arms[objective] = {
            "update_records_match_exactly": True,
            "updates_compared": PREFIX_UPDATES,
            "update_prefix_sha256": canonical_hash(updates),
            "observation_records_match_exactly": True,
            "observations_compared": len(CHECKPOINTS),
            "observation_prefix_sha256": canonical_hash(observations),
            "checkpoints": checkpoints,
        }
    initial_hashes = {arm["checkpoints"]["0"]["parameter_sha256"] for arm in arms.values()}
    if len(initial_hashes) != 1:
        raise ValueError("The objectives did not share a common initial model.")
    result = {
        "comparison": "exact_shared_fixed_scene_training_prefix",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "earlier_run": str(earlier_run), "later_run": str(later_run),
        "earlier_git_head": earlier_provenance["git_head"],
        "later_git_head": later_provenance["git_head"],
        "script_sha256": sha256(Path(__file__)),
        "later_status_at_comparison": read_json(later_run / "run_status.json")["status"],
        "full_run_audit": False,
        "updates_compared_per_arm": PREFIX_UPDATES,
        "later_declared_updates_per_arm": later_budget,
        "checkpoints_compared": CHECKPOINTS,
        "production_source_hashes_match": True,
        "all_settings_except_budget_and_later_checkpoints_match": True,
        "demonstration_records_match_exactly": True,
        "dataset_metadata_and_arrays_match_exactly": True,
        "dataset_sha256": earlier_dataset_hash,
        "common_initial_parameter_sha256": next(iter(initial_hashes)),
        "arms": arms,
        "environment_steps_performed": 0,
        "optimizer_updates_performed": 0,
        "policy_inferences_performed": 0,
        "notes": [
            "Only the common prefix is verified; later updates and outcomes are not audited.",
            "Checkpoint tensors are read directly; no agent or environment is instantiated.",
            "Recorded production source hashes and runner hashes match between the runs.",
            "No training logs, data, or model checkpoints are modified.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--earlier-run", required=True, type=Path)
    parser.add_argument("--later-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = compare(args.earlier_run, args.later_run, args.output)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "shared_prefix_verified": True,
        "updates_per_arm": result["updates_compared_per_arm"],
        "later_status": result["later_status_at_comparison"],
        "full_run_audit": result["full_run_audit"],
    }, indent=2))


if __name__ == "__main__":
    main()
