"""Locate first action divergences of a frozen policy using identical inputs."""

import gzip
import hashlib
import importlib.util
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUTPUT = HERE / "evidence"
PRIOR = HERE.parent / "004_verified_supervision"
SOURCE_HISTORY = HERE.parent / "001_training_coverage/evidence"
DATA = ROOT / "docs/experiments/029_training_coverage/evidence"
MODEL = ROOT / "models/your_method_004_targeted.pt"
MODEL_SHA = "ea42fd1a2d23317c9e15a960c58d90970a38edbcfb7724308a845ead92f6119f"
SNAPSHOT = PRIOR / "resume/targeted_state_14000.pt"
SNAPSHOT_SHA = "ed32355abf23c463a523df6bf00d6a9708b727bcd15fc04f357eaadd6f68fd2c"
SEEDS = [290012, 290021, 290017, 290022]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


training = module("own005training", PRIOR / "run_training.py")
tracer = training.tracer
frozen, equal, require = training.frozen, training.equal, training.require
read, digest = training.read, training.digest
NAMES = frozen.ACTION_NAMES


def comparison_at(row, baseline_action):
    q = row["q_values"]
    chosen = row["action"]
    require(chosen == int(np.argmax(q)), "Current action is not greedy.")
    return {"step": row["step"], "common_prefix_steps": row["step"] - 1,
            "encoded_observation_sha256": row["encoded_observation_sha256"],
            "current_action": chosen, "current_action_name": NAMES[chosen],
            "comparison_action": baseline_action, "comparison_action_name": NAMES[baseline_action],
            "current_chosen_q": q[chosen], "current_comparison_q": q[baseline_action],
            "current_chosen_minus_comparison_q": q[chosen] - q[baseline_action],
            "current_q_values": q}


def first_reference_fork(rows, observations, actions):
    for index, row in enumerate(rows):
        require(index < len(actions), "Policy outlived an identical complete reference prefix.")
        require(row["step"] == index + 1, "Bad trace ordering.")
        require(row["encoded_observation_sha256"] == frozen.encoded_hash(observations[index]),
                "Reference state mismatch before an action fork.")
        if row["action"] != int(actions[index]):
            return comparison_at(row, int(actions[index]))
    return None


def first_policy_fork(current, baseline):
    for index, (row, old) in enumerate(zip(current, baseline)):
        require(row["step"] == old["step"] == index + 1, "Bad paired ordering.")
        for key in ("position", "encoded_observation_sha256", "raw_observation", "rng_before"):
            equal(row[key], old[key], "Common policy input")
        require(old["action"] == int(np.argmax(old["q_values"])), "Baseline is not greedy.")
        if row["action"] != old["action"]:
            result = comparison_at(row, old["action"])
            q = old["q_values"]
            result.update(baseline_chosen_q=q[old["action"]], baseline_current_action_q=q[row["action"]],
                          baseline_chosen_minus_current_action_q=q[old["action"]] - q[row["action"]],
                          baseline_q_values=q, identical_input_verified=True)
            return result
        for key in ("raw_next_observation", "rng_after", "reward", "terminated", "truncated", "info"):
            equal(row[key], old[key], "Identical same-action transition")
    require(len(current) == len(baseline), "Unequal episode lengths without an action fork.")
    return None


def load_trace(directory, record):
    path = directory / record["trace"]
    require(digest(path) == record["trace_compressed_sha256"], "Compressed trace changed.")
    raw = gzip.decompress(path.read_bytes())
    require(hashlib.sha256(raw).hexdigest() == record["trace_sha256"], "Raw trace changed.")
    return [json.loads(line) for line in raw.decode().splitlines()], path


class FrozenSnapshotLoader:
    """Load the exact trained Adam state for the existing read-only trace wrapper."""

    @staticmethod
    def load(path):
        require(Path(path) == MODEL, "Unexpected requested model.")
        source = torch.load(SNAPSHOT, weights_only=True)
        agent, env = training.stages.restore(source)
        equal(training.stages.snapshot(agent, source["environment_config"]), source, "Exact snapshot restore")
        original_close = env.close

        def close():
            equal(training.stages.snapshot(agent, source["environment_config"]), source,
                  "Unchanged full model, Adam, clocks and RNG")
            original_close()

        def forbidden(*args, **kwargs):
            raise RuntimeError("Optimizer steps are forbidden in frozen diagnosis.")

        env.close = close
        agent.learner.optimizer.step = forbidden
        return agent, env, {"frozen_snapshot": True}


def run():
    OUTPUT.mkdir(exist_ok=False)
    started, tick = datetime.now(timezone.utc).isoformat(), perf_counter()
    frozen.save(OUTPUT / "status.json", {"status": "running", "started_at_utc": started})
    records, diagnostics = [], []
    try:
        torch.set_num_threads(1)
        require(digest(MODEL) == MODEL_SHA and digest(SNAPSHOT) == SNAPSHOT_SHA, "Wrong frozen endpoint.")
        payload = torch.load(MODEL, weights_only=True)
        source = torch.load(SNAPSHOT, weights_only=True)
        for key in ("online", "environment_config", "dqn_config"):
            equal(payload[key], source[key], "Inference matches resume state")
        require(source["update_count"] == 14000, "Wrong TD clock.")
        require(source["optimizer_summary"]["step_min"] == source["optimizer_summary"]["step_max"] == 34000,
                "Wrong Adam clock.")
        old_records = read(SOURCE_HISTORY / "episodes.json")
        current_records = read(PRIOR / "evidence/evaluations.json")
        current = {s: next(r for r in current_records if r["arm"] == "targeted" and r["update"] == 2000 and r["seed"] == s)
                   for s in SEEDS}
        controls = {s: next(r for r in current_records if r["arm"] == "control" and r["update"] == 2000 and r["seed"] == s)
                    for s in SEEDS}
        old = {s: next(r for r in old_records if r["arm"] == "targeted" and r["seed"] == s) for s in SEEDS}
        files = list((ROOT / "src").rglob("*.py")) + list((ROOT / "docs/experiments").rglob("*.py"))
        files += [MODEL, SNAPSHOT, HERE / "PROTOCOL.md", SOURCE_HISTORY / "episodes.json",
                  PRIOR / "evidence/evaluations.json", DATA / "dataset.json", DATA / "expanded_demonstrations.npz"]
        for provenance, key in ((SOURCE_HISTORY / "provenance.json", "input_hashes"),
                                (PRIOR / "evidence/provenance.json", "input_hashes")):
            files.append(provenance)
            for path, sha in read(provenance)[key].items():
                if path.startswith(("src/", "models/")):
                    require(digest(ROOT / path) == sha, "Historical runtime code changed.")
                    files.append(ROOT / path)
        control_model = ROOT / "models/your_method_004_control.pt"
        export_manifest = PRIOR / "evidence/model_exports.json"
        require(digest(control_model) == read(export_manifest)[control_model.relative_to(ROOT).as_posix()],
                "Matched control model changed.")
        files += [control_model, export_manifest]
        traces = {}
        for label, rows, directory in (("current", current, PRIOR / "evidence"),
                                       ("control", controls, PRIOR / "evidence"),
                                       ("source", old, SOURCE_HISTORY)):
            traces[label] = {}
            for seed in SEEDS:
                traces[label][seed], path = load_trace(directory, rows[seed])
                files.append(path)
                equal(current[seed]["reset_record"], rows[seed]["reset_record"], "Same seeded initial state")
        metadata = read(DATA / "dataset.json")
        require(digest(DATA / "expanded_demonstrations.npz") == metadata["npz_sha256"], "Dataset changed.")
        hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in files}
        frozen.save(OUTPUT / "provenance.json", {"input_hashes": hashes, "seeds": SEEDS,
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
            "python": platform.python_version(), "torch": torch.__version__, "torch_threads": 1,
            "td_clock": source["update_count"], "adam": source["optimizer_summary"],
            "environment": source["environment_config"], "dqn": source["dqn_config"],
            "optimizer_updates": 0, "action_overrides": 0})
        tracer.MODEL, tracer.OUTPUT, tracer.DQNAgent = MODEL, OUTPUT, FrozenSnapshotLoader
        with np.load(DATA / "expanded_demonstrations.npz") as arrays:
            for seed in SEEDS:
                record, rows = tracer.episode(seed, None, 0, "targeted")
                equal(rows, traces["current"][seed], "Every historical transition, observation, Q and RNG")
                equal(record["metrics"], current[seed]["metrics"], "Historical metrics")
                equal(record["reset_record"], current[seed]["reset_record"], "Historical reset state")
                record.update(full_snapshot_unchanged=True, historical_trace_exact=True)
                records.append(record)
                frozen.save(OUTPUT / "episodes.json", records)
                episode = next(e for e in metadata["episodes"] if e["seed"] == seed)
                start, stop = episode["start"], episode["stop"]
                reference = first_reference_fork(rows, arrays["observations"][start:stop], arrays["actions"][start:stop])
                if reference is not None:
                    reference["dataset_row"] = start + reference["step"] - 1
                diagnostics.append({"seed": seed, "current_metrics": record["metrics"],
                    "source_metrics": old[seed]["metrics"], "control_metrics": controls[seed]["metrics"],
                    "vs_reference": reference, "vs_source": first_policy_fork(rows, traces["source"][seed]),
                    "vs_control": first_policy_fork(rows, traces["control"][seed])})
                frozen.save(OUTPUT / "diagnostics.json", diagnostics)
        for path, sha in hashes.items():
            require(digest(ROOT / path) == sha, "Pinned input changed.")
        frozen.save(OUTPUT / "summary.json", {"episodes": len(records), "optimizer_updates": 0,
            "action_overrides": 0, "episode_actions": sum(r["metrics"]["steps"] for r in records),
            "certification_actions": sum(r["reset_profile"]["certification_actions"] for r in records),
            "certification_calls": sum(r["reset_profile"]["certification_calls"] for r in records),
            "full_snapshot_unchanged": True, "all_historical_replays_exact": True,
            "elapsed_seconds": perf_counter() - tick,
            "scope": "First disagreements only; no corrective replay, learning or causal-failure proof."})
        frozen.save(OUTPUT / "status.json", {"status": "completed", "started_at_utc": started,
                                           "finished_at_utc": datetime.now(timezone.utc).isoformat()})
        frozen.save(OUTPUT / "manifest.json", {p.name: digest(p) for p in sorted(OUTPUT.iterdir()) if p.is_file()})
    except BaseException as exc:
        frozen.save(OUTPUT / "status.json", {"status": "failed", "started_at_utc": started, "error": repr(exc)})
        raise


if __name__ == "__main__":
    run()
