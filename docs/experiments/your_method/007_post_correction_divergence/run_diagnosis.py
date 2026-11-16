"""Locate the next same-state fork after one fixed correction on seed 290017."""

import importlib.util
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUTPUT = HERE / "evidence"
SEED = 290017
HISTORY = HERE.parent / "006_single_action_rescue/evidence"
spec = importlib.util.spec_from_file_location("own007rescue", HERE.parent / "006_single_action_rescue/run_rescue.py")
rescue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rescue)
prior = rescue.prior
frozen, equal, require = prior.frozen, prior.equal, prior.require
read, digest = prior.read, prior.digest


def run():
    OUTPUT.mkdir(exist_ok=False)
    started, tick = datetime.now(timezone.utc).isoformat(), perf_counter()
    frozen.save(OUTPUT / "status.json", {"status": "running", "started_at_utc": started})
    try:
        torch.set_num_threads(1)
        require(digest(prior.MODEL) == prior.MODEL_SHA, "Wrong inference model.")
        require(digest(prior.SNAPSHOT) == prior.SNAPSHOT_SHA, "Wrong full snapshot.")
        require(rescue.CASES[SEED] == (5, 5), "Wrong fixed correction.")
        for name, sha in read(HISTORY / "manifest.json").items():
            require(digest(HISTORY / name) == sha, "Historical evidence changed.")
        for name, sha in read(HISTORY / "provenance.json")["input_hashes"].items():
            require(digest(ROOT / name) == sha, "Historical input changed.")
        old_record = next(r for r in read(HISTORY / "episodes.json")
                          if r["seed"] == SEED and r["arm"] == "corrected")
        old_rows = [json.loads(line) for line in (HISTORY / old_record["trace"]).read_text().splitlines()]
        baseline_rows = [json.loads(line) for line in (HISTORY / f"baseline_0_{SEED}.jsonl").read_text().splitlines()]
        source_dir = HERE.parent / "001_training_coverage/evidence"
        control_dir = HERE.parent / "004_verified_supervision/evidence"
        source_record = next(r for r in read(source_dir / "episodes.json")
                             if r["seed"] == SEED and r["arm"] == "targeted")
        control_record = next(r for r in read(control_dir / "evaluations.json")
                              if r["seed"] == SEED and r["arm"] == "control" and r["update"] == 2000)
        baselines = {}
        paths = list((ROOT / "src").rglob("*.py")) + list((ROOT / "docs/experiments").rglob("*.py"))
        paths += [prior.MODEL, prior.SNAPSHOT, HERE / "PROTOCOL.md",
                  source_dir / "episodes.json", control_dir / "evaluations.json"]
        paths += [p for p in HISTORY.iterdir() if p.is_file()]
        for label, directory, record in (("source", source_dir, source_record),
                                         ("control", control_dir, control_record)):
            require(record["metrics"]["success"], "Comparison policy did not succeed.")
            equal(record["reset_record"], old_record["reset_record"], "Same initial state")
            rows, path = prior.load_trace(directory, record)
            baselines[label] = record, rows
            paths.append(path)
        snapshot = torch.load(prior.SNAPSHOT, weights_only=True)
        require(snapshot["update_count"] == 14000, "Wrong TD clock.")
        require(snapshot["optimizer_summary"]["step_min"] == snapshot["optimizer_summary"]["step_max"] == 34000,
                "Wrong Adam clock.")
        hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}
        frozen.save(OUTPUT / "provenance.json", {"input_hashes": hashes,
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
            "python": platform.python_version(), "torch": torch.__version__, "torch_threads": 1,
            "seed": SEED, "fixed_override": {"step": 5, "action": 5, "name": "south_large"},
            "td_clock": snapshot["update_count"], "adam": snapshot["optimizer_summary"],
            "environment": snapshot["environment_config"], "dqn": snapshot["dqn_config"],
            "optimizer_updates": 0, "new_training_samples": 0})
        rescue.OUTPUT = OUTPUT
        record, rows = rescue.episode(SEED, corrected=True)
        frozen.save(OUTPUT / "episode.json", record)
        equal(rows, old_rows, "Every corrected historical observation, Q, action, transition and RNG")
        equal(record["metrics"], old_record["metrics"], "Exact historical metrics")
        equal(record["reset_record"], old_record["reset_record"], "Exact historical reset")
        check = rescue.audit_pair(SEED, baseline_rows, rows)
        frozen.save(OUTPUT / "intervention_check.json", check)
        diagnostics = {}
        for label, (baseline_record, baseline) in baselines.items():
            fork = prior.first_policy_fork(rows, baseline)
            require(fork is not None and fork["step"] > 5, "No subsequent fork after the fixed correction.")
            diagnostics[label] = {"current_metrics": record["metrics"],
                                  "baseline_metrics": baseline_record["metrics"], "first_fork": fork}
        frozen.save(OUTPUT / "diagnostics.json", diagnostics)
        for name, sha in hashes.items():
            require(digest(ROOT / name) == sha, "Pinned input changed.")
        frozen.save(OUTPUT / "summary.json", {"episodes": 1, "episode_actions": record["metrics"]["steps"],
            "certification_actions": record["reset_profile"]["certification_actions"],
            "certification_calls": record["reset_profile"]["certification_calls"],
            "action_overrides": record["forced_actions"], "additional_action_overrides": 0,
            "optimizer_updates": 0, "full_snapshot_unchanged": True, "historical_replay_exact": True,
            "elapsed_seconds": perf_counter() - tick,
            "scope": "First post-correction disagreements only; no extra rescue or training."})
        frozen.save(OUTPUT / "status.json", {"status": "completed", "started_at_utc": started,
                                            "finished_at_utc": datetime.now(timezone.utc).isoformat()})
        frozen.save(OUTPUT / "manifest.json", {p.name: digest(p) for p in sorted(OUTPUT.iterdir()) if p.is_file()})
    except BaseException as exc:
        frozen.save(OUTPUT / "status.json", {"status": "failed", "started_at_utc": started, "error": repr(exc)})
        raise


if __name__ == "__main__":
    run()
