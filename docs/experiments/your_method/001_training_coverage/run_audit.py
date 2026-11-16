"""Audit two frozen policies on their complete training scene bank."""

import gzip
import importlib.util
import json
import platform
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from me5418_harvesting.envs.scene_cache import CertifiedSceneCache

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUTPUT = HERE / "evidence"
SEEDS = [10000, 10001, 10002, *range(290000, 290030)]
MODELS = {
    "targeted": ("models/supervision_034_targeted.pt",
                 "202fd8d4f6929da61e991962c6f667dd80a9cbcc330b1b744434230f140f8b8b"),
    "control": ("models/supervision_034_control.pt",
                "a8a519b1944afec3bd5ac063b34db6c75136c1661c3c4f8fbc3412859e58e018"),
}
HELPER = ROOT / "docs/experiments/035_scene_cache/run_benchmark.py"
spec = importlib.util.spec_from_file_location("coverage_trace_helper", HELPER)
trace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trace)
frozen = trace.frozen


def save(name, value):
    frozen.save(OUTPUT / name, value)


def aggregate(records):
    metrics = [r["metrics"] for r in records]
    successful = [m for m in metrics if m["success"]]
    return {
        "episodes": len(records),
        "successes": len(successful),
        "success_rate": len(successful) / len(records),
        "damage_failures": sum(m["damage"] for m in metrics),
        "budget_failures": sum(m["budget_exhausted"] for m in metrics),
        "mean_return": sum(m["return"] for m in metrics) / len(records),
        "mean_success_steps": (sum(m["steps"] for m in successful) / len(successful)
                               if successful else None),
        "failure_signatures": dict(Counter(r["failure_signature"] for r in records)),
        "success_seeds": [r["seed"] for r in records if r["metrics"]["success"]],
    }


def run():
    OUTPUT.mkdir(exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    start_clock = perf_counter()
    save("status.json", {"status": "running", "started_at_utc": started})
    records, pairs = [], []
    try:
        torch.set_num_threads(1)
        trace.OUTPUT = OUTPUT
        paths = list((ROOT / "src").rglob("*.py")) + [Path(__file__),
            HERE / "PROTOCOL.md", HELPER,
            ROOT / "docs/experiments/028_frozen_generalization/run_frozen_generalization.py",
            ROOT / "docs/experiments/034_scene10002_supervision/evidence/evaluations.json"]
        paths += [ROOT / path for path, _ in MODELS.values()]
        for path, expected in MODELS.values():
            frozen.require(frozen.digest(ROOT / path) == expected, "Unexpected model hash.")
        hashes = {p.relative_to(ROOT).as_posix(): frozen.digest(p) for p in paths}
        save("provenance.json", {"input_hashes": hashes, "models": MODELS,
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
            "python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "platform": platform.platform(),
            "seeds": SEEDS, "threads": 1, "optimizer_updates": 0, "action_overrides": 0})
        history = json.loads((ROOT / "docs/experiments/034_scene10002_supervision/evidence/evaluations.json").read_text())
        cache = CertifiedSceneCache(33)
        for index, seed in enumerate(SEEDS):
            pair = {}
            order = ("control", "targeted") if index % 2 == 0 else ("targeted", "control")
            for arm in order:
                trace.MODEL = ROOT / MODELS[arm][0]
                record, rows = trace.episode(seed, cache, 0, arm)
                trailing_stationary = 0
                for row in reversed(rows):
                    if np.linalg.norm(row["info"]["executed_motion"]) > 1e-6:
                        break
                    trailing_stationary += 1
                record["trailing_stationary_steps"] = trailing_stationary
                record["failure_signature"] = frozen.failure_signature(record["metrics"], trailing_stationary)
                record["original_metrics_exact"] = None
                if seed < 290000:
                    expected = next(r for r in history if r["arm"] == arm
                                    and r["update"] == 2000 and r["seed"] == seed)
                    trace.require_equal(record["metrics"], expected["metrics"], "Original-scene metrics changed.")
                    record["original_metrics_exact"] = True
                path = OUTPUT / record["trace"]
                raw = path.read_bytes()
                compressed = gzip.compress(raw, mtime=0)
                frozen.require(gzip.decompress(compressed) == raw, "Trace compression changed bytes.")
                compressed_path = path.with_suffix(path.suffix + ".gz")
                compressed_path.write_bytes(compressed)
                path.unlink()
                record["trace"] = compressed_path.name
                record["trace_compressed_sha256"] = frozen.digest(compressed_path)
                records.append(record)
                save("episodes.json", records)
                pair[arm] = record
            trace.require_equal(pair["control"]["reset_record"], pair["targeted"]["reset_record"],
                                "Paired initial scene or RNG differs.")
            pairs.append({"seed": seed, "reset_exact": True,
                          "control_success": pair["control"]["metrics"]["success"],
                          "targeted_success": pair["targeted"]["metrics"]["success"]})
            save("pairs.json", pairs)
        for name, expected in hashes.items():
            frozen.require(frozen.digest(ROOT / name) == expected, "Source or model changed.")
        save("summary.json", {
            "groups": {arm: {
                group: aggregate([r for r in records if r["arm"] == arm and r["seed"] in seeds])
                for group, seeds in (("original", SEEDS[:3]), ("added", SEEDS[3:]), ("all", SEEDS))
            } for arm in MODELS},
            "gained_seeds": [p["seed"] for p in pairs if p["targeted_success"] and not p["control_success"]],
            "lost_seeds": [p["seed"] for p in pairs if p["control_success"] and not p["targeted_success"]],
            "episodes": len(records), "optimizer_updates": 0, "action_overrides": 0,
            "episode_actions": sum(r["metrics"]["steps"] for r in records),
            "certification_actions": sum(r["reset_profile"]["certification_actions"] for r in records),
            "certification_calls": sum(r["reset_profile"]["certification_calls"] for r in records),
            "cache_hits": sum(r["reset_profile"]["cache_status"] == "hit" for r in records),
            "elapsed_seconds": perf_counter() - start_clock,
            "verification": "All 33 reset pairs, six historical metrics, 66 frozen-state checks, greedy actions and input hashes passed.",
            "scope": "Complete training-bank audit; no unseen-scene or teammate comparison."})
        save("status.json", {"status": "completed", "started_at_utc": started,
                             "finished_at_utc": datetime.now(timezone.utc).isoformat()})
        save("manifest.json", {p.name: frozen.digest(p) for p in sorted(OUTPUT.iterdir()) if p.is_file()})
    except BaseException as exc:
        save("status.json", {"status": "failed", "started_at_utc": started,
                             "completed_episodes": len(records), "error": repr(exc)})
        raise


if __name__ == "__main__":
    run()
