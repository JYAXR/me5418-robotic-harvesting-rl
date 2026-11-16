"""Independently verify archived trajectories and complete endpoint exports."""

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
EVIDENCE = HERE / "evidence"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert read(EVIDENCE / "status.json")["status"] == "completed"
    manifest = read(EVIDENCE / "manifest.json")
    for name, sha in manifest.items():
        assert digest(EVIDENCE / name) == sha, name
    records = read(EVIDENCE / "evaluations.json")
    assert len(records) == 120
    assert len({(r["arm"], r["update"], r["seed"]) for r in records}) == 120
    for record in records:
        path = EVIDENCE / record["trace"]
        assert digest(path) == record["trace_compressed_sha256"]
        raw = gzip.decompress(path.read_bytes())
        assert hashlib.sha256(raw).hexdigest() == record["trace_sha256"]
        rows = [json.loads(line) for line in raw.decode().splitlines()]
        total = 0.0
        for step, row in enumerate(rows, start=1):
            assert row["step"] == step
            assert np.isfinite(row["q_values"]).all()
            assert row["action"] == int(np.argmax(row["q_values"]))
            total += row["reward"]
        metrics = record["metrics"]
        assert total == metrics["return"] and len(rows) == metrics["steps"]
        assert sum(r["info"]["collision_blocked"] for r in rows) == metrics["blocked_steps"]
        assert sum(np.linalg.norm(r["info"]["executed_motion"]) <= 1e-6 for r in rows) == metrics["stationary_steps"]
        assert not any(r["terminated"] or r["truncated"] for r in rows[:-1])
        assert rows[-1]["terminated"] or rows[-1]["truncated"]
        for key in ("success", "damage", "budget_exhausted"):
            assert rows[-1]["info"][key] == metrics[key]
        assert record["frozen_state_unchanged"]
    exports = read(EVIDENCE / "model_exports.json")
    for path, sha in exports.items():
        assert digest(ROOT / path) == sha
    for arm in ("control", "targeted"):
        model = torch.load(ROOT / f"models/your_method_004_{arm}.pt", weights_only=True)
        state = torch.load(HERE / "resume" / f"{arm}_state_14000.pt", weights_only=True)
        for key, value in model["online"].items():
            assert torch.equal(value, state["online"][key]) and torch.equal(value, state["target"][key])
        assert model["environment_config"] == state["environment_config"]
        assert model["dqn_config"] == state["dqn_config"]
        assert state["update_count"] == 14000
        assert state["optimizer_summary"]["step_min"] == state["optimizer_summary"]["step_max"] == 34000
    summary = read(EVIDENCE / "summary.json")
    baseline = {r["seed"]: r for r in read(EVIDENCE / "source_baseline.json")}
    assert len(baseline) == 33
    per_scene = []
    for arm in ("control", "targeted"):
        final = [r for r in records if r["arm"] == arm and r["update"] == 2000]
        assert len(final) == 33 and {r["seed"] for r in final} == set(baseline)
        successes = {r["seed"] for r in final if r["metrics"]["success"]}
        old_successes = {s for s, r in baseline.items() if r["metrics"]["success"]}
        assert successes == set(summary["final"][arm]["all"]["success_seeds"])
        assert sorted(successes - old_successes) == summary["final"][arm]["gained_vs_source"]
        assert sorted(old_successes - successes) == summary["final"][arm]["lost_vs_source"]
        for record in final:
            current, old = record["metrics"], baseline[record["seed"]]["metrics"]
            per_scene.append({"arm": arm, "seed": record["seed"],
                "source_success": old["success"], "final_success": current["success"],
                "return_change": current["return"] - old["return"],
                "step_change": current["steps"] - old["steps"],
                "peak_deformation_change": current["peak_deformation"] - old["peak_deformation"]})
    reloads = read(EVIDENCE / "reload_verification.json")
    assert len(reloads) == 12
    for reload in reloads:
        expected = next(r for r in records if r["arm"] == reload["arm"]
                        and r["update"] == 2000 and r["seed"] == reload["seed"])
        assert reload["metrics"] == expected["metrics"]
    assert sum(r["metrics"]["steps"] for r in records + reloads) == summary["episode_actions"]
    assert sum(r["reset_profile"]["certification_actions"] for r in records + reloads) == summary["certification_actions"]
    audits = read(EVIDENCE / "update_verification.json")
    assert len(audits) == 4 and sum(a["updates"] for a in audits) == 40 and all(a["exact"] for a in audits)
    result = {"manifest_files_verified": len(manifest), "primary_traces_verified": len(records),
        "export_files_verified": len(exports), "reload_metrics_verified": len(reloads),
        "review_script_sha256": digest(Path(__file__)), "additional_optimizer_updates": 0,
        "additional_simulator_actions": 0, "per_scene_changes": per_scene}
    with (HERE / "artifact_verification.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Verified {len(manifest)} evidence files, 120 trajectories, four model exports and 12 reload outcomes.")


if __name__ == "__main__":
    main()
