"""Verify archived second-correction trajectories without running the simulator."""

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"
FIXED = {290012: {19: 6}, 290021: {12: 6}}
ADDED = {290012: {26: 4}, 290021: {17: 2}}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert read(EVIDENCE / "status.json")["status"] == "completed"
    manifest = read(EVIDENCE / "manifest.json")
    for name, sha in manifest.items():
        assert digest(EVIDENCE / name) == sha, name
    episodes = read(EVIDENCE / "episodes.json")
    assert {(r["seed"], r["mode"]) for r in episodes} == {
        (seed, mode) for seed in FIXED for mode in ("baseline", "corrected")}
    assert len(episodes) == 4
    verified = []
    for record in episodes:
        seed, mode = record["seed"], record["mode"]
        path = EVIDENCE / str(seed) / record["trace"]
        assert digest(path) == record["trace_sha256"]
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        overrides = {**FIXED[seed], **(ADDED[seed] if mode == "corrected" else {})}
        total, forced_steps = 0.0, []
        for step, row in enumerate(rows, start=1):
            assert row["step"] == step and not row["scripted_reference"]
            assert row["policy_action"] == int(np.argmax(row["q_values"]))
            assert row["forced_action"] == (step in overrides)
            assert row["action"] == overrides.get(step, row["policy_action"])
            if row["forced_action"]:
                forced_steps.append(step)
            total += row["reward"]
        assert forced_steps == sorted(overrides)
        metrics = record["metrics"]
        assert total == metrics["return"] and len(rows) == metrics["steps"]
        assert sum(r["info"]["collision_blocked"] for r in rows) == metrics["blocked_steps"]
        assert sum(np.linalg.norm(r["info"]["executed_motion"]) <= 1e-6 for r in rows) == metrics["stationary_steps"]
        assert not any(r["terminated"] or r["truncated"] for r in rows[:-1])
        assert rows[-1]["terminated"] or rows[-1]["truncated"]
        for key in ("success", "damage", "budget_exhausted"):
            assert rows[-1]["info"][key] == metrics[key]
        verified.append({"seed": seed, "mode": mode, "forced_steps": forced_steps,
                         "metrics_verified": True, "steps": len(rows), "success": metrics["success"]})
    result = {"manifest_files_verified": len(manifest), "primary_episodes_verified": len(verified),
              "review_script_sha256": digest(Path(__file__)), "records": verified,
              "simulator_actions": 0, "optimizer_updates": 0}
    with (HERE / "review.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Verified {len(manifest)} archived files and all four primary trajectories.")


if __name__ == "__main__":
    main()
