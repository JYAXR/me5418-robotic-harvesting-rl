"""Audit completed late-rollout evidence and archive exact JSON bytes."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "results/training/late_rollouts_20261114"
DESTINATION = Path(__file__).parent / "evidence"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert read(RUN / "status.json")["status"] == "completed"
    assert read(RUN / "verification.json")["passed"]
    provenance = read(RUN / "provenance.json")
    assert digest(Path(__file__).with_name("run_late_rollouts.py")) == provenance["script_sha256"]
    for name, expected in read(RUN / "models.json").items():
        assert digest(RUN / "models" / name) == expected
    raw = ROOT / "results/training/td_weight_20261114/treatment/td/sentinel_q.jsonl"
    reference = {r["update"]: r for r in (json.loads(line) for line in raw.read_text().splitlines())}
    rows = read(RUN / "observations.json")
    assert [row["update"] for row in rows] == list(range(9946, 9967))
    compact, failures, successes = [], [], 0
    for row in rows:
        assert [s["seed"] for s in row["scenes"]] == [10000, 10001, 10002]
        focus = next(s for s in row["scenes"] if s["seed"] == 10001)
        decision = next((d for d in focus["decisions"] if d["step"] == 21), None)
        sentinel = next(s for s in reference[row["update"]]["states"] if s["reference_step"] == 21)
        if decision is not None and decision["matches_reference_state"]:
            assert decision["q_values"] == sentinel["q_values"]
            assert decision["action"] == sentinel["greedy_action"]
        for scene in row["scenes"]:
            rollout = scene["rollout"]
            assert [d["action"] for d in scene["decisions"]] == rollout["actions"]
            assert len(scene["decisions"]) == rollout["steps"]
            successes += int(rollout["success"])
            if not rollout["success"]:
                failures.append({"update": row["update"], "seed": scene["seed"],
                                 "rollout": rollout, "first_decisions": scene["decisions"][:24]})
        compact.append({"update": row["update"], "successes": sum(s["rollout"]["success"] for s in row["scenes"]),
                        "reference_step21_gap": sentinel["signed_demonstration_gap"],
                        "actual_step21": decision,
                        "scene10001_outcome": {k: value for k, value in focus["rollout"].items() if k not in ("actions", "robot_path")}})
    accounting = read(RUN / "accounting.json")
    assert sum(s["rollout"]["steps"] for row in rows for s in row["scenes"]) == accounting["main_policy_actions"]
    cert = read(RUN / "reset_certification.json")
    assert len(cert) == 63
    assert sum(r["certification_environment_steps"] for r in cert) == accounting["main_certification_actions"]
    result = {"successful_episodes": successes, "evaluated_episodes": 63,
              "all_scene_success_checkpoints": sum(row["successes"] == 3 for row in compact),
              "checkpoints": compact, "failures": failures,
              "scope": "Targeted retrospective known-scene evaluation; no forced action, independent action-causal rescue, or held-out test.",
              "analysis_script_sha256": digest(Path(__file__))}
    (RUN / "analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    DESTINATION.mkdir(exist_ok=False)
    files = {}
    for source in sorted(RUN.glob("*.json")):
        destination = DESTINATION / source.name
        shutil.copy2(source, destination)
        assert source.read_bytes() == destination.read_bytes()
        files[source.name] = {"sha256": digest(source), "bytes": source.stat().st_size}
    (DESTINATION / "archive_manifest.json").write_text(json.dumps({"files": files,
        "excluded": "21 full model binaries and console/test logs remain local.",
        "source_run": str(RUN)}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"successful_episodes": successes, "all_scene_success_checkpoints": result["all_scene_success_checkpoints"],
                      "failures": [{"update": r["update"], "seed": r["seed"]} for r in failures], "archived_files": len(files)}))


if __name__ == "__main__":
    main()
