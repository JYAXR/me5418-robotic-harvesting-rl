"""Audit actual visits to neighboring reference states and archive complete outcomes."""
import hashlib
import json
import shutil
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "results/training/neighbor_rollouts_20261114"
SOURCE = ROOT / "results/training/half_td_local_supervision_20261114"
ARCHIVE = Path(__file__).parent / "evidence"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert read(RUN / "status.json")["status"] == "completed" and read(RUN / "verification.json")["passed"]
    assert not ARCHIVE.exists()
    for name, expected in read(RUN / "verified_source_hashes.json").items():
        assert digest(Path(name)) == expected
    for name, expected in read(RUN / "model_hashes.json").items():
        assert digest(RUN / "models" / name) == expected
    probes = {r["update"]: r for r in (json.loads(line) for line in (SOURCE / "sentinels.jsonl").read_text().splitlines())}
    selection = read(RUN / "selection.json")
    rows = read(RUN / "observations.json")
    assert [r["update"] for r in rows] == selection["window"]
    summaries, failures = [], []
    successful_episodes = 0
    for row in rows:
        assert [s["seed"] for s in row["scenes"]] == [10000, 10001, 10002]
        focus = row["scenes"][1]
        reference = {r["reference_step"]: r for r in probes[row["update"]]["states"]}
        actual = {}
        for step in (21, 22):
            decision = next((d for d in focus["decisions"] if d["step"] == step), None)
            if decision and decision["matches_reference_state"]:
                assert decision["q_values"] == reference[step]["q_values"]
                assert decision["action"] == reference[step]["greedy_action"]
            actual[str(step)] = decision
        for scene in row["scenes"]:
            result = scene["rollout"]
            assert [d["action"] for d in scene["decisions"]] == result["actions"]
            assert len(scene["decisions"]) == result["steps"]
            successful_episodes += int(result["success"])
            if not result["success"]:
                failures.append({"update": row["update"], "seed": scene["seed"], "rollout": result})
        summaries.append({"update": row["update"], "reference_step22_gap": reference[22]["signed_demonstration_gap"],
                          "reference_step22_action": reference[22]["greedy_action"], "actual_decisions": actual,
                          "scene10001_outcome": {k: value for k, value in focus["rollout"].items() if k not in ("actions", "robot_path")},
                          "scene_success_count": sum(s["rollout"]["success"] for s in row["scenes"])})
    negative = [r for r in summaries if r["reference_step22_gap"] < 0]
    positive = [r for r in summaries if r["reference_step22_gap"] >= 0]
    assert len(negative) == selection["selected_negative_count"] == 11
    mixed = []
    for left, right in pairwise(rows):
        a, b = left["scenes"][1], right["scenes"][1]
        left_gap = next(p for p in probes[left["update"]]["states"] if p["reference_step"] == 22)["signed_demonstration_gap"]
        right_gap = next(p for p in probes[right["update"]]["states"] if p["reference_step"] == 22)["signed_demonstration_gap"]
        if right["update"] != left["update"] + 1 or left_gap >= 0 or right_gap >= 0 or a["rollout"]["success"] == b["rollout"]["success"]:
            continue
        difference = next((i for i, (x, y) in enumerate(zip(a["rollout"]["actions"], b["rollout"]["actions"])) if x != y), None)
        mixed.append({"updates": [left["update"], right["update"]], "successes": [a["rollout"]["success"], b["rollout"]["success"]],
                      "first_executed_action_difference_step": None if difference is None else difference + 1,
                      "decision_pair": None if difference is None else [a["decisions"][difference], b["decisions"][difference]]})
    accounting = read(RUN / "accounting.json")
    assert accounting["main_policy_actions"] == sum(s["rollout"]["steps"] for r in rows for s in r["scenes"])
    assert accounting["main_certification_actions"] == sum(r["certification_environment_steps"] for r in read(RUN / "reset_certification.json"))
    result = {"evaluated_models": len(rows), "successful_episodes": successful_episodes,
              "evaluated_episodes": len(rows) * 3, "all_scene_success_models": sum(r["scene_success_count"] == 3 for r in summaries),
              "negative_gap_models": len(negative), "negative_gap_scene10001_successes": sum(r["scene10001_outcome"]["success"] for r in negative),
              "negative_gap_actual_reference_visits": sum(bool(r["actual_decisions"]["22"] and r["actual_decisions"]["22"]["matches_reference_state"]) for r in negative),
              "positive_gap_models": len(positive), "positive_gap_scene10001_successes": sum(r["scene10001_outcome"]["success"] for r in positive),
              "checkpoints": summaries, "failures": failures, "adjacent_mixed_outcomes": mixed, "analysis_script_sha256": digest(Path(__file__)),
              "scope": "Four intervals chosen from existing Q diagnostics, covering 11/690 negative models. No forced action, isolated-action rescue, or held-out evaluation; outcomes cannot be extrapolated to every reversal."}
    (RUN / "analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    ARCHIVE.mkdir()
    files = {}
    for source in sorted(RUN.glob("*.json")):
        destination = ARCHIVE / source.name
        shutil.copy2(source, destination)
        assert source.read_bytes() == destination.read_bytes()
        files[source.name] = {"sha256": digest(source), "bytes": source.stat().st_size}
    (ARCHIVE / "archive_manifest.json").write_text(json.dumps({"files": files, "source_run": str(RUN),
          "excluded": "19 reconstructed model binaries and console/test logs remain local."}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: value for k, value in result.items() if k not in ("checkpoints", "failures")}))


if __name__ == "__main__":
    main()
