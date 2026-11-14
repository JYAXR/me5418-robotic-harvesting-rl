"""Compare unchanged historical control and local-weight treatment on matched schedules."""
import hashlib
import json
import shutil
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "results/training/half_td_local_supervision_20261114"
CONTROL = ROOT / "results/training/td_weight_20261114"
LATE = ROOT / "results/training/late_rollouts_20261114"
ARCHIVE = Path(__file__).parent / "evidence"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def outcomes(rows):
    failures = [{"update": row["update"], "seed": s["seed"], "rollout": s["rollout"]}
                for row in rows for s in row["scenes"] if not s["rollout"]["success"]]
    return {"checkpoints": len(rows), "episodes": len(rows) * 3,
            "all_scene_success": sum(all(s["rollout"]["success"] for s in row["scenes"]) for row in rows),
            "successful_episodes": len(rows) * 3 - len(failures), "failures": failures,
            "policy_actions": sum(s["rollout"]["steps"] for row in rows for s in row["scenes"])}


def ranking(rows, steps):
    result = {}
    for i, step in enumerate(steps):
        entries = [(row[0], row[2][i]) for row in rows if row[0] > 2000]
        negative = [update for update, gap in entries if gap < 0]
        result[str(step)] = {"negative_updates": len(negative), "first_negative": min(negative, default=None),
                            "last_negative": max(negative, default=None), "negative_update_ids": negative,
                            "minimum_gap": min(gap for _, gap in entries), "final_gap": entries[-1][1]}
    return result


def main():
    assert read(RUN / "status.json")["status"] == "completed"
    assert read(RUN / "verification.json")["passed"]
    assert not ARCHIVE.exists()
    torch.set_num_threads(1)
    for name, source in (("019_td_weight", CONTROL), ("020_late_rollouts", LATE)):
        archive = ROOT / "docs/experiments" / name / "evidence"
        for filename, info in read(archive / "archive_manifest.json")["files"].items():
            if not filename.endswith("_samples.jsonl"):
                assert digest(source / filename) == info["sha256"]
    source_summary = read(CONTROL / "summary.json")
    start = CONTROL / "treatment/td/training_state_2000.pt"
    assert digest(start) == source_summary["treatment"]["checkpoint_sha256"][start.name]
    model_hashes = read(RUN / "model_hashes.json")
    for name, expected in model_hashes.items():
        assert digest(RUN / "models" / name) == expected
        if name.startswith("update_"):
            inference = torch.load(RUN / "models" / name, weights_only=True)
            full = torch.load(RUN / "models" / name.replace("update_", "training_state_"), weights_only=True)
            assert all(torch.equal(value, full["online"][key]) for key, value in inference["online"].items())
    rows = read(RUN / "observations.json")
    original = read(CONTROL / "treatment/td/observations.json")
    late = read(LATE / "observations.json")
    control = sorted(original + late, key=lambda row: row["update"])
    assert [r["update"] for r in rows] == [r["update"] for r in control]
    assert len(rows) == 50
    dense = read(RUN / "dense.json")
    assert [r[0] for r in dense["rows"]] == list(range(2000, 10001))
    old_raw = CONTROL / "treatment/td/sentinel_q.jsonl"
    old_manifest = read(ROOT / "docs/experiments/019_td_weight/evidence/archive_manifest.json")
    assert digest(old_raw) == old_manifest["raw_log_coverage"]["treatment/td/sentinel_q.jsonl"]["raw_sha256"]
    old_dense = [[r["update"], None, [s["signed_demonstration_gap"] for s in r["states"]],
                  [s["greedy_action"] for s in r["states"]]] for r in (json.loads(line) for line in old_raw.read_text().splitlines())]
    comparison = {}
    original_steps = {r["update"] for r in original}
    for name, arm in (("historical_control", control), ("treatment", rows)):
        comparison[name] = {"all_50": outcomes(arm),
                            "original_29": outcomes([r for r in arm if r["update"] in original_steps]),
                            "late_21": outcomes([r for r in arm if 9946 <= r["update"] <= 9966])}
    logs = [json.loads(line) for line in (RUN / "updates.jsonl").read_text().splitlines()]
    assert [r["update"] for r in logs] == list(range(2001,10001))
    result = {"comparison": comparison, "rankings": {"historical_control": ranking(old_dense, dense["reference_steps"]),
              "treatment": ranking(dense["rows"], dense["reference_steps"])},
              "final_fit": rows[-1]["fit"], "final_return_fit": rows[-1]["return_fit"],
              "historical_final_fit": original[-1]["fit"], "historical_final_return_fit": original[-1]["return_fit"],
              "full_batch_perfect_update_count": sum(r[1] == 117 for r in dense["rows"][1:]),
              "full_batch_worst_correct": min(r[1] for r in dense["rows"][1:]),
              "first_update": logs[0], "last_update": logs[-1],
              "clipped_updates": sum(r["gradient_norm"] > 10 for r in logs),
              "nonzero_updates": sum(r["parameter_delta_norm"] > 0 for r in logs),
              "model_files_checked": len(model_hashes), "analysis_script_sha256": digest(Path(__file__)),
              "scope": "One inherited initialization and three known scenes. Historical control and its late diagnostic are reused. Dense ranking is not physical evaluation. Late-window sampling is retrospective and not an unbiased success-rate estimate."}
    (RUN / "analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (RUN / "historical_dense.json").write_text(json.dumps({"reference_steps": dense["reference_steps"], "rows": old_dense}, separators=(",", ":")) + "\n", encoding="utf-8")
    ARCHIVE.mkdir()
    files = {}
    for source in sorted(RUN.glob("*.json")):
        destination = ARCHIVE / source.name
        shutil.copy2(source, destination)
        assert source.read_bytes() == destination.read_bytes()
        files[source.name] = {"sha256": digest(source), "bytes": source.stat().st_size}
    raw = {}
    for name in ("updates.jsonl", "sentinels.jsonl"):
        source = RUN / name
        lines = source.read_text().splitlines()
        selected = [line for i,line in enumerate(lines) if i < 11 or json.loads(line)["update"] % 50 == 0 or i == len(lines)-1]
        dest = ARCHIVE / name.replace(".jsonl", "_samples.jsonl")
        dest.write_text("\n".join(selected) + "\n", encoding="utf-8")
        files[dest.name] = {"sha256": digest(dest), "bytes": dest.stat().st_size}
        raw[name] = {"sha256": digest(source), "records": len(lines), "sampled_records": len(selected)}
    (ARCHIVE / "archive_manifest.json").write_text(json.dumps({"files": files, "raw_log_coverage": raw,
        "source_run": str(RUN), "excluded": "All model binaries and full raw logs remain local; compact dense records retain all updates."}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({arm: {window: {k: value for k, value in data.items() if k != "failures"} for window,data in group.items()}
                      for arm,group in comparison.items()}))


if __name__ == "__main__":
    main()
