"""Audit every frozen evaluation trace and archive all outcomes, including failures."""

import csv
import importlib.util
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location("frozen028_analysis", Path(__file__).with_name("run_frozen_generalization.py"))
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)
RUN = r.OUTPUT
ARCHIVE = Path(__file__).parent / "evidence"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def trace_diagnosis(row):
    if row["status"] != "evaluated":
        return {"seed": row["seed"], "category": row["category"]}
    trace_path = RUN / row["trace"]
    assert r.digest(trace_path) == row["trace_sha256"]
    trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert len(trace) == row["metrics"]["steps"]
    assert [x["step"] for x in trace] == list(range(1, len(trace) + 1))
    assert trace[0]["encoded_observation_sha256"] == row["initial_encoded_sha256"]
    reward = 0.0
    stationary = blocked = trailing = longest = 0
    first = {"blocked": None, "stationary": None, "contact": None}
    actions = Counter()
    for i, x in enumerate(trace):
        assert x["action"] == int(np.asarray(x["q_values"]).argmax())
        assert x["action_name"] == r.ACTION_NAMES[x["action"]]
        q = np.sort(np.asarray(x["q_values"], dtype=np.float32))
        assert x["q_top_gap"] == float(q[-1] - q[-2])
        if i:
            assert x["position"] == trace[i - 1]["next_position"]
            assert x["encoded_observation_sha256"] == trace[i - 1]["next_encoded_observation_sha256"]
        if i < len(trace) - 1:
            assert not x["terminated"] and not x["truncated"]
        stopped = bool(np.linalg.norm(np.asarray(x["info"]["executed_motion"], dtype=np.float32)) <= 1e-6)
        is_blocked = x["info"]["collision_blocked"]
        stationary += int(stopped)
        blocked += int(is_blocked)
        trailing = trailing + 1 if stopped else 0
        longest = max(longest, trailing)
        for name, event in (("blocked", is_blocked), ("stationary", stopped),
                            ("contact", x["info"]["contact_steps"] > 0)):
            if event and first[name] is None:
                first[name] = x["step"]
        actions[x["action_name"]] += 1
        reward += x["reward"]
    assert trace[-1]["terminated"] or trace[-1]["truncated"]
    metrics = {"seed": row["seed"], **r.episode_metrics(reward, len(trace), trace[-1]["info"], stationary, blocked)}
    assert metrics == row["metrics"]
    assert row["category"] == r.failure_signature(metrics, trailing)
    assert trailing == row["trailing_stationary_steps"] and longest == row["longest_stationary_run"]
    assert first == row["first_events"] and actions == row["action_counts"]
    assert row["certification"] == row["reset_info"]["reference_route_checks"]
    assert row["certification"][-1]["success"]
    assert row["frozen_state_verified"]
    # These post-hoc signatures describe physical motion, not a full Markov-state cycle.
    two_step_returns = sum(np.linalg.norm(np.asarray(x["next_position"]) - np.asarray(trace[i - 1]["position"])) <= 1e-6
        and np.linalg.norm(np.asarray(x["next_position"]) - np.asarray(x["position"])) > 1e-6
        for i, x in enumerate(trace) if i)
    return {"seed": row["seed"], "category": row["category"], **metrics,
        "first_events": first, "trailing_stationary_steps": trailing,
        "longest_stationary_run": longest, "two_step_position_returns": int(two_step_returns),
        "first_blocked_transition": next((x for x in trace if x["info"]["collision_blocked"]), None),
        "terminal_action": trace[-1]["action_name"], "terminal_q_gap": trace[-1]["q_top_gap"],
        "final_ten_actions": [x["action_name"] for x in trace[-10:]],
        "final_position": trace[-1]["next_position"],
        "distance_reduction": float(np.linalg.norm(np.asarray(row["initial_geometry"]["target"]) - np.asarray(row["initial_geometry"]["start"]))) - metrics["distance_to_target"]}


def main():
    assert read(RUN / "status.json")["status"] == "completed"
    assert not ARCHIVE.exists()
    rows = read(RUN / "episodes.json")
    controls = read(RUN / "controls.json")
    verification = read(RUN / "verification.json")
    assert [x["seed"] for x in rows] == r.TEST_SEEDS
    assert verification["passed"] and verification["optimizer_updates"] == 0
    diagnoses = [trace_diagnosis(x) for x in rows]
    for control in controls:
        trace_diagnosis(control)
    assert len(controls) == 6 and all(x["category"] == "success" for x in controls)
    for check in verification["records"]:
        original = next(x for x in rows if x["seed"] == check["seed"])
        assert check["metrics"] == original["metrics"] and check["certification"] == original["certification"]
    summary = read(RUN / "summary.json")
    assert summary == r.summary(rows)
    provenance = read(RUN / "provenance.json")
    for name, digest in provenance["source_hashes"].items():
        assert r.digest(r.ROOT / name) == digest, name
    starts = [x["initial_encoded_sha256"] for x in rows if x["status"] == "evaluated"]
    assert len(set(starts)) == len(starts) and not set(starts) & set(provenance["training_initial_hashes"])
    account = read(RUN / "accounting.json")
    assert account["optimizer_updates"] == account["scripted_actions"] == account["training_data_added"] == 0
    assert account["total_policy_actions"] == sum(x["policy_actions"] for x in rows + controls + verification["records"])
    assert account["total_certification_actions"] == sum(c["steps"] for x in rows + controls + verification["records"] for c in x["certification"])
    assert account["total_resets"] == len(rows) + len(controls) + len(verification["records"])
    r.save(RUN / "diagnoses.json", diagnoses)
    valid = [x for x in rows if x["status"] == "evaluated"]
    groups = {}
    for category in sorted({x["category"] for x in rows}):
        selected = [x for x in valid if x["category"] == category]
        groups[category] = {"seeds": [x["seed"] for x in rows if x["category"] == category],
            "mean_steps": float(np.mean([x["metrics"]["steps"] for x in selected])) if selected else None,
            "mean_blocked_fraction": float(np.mean([x["metrics"]["blocked_steps"] / x["metrics"]["steps"] for x in selected])) if selected else None,
            "mean_stationary_fraction": float(np.mean([x["metrics"]["stationary_steps"] / x["metrics"]["steps"] for x in selected])) if selected else None,
            "max_peak_deformation": max((x["metrics"]["peak_deformation"] for x in selected), default=None)}
    analysis = {"summary": summary, "groups": groups,
        "distinct_test_initial_states": len(set(starts)), "training_initial_state_overlap": 0,
        "accepted_scenes_have_certified_routes": all(x["certification"][-1]["success"] for x in valid),
        "main_damage_count": sum(x["metrics"]["damage"] for x in valid),
        "main_budget_exhausted_count": sum(x["metrics"]["budget_exhausted"] for x in valid),
        "main_undamaged_budget_failures": sum(not x["metrics"]["success"] and not x["metrics"]["damage"] and x["metrics"]["budget_exhausted"] for x in valid),
        "normal_evaluator_reloads": len(verification["records"]), "all_trace_audits_passed": True,
        "model_sha256": r.digest(r.MODEL), "analysis_source_sha256": r.digest(Path(__file__)),
        "scope": "Frozen within-distribution holdout evaluation. Failure signatures are descriptive, not intervention-based causal diagnoses. No policy changes."}
    r.save(RUN / "analysis.json", analysis)
    with (RUN / "per_scene.csv").open("x", newline="", encoding="utf-8") as f:
        fields = ["seed", "category", "success", "damage", "budget_exhausted", "steps", "return",
                  "blocked_steps", "stationary_steps", "distance_to_target", "peak_deformation", "trailing_stationary_steps"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(diagnoses)
    ARCHIVE.mkdir()
    files = {}
    for path in sorted(RUN.iterdir()):
        if path.is_file():
            target = ARCHIVE / path.name
            shutil.copy2(path, target)
            assert target.read_bytes() == path.read_bytes()
            files[path.name] = {"sha256": r.digest(path), "bytes": path.stat().st_size}
    r.save(ARCHIVE / "archive_manifest.json", {"source_run": str(RUN), "files": files,
        "coverage": "All 100 requested-scene records, all emitted action traces, six known-scene controls and all selected reload metrics. No failed scenes omitted."})
    print(json.dumps({"summary": summary, "groups": {k:len(v["seeds"]) for k,v in groups.items()},
        "archived_files": len(files), "archive_bytes": sum(v["bytes"] for v in files.values())}))


if __name__ == "__main__":
    main()
