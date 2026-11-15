"""Audit paired coverage expansion, original-scene retention and final holdout outcomes."""

import csv

# Load the runner without executing its entry point.
import importlib.util
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch

SPEC = importlib.util.spec_from_file_location("coverage029analysis", Path(__file__).with_name("run_coverage.py"))
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)
RUN = r.OUTPUT
ARCHIVE = Path(__file__).parent / "evidence"
audit = r.module("coverage029traceaudit", r.ROOT / "docs/experiments/028_frozen_generalization/analyze_generalization.py")
audit.RUN = RUN


def main():
    assert r.read(RUN / "status.json")["status"] == "completed"
    assert not ARCHIVE.exists()
    records = r.read(RUN / "evaluations.json")
    diagnoses = []
    for row in records:
        diagnoses.append({"arm": row["arm"], "phase": row["phase"], "relative_update": row["relative_update"], **audit.trace_diagnosis(row)})
    r.save("diagnoses.json", diagnoses)
    source = torch.load(r.SOURCE, weights_only=True)
    data = r.read(RUN / "dataset.json")
    assert data["seeds"] == r.OLD_SEEDS + r.NEW_SEEDS
    assert r.digest(RUN / "expanded_demonstrations.npz") == data["npz_sha256"]
    with np.load(RUN / "expanded_demonstrations.npz") as expanded, np.load(r.ORIGINAL) as old:
        for key in old.files:
            assert np.array_equal(expanded[key][:117], old[key])
        assert len(expanded["actions"]) == data["rows"]
    holdout, retention = {}, {}
    for arm in ("control", "expanded"):
        held = [x for x in records if x["arm"] == arm and x["phase"] == "holdout"]
        assert [x["seed"] for x in held] == r.TEST_SEEDS
        holdout[arm] = r.frozen.summary(held)
        assert holdout[arm] == r.read(RUN / f"{arm}_holdout_summary.json")
        retained = [x for x in records if x["arm"] == arm and x["phase"] == "retention"]
        assert [(x["relative_update"], x["seed"]) for x in retained] == [(u,s) for u in r.CHECKPOINTS for s in r.OLD_SEEDS]
        retention[arm] = {"by_update": [{"relative_update": u, "successes": sum(x["category"] == "success" for x in retained if x["relative_update"] == u),
            "failures": [{"seed": x["seed"], "category": x["category"]} for x in retained if x["relative_update"] == u and x["category"] != "success"]} for u in r.CHECKPOINTS],
            "summary": r.frozen.summary(retained)}
        initial = [x for x in retained if x["relative_update"] == 0]
        expected = r.read(r.ROOT / "models/step6_repair_027.json")["evaluation"]["records"]
        assert [x["metrics"] for x in initial] == expected
        logs = [json.loads(line) for line in (RUN / arm / "updates.jsonl").read_text().splitlines()]
        assert len(logs) == r.UPDATES and [x["absolute_update"] for x in logs] == list(range(10001,18001))
        assert all(x["parameter_delta_norm"] > 0 for x in logs)
        assert all(x["td_weight"] == 0.5 and x["step6_weight"] == 2 and x["selected_weight"] == 2 and x["recovery_weight"] == 1 for x in logs)
        for key, digest in r.read(RUN / f"{arm}_model_hashes.json").items():
            assert r.digest(RUN / arm / key) == digest
        final = torch.load(RUN / arm / "state_8000.pt", weights_only=True)
        inference = torch.load(RUN / arm / "final.pt", weights_only=True)
        r.equal(final["online"], inference["online"], "Matching exported final weights")
        r.equal(final["dqn_config"], source["dqn_config"], "Preserved hyperparameters")
        r.equal(final["optimizer_summary"]["parameter_groups"], source["optimizer_summary"]["parameter_groups"], "Preserved Adam settings")
        assert final["optimizer_summary"]["step_min"] == final["optimizer_summary"]["step_max"] == 38000
    acquisition = [x for x in records if x["phase"] == "acquisition"]
    assert [x["seed"] for x in acquisition] == r.NEW_SEEDS
    control = {x["seed"]:x for x in records if x["phase"] == "holdout" and x["arm"] == "control"}
    treatment = {x["seed"]:x for x in records if x["phase"] == "holdout" and x["arm"] == "expanded"}
    pairs = []
    for seed in r.TEST_SEEDS:
        a,b = control[seed], treatment[seed]
        assert a["status"] == b["status"]
        if a["status"] == "evaluated":
            assert a["initial_encoded_sha256"] == b["initial_encoded_sha256"]
            assert a["initial_geometry"] == b["initial_geometry"] and a["certification"] == b["certification"]
        pairs.append({"seed": seed, "control_category": a["category"], "expanded_category": b["category"],
            "control_success": a["category"] == "success", "expanded_success": b["category"] == "success"})
    paired_counts = dict(Counter("both_success" if x["control_success"] and x["expanded_success"] else
        "expanded_only" if x["expanded_success"] else "control_only" if x["control_success"] else "neither_success" for x in pairs))
    collection = r.read(RUN / "collection.json")
    reloads = r.read(RUN / "reload_verification.json")
    for check in reloads:
        expected = next(x for x in records if x["arm"] == check["arm"] and x["relative_update"] == 8000 and x["phase"] == ("retention" if check["seed"] in r.OLD_SEEDS else "holdout") and x["seed"] == check["seed"])
        assert check["metrics"] == expected["metrics"] and check["certification"] == expected["certification"]
    phases = {"collection": {"resets": len(collection), "executed_actions": sum(x["collection"]["collection_environment_steps"] for x in collection), "certification_actions": sum(c["steps"] for x in collection for c in x["certification"])}}
    for phase in ("retention", "acquisition", "holdout"):
        selected = [x for x in records if x["phase"] == phase]
        phases[phase] = {"resets": len(selected), "executed_actions": sum(x["policy_actions"] for x in selected), "certification_actions": sum(c["steps"] for x in selected for c in x["certification"])}
    phases["reloads"] = {"resets": len(reloads), "executed_actions": sum(x["metrics"]["steps"] for x in reloads), "certification_actions": sum(c["steps"] for x in reloads for c in x["certification"])}
    account = {"phases": phases, "resets": sum(x["resets"] for x in phases.values()), "executed_actions": sum(x["executed_actions"] for x in phases.values()), "certification_actions": sum(x["certification_actions"] for x in phases.values()),
        "training_optimizer_updates": 16000, "gate_optimizer_updates": 4, "independent_optimizer_updates": 40, "total_optimizer_updates": 16044,
        "main_holdout_scenes": 100, "main_holdout_policy_episodes": 200, "online_training_steps": 0,
        "scripted_collection_actions": phases["collection"]["executed_actions"], "forced_evaluation_actions": 0, "unit_tests_excluded": True}
    train_checks = r.read(RUN / "training_verification.json")
    assert len(train_checks) == 2 and all(x["passed"] and x["independent_optimizer_updates"] == 20 for x in train_checks)
    provenance = r.read(RUN / "provenance.json")
    for name, sha in provenance["source_hashes"].items():
        assert r.digest(r.ROOT / name) == sha, name
    boundary = r.read(RUN / "frozen_test_boundary.json")
    exports = {}
    for arm in ("control", "expanded"):
        src = RUN / arm / "final.pt"
        assert r.digest(src) == boundary["final_model_hashes"][arm]
        dest = r.ROOT / "models" / f"coverage_029_{arm}.pt"
        assert not dest.exists()
        shutil.copy2(src, dest)
        assert dest.read_bytes() == src.read_bytes()
        exports[arm] = {"path": dest.relative_to(r.ROOT).as_posix(), "sha256": r.digest(dest),
            "scope": "Predeclared final experimental candidate; not a replacement for the previous policy.",
            "holdout": holdout[arm], "final_original_scenes": retention[arm]["by_update"][-1]}
        r.frozen.save(dest.with_suffix(".json"), exports[arm])
    result = {"holdout": holdout, "retention": retention, "new_training_acquisition": r.frozen.summary(acquisition),
        "paired_counts": paired_counts, "pairs": pairs, "original_sample_fraction": data["old_row_fraction"],
        "training_rows": data["rows"], "final_fit": {arm:r.read(RUN / f"{arm}_fit.json")[-1] for arm in ("control", "expanded")},
        "exports": exports, "all_trace_audits_passed": True, "sources_unchanged": True,
        "analysis_source_sha256": r.digest(Path(__file__)), "interpretation": "Matched fixed-data continuation; no retraining of supervised/calibration stages. Old-sample dilution follows the unchanged full-data mean rule."}
    r.save("analysis.json", result)
    r.save("accounting.json", account)
    with (RUN / "per_scene.csv").open("x", newline="", encoding="utf-8") as handle:
        fields = ["arm","phase","relative_update","seed","category","success","damage","budget_exhausted","steps","return","blocked_steps","stationary_steps","peak_deformation","distance_to_target"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader();writer.writerows(diagnoses)
    ARCHIVE.mkdir()
    files = {}
    paths = [p for p in RUN.iterdir() if p.is_file()] + [RUN / arm / "updates.jsonl" for arm in ("control", "expanded")]
    for src in sorted(paths):
        relative = src.relative_to(RUN)
        dest = ARCHIVE / relative
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(src, dest)
        assert dest.read_bytes() == src.read_bytes()
        files[relative.as_posix()] = {"sha256": r.digest(dest), "bytes": dest.stat().st_size}
    r.frozen.save(ARCHIVE / "archive_manifest.json", {"source_run": str(RUN), "files": files,
        "coverage": "All main/control/acquisition traces, all 16000 update logs, complete expanded demonstration arrays, failures, provenance, checks and accounting. Full resume snapshots stay local; both final inference candidates are separately tracked under models/."})
    print(json.dumps({"holdout": holdout, "final_retention": {arm:retention[arm]["by_update"][-1] for arm in retention},
        "acquisition": result["new_training_acquisition"], "paired_counts": paired_counts, "accounting": account, "archive_files": len(files)}))


if __name__ == "__main__":
    main()
