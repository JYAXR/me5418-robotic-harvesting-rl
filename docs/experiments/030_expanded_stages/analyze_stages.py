"""Audit all staged-training outcomes and archive both successes and failures."""

import csv
import importlib.util
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch

SPEC = importlib.util.spec_from_file_location("analysis030", Path(__file__).with_name("run_stages.py"))
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)
RUN = r.OUTPUT
ARCHIVE = Path(__file__).parent / "evidence"
audit = r.module("traceaudit030", r.ROOT / "docs/experiments/028_frozen_generalization/analyze_generalization.py")
audit.RUN = RUN


def main():
    assert r.read(RUN / "status.json")["status"] == "completed"
    assert not ARCHIVE.exists()
    rows = r.read(RUN / "evaluations.json")
    diagnoses = [{"arm": x["arm"], "phase": x["phase"], "update": x["update"],
                  **audit.trace_diagnosis(x)} for x in rows]
    r.save("diagnoses.json", diagnoses)
    source, _batch, returns, _, metadata, _ = r.inputs()
    assert np.array_equal(np.load(RUN / "actual_returns.npy"), returns.numpy())
    previous = source
    phases = {}
    for phase, budget in r.PHASES.items():
        start = torch.load(RUN / phase / "state_0.pt", weights_only=True)
        final = torch.load(RUN / phase / f"state_{budget}.pt", weights_only=True)
        inference = torch.load(RUN / phase / "final.pt", weights_only=True)
        r.equal(previous["online"], start["online"], "Continuous learned weights between phases")
        if phase == "td":
            r.equal(previous["optimizer"], start["optimizer"], "Preserved calibration Adam state")
        else:
            assert not start["optimizer"]["state"]
        r.equal(start["target"], start["online"], "Boundary target")
        r.equal(final["online"], inference["online"], "Inference equals final snapshot")
        r.equal(source["dqn_config"], final["dqn_config"], "Unchanged network and hyperparameters")
        r.equal(source["environment_config"], inference["environment_config"], "Unchanged environment")
        r.equal(source["optimizer_summary"]["parameter_groups"],
                final["optimizer_summary"]["parameter_groups"], "Unchanged optimizer settings")
        for name, sha in r.read(RUN / f"{phase}_model_hashes.json").items():
            assert r.digest(RUN / phase / name) == sha
        logs = [json.loads(x) for x in (RUN / phase / "updates.jsonl").read_text().splitlines()]
        assert [x["update"] for x in logs] == list(range(1, budget + 1))
        assert all(x["phase"] == phase for x in logs)
        assert sum(x["target_synchronized"] for x in logs) == (50 if phase == "td" else 0)
        if phase == "td":
            assert all(x["td_weight"] == 0.5 and x["step6_weight"] == 2
                       and x["selected_weight"] == 2 and x["recovery_weight"] == 1 for x in logs)
        elif phase == "supervised":
            assert all(x["return_loss_before"] == x["return_loss_after"] == 0 for x in logs)
        old = [x for x in rows if x["phase"] == phase]
        assert [(x["update"], x["seed"]) for x in old] == [
            (u, s) for u in r.CHECKS[phase] for s in r.OLD_SEEDS]
        new = [x for x in rows if x["phase"] == phase + "_acquisition"]
        assert [x["seed"] for x in new] == r.NEW_SEEDS
        final_old = [x for x in old if x["update"] == budget]
        phases[phase] = {
            "original_by_update": [{"update": u, "successes": sum(
                x["category"] == "success" for x in old if x["update"] == u),
                "failures": [{"seed": x["seed"], "category": x["category"]}
                             for x in old if x["update"] == u and x["category"] != "success"]}
                for u in r.CHECKS[phase]],
            "original_all_checks": r.frozen.summary(old),
            "original_final": r.frozen.summary(final_old), "added_training_final": r.frozen.summary(new),
            "all_training_final": r.frozen.summary(final_old + new),
            "final_fit": r.read(RUN / f"{phase}_fit.json")[-1],
            "final_adam": final["optimizer_summary"], "update_rows": len(logs),
            "zero_parameter_changes": sum(x["parameter_delta_norm"] == 0 for x in logs),
        }
        previous = final
    holdout, by_arm = {}, {}
    for arm in ("baseline", "staged"):
        held = [x for x in rows if x["phase"] == "holdout" and x["arm"] == arm]
        assert [x["seed"] for x in held] == r.TEST_SEEDS
        holdout[arm] = r.frozen.summary(held)
        assert holdout[arm] == r.read(RUN / f"{arm}_holdout_summary.json")
        by_arm[arm] = {x["seed"]: x for x in held}
    pairs = []
    excluded = set(metadata["initial_encoded_hashes"]) | r.previous_test_hashes()
    seen = set()
    for seed in r.TEST_SEEDS:
        a, b = by_arm["baseline"][seed], by_arm["staged"][seed]
        assert a["status"] == b["status"]
        if a["status"] == "evaluated":
            assert a["initial_encoded_sha256"] == b["initial_encoded_sha256"]
            assert a["initial_geometry"] == b["initial_geometry"] and a["certification"] == b["certification"]
            key = a["initial_encoded_sha256"]
            assert key not in seen and key not in excluded
            seen.add(key)
        pairs.append({"seed": seed, "baseline_category": a["category"], "staged_category": b["category"],
                      "baseline_success": a["category"] == "success", "staged_success": b["category"] == "success"})
    paired_counts = dict(Counter(
        "both_success" if x["baseline_success"] and x["staged_success"] else
        "staged_only" if x["staged_success"] else "baseline_only" if x["baseline_success"] else
        "neither_success" for x in pairs))
    checks = r.read(RUN / "training_verification.json")
    assert [x["phase"] for x in checks] == list(r.PHASES)
    assert all(x["passed"] and x["independent_optimizer_updates"] == 20 for x in checks)
    reloads = r.read(RUN / "reload_verification.json")
    for check in reloads:
        phase = ("td" if check["arm"] == "staged" else "original") if check["seed"] in r.OLD_SEEDS else "holdout"
        step = 10000 if check["arm"] == "staged" else 0
        expected = next(x for x in rows if x["arm"] == check["arm"] and x["phase"] == phase
                        and x["update"] == step and x["seed"] == check["seed"])
        assert check["metrics"] == expected["metrics"] and check["certification"] == expected["certification"]
    counts = {}
    for phase in dict.fromkeys(x["phase"] for x in rows):
        selected = [x for x in rows if x["phase"] == phase]
        counts[phase] = {"resets": len(selected), "executed_actions": sum(x["policy_actions"] for x in selected),
                        "certification_actions": sum(c["steps"] for x in selected for c in x["certification"])}
    counts["reloads"] = {"resets": len(reloads), "executed_actions": sum(x["metrics"]["steps"] for x in reloads),
                        "certification_actions": sum(c["steps"] for x in reloads for c in x["certification"])}
    accounting = {"phases": counts, "resets": sum(x["resets"] for x in counts.values()),
        "executed_actions": sum(x["executed_actions"] for x in counts.values()),
        "certification_actions": sum(x["certification_actions"] for x in counts.values()),
        "training_optimizer_updates": 50000, "independent_optimizer_updates": 60,
        "total_optimizer_updates": 50060, "unit_tests_excluded": True,
        "new_collection_actions": 0, "reused_demonstration_rows": 1440, "online_training_actions": 0,
        "forced_evaluation_actions": 0, "holdout_requested_scenes": 100, "holdout_episodes": 200}
    provenance = r.read(RUN / "provenance.json")
    for name, sha in provenance["source_hashes"].items():
        assert r.digest(r.ROOT / name) == sha, name
    boundary = r.read(RUN / "frozen_test_boundary.json")
    assert r.digest(r.BASELINE) == boundary["model_hashes"]["baseline"]
    assert r.digest(RUN / "td/final.pt") == boundary["model_hashes"]["staged"]
    exports = {}
    for phase in r.PHASES:
        src = RUN / phase / "final.pt"
        dest = r.ROOT / "models" / f"stages_030_{phase}.pt"
        assert not dest.exists()
        shutil.copy2(src, dest)
        assert dest.read_bytes() == src.read_bytes()
        exports[phase] = {"path": dest.relative_to(r.ROOT).as_posix(), "sha256": r.digest(dest),
            "phase": phase, "selection": "Fixed endpoint; no best-checkpoint or holdout selection.",
            "training_scenes": phases[phase]["all_training_final"],
            "original_scenes": phases[phase]["original_final"],
            "holdout": holdout["staged"] if phase == "td" else None,
            "scope": "Experimental inference checkpoint; no Adam resume state."}
        r.frozen.save(dest.with_suffix(".json"), exports[phase])
    result = {"phases": phases, "holdout": holdout, "paired_counts": paired_counts, "pairs": pairs,
        "exports": exports, "all_trace_audits_passed": True, "sources_unchanged": True,
        "analysis_source_sha256": r.digest(Path(__file__)),
        "interpretation": "Paired frozen inference comparison after an additional fixed 50000-update staged procedure; not a matched-compute or individual-component ablation."}
    r.save("analysis.json", result)
    r.save("accounting.json", accounting)
    with (RUN / "per_scene.csv").open("x", newline="", encoding="utf-8") as handle:
        fields = ["arm", "phase", "update", "seed", "category", "success", "damage", "budget_exhausted",
                  "steps", "return", "blocked_steps", "stationary_steps", "peak_deformation", "distance_to_target"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(diagnoses)
    ARCHIVE.mkdir()
    files = {}
    paths = [p for p in RUN.iterdir() if p.is_file()] + [RUN / p / "updates.jsonl" for p in r.PHASES]
    for src in sorted(paths):
        relative = src.relative_to(RUN)
        dest = ARCHIVE / relative
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(src, dest)
        assert src.read_bytes() == dest.read_bytes()
        files[relative.as_posix()] = {"sha256": r.digest(dest), "bytes": dest.stat().st_size}
    r.frozen.save(ARCHIVE / "archive_manifest.json", {"source_run": str(RUN), "files": files,
        "coverage": "All stage/holdout traces, 50000 update logs, actual returns, provenance, checks, failures and accounting. Source full resume input is tracked separately; intermediate resumes remain local; three phase inference endpoints are tracked under models/."})
    print(json.dumps({"holdout": holdout, "paired_counts": paired_counts,
        "phases": {p: {"original": v["original_final"], "all_training": v["all_training_final"],
                      "fit": v["final_fit"]} for p, v in phases.items()},
        "accounting": accounting, "archive_files": len(files)}))


if __name__ == "__main__":
    main()
