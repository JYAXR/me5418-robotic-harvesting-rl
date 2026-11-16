"""Compare five verified supervision weights with matched unchanged continuation."""

import gzip
import importlib.util
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from me5418_harvesting.agents.dqn_agent import DQNAgent
from me5418_harvesting.agents.dqn_learning import TransitionBatch
from me5418_harvesting.agents.training import evaluate
from me5418_harvesting.envs.scene_cache import CertifiedSceneCache

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUTPUT = ROOT / "results/training/your_method_004_verified_supervision"
SOURCE = ROOT / "docs/experiments/034_scene10002_supervision/checkpoints/targeted_state_12000.pt"
SOURCE_SHA = "fe8e29438165cd4d99c6e143f819a37a2febc5e704a47efb9a895bb1074c610f"
MODEL = ROOT / "models/supervision_034_targeted.pt"
MODEL_SHA = "202fd8d4f6929da61e991962c6f667dd80a9cbcc330b1b744434230f140f8b8b"
DATA = ROOT / "docs/experiments/029_training_coverage/evidence"
HISTORY = HERE.parent / "001_training_coverage/evidence"
SEEDS = [10000, 10001, 10002, *range(290000, 290030)]
DIAGNOSTIC = [10000, 10001, 10002, 290007, 290012, 290021]
TARGETS = [(290007, 8, 430, 6), (290012, 19, 653, 6), (290012, 26, 660, 4),
           (290021, 12, 1052, 6), (290021, 17, 1057, 2)]
UPDATES = 2000
CHECKS = [10, 100, 500, 1000, 2000]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


prior = module("own004prior", ROOT / "docs/experiments/034_scene10002_supervision/run_training.py")
core = module("own004core", HERE / "focused_update.py")
tracer = module("own004trace", ROOT / "docs/experiments/035_scene_cache/run_benchmark.py")
stages, frozen, equal, require = prior.stages, prior.frozen, prior.equal, prior.require
read, digest = prior.read, prior.digest


def save(name, value):
    frozen.save(OUTPUT / name, value)


def inputs():
    require(digest(SOURCE) == SOURCE_SHA and digest(MODEL) == MODEL_SHA, "Source model changed.")
    source = torch.load(SOURCE, weights_only=True)
    inference = torch.load(MODEL, weights_only=True)
    for key in ("online", "environment_config", "dqn_config"):
        equal(source[key], inference[key], "Source snapshot matches policy")
    require(source["update_count"] == 12000, "Wrong source TD clock.")
    require(source["optimizer_summary"]["step_min"] == source["optimizer_summary"]["step_max"] == 32000,
            "Wrong source Adam clock.")
    metadata = read(DATA / "dataset.json")
    require(metadata["seeds"] == SEEDS and digest(DATA / "expanded_demonstrations.npz") == metadata["npz_sha256"],
            "Dataset changed.")
    with np.load(DATA / "expanded_demonstrations.npz") as arrays:
        batch = TransitionBatch(*(torch.from_numpy(arrays[k].copy()) for k in TransitionBatch.__dataclass_fields__))
    require(len(batch.actions) == 1440, "Wrong dataset size.")
    sample = read(ROOT / "docs/experiments/027_step6_supervision/evidence/recovery_sample.json")
    require(sample["action"] == 4, "Existing recovery label changed.")
    recovery = torch.tensor([sample["observation"]], dtype=torch.float32)
    labels = []
    for seed, step, index, action in TARGETS:
        attempt = "002_first_action_rescue" if seed == 290007 else "003_second_action_rescue"
        path = HERE.parent / attempt / "evidence" / str(seed) / "corrected.jsonl"
        record = read(path.with_suffix(".json"))
        require(digest(path) == record["trace_sha256"] and record["metrics"]["success"], "Unverified correction trace.")
        row = json.loads(path.read_text(encoding="utf-8").splitlines()[step - 1])
        require(row["forced_action"] and row["action"] == int(batch.actions[index]) == action, "Wrong verified label.")
        state_hash = frozen.encoded_hash(batch.observations[index].numpy())
        require(state_hash == row["encoded_observation_sha256"], "Verified state is not the stored row.")
        labels.append({"seed": seed, "step": step, "row": index, "action": action,
                       "state_sha256": state_hash, "source_trace": path.relative_to(ROOT).as_posix()})
    historical = {r["seed"]: r for r in read(HISTORY / "episodes.json") if r["arm"] == "targeted"}
    require(sorted(historical) == sorted(SEEDS), "Missing full source evaluation.")
    for path, sha in read(HISTORY / "provenance.json")["input_hashes"].items():
        if path.startswith("src/") or path == MODEL.relative_to(ROOT).as_posix():
            require(digest(ROOT / path) == sha, "Historical baseline source changed.")
    return source, batch, recovery, metadata, labels, historical


def probes(agent, batch):
    rows = sorted({i + delta for i in core.VERIFIED_ROWS for delta in (-1, 0, 1)})
    records = []
    with torch.no_grad():
        for index in rows:
            q = agent.learner.online(batch.observations[index:index + 1])[0]
            label = int(batch.actions[index])
            other = q.clone()
            other[label] = -torch.inf
            records.append({"row": index, "label": label, "greedy_action": int(q.argmax()),
                            "reference_margin": float(q[label] - other.max()), "q_values": q.tolist()})
    return records


def evaluate_export(model, arm, update, seeds, cache, historical, evaluations):
    directory = OUTPUT / "traces" / f"{arm}_{update}"
    directory.mkdir(parents=True, exist_ok=False)
    tracer.MODEL, tracer.OUTPUT = model, directory
    # Model construction uses Torch RNG; restore the training stream afterwards.
    with torch.random.fork_rng(devices=[]):
        for seed in seeds:
            record, _ = tracer.episode(seed, cache, update, arm)
            equal(record["reset_record"], historical[seed]["reset_record"], "Historical initial state")
            if arm == "source":
                equal(record["metrics"], historical[seed]["metrics"], "Historical source metrics")
            path = directory / record["trace"]
            raw = path.read_bytes()
            packed = gzip.compress(raw, mtime=0)
            require(gzip.decompress(packed) == raw, "Lossy compression.")
            compressed = path.with_suffix(".jsonl.gz")
            compressed.write_bytes(packed)
            path.unlink()
            record.update(arm=arm, update=update, trace=compressed.relative_to(OUTPUT).as_posix(),
                          trace_compressed_sha256=digest(compressed))
            evaluations.append(record)
            save("evaluations.json", evaluations)


def group_summary(rows):
    metrics = [r["metrics"] for r in rows]
    successful = [m for m in metrics if m["success"]]
    return {"episodes": len(metrics), "successes": len(successful),
            "success_seeds": [m["seed"] for m in successful],
            "damage_failures": sum(m["damage"] for m in metrics),
            "budget_failures": sum(m["budget_exhausted"] for m in metrics),
            "mean_return": sum(m["return"] for m in metrics) / len(metrics),
            "mean_success_steps": sum(m["steps"] for m in successful) / len(successful) if successful else None}


def run():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    started, tick = datetime.now(timezone.utc).isoformat(), perf_counter()
    save("status.json", {"status": "running", "started_at_utc": started})
    evaluations, fits, audits, reloads, exports = [], [], [], [], {}
    train_seconds = {}
    try:
        torch.set_num_threads(1)
        source, batch, recovery, metadata, labels, historical = inputs()
        config = source["environment_config"]
        files = list((ROOT / "src").rglob("*.py")) + list((ROOT / "docs/experiments").rglob("*.py"))
        files += [SOURCE, MODEL, HERE / "PROTOCOL.md", DATA / "dataset.json", DATA / "expanded_demonstrations.npz",
                  HISTORY / "episodes.json", HISTORY / "provenance.json",
                  ROOT / "docs/experiments/027_step6_supervision/evidence/recovery_sample.json"]
        files += [ROOT / item["source_trace"] for item in labels]
        hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in files}
        save("provenance.json", {"input_hashes": hashes, "labels": labels,
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
            "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "torch_threads": 1, "training_seeds": SEEDS, "environment": config, "dqn": source["dqn_config"],
            "dataset": metadata, "initial_adam": source["optimizer_summary"], "updates_per_arm": UPDATES,
            "checks": CHECKS, "weights": {"control": 1, "targeted": 2}, "action_overrides": 0})
        save("source_baseline.json", list(historical.values()))
        cache = CertifiedSceneCache(33)
        evaluate_export(MODEL, "source", 0, DIAGNOSTIC, cache, historical, evaluations)
        for arm, weight in (("control", 1.0), ("targeted", 2.0)):
            core.VERIFIED_WEIGHT = weight
            directory = OUTPUT / arm
            directory.mkdir()
            agent, env = stages.restore(source)
            train_seconds[arm] = 0.0
            equal(stages.snapshot(agent, config), source, "Exact common starting state")
            torch.save(source, directory / "state_0.pt")
            with (directory / "updates.jsonl").open("x", encoding="utf-8", newline="\n") as log:
                for update in range(1, UPDATES + 1):
                    start = perf_counter()
                    stats = core.recovery_td_update(agent.learner, batch, recovery)
                    train_seconds[arm] += perf_counter() - start
                    log.write(json.dumps({"update": update, **stats}, allow_nan=False) + "\n")
                    if update in CHECKS or update == UPDATES - 10:
                        torch.save(stages.snapshot(agent, config), directory / f"state_{update}.pt")
                    if update in CHECKS:
                        before = stages.snapshot(agent, config)
                        fits.append({"arm": arm, "update": update, "probes": probes(agent, batch)})
                        save("probes.json", fits)
                        model = directory / f"policy_{update}.pt"
                        agent.save(model, config, {"attempt": "your_method_004", "arm": arm,
                            "verified_weight": weight, "additional_updates": update, "training_seeds": SEEDS})
                        evaluate_export(model, arm, update, SEEDS if update == UPDATES else DIAGNOSTIC,
                                        cache, historical, evaluations)
                        equal(stages.snapshot(agent, config), before, "Evaluation preserves full training state")
                    if update % 100 == 0:
                        log.flush()
                        save("progress.json", {"arm": arm, "update": update, "budget": UPDATES})
                        print(f"{arm}: {update}/{UPDATES} updates; training seconds {train_seconds[arm]:.1f}", flush=True)
            env.close()
            logs = [json.loads(line) for line in (directory / "updates.jsonl").read_text().splitlines()]
            require(len(logs) == UPDATES and all(row["parameter_delta_norm"] > 0 for row in logs), "Missing update.")
            for start in (0, UPDATES - 10):
                agent, env = stages.restore(torch.load(directory / f"state_{start}.pt", weights_only=True))
                for update in range(start + 1, start + 11):
                    stats = core.independent_update(agent.learner, batch, recovery)
                    equal({"update": update, **stats}, logs[update - 1], "Independent update metrics")
                equal(stages.snapshot(agent, config), torch.load(directory / f"state_{start + 10}.pt", weights_only=True),
                      "Independent complete state")
                env.close()
                audits.append({"arm": arm, "start": start, "updates": 10, "exact": True})
                save("update_verification.json", audits)
            final = torch.load(directory / f"state_{UPDATES}.pt", weights_only=True)
            require(final["update_count"] == 14000, "Wrong final TD clock.")
            require(final["optimizer_summary"]["step_min"] == final["optimizer_summary"]["step_max"] == 34000,
                    "Wrong final Adam clock.")
            equal(final["online"], final["target"], "Final target synchronization")
            equal(final["dqn_config"], source["dqn_config"], "Training parameters changed")
            exports[arm] = {"model": directory / f"policy_{UPDATES}.pt", "resume": directory / f"state_{UPDATES}.pt"}
            for seed in DIAGNOSTIC:
                agent, env, _ = DQNAgent.load(exports[arm]["model"])
                before, timings = frozen.state(agent), []
                env.profile_reset = True
                result = evaluate(agent, env, 1, seed, timings=timings)["records"][0]
                expected = next(r for r in evaluations if r["arm"] == arm and r["update"] == UPDATES and r["seed"] == seed)
                equal(result, expected["metrics"], "Endpoint normal evaluator reload")
                require(frozen.equal(before, frozen.state(agent)), "Reload changes learner state.")
                reloads.append({"arm": arm, "seed": seed, "metrics": result, "timing": timings[0],
                                "reset_profile": dict(env.last_reset_profile)})
                save("reload_verification.json", reloads)
                env.close()
        for path, sha in hashes.items():
            require(digest(ROOT / path) == sha, "Pinned input changed.")
        final_rows = {arm: [r for r in evaluations if r["arm"] == arm and r["update"] == UPDATES]
                      for arm in exports}
        base_success = {s for s, r in historical.items() if r["metrics"]["success"]}
        outcomes = {}
        for arm, rows in final_rows.items():
            successes = {r["seed"] for r in rows if r["metrics"]["success"]}
            outcomes[arm] = {"all": group_summary(rows),
                "original": group_summary([r for r in rows if r["seed"] in SEEDS[:3]]),
                "targeted": group_summary([r for r in rows if r["seed"] in DIAGNOSTIC[3:]]),
                "gained_vs_source": sorted(successes - base_success), "lost_vs_source": sorted(base_success - successes)}
        control_success = set(outcomes["control"]["all"]["success_seeds"])
        target_success = set(outcomes["targeted"]["all"]["success_seeds"])
        save("summary.json", {"final": outcomes, "source_successes": sorted(base_success),
            "targeted_gained_vs_control": sorted(target_success - control_success),
            "targeted_lost_vs_control": sorted(control_success - target_success),
            "main_optimizer_updates": 4000, "verification_optimizer_updates": 40, "unit_test_optimizer_updates": 4,
            "primary_episodes": len(evaluations), "reload_episodes": len(reloads), "action_overrides": 0,
            "episode_actions": sum(r["metrics"]["steps"] for r in evaluations + reloads),
            "certification_actions": sum(r["reset_profile"]["certification_actions"] for r in evaluations + reloads),
            "certification_calls": sum(r["reset_profile"]["certification_calls"] for r in evaluations + reloads),
            "training_seconds": train_seconds, "elapsed_seconds": perf_counter() - tick,
            "scope": "Single continuation seed; fixed final endpoints; training-bank evaluation only."})
        save("status.json", {"status": "completed", "started_at_utc": started,
                             "finished_at_utc": datetime.now(timezone.utc).isoformat()})
    except BaseException as exc:
        save("status.json", {"status": "failed", "started_at_utc": started, "error": repr(exc)})
        raise
    archive = HERE / "evidence"
    archive.mkdir(exist_ok=False)
    for path in OUTPUT.rglob("*"):
        if path.is_file() and path.suffix in (".json", ".jsonl", ".gz"):
            dest = archive / path.relative_to(OUTPUT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            require(path.read_bytes() == dest.read_bytes(), "Archive changed bytes.")
    (HERE / "resume").mkdir(exist_ok=False)
    models = {}
    for arm, paths in exports.items():
        model = ROOT / f"models/your_method_004_{arm}.pt"
        resume = HERE / "resume" / f"{arm}_state_14000.pt"
        require(not model.exists(), "Model export already exists.")
        shutil.copy2(paths["model"], model)
        shutil.copy2(paths["resume"], resume)
        for source_path, dest in ((paths["model"], model), (paths["resume"], resume)):
            require(digest(source_path) == digest(dest), "Export changed bytes.")
            models[dest.relative_to(ROOT).as_posix()] = digest(dest)
    frozen.save(archive / "model_exports.json", models)
    frozen.save(archive / "manifest.json", {p.relative_to(archive).as_posix(): digest(p)
                                            for p in sorted(archive.rglob("*")) if p.is_file()})


if __name__ == "__main__":
    run()
