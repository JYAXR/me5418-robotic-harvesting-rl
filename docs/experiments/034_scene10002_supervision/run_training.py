"""Compare targeted action supervision with matched unchanged TD continuation."""
import importlib.util
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from me5418_harvesting.agents.dqn_agent import DQNAgent
from me5418_harvesting.agents.training import evaluate

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).parent
OUTPUT = ROOT / "results/training/scene10002_supervision_034_20261116"
UPDATES = 2000
CHECKS = [0, 10, 100, 500, 1000, 2000]
SEEDS = [10000, 10001, 10002]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


prior = module("focus034prior", ROOT / "docs/experiments/033_scene10002_step28/run_step28.py")
core = module("focus034core", HERE / "focused_update.py")
stages = prior.stages
frozen, equal, require = prior.frozen, prior.equal, prior.require
read, digest = prior.read, prior.digest


def save(name, value):
    frozen.save(OUTPUT / name, value)


def inputs():
    source, _, _, _, _ = prior.inputs()
    _, batch, _, recovery, metadata, _ = stages.inputs()
    trace_path = ROOT / "docs/experiments/033_scene10002_step28/evidence/corrected.jsonl"
    manifest = read(trace_path.parent / "archive_manifest.json")
    require(digest(trace_path) == manifest["files"][trace_path.name]["sha256"], "Changed rescue trace.")
    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    for step, index, action in ((23, 98, 4), (27, 102, 0), (28, 103, 4)):
        require(frozen.encoded_hash(batch.observations[index].numpy()) ==
                rows[step - 1]["encoded_observation_sha256"], "Target state is not the demonstrated state.")
        require(int(batch.actions[index]) == action == rows[step - 1]["action"], "Wrong target label.")
    return source, batch, recovery, metadata


def probes(agent, batch):
    result = []
    for index in (97, 98, 99, 101, 102, 103, 104):
        with torch.no_grad():
            q = agent.learner.online(batch.observations[index:index + 1])[0]
        label = int(batch.actions[index])
        other = q.clone()
        other[label] = -torch.inf
        result.append({"row": index, "scene": 10002, "step": index - 75,
                       "label": label, "greedy_action": int(q.argmax()),
                       "reference_margin": float(q[label] - other.max()), "q_values": q.tolist()})
    return result


def run():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    save("status.json", {"status": "running", "started_at_utc": started})
    try:
        torch.set_num_threads(1)
        source, batch, recovery, metadata = inputs()
        config = source["environment_config"]
        paths = list((ROOT / "src/me5418_harvesting").rglob("*.py"))
        paths += list((ROOT / "docs/experiments").rglob("*.py"))
        paths += [prior.SOURCE, prior.MODEL, stages.SOURCE,
                  stages.DATA / "expanded_demonstrations.npz", stages.DATA / "dataset.json",
                  ROOT / "docs/experiments/027_step6_supervision/evidence/recovery_sample.json",
                  HERE.parent / "034_scene10002_supervision.md", ROOT / "tests/test_scene10002_supervision.py"]
        hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}
        save("provenance.json", {"git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(), "source_hashes": hashes,
            "started_at_utc": started, "torch": torch.__version__, "torch_threads": 1,
            "environment": config, "dqn": source["dqn_config"], "initial_adam": source["optimizer_summary"],
            "training_seeds": metadata["seeds"], "evaluation_seeds": SEEDS,
            "updates_per_arm": UPDATES, "checkpoints": CHECKS, "focus_rows": [98, 102, 103],
            "labels": [4, 0, 4], "weights": {"control": 1, "targeted": 2}, "action_overrides": 0})
        evaluations, fits, audits, reloads, model_hashes = [], [], [], [], {}
        for arm, weight in (("control", 1.0), ("targeted", 2.0)):
            core.FOCUS_WEIGHT = weight
            directory = OUTPUT / arm
            directory.mkdir()
            agent, env = stages.restore(source)
            equal(stages.snapshot(agent, config), source, "Exact common starting state")
            with (directory / "updates.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
                for step in range(UPDATES + 1):
                    if step:
                        stats = core.recovery_td_update(agent.learner, batch, recovery)
                        handle.write(json.dumps({"arm": arm, "update": step, **stats}) + "\n")
                    if step in CHECKS or step == UPDATES - 10:
                        path = directory / f"state_{step}.pt"
                        torch.save(stages.snapshot(agent, config), path)
                        model_hashes[f"{arm}/{path.name}"] = digest(path)
                        save("model_hashes.json", model_hashes)
                    if step in CHECKS:
                        fits.append({"arm": arm, "update": step, "probes": probes(agent, batch)})
                        save("probes.json", fits)
                        for seed in SEEDS:
                            row = frozen.rollout(agent, env, seed, OUTPUT / f"{arm}_{step}_{seed}.jsonl")
                            evaluations.append({"arm": arm, "update": step, **row})
                            save("evaluations.json", evaluations)
                            print(f"{arm} update {step}: scene {seed}: {row['category']}.", flush=True)
                    if step % 100 == 0:
                        handle.flush()
                        save("progress.json", {"arm": arm, "update": step, "budget": UPDATES})
            agent.save(directory / "final.pt", config, {"attempt": 34, "arm": arm,
                "focus_weight": weight, "additional_updates": UPDATES, "training_seeds": metadata["seeds"],
                "selection": "Predeclared final endpoint, no holdout selection."})
            env.close()
            logs = [json.loads(line) for line in (directory / "updates.jsonl").read_text().splitlines()]
            require(len(logs) == UPDATES and all(x["parameter_delta_norm"] > 0 for x in logs),
                    "Missing or zero training update.")
            for start in (0, UPDATES - 10):
                agent, env = stages.restore(torch.load(directory / f"state_{start}.pt", weights_only=True))
                for step in range(start + 1, start + 11):
                    stats = core.independent_update(agent.learner, batch, recovery)
                    equal({"arm": arm, "update": step, **stats}, logs[step - 1], "Independent update metrics")
                equal(stages.snapshot(agent, config),
                      torch.load(directory / f"state_{start + 10}.pt", weights_only=True), "Independent full state")
                env.close()
            final = torch.load(directory / f"state_{UPDATES}.pt", weights_only=True)
            require(final["update_count"] == 12000 and
                    final["optimizer_summary"]["step_min"] == final["optimizer_summary"]["step_max"] == 32000,
                    "Incorrect retained optimizer/TD clocks.")
            equal(final["online"], final["target"], "Final synchronized target")
            equal(final["dqn_config"], source["dqn_config"], "Unchanged hyperparameters")
            audits.append({"arm": arm, "passed": True, "independent_updates": 20})
            save("update_verification.json", audits)
            for seed in SEEDS:
                agent, env, _ = DQNAgent.load(directory / "final.pt")
                before, cert = frozen.state(agent), []
                with frozen.certification_checks(cert):
                    metrics = evaluate(agent, env, 1, seed)["records"][0]
                expected = next(r for r in evaluations if r["arm"] == arm and r["update"] == UPDATES and r["seed"] == seed)
                equal(metrics, expected["metrics"], "Normal evaluator endpoint reload")
                equal(cert, expected["certification"], "Reload certification")
                require(frozen.equal(before, frozen.state(agent)), "Evaluation changed model state.")
                reloads.append({"arm": arm, "metrics": metrics, "certification": cert})
                save("reload_verification.json", reloads)
                env.close()
        for name, sha in hashes.items():
            require(digest(ROOT / name) == sha, "Source/input changed.")
        save("summary.json", {"main_optimizer_updates": 2 * UPDATES, "independent_optimizer_updates": 40,
            "physical_episodes": len(evaluations), "reload_episodes": len(reloads), "action_overrides": 0,
            "episode_actions": sum(r["metrics"]["steps"] for r in evaluations) + sum(r["metrics"]["steps"] for r in reloads),
            "certification_actions": sum(c["steps"] for r in evaluations + reloads for c in r["certification"]),
            "final_metrics": {arm: [r["metrics"] for r in evaluations if r["arm"] == arm and r["update"] == UPDATES]
                              for arm in ("control", "targeted")}, "verification_passed": True,
            "source_unchanged": True, "scope": "Known-scene acquisition and sampled retention; no holdout test."})
        save("status.json", {"status": "completed", "started_at_utc": started,
                             "finished_at_utc": datetime.now(timezone.utc).isoformat()})
    except BaseException as exc:
        save("status.json", {"status": "failed", "started_at_utc": started,
                             "error": str(exc), "exception": type(exc).__name__})
        raise
    archive = HERE / "evidence"
    archive.mkdir()
    files = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.suffix in (".json", ".jsonl"):
            relative = path.relative_to(OUTPUT)
            destination = archive / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            require(path.read_bytes() == destination.read_bytes(), "Archive bytes changed.")
            files[relative.as_posix()] = {"sha256": digest(path), "bytes": path.stat().st_size}
    frozen.save(archive / "archive_manifest.json", {"source_run": OUTPUT.relative_to(ROOT).as_posix(), "files": files})


if __name__ == "__main__":
    run()
