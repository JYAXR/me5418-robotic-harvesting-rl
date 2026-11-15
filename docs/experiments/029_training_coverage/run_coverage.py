"""Compare unchanged TD continuation with three versus thirty-three training scenes."""

import importlib.util
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from me5418_harvesting.agents.demonstrations import seed_replay_from_reference
from me5418_harvesting.agents.dqn_learning import TransitionBatch
from me5418_harvesting.agents.multiscene_demonstrations import (
    build_multiscene_batch,
    slice_transition_batch,
)
from me5418_harvesting.agents.replay_buffer import ReplayBuffer
from me5418_harvesting.agents.training import evaluate

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "results/training/coverage_029_20261115"
SOURCE = ROOT / "results/training/step6_supervision_20261115/models/training_state_10000.pt"
ORIGINAL = ROOT / "results/training/multiscene_retention_20261112/demonstration.npz"
OLD_SEEDS = [10000, 10001, 10002]
NEW_SEEDS = list(range(290000, 290030))
TEST_SEEDS = list(range(390000, 390100))
CHECKPOINTS = [0, 1, 10, 100, 500, 1000, 2000, 4000, 6000, 8000]
UPDATES = 8000


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


core = module("coverage029core", Path(__file__).with_name("coverage_update.py"))
legacy = module("coverage029legacy", ROOT / "docs/experiments/027_step6_supervision/run_step6_supervision.py")
frozen = module("coverage029eval", ROOT / "docs/experiments/028_frozen_generalization/run_frozen_generalization.py")
require, equal, digest = frozen.require, legacy.equal, frozen.digest
read = lambda path: json.loads(path.read_text(encoding="utf-8"))


def save(name, value):
    frozen.save(OUTPUT / name, value)


def restore(source):
    return legacy.v.prior.independent_restore(source)


def snapshot(agent, config):
    return legacy.b.snapshot(agent, config)


def replay_batch(replay):
    cols = list(zip(*replay.records))
    return TransitionBatch(torch.from_numpy(np.stack(cols[0])), torch.tensor(cols[1], dtype=torch.long),
        torch.tensor(cols[2], dtype=torch.float32), torch.from_numpy(np.stack(cols[3])),
        torch.tensor(cols[4], dtype=torch.bool), torch.tensor(cols[5], dtype=torch.bool))


def source_inputs():
    manifest = read(ROOT / "docs/experiments/027_step6_supervision/evidence/model_hashes.json")
    require(digest(SOURCE) == manifest[SOURCE.name], "Source snapshot hash mismatch.")
    dataset = read(ROOT / "docs/experiments/011_multiscene_retention/evidence/dataset.json")
    require(digest(ORIGINAL) == dataset["sha256"], "Original demonstration hash mismatch.")
    with np.load(ORIGINAL) as data:
        batch = TransitionBatch(*(torch.from_numpy(data[k].copy()) for k in TransitionBatch.__dataclass_fields__))
    source = torch.load(SOURCE, weights_only=True)
    inference = torch.load(ROOT / "models/step6_repair_027.pt", weights_only=True)
    equal(source["online"], inference["online"], "Frozen candidate equals continuation source")
    require(source["update_count"] == 10000 and len(batch.actions) == 117, "Wrong continuation boundary.")
    sample = read(ROOT / "docs/experiments/027_step6_supervision/evidence/recovery_sample.json")
    recovery = torch.tensor([sample["observation"]], dtype=torch.float32)
    require(sample["action"] == 4, "Recovery label changed.")
    return source, batch, recovery, dataset


def collect(source, original, dataset):
    agent, env = restore(source)
    config = source["environment_config"]
    trajectories = [slice_transition_batch(original, e["start"], e["stop"]) for e in dataset["episodes"]]
    records = []
    before = snapshot(agent, config)
    try:
        for seed in NEW_SEEDS:
            replay = ReplayBuffer(200)
            checks = []
            with frozen.certification_checks(checks):
                result = seed_replay_from_reference(config, agent.encoder, replay, 1, seed)
            records.append({"seed": seed, "collection": result, "certification": checks})
            save("collection.json", records)
            require(result["successful_episodes"] == 1, "A predeclared training scene failed collection; no substitution.")
            trajectories.append(replay_batch(replay))
            print(f"Collected training seed {seed}: {len(replay)} transitions.", flush=True)
        equal(before, snapshot(agent, config), "Collection preserves inherited learner")
    finally:
        env.close()
    combined, returns, episodes = build_multiscene_batch(trajectories, OLD_SEEDS + NEW_SEEDS, source["dqn_config"]["gamma"])
    for key in TransitionBatch.__dataclass_fields__:
        require(torch.equal(getattr(combined, key)[:117], getattr(original, key)), "Original rows changed.")
    np.savez_compressed(OUTPUT / "expanded_demonstrations.npz", **{k:getattr(combined, k).numpy() for k in TransitionBatch.__dataclass_fields__})
    save("dataset.json", {"seeds": OLD_SEEDS + NEW_SEEDS, "episodes": episodes,
        "rows": len(combined.actions), "old_rows": 117, "old_row_fraction": 117 / len(combined.actions),
        "full_data_mean": True, "recovery_denominator": 117, "returns_used_for_training": False,
        "initial_encoded_hashes": [frozen.encoded_hash(combined.observations[e["start"]].numpy()) for e in episodes],
        "npz_sha256": digest(OUTPUT / "expanded_demonstrations.npz"), "return_targets_sha256": frozen.encoded_hash(returns.numpy())})
    return combined


def metrics(agent, batch, recovery):
    with torch.no_grad():
        values = agent.learner.online(batch.observations)
        chosen = values.argmax(1)
        correct = chosen == batch.actions
        q = agent.learner.online(recovery)[0]
    return {"rows": len(correct), "correct": int(correct.sum()), "original_correct": int(correct[:117].sum()),
        "new_correct": int(correct[117:].sum()), "recovery_action": int(q.argmax()),
        "recovery_gap": float(q[4] - torch.cat((q[:4], q[5:])).max())}


def physical(agent, env, seed, arm, phase, step, records):
    row = frozen.rollout(agent, env, seed, OUTPUT / f"{arm}_{phase}_{step}_{seed}.jsonl")
    records.append({"arm": arm, "phase": phase, "relative_update": step, **row})
    save("evaluations.json", records)
    return row


def gates(source, original, expanded, recovery):
    agents = [restore(source) for _ in range(4)]
    try:
        old = legacy.core.recovery_td_update(agents[0][0].learner, original, recovery)
        new = core.recovery_td_update(agents[1][0].learner, original, recovery)
        equal(old, new, "Exact N=117 old update")
        equal(snapshot(agents[0][0], source["environment_config"]), snapshot(agents[1][0], source["environment_config"]), "Exact control state")
        new = core.recovery_td_update(agents[2][0].learner, expanded, recovery)
        independent = core.independent_update(agents[3][0].learner, expanded, recovery)
        equal(new, independent, "Independent expanded update")
        equal(snapshot(agents[2][0], source["environment_config"]), snapshot(agents[3][0], source["environment_config"]), "Independent expanded state")
        save("gates.json", {"passed": True, "optimizer_updates": 4, "control": old, "expanded": new})
    finally:
        for _, env in agents:
            env.close()


def train_arm(arm, source, batch, expanded, recovery, evaluations):
    agent, env = restore(source)
    config = source["environment_config"]
    directory = OUTPUT / arm
    directory.mkdir()
    fit, logs, hashes = [], [], {}
    try:
        equal(snapshot(agent, config), source, "Exact common continuation state")
        with (directory / "updates.jsonl").open("x", encoding="utf-8") as log:
            for step in range(UPDATES + 1):
                if step:
                    stats = core.recovery_td_update(agent.learner, batch, recovery)
                    row = {"relative_update": step, "absolute_update": 10000 + step, **stats}
                    logs.append(row)
                    log.write(json.dumps(row) + "\n")
                if step in CHECKPOINTS:
                    fit.append({"arm": arm, "relative_update": step, **metrics(agent, expanded, recovery)})
                    save(f"{arm}_fit.json", fit)
                    rows = [physical(agent, env, seed, arm, "retention", step, evaluations) for seed in OLD_SEEDS]
                    successes = sum(row["category"] == "success" for row in rows)
                    print(f"{arm} update {step}: original success {successes}/3, original fit {fit[-1]['original_correct']}/117.", flush=True)
                if step in CHECKPOINTS or step == UPDATES - 10:
                    path = directory / f"state_{step}.pt"
                    torch.save(snapshot(agent, config), path)
                    hashes[path.name] = digest(path)
                if step % 100 == 0:
                    log.flush()
                    save("progress.json", {"phase": "training", "arm": arm, "relative_update": step})
            agent.save(directory / "final.pt", config, {"attempt": 29, "arm": arm,
                "training_seeds": OLD_SEEDS if arm == "control" else OLD_SEEDS + NEW_SEEDS,
                "absolute_td_update": 18000, "prior_updates": 50000, "additional_updates": UPDATES,
                "selection": "Predeclared final model, not selected by evaluation score."})
            hashes["final.pt"] = digest(directory / "final.pt")
        save(f"{arm}_model_hashes.json", hashes)
    finally:
        env.close()
    return logs


def verify_updates(arm, source, batch, recovery, logs):
    comparisons = 0
    for start in (0, UPDATES - 10):
        payload = source if start == 0 else torch.load(OUTPUT / arm / f"state_{start}.pt", weights_only=True)
        agent, env = restore(payload)
        try:
            for step in range(start + 1, start + 11):
                stats = core.independent_update(agent.learner, batch, recovery)
                equal({"relative_update": step, "absolute_update": 10000 + step, **stats}, logs[step - 1], "Independent update metrics")
                path = OUTPUT / arm / f"state_{step}.pt"
                if path.exists():
                    equal(snapshot(agent, source["environment_config"]), torch.load(path, weights_only=True), "Independent full snapshot")
                comparisons += 1
        finally:
            env.close()
    require(sum(row["target_synchronized"] for row in logs) == 40, "Target synchronization count changed.")
    final = torch.load(OUTPUT / arm / "state_8000.pt", weights_only=True)
    require(final["optimizer_summary"]["step_min"] == final["optimizer_summary"]["step_max"] == 38000, "Adam clock changed.")
    equal(final["optimizer_summary"]["parameter_groups"], source["optimizer_summary"]["parameter_groups"], "Unchanged Adam settings")
    equal(final["dqn_config"], source["dqn_config"], "Unchanged DQN configuration")
    equal(final["target"], final["online"], "Final target synchronization")
    return {"arm": arm, "passed": True, "independent_optimizer_updates": comparisons}


def frozen_final(arm, config):
    from me5418_harvesting.agents.dqn_agent import DQNAgent
    path = OUTPUT / arm / "final.pt"
    agent, env, _ = DQNAgent.load(path)
    full = torch.load(OUTPUT / arm / "state_8000.pt", weights_only=True)
    equal(agent.learner.online.state_dict(), full["online"], "Inference matches final trained state")
    equal(torch.load(path, weights_only=True)["environment_config"], config, "Unchanged inference environment")
    agent.learner.online.requires_grad_(False)
    agent.learner.target.requires_grad_(False)
    def forbidden(*args, **kwargs):
        raise RuntimeError("Evaluation cannot update the optimizer.")
    agent.learner.optimizer.step = forbidden
    return agent, env


def run():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    save("status.json", {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()})
    torch.set_num_threads(1)
    require(not set(OLD_SEEDS + NEW_SEEDS) & set(TEST_SEEDS) and not set(NEW_SEEDS + TEST_SEEDS) & set(range(280000, 280100)), "Seed sets overlap.")
    source, original, recovery, dataset = source_inputs()
    paths = list((ROOT / "src/me5418_harvesting").rglob("*.py")) + list((ROOT / "docs/experiments").rglob("*.py"))
    paths += [SOURCE, ORIGINAL, ROOT / "models/step6_repair_027.pt", ROOT / "docs/experiments/029_training_coverage.md", ROOT / "tests/test_training_coverage.py"]
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}
    save("provenance.json", {"source_hashes": hashes, "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "source_snapshot_sha256": digest(SOURCE), "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "environment": source["environment_config"], "dqn": source["dqn_config"], "source_adam": source["optimizer_summary"],
        "training_seeds_control": OLD_SEEDS, "training_seeds_expanded": OLD_SEEDS + NEW_SEEDS,
        "test_seeds": TEST_SEEDS, "updates_per_arm": UPDATES, "retention_checkpoints": CHECKPOINTS})
    expanded = collect(source, original, dataset)
    gates(source, original, expanded, recovery)
    evaluations, checks = [], []
    for arm, batch in (("control", original), ("expanded", expanded)):
        logs = train_arm(arm, source, batch, expanded, recovery, evaluations)
        checks.append(verify_updates(arm, source, batch, recovery, logs))
        save("training_verification.json", checks)
    final_hashes = {arm: digest(OUTPUT / arm / "final.pt") for arm in ("control", "expanded")}
    save("frozen_test_boundary.json", {"final_model_hashes": final_hashes, "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "test_episodes_started": 0})
    agent, env = frozen_final("expanded", source["environment_config"])
    try:
        for seed in NEW_SEEDS:
            row = physical(agent, env, seed, "expanded", "acquisition", UPDATES, evaluations)
            print(f"Training acquisition {seed}: {row['category']}", flush=True)
    finally:
        env.close()
    train_hashes = set(read(OUTPUT / "dataset.json")["initial_encoded_hashes"])
    verification = []
    for arm in ("control", "expanded"):
        require(digest(OUTPUT / arm / "final.pt") == final_hashes[arm], "Final model changed before testing.")
        agent, env = frozen_final(arm, source["environment_config"])
        main, seen = [], set()
        try:
            for seed in TEST_SEEDS:
                row = physical(agent, env, seed, arm, "holdout", UPDATES, evaluations)
                main.append(row)
                if row["status"] == "evaluated":
                    h = row["initial_encoded_sha256"]
                    require(h not in train_hashes and h not in seen, "Test scene duplicates training or another test scene.")
                    seen.add(h)
                summary = frozen.summary(main)
                save(f"{arm}_holdout_summary.json", summary)
                save("progress.json", {"phase": "holdout", "arm": arm, "completed": len(main), "successes": summary["successes"]})
                print(f"Holdout {arm} {len(main)}/100: seed {seed}, {row['category']}, successes {summary['successes']}.", flush=True)
        finally:
            env.close()
        valid = [row for row in main if row["status"] == "evaluated"]
        selected = set(OLD_SEEDS + [row["seed"] for row in valid[:5]])
        for category in {row["category"] for row in valid}:
            selected.add(min(row["seed"] for row in valid if row["category"] == category))
        for seed in sorted(selected):
            agent, env = frozen_final(arm, source["environment_config"])
            before, cert = frozen.state(agent), []
            try:
                with frozen.certification_checks(cert):
                    actual = evaluate(agent, env, 1, seed)["records"][0]
                expected = next(row for row in evaluations if row["arm"] == arm and row["relative_update"] == UPDATES
                    and row["phase"] == ("retention" if seed in OLD_SEEDS else "holdout") and row["seed"] == seed)
                equal(actual, expected["metrics"], "Exact normal evaluation reload")
                equal(cert, expected["certification"], "Exact reset certification")
                require(frozen.equal(before, frozen.state(agent)), "Frozen state changed on reload.")
                verification.append({"arm": arm, "seed": seed, "metrics": actual, "certification": cert})
                save("reload_verification.json", verification)
                print(f"Verified {arm} final reload seed {seed}.", flush=True)
            finally:
                env.close()
    for name, sha in hashes.items():
        require(digest(ROOT / name) == sha, f"Bound source changed: {name}")
    for arm, sha in final_hashes.items():
        require(digest(OUTPUT / arm / "final.pt") == sha, "Final policy changed after testing.")
    save("status.json", {"status": "completed", "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources_and_frozen_policies_unchanged": True, "training_optimizer_updates": 16000,
        "gate_optimizer_updates": 4, "independent_optimizer_updates": 40})


if __name__ == "__main__":
    require(not OUTPUT.exists(), "Refusing to overwrite attempt 029.")
    try:
        run()
    except BaseException as exc:
        if OUTPUT.exists():
            save("failure.json", {"type": type(exc).__name__, "message": str(exc)})
            save("status.json", {"status": "failed", "finished_at_utc": datetime.now(timezone.utc).isoformat()})
        raise
