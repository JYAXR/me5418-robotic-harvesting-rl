"""Run fixed supervised, actual-return and Adam-preserving TD stages on 33 scenes."""

import importlib.util
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from me5418_harvesting.agents.dqn_agent import DQNAgent
from me5418_harvesting.agents.dqn_learning import TransitionBatch
from me5418_harvesting.agents.multiscene_demonstrations import (
    build_multiscene_batch,
    slice_transition_batch,
)
from me5418_harvesting.agents.optimizer_transition import begin_td_with_optimizer
from me5418_harvesting.agents.training import evaluate

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).parent
OUTPUT = ROOT / "results/training/expanded_stages_030_20261115"
SOURCE = HERE / "inputs/source_state.pt"
SOURCE_SHA = "8890fea9630badb438c0e8a380f64c9f1dddfa7d5911361b0bdeb2835503b559"
DATA = ROOT / "docs/experiments/029_training_coverage/evidence"
BASELINE = ROOT / "models/coverage_029_expanded.pt"
OLD_SEEDS = [10000, 10001, 10002]
NEW_SEEDS = list(range(290000, 290030))
TEST_SEEDS = list(range(490000, 490100))
PHASES = {"supervised": 20000, "calibration": 20000, "td": 10000}
CHECKS = {
    "supervised": [0, 5000, 20000], "calibration": [0, 5000, 20000],
    "td": [0, 1, 10, 100, 500, 1000, 2000, 4000, 6000, 8000, 10000],
}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


prior = module("stages030prior", ROOT / "docs/experiments/029_training_coverage/run_coverage.py")
core = module("stages030core", HERE / "stage_update.py")
frozen, require, equal, digest = prior.frozen, prior.require, prior.equal, prior.digest
read, restore, snapshot = prior.read, prior.restore, prior.snapshot


def save(name, value):
    frozen.save(OUTPUT / name, value)


def inputs():
    require(digest(SOURCE) == SOURCE_SHA, "Wrong attempt-029 expanded source snapshot.")
    source = torch.load(SOURCE, weights_only=True)
    equal(source["online"], torch.load(BASELINE, weights_only=True)["online"], "Baseline source weights")
    require(source["update_count"] == 18000, "Unexpected source TD clock.")
    require(source["dqn_config"]["learning_rate"] == 0.0001, "Unexpected learning rate.")
    metadata = read(DATA / "dataset.json")
    require(metadata["seeds"] == OLD_SEEDS + NEW_SEEDS, "Training seed/order mismatch.")
    require(digest(DATA / "expanded_demonstrations.npz") == metadata["npz_sha256"], "Changed data.")
    with np.load(DATA / "expanded_demonstrations.npz") as arrays:
        batch = TransitionBatch(*(torch.from_numpy(arrays[k].copy())
                                  for k in TransitionBatch.__dataclass_fields__))
    require(len(batch.actions) == 1440, "Wrong expanded demonstration count.")
    trajectories = [slice_transition_batch(batch, e["start"], e["stop"]) for e in metadata["episodes"]]
    assembled, returns, episodes = build_multiscene_batch(
        trajectories, metadata["seeds"], source["dqn_config"]["gamma"])
    for key in TransitionBatch.__dataclass_fields__:
        equal(getattr(assembled, key), getattr(batch, key), "Unchanged training arrays")
    equal(episodes, metadata["episodes"], "Per-episode boundaries and recorded returns")
    require(frozen.encoded_hash(returns.numpy()) == metadata["return_targets_sha256"],
            "Actual returns disagree with the historical data manifest.")
    # Independent float64 sums verify each endpoint rather than a concatenated recurrence.
    independent = []
    for e in episodes:
        rewards = batch.rewards[e["start"]:e["stop"]].double().numpy()
        independent.extend(float(np.dot(rewards[i:], 0.99 ** np.arange(len(rewards) - i)))
                           for i in range(len(rewards)))
        require(returns[e["stop"] - 1] == batch.rewards[e["stop"] - 1],
                "Final reward must be included exactly once.")
    error = float(np.max(np.abs(returns.numpy() - np.asarray(independent))))
    require(error < 0.0002, "Float32 targets disagree with independent complete returns.")
    sample = read(ROOT / "docs/experiments/027_step6_supervision/evidence/recovery_sample.json")
    require(sample["action"] == 4, "Recovery action changed.")
    recovery = torch.tensor([sample["observation"]], dtype=torch.float32)
    groups = [set(OLD_SEEDS), set(NEW_SEEDS), set(TEST_SEEDS),
              set(range(280000, 280100)), set(range(390000, 390100))]
    require(all(not a & b for i, a in enumerate(groups) for b in groups[i + 1:]),
            "Training and historical/fresh test seeds overlap.")
    return source, batch, returns, recovery, metadata, error


def fit(agent, batch, returns, recovery):
    result = prior.metrics(agent, batch, recovery)
    with torch.no_grad():
        q = agent.learner.online(batch.observations)
        chosen = q.gather(1, batch.actions[:, None]).squeeze(1)
        result["return_mae"] = float((chosen - returns).abs().mean())
        result["return_rmse"] = float((chosen - returns).square().mean().sqrt())
        result["minimum_q"] = float(q.min())
        result["maximum_q"] = float(q.max())
        result["probes"] = []
        for index in [39, 40, 41, 54, 55, 56, 57, 58, 59]:
            action = int(batch.actions[index])
            competing = q[index].clone()
            competing[action] = -torch.inf
            result["probes"].append({"row": index, "scene": 10001, "step": index - 34,
                "demonstrated_action": action, "greedy_action": int(q[index].argmax()),
                "gap": float(q[index, action] - competing.max())})
    return result


def physical(agent, env, seed, arm, phase, step, records):
    row = frozen.rollout(agent, env, seed, OUTPUT / f"{arm}_{phase}_{step}_{seed}.jsonl")
    records.append({"arm": arm, "phase": phase, "update": step, **row})
    save("evaluations.json", records)
    return row


def transition(agent, phase, config):
    before = snapshot(agent, config)
    result = begin_td_with_optimizer(agent.learner, preserve_optimizer=phase == "td")
    after = snapshot(agent, config)
    for key in ("online", "dqn_config", "agent_rng_state", "torch_rng_state"):
        equal(before[key], after[key], "Boundary preserves weights, settings and RNG")
    equal(after["online"], after["target"], "Synchronized phase target")
    equal(before["optimizer_summary"]["parameter_groups"],
          after["optimizer_summary"]["parameter_groups"], "Unchanged Adam settings")
    if phase == "td":
        equal(before["optimizer"], after["optimizer"], "Exact retained Adam tensors and clocks")
        equal(result["optimizer_before"], result["optimizer_after"], "Exact named Adam hash")
        require(after["optimizer_summary"]["step_min"] == 20000, "Wrong calibration Adam clock.")
    else:
        require(after["optimizer_summary"]["initialized_parameter_count"] == 0,
                "Pre-TD stage must start with a fresh Adam state.")
    save(f"{phase}_boundary.json", result)
    return result


def update(agent, phase, batch, returns, recovery, independent=False):
    if phase == "td":
        function = core.coverage.independent_update if independent else core.coverage.recovery_td_update
        return function(agent.learner, batch, recovery)
    function = core.independent_update if independent else core.stage_update
    return function(agent.learner, batch, recovery, returns if phase == "calibration" else None)


def train_phase(agent, env, phase, config, batch, returns, recovery, evaluations):
    transition(agent, phase, config)
    directory = OUTPUT / phase
    directory.mkdir()
    fits, hashes = [], {}
    budget = PHASES[phase]
    with (directory / "updates.jsonl").open("x", encoding="utf-8") as handle:
        for step in range(budget + 1):
            if step:
                stats = update(agent, phase, batch, returns, recovery)
                handle.write(json.dumps({"phase": phase, "update": step, **stats}, allow_nan=False) + "\n")
            if step in CHECKS[phase]:
                measured = {"phase": phase, "update": step, **fit(agent, batch, returns, recovery)}
                fits.append(measured)
                save(f"{phase}_fit.json", fits)
                rows = [physical(agent, env, seed, "staged", phase, step, evaluations) for seed in OLD_SEEDS]
                successes = sum(x["category"] == "success" for x in rows)
                print(f"{phase} {step}/{budget}: original {successes}/3, "
                      f"labels {measured['original_correct']}/117 + {measured['new_correct']}/1323, "
                      f"return MAE {measured['return_mae']:.4f}.", flush=True)
            if step in CHECKS[phase] or step in (10, budget - 10):
                path = directory / f"state_{step}.pt"
                torch.save(snapshot(agent, config), path)
                hashes[path.name] = digest(path)
            if step % 100 == 0:
                handle.flush()
                save("progress.json", {"phase": phase, "update": step, "budget": budget})
                if step and step % 1000 == 0 and step not in CHECKS[phase]:
                    print(f"{phase} {step}/{budget}: loss {stats['loss_after']:.6f}.", flush=True)
    agent.save(directory / "final.pt", config, {
        "attempt": 30, "phase": phase, "training_seeds": OLD_SEEDS + NEW_SEEDS,
        "source_sha256": SOURCE_SHA, "phase_updates": budget,
        "phase_budgets": PHASES, "selection": "Fixed budget endpoint; no holdout model selection."})
    hashes["final.pt"] = digest(directory / "final.pt")
    save(f"{phase}_model_hashes.json", hashes)
    for seed in NEW_SEEDS:
        row = physical(agent, env, seed, "staged", phase + "_acquisition", budget, evaluations)
        print(f"{phase} training acquisition {seed}: {row['category']}.", flush=True)


def verify_phase(phase, config, batch, returns, recovery):
    budget = PHASES[phase]
    logs = [json.loads(line) for line in (OUTPUT / phase / "updates.jsonl").read_text().splitlines()]
    require(len(logs) == budget, "Wrong phase update count.")
    for start in (0, budget - 10):
        payload = torch.load(OUTPUT / phase / f"state_{start}.pt", weights_only=True)
        agent, env = restore(payload)
        try:
            for step in range(start + 1, start + 11):
                result = update(agent, phase, batch, returns, recovery, independent=True)
                equal({"phase": phase, "update": step, **result}, logs[step - 1],
                      "Independent first/last update metrics")
                path = OUTPUT / phase / f"state_{step}.pt"
                if path.exists():
                    equal(snapshot(agent, config), torch.load(path, weights_only=True),
                          "Independent complete training state")
        finally:
            env.close()
    final = torch.load(OUTPUT / phase / f"state_{budget}.pt", weights_only=True)
    start = torch.load(OUTPUT / phase / "state_0.pt", weights_only=True)
    expected_clock = budget + (20000 if phase == "td" else 0)
    require(final["optimizer_summary"]["step_min"] == final["optimizer_summary"]["step_max"]
            == expected_clock, "Incorrect Adam clock.")
    require(sum(x["target_synchronized"] for x in logs) == (50 if phase == "td" else 0),
            "Incorrect target synchronization count.")
    equal(final["target"], final["online"] if phase == "td" else start["target"],
          "Expected phase target state")
    equal(final["dqn_config"], start["dqn_config"], "Unchanged network and hyperparameters")
    return {"phase": phase, "passed": True, "independent_optimizer_updates": 20,
            "final_adam_clock": expected_clock, "target_syncs": 50 if phase == "td" else 0}


def frozen_model(arm):
    path = BASELINE if arm == "baseline" else OUTPUT / "td/final.pt"
    agent, env, _ = DQNAgent.load(path)
    for net in (agent.learner.online, agent.learner.target):
        net.eval()
        net.requires_grad_(False)
    def forbidden(*args, **kwargs):
        raise RuntimeError("Optimizer steps are forbidden during frozen evaluation.")
    agent.learner.optimizer.step = forbidden
    return agent, env


def previous_test_hashes():
    old = read(ROOT / "docs/experiments/028_frozen_generalization/evidence/episodes.json")
    old += [x for x in read(DATA / "evaluations.json") if x["phase"] == "holdout"]
    return {x["initial_encoded_sha256"] for x in old if x["status"] == "evaluated"}


def test_final(source, metadata, evaluations):
    model_hashes = {"baseline": digest(BASELINE), "staged": digest(OUTPUT / "td/final.pt")}
    save("frozen_test_boundary.json", {"model_hashes": model_hashes,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "holdout_episodes_started": 0})
    excluded = set(metadata["initial_encoded_hashes"]) | previous_test_hashes()
    reloads = []
    for arm in ("baseline", "staged"):
        agent, env = frozen_model(arm)
        main, seen = [], set()
        try:
            equal(agent.learner.config.__dict__, source["dqn_config"], "Frozen inference configuration")
            if arm == "baseline":
                for seed in OLD_SEEDS:
                    row = physical(agent, env, seed, arm, "original", 0, evaluations)
                    old = next(x for x in read(DATA / "evaluations.json") if x["arm"] == "expanded"
                               and x["phase"] == "retention" and x["relative_update"] == 8000 and x["seed"] == seed)
                    equal(row["metrics"], old["metrics"], "Reproduced source policy outcomes")
            for seed in TEST_SEEDS:
                row = physical(agent, env, seed, arm, "holdout", PHASES["td"] if arm == "staged" else 0, evaluations)
                main.append(row)
                if row["status"] == "evaluated":
                    key = row["initial_encoded_sha256"]
                    require(key not in excluded and key not in seen, "Test scene duplicates observed data.")
                    seen.add(key)
                summary = frozen.summary(main)
                save(f"{arm}_holdout_summary.json", summary)
                save("progress.json", {"phase": "holdout", "arm": arm, "completed": len(main),
                                      "successes": summary["successes"]})
                print(f"Holdout {arm} {len(main)}/100: {seed} {row['category']}, "
                      f"successes {summary['successes']}.", flush=True)
        finally:
            env.close()
        valid = [row for row in main if row["status"] == "evaluated"]
        selected = set(OLD_SEEDS)
        for category in {row["category"] for row in valid}:
            selected.add(min(row["seed"] for row in valid if row["category"] == category))
        for seed in sorted(selected):
            agent, env = frozen_model(arm)
            before, cert = frozen.state(agent), []
            try:
                with frozen.certification_checks(cert):
                    actual = evaluate(agent, env, 1, seed)["records"][0]
                phase = ("td" if arm == "staged" else "original") if seed in OLD_SEEDS else "holdout"
                step = PHASES["td"] if arm == "staged" else 0
                expected = next(x for x in evaluations if x["arm"] == arm and x["phase"] == phase
                                and x["update"] == step and x["seed"] == seed)
                equal(actual, expected["metrics"], "Normal evaluator reload metrics")
                equal(cert, expected["certification"], "Normal evaluator reset certification")
                require(frozen.equal(before, frozen.state(agent)), "Frozen reload changed learner state.")
                reloads.append({"arm": arm, "seed": seed, "metrics": actual, "certification": cert})
                save("reload_verification.json", reloads)
                print(f"Verified {arm} reload {seed}.", flush=True)
            finally:
                env.close()
    require(digest(BASELINE) == model_hashes["baseline"]
            and digest(OUTPUT / "td/final.pt") == model_hashes["staged"], "Frozen policy changed.")


def run():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    save("status.json", {"status": "running", "started_at_utc": started})
    torch.set_num_threads(1)
    source, batch, returns, recovery, metadata, error = inputs()
    paths = list((ROOT / "src/me5418_harvesting").rglob("*.py"))
    paths += list((ROOT / "docs/experiments").rglob("*.py"))
    paths += [SOURCE, BASELINE, DATA / "dataset.json", DATA / "expanded_demonstrations.npz",
              ROOT / "docs/experiments/027_step6_supervision/evidence/recovery_sample.json",
              ROOT / "docs/experiments/030_expanded_stages.md", ROOT / "tests/test_expanded_stages.py"]
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}
    save("provenance.json", {"source_hashes": hashes,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "started_at_utc": started, "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "environment": source["environment_config"], "dqn": source["dqn_config"],
        "source_adam": source["optimizer_summary"], "training_seeds": metadata["seeds"],
        "test_seeds": TEST_SEEDS, "phase_budgets": PHASES, "checkpoints": CHECKS,
        "independent_return_max_error": error, "new_collection_steps": 0})
    np.save(OUTPUT / "actual_returns.npy", returns.numpy())
    save("dataset.json", {**metadata, "returns_used_for_training": True,
        "return_array_sha256": digest(OUTPUT / "actual_returns.npy"),
        "independent_float64_max_error": error, "collection_reused_from_attempt": 29})
    evaluations, checks = [], []
    agent, env = restore(source)
    try:
        equal(snapshot(agent, source["environment_config"]), source, "Exact source restoration")
        for phase in PHASES:
            train_phase(agent, env, phase, source["environment_config"], batch, returns, recovery, evaluations)
            # Preserve the training RNG around disposable independent replay construction.
            state = snapshot(agent, source["environment_config"])
            checks.append(verify_phase(phase, source["environment_config"], batch, returns, recovery))
            torch.set_rng_state(state["torch_rng_state"])
            equal(snapshot(agent, source["environment_config"]), state, "Independent replay preserves main state")
            save("training_verification.json", checks)
    finally:
        env.close()
    test_final(source, metadata, evaluations)
    for name, sha in hashes.items():
        require(digest(ROOT / name) == sha, f"Bound source changed: {name}")
    save("status.json", {"status": "completed", "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_optimizer_updates": sum(PHASES.values()), "independent_optimizer_updates": 60,
        "sources_and_frozen_policies_unchanged": True})


if __name__ == "__main__":
    require(not OUTPUT.exists(), "Refusing to overwrite attempt 030.")
    try:
        run()
    except BaseException as exc:
        if OUTPUT.exists():
            save("failure.json", {"type": type(exc).__name__, "message": str(exc)})
            save("status.json", {"status": "failed", "finished_at_utc": datetime.now(timezone.utc).isoformat()})
        raise
