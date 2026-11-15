"""Diagnose one frozen policy with exactly one predeclared action replacement."""

import importlib.util
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from me5418_harvesting.agents.training import episode_metrics, evaluate

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).parent
OUTPUT = ROOT / "results/evaluation/scene10002_rescue_031_20261116"
SOURCE = HERE / "inputs/final_td_state.pt"
SOURCE_SHA = "1a63fba8f5c12b8cf13129b6dc290264c709d639e9a2b435d502d87d31ae7394"
MODEL = ROOT / "models/stages_030_td.pt"
MODEL_SHA = "e236374791ce5b7bb59c00d6e99c3ab557821fc06ccddbcbed6686b638b67868"
DATA = ROOT / "docs/experiments/029_training_coverage/evidence"
HISTORY = ROOT / "docs/experiments/030_expanded_stages/evidence"
TRACE = HISTORY / "staged_td_10000_10002.jsonl"
TRACE_SHA = "b90608cd468550b8c7f4dd0d75b8f2332a123cdb2e40c6183615831c897e8290"
SEED, FORK_STEP, REPLACEMENT = 10002, 23, 4

spec = importlib.util.spec_from_file_location(
    "scene10002_stages", ROOT / "docs/experiments/030_expanded_stages/run_stages.py")
stages = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stages)
frozen, equal, require = stages.frozen, stages.equal, stages.require
read, digest = stages.read, stages.digest
NAMES = frozen.ACTION_NAMES


def save(name, value):
    frozen.save(OUTPUT / name, value)


def first_fork(rows, reference):
    """Stop at the first differing action; do not compare states after a fork."""
    for index, row in enumerate(rows):
        if index == len(reference["actions"]):
            return None
        require(row["step"] == index + 1, "Trace step ordering changed.")
        require(row["encoded_observation_sha256"] ==
                frozen.encoded_hash(reference["observations"][index]),
                "State differs before the first action disagreement.")
        action = int(reference["actions"][index])
        if row["action"] != action:
            q = row["q_values"]
            return {"step": index + 1, "policy_action": row["action"],
                    "policy_action_name": NAMES[row["action"]],
                    "reference_action": action, "reference_action_name": NAMES[action],
                    "policy_q": q[row["action"]], "reference_q": q[action],
                    "policy_minus_reference": q[row["action"]] - q[action],
                    "q_values": q, "encoded_observation_sha256":
                    row["encoded_observation_sha256"], "identical_prefix_steps": index}
    return None


def audit_pair(control, corrected):
    """Reject hidden interventions or changes to the common prefix/fork input."""
    require(len(control) >= FORK_STEP and len(corrected) >= FORK_STEP,
            "Both arms must reach the planned fork.")
    equal(control[:FORK_STEP - 1], corrected[:FORK_STEP - 1], "Identical prefix")
    for key in ("step", "position", "encoded_observation_sha256", "q_values",
                "policy_action", "q_top_gap"):
        equal(control[FORK_STEP - 1][key], corrected[FORK_STEP - 1][key],
              "Identical fork input and policy")
    for mode, rows in (("baseline", control), ("corrected", corrected)):
        for index, row in enumerate(rows):
            require(row["step"] == index + 1, "Trace step ordering changed.")
            require(not row["scripted_reference"], "Diagnostic arm used scripted reference actions.")
            forced = mode == "corrected" and row["step"] == FORK_STEP
            require(row["forced_action"] == forced, "Unexpected override count/location.")
            require(row["action"] == (REPLACEMENT if forced else row["policy_action"]),
                    "An action outside the predeclared intervention changed.")
            require(row["policy_action"] == int(np.argmax(row["q_values"])),
                    "Unforced policy decision is not greedy.")
    require(control[FORK_STEP - 1]["action"] != REPLACEMENT,
            "The planned correction must change the selected action.")
    return {"identical_prefix_steps": FORK_STEP - 1, "identical_fork_input_and_q": True,
            "baseline_overrides": 0, "corrected_overrides": 1,
            "all_other_actions_autonomous": True}


def inputs():
    require(digest(SOURCE) == SOURCE_SHA, "Full snapshot hash mismatch.")
    require(digest(MODEL) == MODEL_SHA, "Inference policy hash mismatch.")
    require(digest(TRACE) == TRACE_SHA, "Historical trace hash mismatch.")
    for name, sha in read(HISTORY / "provenance.json")["source_hashes"].items():
        if name.startswith("src/"):
            require(digest(ROOT / name) == sha, "Simulator/agent code changed since attempt 030.")
    source = torch.load(SOURCE, weights_only=True)
    inference = torch.load(MODEL, weights_only=True)
    for key in ("online", "environment_config", "dqn_config"):
        equal(source[key], inference[key], "Full snapshot matches inference policy")
    equal(source["online"], source["target"], "Final synchronized target")
    require(source["update_count"] == 10000, "Wrong final TD clock.")
    require(source["optimizer_summary"]["step_min"] == 30000 and
            source["optimizer_summary"]["step_max"] == 30000, "Wrong final Adam clock.")
    dataset = read(DATA / "dataset.json")
    require(digest(DATA / "expanded_demonstrations.npz") == dataset["npz_sha256"],
            "Reference data hash mismatch.")
    episode = next(e for e in dataset["episodes"] if e["seed"] == SEED)
    require((episode["start"], episode["stop"]) == (76, 117), "Reference slice changed.")
    with np.load(DATA / "expanded_demonstrations.npz") as arrays:
        reference = {k: arrays[k][episode["start"]:episode["stop"]].copy() for k in arrays}
    history = [json.loads(line) for line in TRACE.read_text().splitlines()]
    original = next(r for r in read(HISTORY / "evaluations.json")
                    if r["arm"] == "staged" and r["phase"] == "td" and
                    r["update"] == 10000 and r["seed"] == SEED)
    fork = first_fork(history, reference)
    require(fork["step"] == FORK_STEP and fork["reference_action"] == REPLACEMENT,
            "Historical first disagreement changed.")
    require(fork["policy_action"] == 0 and len(history) == 58 and
            original["metrics"]["damage"], "Historical failure changed.")
    return source, reference, history, original, fork


def load_frozen(source):
    agent, env = stages.restore(source)
    equal(stages.snapshot(agent, source["environment_config"]), source,
          "Exact restored full snapshot")
    for network in (agent.learner.online, agent.learner.target):
        network.eval()
        network.requires_grad_(False)

    def forbidden(*args, **kwargs):
        raise RuntimeError("Optimizer steps are forbidden during attempt 031.")

    agent.learner.optimizer.step = forbidden
    return agent, env


def finish_state(agent, source):
    equal(stages.snapshot(agent, source["environment_config"]), source,
          "Unchanged weights, target, Adam, clocks, configuration and RNG")
    require(all(not p.requires_grad for net in (agent.learner.online, agent.learner.target)
                for p in net.parameters()), "Gradients were enabled.")


def main_rollout(source, reference, mode):
    agent, env = load_frozen(source)
    records, checks = [], []
    try:
        with frozen.certification_checks(checks):
            obs, reset_info = env.reset(seed=SEED)
        require(checks == reset_info["reference_route_checks"], "Reset accounting mismatch.")
        initial = frozen.plain(frozen.geometry(env))
        total, stationary, blocked = 0.0, 0, 0
        path = OUTPUT / f"{mode}.jsonl"
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            for step in range(1, env.max_steps + 1):
                encoded = agent.encoder(obs)
                with torch.no_grad():
                    values = agent.learner.online(torch.from_numpy(encoded).unsqueeze(0))[0]
                policy = agent.act(obs)
                require(policy == int(values.argmax()), "Policy is not greedy.")
                forced = mode == "corrected" and step == FORK_STEP
                action = REPLACEMENT if forced else policy
                if mode == "reference":
                    require(step <= len(reference["actions"]), "Reference failed to terminate.")
                    action = int(reference["actions"][step - 1])
                    require(np.array_equal(encoded, reference["observations"][step - 1]),
                            "Stored reference observation changed.")
                position = env.scene.end_effector.copy()
                next_obs, reward, terminated, truncated, info = env.step(action)
                next_encoded = agent.encoder(next_obs)
                if mode == "reference":
                    require(np.array_equal(next_encoded, reference["next_observations"][step - 1]),
                            "Stored reference successor changed.")
                    require(np.float32(reward) == reference["rewards"][step - 1] and
                            terminated == reference["terminated"][step - 1] and
                            truncated == reference["truncated"][step - 1],
                            "Stored reference reward/end flags changed.")
                sorted_q = values.sort(descending=True).values
                row = frozen.plain({"step": step, "position": position,
                    "next_position": env.scene.end_effector.copy(),
                    "encoded_observation_sha256": frozen.encoded_hash(encoded),
                    "next_encoded_observation_sha256": frozen.encoded_hash(next_encoded),
                    "action": action, "action_name": NAMES[action], "policy_action": policy,
                    "forced_action": forced, "scripted_reference": mode == "reference",
                    "q_values": values.tolist(), "q_top_gap": float(sorted_q[0] - sorted_q[1]),
                    "reward": reward, "terminated": terminated, "truncated": truncated,
                    "info": info})
                handle.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
                handle.flush()
                records.append(row)
                total += reward
                stationary += int(np.linalg.norm(info["executed_motion"]) <= 1e-6)
                blocked += int(info["collision_blocked"])
                obs = next_obs
                if terminated or truncated:
                    break
            else:
                raise AssertionError("Task budget did not end the episode.")
        finish_state(agent, source)
        metrics = {"seed": SEED, **episode_metrics(total, step, info, stationary, blocked)}
        if mode == "reference":
            require(metrics["success"] and step == 41, "Reference positive control failed.")
        result = {"mode": mode, "metrics": metrics, "initial_geometry": initial,
                  "reset_info": reset_info, "certification": checks,
                  "forced_actions": sum(r["forced_action"] for r in records),
                  "scripted_actions": step if mode == "reference" else 0,
                  "frozen_full_state_verified": True,
                  "trace": path.name, "trace_sha256": digest(path)}
        save(f"{mode}.json", result)
        print(json.dumps({"mode": mode, **metrics}), flush=True)
        return records, result
    finally:
        env.close()


def independent_replay(source, expected, expected_result, corrected):
    """Audit normal evaluate(), installing the step observer only after reset."""
    agent, env = load_frozen(source)
    original_act, original_step = agent.act, env.step
    seen, checks, pending = [], [], {}

    def observed_step(action):
        index = len(seen)
        row = expected[index]
        require(action == row["action"], "Independent executed action differs.")
        next_obs, reward, terminated, truncated, info = original_step(action)
        actual = {**pending, "next_position": env.scene.end_effector.tolist(),
                  "next_encoded_observation_sha256": frozen.encoded_hash(agent.encoder(next_obs)),
                  "reward": reward, "terminated": terminated, "truncated": truncated,
                  "info": frozen.plain(info)}
        equal(actual, row, "Independent full transition")
        seen.append(actual)
        return next_obs, reward, terminated, truncated, info

    def audited_act(obs, epsilon=0.0):
        require(epsilon == 0.0, "Independent evaluator enabled exploration.")
        step = len(seen) + 1
        require(step <= len(expected), "Independent episode exceeds primary length.")
        if step == 1:
            equal(frozen.plain(frozen.geometry(env)), expected_result["initial_geometry"],
                  "Independent reset geometry")
            env.step = observed_step
        encoded = agent.encoder(obs)
        with torch.no_grad():
            values = agent.learner.online(torch.from_numpy(encoded).unsqueeze(0))[0]
        policy = original_act(obs, epsilon=0.0)
        forced = corrected and step == 23
        action = 4 if forced else policy
        sorted_q = torch.sort(values, descending=True).values
        pending.clear()
        pending.update({"step": step, "position": env.scene.end_effector.tolist(),
                        "encoded_observation_sha256": frozen.encoded_hash(encoded),
                        "q_values": values.tolist(), "q_top_gap": float(sorted_q[0] - sorted_q[1]),
                        "policy_action": policy, "action": action, "action_name": NAMES[action],
                        "forced_action": forced, "scripted_reference": False})
        return action

    agent.act = audited_act
    try:
        with frozen.certification_checks(checks):
            metrics = evaluate(agent, env, episodes=1, seed=SEED)["records"][0]
        require(len(seen) == len(expected), "Independent episode ended too soon.")
        equal(metrics, expected_result["metrics"], "Independent episode outcome")
        equal(checks, expected_result["certification"], "Independent certification")
        finish_state(agent, source)
        mode = "corrected" if corrected else "baseline"
        result = {"mode": mode, "metrics": metrics, "certification": checks,
                  "full_trace_matches": True, "frozen_full_state_verified": True,
                  "forced_actions": sum(r["forced_action"] for r in seen)}
        save(f"verification_{mode}.json", result)
        print(json.dumps({"verification": mode, "steps": len(seen),
                          "success": metrics["success"]}), flush=True)
        return result
    finally:
        del agent.act
        if "step" in env.__dict__:
            del env.step
        env.close()


def archive():
    target = HERE / "evidence"
    target.mkdir(exist_ok=False)
    files = {}
    for source in sorted(OUTPUT.iterdir()):
        destination = target / source.name
        shutil.copy2(source, destination)
        require(source.read_bytes() == destination.read_bytes(), "Archive bytes changed.")
        files[source.name] = {"sha256": digest(source), "bytes": source.stat().st_size}
    frozen.save(target / "archive_manifest.json",
                {"source_run": OUTPUT.relative_to(ROOT).as_posix(), "files": files})


def run():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    save("status.json", {"status": "running", "started_at_utc": started})
    try:
        torch.set_num_threads(1)
        source, reference, history, original, fork = inputs()
        paths = list((ROOT / "src/me5418_harvesting").rglob("*.py"))
        paths += list((ROOT / "docs/experiments").rglob("*.py"))
        paths += [SOURCE, MODEL, DATA / "dataset.json", DATA / "expanded_demonstrations.npz",
                  TRACE, HISTORY / "evaluations.json", HERE.parent / "031_scene10002_rescue.md",
                  ROOT / "tests/test_scene10002_rescue.py"]
        hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}
        save("provenance.json", {"git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
            "source_hashes": hashes, "python": platform.python_version(),
            "torch": torch.__version__, "numpy": np.__version__, "torch_threads": 1,
            "started_at_utc": started, "scene_seed": SEED, "fork": fork,
            "environment": source["environment_config"], "dqn": source["dqn_config"],
            "adam": source["optimizer_summary"], "td_clock": source["update_count"],
            "new_optimizer_updates": 0, "new_training_samples": 0,
            "reference_rows": [76, 117], "reference_source_attempt": 29})
        save("first_fork.json", fork)
        traces, results = {}, {}
        for mode in ("baseline", "corrected", "reference"):
            traces[mode], results[mode] = main_rollout(source, reference, mode)
            if mode == "baseline":
                require(len(traces[mode]) == len(history), "Historical episode length changed.")
                for actual, old in zip(traces[mode], history, strict=True):
                    equal({k: actual[k] for k in old}, old, "Exact historical transition")
                equal(results[mode]["metrics"], original["metrics"], "Exact historical outcome")
                equal(results[mode]["reset_info"], original["reset_info"], "Historical reset")
                equal(first_fork(traces[mode], reference), fork, "Historical first fork")
        pair = audit_pair(traces["baseline"], traces["corrected"])
        # This records any later disagreement without changing another action.
        next_fork = first_fork(traces["corrected"], reference)
        save("pair_audit.json", {**pair, "next_reference_disagreement": next_fork})
        verified = [independent_replay(source, traces[mode], results[mode], mode == "corrected")
                    for mode in ("baseline", "corrected")]
        for name, sha in hashes.items():
            require(digest(ROOT / name) == sha, "An input/source file changed during evaluation.")
        main_steps = sum(results[m]["metrics"]["steps"] for m in ("baseline", "corrected"))
        reference_steps = results["reference"]["metrics"]["steps"]
        verification_steps = sum(v["metrics"]["steps"] for v in verified)
        certification = {m: sum(c["steps"] for c in r["certification"])
                         for m, r in results.items()}
        verification_certification = sum(c["steps"] for v in verified for c in v["certification"])
        save("summary.json", {"first_fork": fork, "pair_audit": pair,
            "metrics": {m: r["metrics"] for m, r in results.items()},
            "correction_restores_success": results["corrected"]["metrics"]["success"],
            "next_reference_disagreement": next_fork, "optimizer_updates": 0,
            "main_diagnostic_episodes": 2, "reference_episodes": 1, "verification_episodes": 2,
            "main_executed_actions": main_steps, "scripted_reference_actions": reference_steps,
            "verification_executed_actions": verification_steps,
            "main_forced_actions": 1, "verification_forced_actions": 1,
            "certification_actions_by_main_mode": certification,
            "verification_certification_actions": verification_certification,
            "total_simulator_actions": main_steps + reference_steps + verification_steps +
            sum(certification.values()) + verification_certification,
            "all_full_states_unchanged": True, "all_inputs_unchanged": True,
            "independent_verification_passed": True,
            "scope": "One frozen model and known scene; assisted diagnostic, not learned repair."})
        save("status.json", {"status": "completed", "started_at_utc": started,
                             "finished_at_utc": datetime.now(timezone.utc).isoformat()})
    except BaseException as exc:
        save("status.json", {"status": "failed", "started_at_utc": started,
                             "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                             "exception_type": type(exc).__name__, "error": str(exc)})
        raise
    archive()


if __name__ == "__main__":
    run()
