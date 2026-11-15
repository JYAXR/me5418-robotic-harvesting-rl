"""Test one extra frozen action correction while holding the two previous corrections fixed."""

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
OUTPUT = ROOT / "results/evaluation/scene10002_step28_033_20261116"
PRIOR = ROOT / "docs/experiments/032_scene10002_step27"
HISTORY = PRIOR / "evidence"
TRACE = HISTORY / "corrected.jsonl"
TRACE_SHA = "a31adeba588d0f08bd7559ca0c4789ff1dccfcc067e32d06098b2b03e8647c22"
SEED, FORK_STEP, REPLACEMENT = 10002, 28, 4

spec = importlib.util.spec_from_file_location("step28_prior", PRIOR / "run_step27.py")
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
stages, frozen = prior.stages, prior.frozen
equal, require, read, digest = prior.equal, prior.require, prior.read, prior.digest
NAMES, SOURCE, MODEL, DATA = prior.NAMES, prior.SOURCE, prior.MODEL, prior.DATA
first_fork, finish_state = prior.first_fork, prior.finish_state


def save(name, value):
    frozen.save(OUTPUT / name, value)


def selected_action(policy, step, mode):
    """Keep the common intervention fixed and add exactly one treatment override."""
    require(mode in ("baseline", "corrected"), "Unknown experimental arm.")
    overrides = {23: 4, 27: 0}
    if mode == "corrected":
        overrides[FORK_STEP] = REPLACEMENT
    return overrides.get(step, policy), step in overrides


def audit_pair(control, corrected):
    require(len(control) >= FORK_STEP and len(corrected) >= FORK_STEP,
            "Both arms must reach the additional correction.")
    equal(control[:FORK_STEP - 1], corrected[:FORK_STEP - 1], "Identical prefix through step 27")
    for key in ("step", "position", "encoded_observation_sha256", "q_values",
                "policy_action", "q_top_gap"):
        equal(control[FORK_STEP - 1][key], corrected[FORK_STEP - 1][key],
              "Identical step-28 input and policy")
    counts = {}
    for mode, rows in (("baseline", control), ("corrected", corrected)):
        for index, row in enumerate(rows):
            require(row["step"] == index + 1, "Trace step ordering changed.")
            require(not row["scripted_reference"], "Unexpected scripted trajectory.")
            # Independently encode the predeclared intervention, rather than call the selector.
            forced = index in (22, 26) or (mode == "corrected" and index == 27)
            expected = 0 if index == 26 else (4 if forced else row["policy_action"])
            require(row["forced_action"] == forced, "Unexpected override location.")
            require(row["action"] == expected, "Unexpected executed action.")
            require(row["policy_action"] == int(np.argmax(row["q_values"])), "Policy is not greedy.")
        counts[mode] = sum(r["forced_action"] for r in rows)
    require(counts == {"baseline": 2, "corrected": 3}, "Wrong override totals.")
    require(control[22]["policy_action"] == 0 and control[26]["policy_action"] == 6 and control[27]["policy_action"] == 6,
            "The frozen policy no longer makes the predeclared choices.")
    return {"identical_prefix_steps": 27, "identical_fork_input_and_q": True,
            "fixed_corrections": [{"step": 23, "action": 4}, {"step": 27, "action": 0}],
            "additional_correction": {"step": 28, "action": 4},
            "override_counts": counts, "all_other_actions_autonomous": True}


def inputs():
    source, reference, _, _, _ = prior.inputs()
    require(digest(TRACE) == TRACE_SHA, "Attempt-032 control trace changed.")
    manifest = read(HISTORY / "archive_manifest.json")
    for name, record in manifest["files"].items():
        path = HISTORY / name
        require(digest(path) == record["sha256"] and path.stat().st_size == record["bytes"],
                "Attempt-032 evidence changed.")
    require(read(HISTORY / "status.json")["status"] == "completed", "Incomplete source run.")
    require(read(HISTORY / "summary.json")["independent_verification_passed"],
            "Source run lacks independent verification.")
    history = [json.loads(line) for line in TRACE.read_text().splitlines()]
    original = read(HISTORY / "corrected.json")
    fork = first_fork(history, reference)
    equal(fork, read(HISTORY / "pair_audit.json")["next_reference_disagreement"],
          "Exact source step-28 disagreement")
    require(fork["step"] == 28 and fork["policy_action"] == 6 and
            fork["reference_action"] == 4 and len(history) == 58,
            "Wrong source control trajectory.")
    require(original["forced_actions"] == 2 and original["metrics"]["damage"] and
            not original["metrics"]["success"], "Wrong control outcome.")
    return source, reference, history, original, fork


def load_frozen(source):
    agent, env = prior.load_frozen(source)

    def forbidden(*args, **kwargs):
        raise RuntimeError("Optimizer steps are forbidden during attempt 033.")

    agent.learner.optimizer.step = forbidden
    return agent, env


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
                action, forced = selected_action(policy, step, mode)
                position = env.scene.end_effector.copy()
                next_obs, reward, terminated, truncated, info = env.step(action)
                next_encoded = agent.encoder(next_obs)
                sorted_q = values.sort(descending=True).values
                row = frozen.plain({"step": step, "position": position,
                    "next_position": env.scene.end_effector.copy(),
                    "encoded_observation_sha256": frozen.encoded_hash(encoded),
                    "next_encoded_observation_sha256": frozen.encoded_hash(next_encoded),
                    "action": action, "action_name": NAMES[action], "policy_action": policy,
                    "forced_action": forced, "scripted_reference": False,
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
        result = {"mode": mode, "metrics": metrics, "initial_geometry": initial,
                  "reset_info": reset_info, "certification": checks,
                  "forced_actions": sum(r["forced_action"] for r in records),
                  "scripted_actions": 0,
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
        forced = step in (23, 27) or (corrected and step == 28)
        action = 0 if step == 27 else (4 if forced else policy)
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
        paths += list(HISTORY.iterdir())
        paths += [SOURCE, MODEL, DATA / "dataset.json", DATA / "expanded_demonstrations.npz",
                  HERE.parent / "033_scene10002_step28.md", ROOT / "tests/test_scene10002_step28.py"]
        hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}
        save("provenance.json", {"git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
            "source_hashes": hashes, "python": platform.python_version(),
            "torch": torch.__version__, "numpy": np.__version__, "torch_threads": 1,
            "started_at_utc": started, "scene_seed": SEED, "additional_fork": fork,
            "fixed_corrections": [{"step": 23, "action": 4}, {"step": 27, "action": 0}],
            "environment": source["environment_config"], "dqn": source["dqn_config"],
            "adam": source["optimizer_summary"], "td_clock": source["update_count"],
            "new_optimizer_updates": 0, "new_training_samples": 0,
            "reference_rows": [76, 117], "reference_source_attempt": 29})
        save("additional_fork.json", fork)
        traces, results = {}, {}
        for mode in ("baseline", "corrected"):
            traces[mode], results[mode] = main_rollout(source, reference, mode)
            if mode == "baseline":
                equal(traces[mode], history, "All 58 attempt-032 control transitions")
                for key in ("metrics", "initial_geometry", "reset_info", "certification"):
                    equal(results[mode][key], original[key], "Exact source control reproduction")
                equal(first_fork(traces[mode], reference), fork, "Exact source fork")
        pair = audit_pair(traces["baseline"], traces["corrected"])
        next_fork = first_fork(traces["corrected"], reference)
        save("pair_audit.json", {**pair, "next_reference_disagreement": next_fork})
        verified = [independent_replay(source, traces[mode], results[mode], mode == "corrected")
                    for mode in ("baseline", "corrected")]
        for name, sha in hashes.items():
            require(digest(ROOT / name) == sha, "An input/source file changed during evaluation.")
        main_steps = sum(r["metrics"]["steps"] for r in results.values())
        verification_steps = sum(v["metrics"]["steps"] for v in verified)
        certification = {m: sum(c["steps"] for c in r["certification"])
                         for m, r in results.items()}
        verification_certification = sum(c["steps"] for v in verified for c in v["certification"])
        save("summary.json", {"additional_fork": fork, "pair_audit": pair,
            "metrics": {m: r["metrics"] for m, r in results.items()},
            "additional_correction_restores_success": results["corrected"]["metrics"]["success"],
            "next_reference_disagreement": next_fork, "optimizer_updates": 0,
            "main_diagnostic_episodes": 2, "verification_episodes": 2,
            "main_executed_actions": main_steps, "verification_executed_actions": verification_steps,
            "main_forced_actions": sum(r["forced_actions"] for r in results.values()),
            "verification_forced_actions": sum(v["forced_actions"] for v in verified),
            "certification_actions_by_main_mode": certification,
            "verification_certification_actions": verification_certification,
            "total_simulator_actions": main_steps + verification_steps +
            sum(certification.values()) + verification_certification,
            "all_full_states_unchanged": True, "all_inputs_unchanged": True,
            "independent_verification_passed": True,
            "scope": "One known scene; step-28 contrast conditional on fixed steps 23 and 27; no learning."})
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
