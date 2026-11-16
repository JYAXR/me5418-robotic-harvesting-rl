"""Verify recorded interventions and describe the next reference disagreement."""

import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("own_rescue_review", HERE / "run_rescue.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def main():
    evidence = HERE / "evidence"
    read = runner.legacy.read
    require = runner.require
    require(read(evidence / "status.json")["status"] == "completed", "Incomplete run.")
    manifest = read(evidence / "manifest.json")
    for name, digest in manifest.items():
        require(runner.frozen.digest(evidence / name) == digest, "Archived file hash mismatch.")
    _, references, _, _, _ = runner.inputs()
    episodes = read(evidence / "episodes.json")
    review = []
    for record in episodes:
        seed, mode = record["seed"], record["mode"]
        path = evidence / str(seed) / record["trace"]
        require(runner.frozen.digest(path) == record["trace_sha256"], "Trace hash mismatch.")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        total, forced_steps = 0.0, []
        for index, row in enumerate(rows, start=1):
            require(row["step"] == index, "Wrong step order.")
            forced = mode == "corrected" and index == runner.CASES[seed][0]
            require(row["forced_action"] == forced and not row["scripted_reference"], "Unexpected intervention.")
            require(row["policy_action"] == int(np.argmax(row["q_values"])), "Non-greedy policy.")
            require(row["action"] == (runner.REPLACEMENT if forced else row["policy_action"]), "Wrong action.")
            if forced:
                forced_steps.append(index)
            total += row["reward"]
        metrics = record["metrics"]
        require(total == metrics["return"] and len(rows) == metrics["steps"], "Return/length mismatch.")
        require(sum(r["info"]["collision_blocked"] for r in rows) == metrics["blocked_steps"], "Blocking mismatch.")
        require(sum(np.linalg.norm(r["info"]["executed_motion"]) <= 1e-6 for r in rows)
                == metrics["stationary_steps"], "Stationary count mismatch.")
        require(not any(r["terminated"] or r["truncated"] for r in rows[:-1]), "Early terminal step.")
        require(rows[-1]["terminated"] or rows[-1]["truncated"], "Missing terminal step.")
        for key in ("success", "damage", "budget_exhausted"):
            require(rows[-1]["info"][key] == metrics[key], "Terminal outcome mismatch.")
        review.append({"seed": seed, "mode": mode, "forced_steps": forced_steps,
                       "metrics_verified": True,
                       "first_remaining_reference_disagreement": runner.legacy.first_fork(rows, references[seed])})
    result = {"manifest_files_verified": len(manifest), "primary_episodes_verified": len(review),
              "simulator_actions": 0, "optimizer_updates": 0,
              "review_script_sha256": runner.frozen.digest(Path(__file__)), "records": review,
              "scope": "Post-hoc trace description; later disagreements are not proven failure causes."}
    with (HERE / "review.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"verified_files": len(manifest), "corrected_next_disagreements": [
        {"seed": r["seed"], "fork": r["first_remaining_reference_disagreement"]}
        for r in review if r["mode"] == "corrected"]}, indent=2))


if __name__ == "__main__":
    main()
