"""Export the final policy only after the declared repair and retention checks pass."""
import importlib.util
import json
import shutil
from copy import deepcopy
from pathlib import Path

import torch

from me5418_harvesting.agents.dqn_agent import DQNAgent
from me5418_harvesting.agents.training import evaluate

SPEC = importlib.util.spec_from_file_location("export027", Path(__file__).with_name("run_step6_supervision.py"))
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)


def main():
    torch.set_num_threads(1)
    r.require(r.read(r.OUTPUT / "status.json")["status"] == "completed", "completed training")
    r.require(r.read(r.OUTPUT / "verification.json")["passed"], "passed independent verification")
    rows = r.read(r.OUTPUT / "observations.json")
    tails = r.read(r.OUTPUT / "conditional.json")
    dense = r.read(r.OUTPUT / "dense.json")
    schedule = r.read(r.OUTPUT / "protocol.json")["checkpoints"]
    r.require([row["update"] for row in rows] == [row["update"] for row in tails] == schedule, "complete evaluation schedule")
    r.require([row["update"] for row in dense] == list(range(2000, 10001)), "complete dense update schedule")
    r.require(len(rows) == 105 and all(row["aggregate"]["success_count"] == 3 for row in rows), "all declared autonomous checks pass")
    r.require(all(row["rollout"]["success"] and row["recovery_action"] == 4 for row in tails if row["update"] > 2000), "retained recovery")
    r.require(all(next(p for p in row["reference"]["states"] if p["reference_step"] == 6)["greedy_action"] == 1
                  and row["recovery"]["greedy_action"] == 4 for row in dense[1:]), "no dense step-six or recovery ranking regressions")
    source = r.OUTPUT / "models/update_10000.pt"
    full_path = r.OUTPUT / "models/training_state_10000.pt"
    r.require(r.v.file_hash(full_path) == r.read(r.OUTPUT / "model_hashes.json")[full_path.name], "verified final full snapshot")
    full = torch.load(full_path, weights_only=True)
    target = r.ROOT / "models/step6_repair_027.pt"
    metadata_path = target.with_suffix(".json")
    r.require(not target.exists() and not metadata_path.exists(), "no overwritten policy export")
    agent, env, _ = DQNAgent.load(source)
    records, incomplete = [], []
    try:
        r.equal(agent.learner.online.state_dict(), full["online"], "inference export matches final trained weights")
        rng = deepcopy(agent.rng.bit_generator.state)
        torch_rng = torch.get_rng_state().clone()
        with r.v.capture_resets(env, 10000, records, incomplete):
            result = evaluate(agent, env, 3, 10000)
        r.require(result["success_rate"] == 1.0 and not incomplete and len(records) == 3, "normal inference entry point succeeds")
        for actual, scene in zip(result["records"], rows[-1]["scenes"], strict=True):
            r.require(all(value == scene["rollout"][key] for key, value in actual.items()), "exact normal evaluator episode metrics")
        r.equal(agent.learner.online.state_dict(), full["online"], "no inference weight update")
        r.require(agent.rng.bit_generator.state == rng and torch.equal(torch.get_rng_state(), torch_rng), "no inference RNG update")
    finally:
        env.close()
    target.parent.mkdir(exist_ok=True)
    shutil.copy2(source, target)
    r.require(source.read_bytes() == target.read_bytes(), "byte-identical final inference export")
    evidence = {"passed": True, "checkpoint": target.relative_to(r.ROOT).as_posix(), "sha256": r.v.file_hash(target),
        "source_checkpoint": str(source), "source_update": 10000, "optimizer_updates": 0,
        "evaluation": result, "reset_certification": records,
        "policy_actions": sum(row["steps"] for row in result["records"]),
        "certification_actions": sum(row["certification_environment_steps"] for row in records),
        "script_sha256": r.v.file_hash(Path(__file__)),
        "scope": "Verified on three known training scenes only. No held-out generalization claim. This is an inference checkpoint, not an Adam-resume snapshot."}
    metadata_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    r.save("delivery_verification.json", evidence)
    print(json.dumps({"checkpoint": str(target), "success_rate": result["success_rate"], "sha256": evidence["sha256"]}))


if __name__ == "__main__":
    main()
