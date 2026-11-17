"""Offline retained-action audit and a two-step synthetic process-pool smoke test."""

import gzip
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import run_study as study
import torch

from me5418_harvesting.agents.dqn import DQNConfig
from me5418_harvesting.agents.recovery_policy import RecoveryAgent
from me5418_harvesting.envs.harvesting_env import HarvestingEnv


def main():
    torch.set_num_threads(1)
    directory = study.HERE / "preparation"
    directory.mkdir(exist_ok=False)
    agent, env, _ = RecoveryAgent.from_source(study.c.SOURCE)
    history = study.HERE.parent / "004_verified_supervision/evidence"
    count, seeds = 0, []
    for record in study.c.read(history / "evaluations.json"):
        if record["arm"] != "targeted" or record["update"] != 2000 or not record["metrics"]["success"]:
            continue
        seeds.append(record["seed"])
        for line in gzip.decompress((history / record["trace"]).read_bytes()).decode().splitlines():
            row = json.loads(line)
            assert agent.act(np.asarray(row["raw_observation"], dtype=np.float32)) == row["action"]
            count += 1
    env.close()
    with np.load(study.c.BASE_DATA / "expanded_demonstrations.npz") as base:
        features = np.stack([agent.encoder.extend(x) for x in base["observations"]])
    assert features.shape == (1440, 381) and np.isfinite(features).all()
    toy_config = {"max_steps": 2}
    toy_env = HarvestingEnv(toy_config)
    toy = RecoveryAgent(toy_env, DQNConfig(hidden_sizes=(16, 16)), seed=0)
    toy.save(directory / "toy.pt", toy_config, {"purpose": "Synthetic infrastructure smoke test"})
    toy_env.close()
    with ProcessPoolExecutor(max_workers=1, initializer=study.worker_init) as pool:
        record = pool.submit(study.evaluation_task, directory / "toy.pt", 0, directory).result()
    assert record["metrics"]["steps"] == 2 and record["frozen_state_unchanged"]
    study.c.save(directory / "checks.json", {"retained_actions": count, "retained_seeds": seeds,
                 "base_feature_shape": list(features.shape), "worker_smoke": record,
                 "training_updates": 0, "experiment_seed_rollouts": 0})
    print(json.dumps({"retained_actions": count, "seeds": len(seeds), "process_smoke_steps": 2}))


if __name__ == "__main__":
    main()
