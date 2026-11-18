"""Read-only, deterministic evaluation of the seven archived inference models."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

import torch
from coordinate_policy import CoordinateAgent


def evaluate(name, first_seed, episodes):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    models = json.loads((HERE / "models.json").read_text())
    entry = models[name]
    path = HERE / entry["file"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry["sha256"]:
        raise ValueError(f"Model hash mismatch: {name}")
    torch.set_num_threads(1)
    agent, env, _ = CoordinateAgent.load(path)
    agent.learner.online.eval()
    env.noise_fraction = 0.0
    rows = []
    try:
        for seed in range(first_seed, first_seed + episodes):
            obs, _ = env.reset(seed=seed)
            total = 0.0
            with torch.inference_mode():
                for steps in range(1, env.max_steps + 1):
                    action = agent.act(obs, epsilon=0.0)
                    obs, reward, terminated, truncated, info = env.step(action)
                    total += float(reward)
                    if terminated or truncated:
                        break
                else:
                    raise RuntimeError("Environment did not terminate within its budget")
            success, damage = bool(info["success"]), bool(info["damage"])
            rows.append(dict(seed=seed, success=success, damage=damage,
                timeout=bool(info["budget_exhausted"] and not success and not damage),
                steps=steps, raw_return=total, travel=float(info["travel_distance"]),
                peak_deformation=float(info["peak_normalized_deformation"])))
    finally:
        env.close()
    return dict(model=name, sha256=digest, condition="deterministic, greedy, original reward",
        episodes=episodes, success=sum(r["success"] for r in rows),
        damage=sum(r["damage"] for r in rows), timeout=sum(r["timeout"] for r in rows), rows=rows)


if __name__ == "__main__":
    names = json.loads((HERE / "models.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=names, default="prepared")
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.model, args.seed, args.episodes), indent=2))
