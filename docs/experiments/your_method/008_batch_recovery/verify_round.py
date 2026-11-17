"""Independently audit saved recovery labels, autonomous traces and checkpoints."""

import argparse
import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("batch_audit", HERE / "run_batch.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def trace(record):
    path = HERE / record["trace"]
    assert m.sha(path) == record["trace_sha256"]
    raw = gzip.decompress(path.read_bytes())
    assert m.hashlib.sha256(raw).hexdigest() == record["raw_trace_sha256"]
    return [json.loads(line) for line in raw.decode().splitlines() if line]


def verify(number):
    folder = HERE / f"round_{number}"
    assert m.read(folder / "status.json")["status"] == "completed"
    manifest = m.read(folder / "manifest.json")
    for name, digest in manifest.items():
        assert m.sha(folder / name) == digest, name
    provenance = m.read(folder / "provenance.json")
    for name, digest in provenance["input_hashes"].items():
        assert m.sha(m.ROOT / name) == digest, name
    collections = m.read(folder / "collection.json")
    evals = m.read(folder / "evaluations.json")
    assert sorted(r["seed"] for r in collections) == sorted(m.TRAIN)
    assert sorted(r["seed"] for r in evals) == sorted(m.TRAIN + m.VALIDATION)
    baseline = m.read(HERE / "round_1/baseline_validation.json")
    all_evals = evals + (baseline if number == 1 else [])
    eval_actions = recovery_actions = labels = 0
    for r in all_evals:
        assert r["frozen_state_unchanged"] and r["action_overrides"] == 0
        if "reset_error" in r:
            assert not r["metrics"]["success"]
            continue
        rows = trace(r)
        reward = 0.0
        for index, row in enumerate(rows, 1):
            assert row["step"] == index
            assert row["action"] == int(np.argmax(row["q_values"]))
            reward += row["reward"]
            if index < len(rows):
                assert not row["terminated"] and not row["truncated"]
                assert row["next_observation"] == rows[index]["observation"]
                assert row["rng_after"] == rows[index]["rng_before"]
        assert reward == r["metrics"]["return"]
        assert len(rows) == r["metrics"]["steps"]
        assert rows[-1]["terminated"] or rows[-1]["truncated"]
        for key in ("success", "damage", "budget_exhausted", "travel_distance", "distance_to_target"):
            assert rows[-1]["info"][key] == r["metrics"][key]
        assert sum(row["info"]["collision_blocked"] for row in rows) == r["metrics"]["blocked_steps"]
        assert sum(np.linalg.norm(row["info"]["executed_motion"]) <= 1e-6 for row in rows) == r["metrics"]["stationary_steps"]
        eval_actions += len(rows)
    for r in collections:
        assert r["seed"] not in m.VALIDATION and r["frozen_state_unchanged"]
        if "dataset" not in r:
            assert "reset_error" in r
            continue
        path = HERE / r["dataset"]
        assert m.sha(path) == r["dataset_sha256"]
        prefix = trace(r)
        assert len(prefix) == r["prefix_actions"]
        with np.load(path) as data:
            accepted = 0
            for query in r["queries"]:
                rows = trace(query)
                assert len(rows) == query["executed_actions"]
                recovery_actions += len(rows)
                if rows:
                    assert rows[0]["step"] == query["step"]
                    for i, row in enumerate(rows):
                        assert row["step"] == query["step"] + i
                        if i + 1 < len(rows):
                            assert row["next_observation"] == rows[i + 1]["observation"]
                            assert row["rng_after"] == rows[i + 1]["rng_before"]
                            assert not row["terminated"] and not row["truncated"]
                if query["success"]:
                    assert rows[-1]["info"]["success"] and rows[-1]["terminated"]
                    assert query["stop"] - query["start"] == len(rows) == query["accepted_samples"]
                    np.testing.assert_array_equal(data["actions"][query["start"]:query["stop"]], [v["action"] for v in rows])
                    np.testing.assert_array_equal(data["observations"][query["start"]:query["stop"]],
                                                  np.asarray([v["encoded_observation"] for v in rows], dtype=np.float32))
                    accepted += len(rows)
                else:
                    assert query["start"] == query["stop"] and query["accepted_samples"] == 0
            assert accepted == r["samples"] == len(data["actions"])
            labels += accepted
    fit = m.read(folder / "training.json")
    data_meta = m.read(folder / "training_data.json")
    assert m.sha(folder / "training_data.npz") == data_meta["sha256"]
    assert data_meta["validation_samples"] == 0 and data_meta["base_rows"] == 1441
    assert all(v["seed"] in m.TRAIN for v in data_meta["sources"])
    assert m.sha(m.ROOT / fit["model"]) == fit["model_sha256"]
    assert m.sha(folder / "resume.pt") == fit["resume_sha256"]
    resume = torch.load(folder / "resume.pt", weights_only=True)
    source = torch.load(m.SOURCE_STATE, weights_only=True)
    inference = torch.load(m.ROOT / fit["model"], weights_only=True)
    assert m._digest(inference["online"]) == m._digest(resume["online"])
    assert m._digest(source["target"]) == m._digest(resume["target"])
    assert resume["td_updates"] == 14000 and resume["supervised_updates"] == number * 6000
    assert fit["optimizer_after"]["step_min"] == fit["optimizer_after"]["step_max"] == number * 6000
    updates = [json.loads(line) for line in (folder / "updates.jsonl").read_text().splitlines()]
    assert len(updates) == 6000
    assert [r["update"] for r in updates] == list(range((number - 1) * 6000 + 1, number * 6000 + 1))
    assert all(np.isfinite([r["cross_entropy"], r["gradient_norm"]]).all() for r in updates)
    if number == 2:
        previous = m.read(HERE / "round_1/training.json")
        assert fit["optimizer_before"] == previous["optimizer_after"]
    summary = m.read(folder / "summary.json")
    for name, seeds in (("training", m.TRAIN), ("validation", m.VALIDATION), ("original_three", m.TRAIN[:3])):
        assert m.group(evals, seeds) == summary[name]
    assert labels == summary["new_recovery_samples"] and recovery_actions == summary["recovery_physics_actions"]
    assert eval_actions == summary["evaluation_actions"] + summary["fresh_baseline_actions"]
    result = {"round": number, "manifest_files": len(manifest), "pinned_inputs": len(provenance["input_hashes"]),
              "fresh_evaluation_episodes": len(all_evals), "greedy_actions_verified": eval_actions,
              "recovery_actions_verified": recovery_actions, "accepted_labels_verified": labels,
              "supervised_updates_verified": len(updates), "td_and_target_unchanged": True,
              "resume_and_inference_weights_equal": True, "validation_labels_in_training": 0,
              "verification_failures": 0}
    m.save(HERE / f"verification_round{number}.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, choices=(1, 2), required=True)
    verify(parser.parse_args().round)
