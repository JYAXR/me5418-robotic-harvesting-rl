"""Verify archived artifacts and greedy decisions without further learning."""

import gzip
import hashlib
import json
from collections import Counter

import numpy as np
import run_study as study
import torch

from me5418_harvesting.agents.recovery_policy import RecoveryAgent

HERE, EVIDENCE, c = study.HERE, study.OUTPUT, study.c


def rows(record):
    path = HERE / record["trace"]
    assert c.sha(path) == record["trace_sha256"]
    raw = gzip.decompress(path.read_bytes())
    assert hashlib.sha256(raw).hexdigest() == record["raw_trace_sha256"]
    return [json.loads(line) for line in raw.decode().splitlines() if line]


def main():
    torch.set_num_threads(1)
    assert c.read(EVIDENCE / "status.json")["status"] == "completed"
    manifest = c.read(EVIDENCE / "manifest.json")
    for name, digest in manifest.items():
        assert c.sha(EVIDENCE / name) == digest, name
    provenance = c.read(EVIDENCE / "provenance.json")
    for name, digest in provenance["input_hashes"].items():
        assert c.sha(study.ROOT / name) == digest, name
    data = np.load(EVIDENCE / "training_data.npz")
    for name in ("base_groups", "query_groups", "anchor_groups"):
        assert set(data[name]).issubset(study.TRAIN)
        assert not set(data[name]) & set(study.VALIDATION)
    agent, env, _ = RecoveryAgent.load(study.MODEL)
    source = torch.load(c.SOURCE, map_location="cpu", weights_only=True)
    for key, value in agent.learner.online.inherited.state_dict().items():
        assert torch.equal(value, source["online"][key])
    anchor_predictions = study.point_scores(agent.learner.online, torch.from_numpy(data["anchor_x"])).argmax(1).numpy()
    np.testing.assert_array_equal(anchor_predictions, data["anchor_y"])
    anchors_by_input = {x.tobytes(): int(action) for x, action in zip(data["anchor_x"], data["anchor_y"])}
    overlapping, conflicts, risk_conflicts = 0, [], 0
    for i, (x, label) in enumerate(zip(data["query_x"], data["query_y"])):
        retained_action = anchors_by_input.get(x.tobytes())
        if retained_action is None:
            continue
        overlapping += 1
        risk_conflicts += int(data["bad_masks"][i, retained_action])
        if label >= 0 and label != retained_action:
            conflicts.append({"query_index": i, "seed": int(data["query_groups"][i]),
                              "planner_action": int(label), "retained_successful_action": retained_action})
    queries, verified, unknown, steps, kinds = 0, 0, 0, [], Counter()
    collection = c.read(EVIDENCE / "collection.json")
    for record in collection:
        assert record["frozen_state_unchanged"]
        if "reset_error" in record:
            continue
        trajectory = rows(record)
        assert len(trajectory) == record["metrics"]["steps"]
        with np.load(HERE / record["dataset"]) as samples:
            assert len(samples["actions"]) == len(record["queries"])
            for query in record["queries"]:
                queries += 1
                steps.append(query["step"])
                kinds[query["kind"]] += 1
                row = trajectory[query["step"] - 1]
                assert row["observation"] == query["state"]["observation"]
                np.testing.assert_array_equal(agent.encoder(np.asarray(row["observation"], dtype=np.float32)), query["feature"])
                i, label = query["sample_index"], query["label"]
                np.testing.assert_array_equal(samples["observations"][i], query["feature"])
                assert samples["actions"][i] == label
                witness = rows(query)
                if label >= 0:
                    verified += 1
                    assert query["success"] and query["used_route_labels"] == 1
                    assert witness[0]["observation"] == row["observation"]
                    assert witness[0]["action"] == label
                    assert witness[-1]["info"]["success"]
                else:
                    unknown += 1
                    assert label == -1 and not query["success"] and query["used_route_labels"] == 0
                flags = []
                for action, check in enumerate(query["action_checks"]):
                    assert action == check["action"]
                    info = check["info"]
                    flags.append(bool(info["damage"] or (np.linalg.norm(info["executed_motion"]) <= 1e-6 and not info["success"])))
                assert flags == query["raw_bad_mask"]
                if label >= 0:
                    assert not query["action_checks"][label]["info"]["damage"]
                    flags[label] = False
                assert flags == query["training_bad_mask"]
                np.testing.assert_array_equal(flags, samples["bad_actions"][i])
    evaluations = c.read(EVIDENCE / "evaluations.json")
    assert sorted(r["seed"] for r in evaluations) == sorted(study.TRAIN + study.VALIDATION)
    actions, identical_training_rollouts = 0, []
    for record in evaluations:
        assert record["action_overrides"] == 0 and record["frozen_state_unchanged"]
        if "reset_error" in record:
            continue
        trajectory = rows(record)
        if record["seed"] in study.TRAIN:
            old = rows(next(r for r in collection if r["seed"] == record["seed"]))
            fields = ("observation", "action", "next_observation", "reward", "terminated", "truncated", "info", "rng_before", "rng_after")
            if len(old) == len(trajectory) and all(all(a[key] == b[key] for key in fields) for a, b in zip(old, trajectory)):
                identical_training_rollouts.append(record["seed"])
        assert len(trajectory) == record["metrics"]["steps"]
        assert abs(sum(row["reward"] for row in trajectory) - record["metrics"]["return"]) < 1e-10
        assert trajectory[-1]["terminated"] or trajectory[-1]["truncated"]
        assert trajectory[-1]["info"]["success"] == record["metrics"]["success"]
        for row in trajectory:
            x = torch.from_numpy(agent.encoder(np.asarray(row["observation"], dtype=np.float32)))[None]
            with torch.no_grad():
                scores = agent.learner.online(x)[0].numpy()
            np.testing.assert_array_equal(scores, row["q_values"])
            assert int(scores.argmax()) == row["action"]
            actions += 1
    fit = c.read(EVIDENCE / "training.json")
    blocks = c.read(EVIDENCE / "blocks.json")
    updates = [json.loads(line) for line in (EVIDENCE / "updates.jsonl").read_text().splitlines()]
    assert [row["attempted_update"] for row in updates] == list(range(1, fit["attempted_updates"] + 1))
    assert all(row["retention_accuracy"] == 1 for row in blocks)
    accepted = sum(row["accepted_scale"] > 0 for row in blocks)
    assert accepted == fit["accepted_blocks"]
    assert fit["optimizer_steps"] == ([accepted * study.BLOCK] if accepted else [])
    assert fit["attempted_updates"] == (study.MAX_UPDATES if fit["pilot_passed"] else study.PILOT)
    resume = torch.load(EVIDENCE / "resume.pt", map_location="cpu", weights_only=True)
    assert c._digest(resume["online"]) == c._digest(agent.learner.online.state_dict())
    assert sorted({int(v["step"]) for v in resume["optimizer"]["state"].values()}) == fit["optimizer_steps"]
    assert c.sha(study.MODEL) == fit["model_sha256"]
    result = {"manifest_files_verified": len(manifest), "pinned_inputs_verified": len(provenance["input_hashes"]),
              "queries_verified": queries, "verified_root_labels": verified, "unknown_root_labels": unknown,
              "query_categories": dict(kinds), "query_steps": dict(Counter(steps)),
              "queries_after_step_40": sum(step > 40 for step in steps),
              "queries_after_step_100": sum(step > 100 for step in steps),
              "retention_actions": len(anchor_predictions), "evaluation_episodes": len(evaluations),
              "greedy_decisions_recomputed": actions, "accepted_scale_counts": dict(Counter(str(row["accepted_scale"]) for row in blocks)),
              "attempted_updates_verified": len(updates), "surviving_optimizer_steps": fit["optimizer_steps"],
              "new_training_updates_during_audit": 0, "new_environment_steps_during_audit": 0,
              "source_weights_unchanged": True, "validation_samples_in_training": 0}
    result.update(query_inputs_also_in_retention_bank=overlapping,
                  planner_labels_conflicting_with_exact_retention=conflicts,
                  adverse_masks_conflicting_with_retained_actions=risk_conflicts,
                  training_rollouts_identical_to_source=identical_training_rollouts)
    c.save(HERE / "audit.json", result)
    data.close()
    env.close()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
