"""Independently verify saved traces and describe first reference disagreements.

This post-hoc review performs no environment steps or optimizer updates. A
reference disagreement is not proof of an incorrect or failure-causing action.
"""

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
EVIDENCE = HERE / "evidence"
DATA = ROOT / "docs/experiments/029_training_coverage/evidence"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    assert read(EVIDENCE / "status.json")["status"] == "completed"
    manifest = read(EVIDENCE / "manifest.json")
    assert all(digest(EVIDENCE / name) == sha for name, sha in manifest.items())
    metadata = read(DATA / "dataset.json")
    assert digest(DATA / "expanded_demonstrations.npz") == metadata["npz_sha256"]
    bounds = {e["seed"]: e for e in metadata["episodes"]}
    records = read(EVIDENCE / "episodes.json")
    assert len(records) == 66 and len({(r["arm"], r["seed"]) for r in records}) == 66
    diagnostics = []
    with np.load(DATA / "expanded_demonstrations.npz") as data:
        for record in records:
            raw = gzip.decompress((EVIDENCE / record["trace"]).read_bytes())
            assert hashlib.sha256(raw).hexdigest() == record["trace_sha256"]
            rows = [json.loads(line) for line in raw.decode().splitlines()]
            metrics = record["metrics"]
            assert len(rows) == metrics["steps"]
            # Match the evaluator's sequential accumulation, not Python's sum.
            sequential_return = 0.0
            for row in rows:
                sequential_return += row["reward"]
            assert sequential_return == metrics["return"]
            assert sum(r["info"]["collision_blocked"] for r in rows) == metrics["blocked_steps"]
            assert sum(np.linalg.norm(r["info"]["executed_motion"]) <= 1e-6 for r in rows) == metrics["stationary_steps"]
            assert all(r["action"] == int(np.argmax(r["q_values"])) for r in rows)
            assert not any(r["terminated"] or r["truncated"] for r in rows[:-1])
            assert rows[-1]["terminated"] or rows[-1]["truncated"]
            for key in ("success", "damage", "budget_exhausted"):
                assert rows[-1]["info"][key] == metrics[key]
            episode = bounds[record["seed"]]
            diagnostic = {"arm": record["arm"], "seed": record["seed"],
                          "outcome": record["failure_signature"], "first_reference_disagreement": None}
            for offset, row in enumerate(rows):
                index = episode["start"] + offset
                if index >= episode["stop"]:
                    diagnostic["prefix_end_reason"] = "reference_exhausted"
                    break
                expected_hash = hashlib.sha256(data["observations"][index].astype(np.float32).tobytes()).hexdigest()
                if expected_hash != row["encoded_observation_sha256"]:
                    diagnostic["prefix_end_reason"] = "state_mismatch_before_action_comparison"
                    break
                label = int(data["actions"][index])
                if row["action"] != label:
                    q = row["q_values"]
                    diagnostic["first_reference_disagreement"] = {
                        "step": offset + 1, "dataset_row": index,
                        "policy_action": row["action"], "reference_action": label,
                        "reference_minus_best_other_q": q[label] - max(q[a] for a in range(len(q)) if a != label),
                        "q_values": q, "same_encoded_state": True}
                    diagnostic["prefix_end_reason"] = "action_disagreement"
                    break
            else:
                diagnostic["prefix_end_reason"] = "policy_episode_ended_with_matching_prefix"
            diagnostics.append(diagnostic)
    result = {"episodes_verified": len(records), "manifest_files_verified": len(manifest),
              "dataset_sha256": metadata["npz_sha256"], "review_script_sha256": digest(Path(__file__)),
              "optimizer_updates": 0, "simulator_actions": 0,
              "interpretation": "Post-hoc description only; first disagreements may be valid alternative actions.",
              "diagnostics": diagnostics}
    destination = HERE / "review.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Verified {len(records)} episodes and {len(manifest)} archived files.")


if __name__ == "__main__":
    main()
