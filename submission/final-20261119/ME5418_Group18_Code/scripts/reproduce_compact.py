"""Regenerate demonstrations, train the compact DQN, and compare frozen evaluations."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import subprocess
import sys

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
DEFAULT_REPO = HERE.parent
RELEASED = Path("models/compact_guided_generalization.pt")
RELEASED_SHA256 = "75d7d7ca728b1c85c8ba6fa226076f85f4f72bdce3cfe4c6c0177946e5fe51e6"


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def run(repo, *argv):
    env = dict(os.environ, PYTHONPATH=str(repo / "src"), PYTHONUTF8="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    argv = [sys.executable, "-u", *map(str, argv)]
    print(subprocess.list2cmdline(argv), flush=True)
    subprocess.run(argv, cwd=repo, env=env, check=True)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def data_paths(directory, metadata):
    return [directory / PureWindowsPath(entry["path"]).name for entry in metadata["new_data"]]


def collect(repo, files, metadata, workers):
    files[0].parent.mkdir(parents=True, exist_ok=True)
    if not files[0].exists():
        run(repo, "scripts/collect_generalization_demos.py", "--seed", 5418000, "--count", 320,
            "--workers", workers, "--output", files[0])
    if not files[1].exists():
        run(repo, "scripts/collect_generalization_recovery.py", "--checkpoint", metadata["source"],
            "--seed", 5418400, "--count", 32, "--workers", workers, "--output", files[1])


def audit_inputs(repo, out, files, metadata):
    teacher_hash = digest(repo / metadata["source"])
    assert teacher_hash == metadata["source_sha256"]
    assert (repo / metadata["old_data"]["path"]).is_file()
    comparisons = []
    for path, wanted in zip(files, metadata["new_data"], strict=True):
        with np.load(path, allow_pickle=False) as dataset:
            rows = len(dataset["actions"])
            seeds = np.unique(dataset["scene_seeds"]).tolist()
            hash_ = digest(path)
            comparison = dict(file=path.name, rows=rows, expected_rows=wanted["transitions"],
                              seeds=seeds, expected_seeds=wanted["seeds"], sha256=hash_,
                              expected_sha256=wanted["sha256"], exact_file_match=hash_ == wanted["sha256"])
            comparisons.append(comparison)
            print(path.name, rows, "transitions from", len(seeds), "layouts; exact file match:",
                  comparison["exact_file_match"], flush=True)
    write_json(out / "data-comparison.json", dict(teacher_sha256=teacher_hash, datasets=comparisons))
    if any(r["rows"] != r["expected_rows"] or r["seeds"] != r["expected_seeds"] for r in comparisons):
        print("Warning: regenerated input rows or seed membership differ from the release.", flush=True)


def train(repo, out, files, metadata):
    run(repo, "scripts/tune_coordinate_dqn.py", "--compact", "--checkpoint", metadata["source"],
        "--demos", files[0], "--demos", files[1], "--output-dir", out / "training",
        "--updates", metadata["updates"], "--seed", metadata["seed"],
        "--learning-rate", metadata["learning_rate"], "--batch-size", metadata["batch_size_per_group"],
        "--old-weight", metadata["old_weight"], "--td-weight", metadata["td_weight"],
        "--margin", metadata["margin"], "--target-sync", metadata["target_sync_interval"],
        "--supervision", metadata["supervision"], "--demo-weights", *metadata["demo_weights"])


def compare_weights(out, retrained, original):
    actual = torch.load(retrained, map_location="cpu", weights_only=True)
    assert (actual["environment_config"], actual["dqn_config"]) == (
        original["environment_config"], original["dqn_config"],
    )
    original_tensors, new_tensors = original["online"], actual["online"]
    assert original_tensors.keys() == new_tensors.keys()
    comparison = {
        key: {
            "exact": torch.equal(original_tensors[key], value),
            "max_abs_error": float((original_tensors[key] - value).abs().max()),
        }
        for key, value in new_tensors.items()
    }
    result = dict(retrained_sha256=digest(retrained), released_sha256=RELEASED_SHA256,
                  exact_model_tensors=all(r["exact"] for r in comparison.values()),
                  tensors=comparison)
    write_json(out / "weight-comparison.json", result)
    print("Exact equality across all network tensors:", result["exact_model_tensors"], flush=True)


def evaluate(repo, out, retrained, workers):
    name = out / "evaluation-133.json"
    if not name.exists():
        run(repo, "scripts/evaluate_coordinate_models.py", "--model", f"released={RELEASED.as_posix()}",
            "--model", f"regenerated={retrained}", "--seed", 5420000, "--episodes", 100,
            "--known-dataset", "--workers", workers, "--output", name)
    data = json.loads(name.read_text(encoding="utf-8"))
    assert data["models"]["released"]["sha256"] == RELEASED_SHA256
    assert data["models"]["regenerated"]["sha256"] == digest(retrained)
    for model in ("released", "regenerated"):
        assert len([r for r in data["records"] if r["model"] == model]) == 133
    mapped = {
        model: {(r["group"], r["seed"]): {k: v for k, v in r.items() if k != "model"}
                for r in data["records"] if r["model"] == model}
        for model in ("released", "regenerated")
    }
    assert mapped["released"].keys() == mapped["regenerated"].keys()
    diff = [list(k) for k in mapped["released"] if mapped["released"][k] != mapped["regenerated"][k]]
    comparison = dict(summary=data["summary"], per_seed_record_differences=diff)
    write_json(out / "evaluation-comparison.json", comparison)
    print(json.dumps(comparison, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--regenerate-data", action="store_true",
                        help="Regenerate demonstrations in the output directory.")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--stage", choices=("data", "train", "evaluate", "all"), default="all")
    args = parser.parse_args()
    if args.workers < 1 or not __debug__:
        parser.error("Use positive workers and leave Python assertions enabled.")
    repo, out = args.repo.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    assert digest(repo / RELEASED) == RELEASED_SHA256
    original = torch.load(repo / RELEASED, map_location="cpu", weights_only=True)
    metadata = original["metadata"]
    files = data_paths(out / "data" if args.regenerate_data else repo / "data", metadata)
    if args.regenerate_data and args.stage in ("data", "train", "all"):
        collect(repo, files, metadata, args.workers)
    audit_inputs(repo, out, files, metadata)
    retrained = out / "training" / f"update_{metadata['updates']:05d}.pt"
    if args.stage in ("train", "all") and not retrained.exists():
        train(repo, out, files, metadata)
    if args.stage in ("train", "evaluate", "all"):
        compare_weights(out, retrained, original)
    if args.stage in ("evaluate", "all"):
        evaluate(repo, out, retrained, args.workers)


if __name__ == "__main__":
    main()
