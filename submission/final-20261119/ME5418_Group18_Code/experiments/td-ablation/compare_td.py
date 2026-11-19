"""Isolate the TD objective in the compact DQN under paired seeds."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
from statistics import mean
import subprocess
import sys

import torch


HERE = Path(__file__).resolve().parent
DEFAULT_REPO = HERE.parents[1]
TEACHER_TRAINING = DEFAULT_REPO
RELEASED = "models/compact_guided_generalization.pt"
RELEASED_HASH = "75d7d7ca728b1c85c8ba6fa226076f85f4f72bdce3cfe4c6c0177946e5fe51e6"
KNOWN = "docs/experiments/030_expanded_stages/evidence/dataset.json"
TRAIN_SEEDS = (5418, 5419, 5420)
VARIANTS = {"no_td": 0.0, "td": 0.03}
FRESH_SEEDS = tuple(range(5420200, 5420300))


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(repo, args, **options):
    env = dict(os.environ, PYTHONPATH=str(repo / "src"), PYTHONUTF8="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    subprocess.run([sys.executable, "-u", *map(str, args)], cwd=repo, env=env,
                   check=True, **options)


def model_name(seed, variant):
    return f"s{seed}-{variant}"


def audit_checkpoint(path, seed, variant, release, data):
    actual = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("format_version", "base_encoder_version", "relative", "environment_config", "dqn_config"):
        assert actual[key] == release[key], (path, key)
    config = actual["metadata"]
    baseline = release["metadata"]
    for key in ("updates", "source_sha256", "learning_rate", "batch_size_per_group",
                "old_weight", "margin", "target_sync_interval", "observation", "supervision",
                "optimizer_start", "demo_weights"):
        assert config[key] == baseline[key], (path, key)
    assert config["seed"] == seed and config["td_weight"] == VARIANTS[variant]
    assert config["old_data"]["seeds"] == baseline["old_data"]["seeds"]
    inputs = ("sha256", "seeds", "transitions")
    for expected, recorded, file in zip(baseline["new_data"], config["new_data"], data, strict=True):
        assert recorded["sha256"] == sha256(file)
        assert all(recorded[key] == expected[key] for key in inputs)
    assert variant != "no_td" or config["td_loss"] == 0.0
    return config


def fit(repo, out, target, seed, variant, release, data):
    m = release["metadata"]
    cmd = (
        "scripts/tune_coordinate_dqn.py", "--compact", "--checkpoint", m["source"],
        "--demos", data[0], "--demos", data[1], "--output-dir", target.parent,
        "--updates", m["updates"], "--seed", seed,
        "--learning-rate", m["learning_rate"], "--batch-size", m["batch_size_per_group"],
        "--old-weight", m["old_weight"], "--td-weight", VARIANTS[variant],
        "--margin", m["margin"], "--target-sync", m["target_sync_interval"],
        "--supervision", m["supervision"], "--demo-weights", *m["demo_weights"],
    )
    if not target.is_file():
        log = out / f"{model_name(seed, variant)}.log"
        with log.open("w", encoding="utf-8") as stream:
            print(subprocess.list2cmdline([sys.executable, *map(str, cmd)]), file=stream, flush=True)
            run(repo, cmd, stdout=stream, stderr=subprocess.STDOUT)
    config = audit_checkpoint(target, seed, variant, release, data)
    print(f"{model_name(seed, variant)}: completed {config['updates']} updates "
          f"(final TD loss {config['td_loss']:.5f})", flush=True)


def count_pairs(result, groups, checkpoints, out):
    assert result["groups"] == groups
    assert result["environment_config"]["execution_noise_fraction"] == 0.0
    assert result["environment_config"]["max_steps"] == 200
    assert set(result["models"]) == set(checkpoints)
    assert all(result["models"][key]["sha256"] == sha256(path) for key, path in checkpoints.items())
    records = {}
    for row in result["records"]:
        key = row["model"], row["group"], row["seed"]
        assert key not in records and row["action_overrides"] == 0
        assert sum(row[k] for k in ("success", "damage", "timeout")) == 1
        assert not row["timeout"] or row["steps"] == 200
        records[key] = row
    assert len(records) == len(checkpoints) * sum(map(len, groups.values()))

    rows, pairs = [], []
    for group, layout_seeds in groups.items():
        for seed in TRAIN_SEEDS:
            before = [records[model_name(seed, "no_td"), group, layout] for layout in layout_seeds]
            after = [records[model_name(seed, "td"), group, layout] for layout in layout_seeds]
            wins = sum(not a["success"] and b["success"] for a, b in zip(before, after, strict=True))
            losses = sum(a["success"] and not b["success"] for a, b in zip(before, after, strict=True))
            row = dict(group=group, training_seed=seed, layouts=len(layout_seeds),
                       td_failure_to_success=wins, td_success_to_failure=losses,
                       success_delta_pp=100 * (wins - losses) / len(layout_seeds))
            for label, policy in (("no_td", before), ("td", after)):
                row.update({f"{label}_{k}": sum(r[k] for r in policy)
                            for k in ("success", "damage", "timeout")})
            assert row["td_success"] - row["no_td_success"] == wins - losses
            rows.append(row)
            for a, b in zip(before, after, strict=True):
                pair = dict(group=group, training_seed=seed, layout_seed=a["seed"])
                for label, record in (("no_td", a), ("td", b)):
                    pair.update({f"{label}_{k}": record[k] for k in
                                 ("success", "damage", "timeout", "steps", "return",
                                  "travel_distance", "peak_deformation", "contact_steps")})
                pairs.append(pair)
    save_csv(out / "paired.csv", pairs)
    save_csv(out / "summary.csv", rows)
    averages = {}
    for group in groups:
        group_rows = [r for r in rows if r["group"] == group]
        averages[group] = {
            "training_seed_pairs": len(group_rows),
            "layouts_per_pair": group_rows[0]["layouts"],
            "mean_paired_delta_pp": mean(r["success_delta_pp"] for r in group_rows),
            "mean_no_td_success_rate": mean(r["no_td_success"] / r["layouts"] for r in group_rows),
            "mean_td_success_rate": mean(r["td_success"] / r["layouts"] for r in group_rows),
            "failure_to_success": sum(r["td_failure_to_success"] for r in group_rows),
            "success_to_failure": sum(r["td_success_to_failure"] for r in group_rows),
        }
    save_json(out / "averages.json", averages)
    for row in rows:
        print(row)
    print(json.dumps(averages, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--inputs", type=Path, default=TEACHER_TRAINING)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-workers", type=int, default=5)
    parser.add_argument("--eval-workers", type=int, default=8)
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()
    if not __debug__ or min(args.train_workers, args.eval_workers) < 1:
        parser.error("Leave assertions enabled and use positive worker counts.")
    repo, inputs, out = (p.resolve() for p in (args.repo, args.inputs, args.out))
    assert sha256(repo / RELEASED) == RELEASED_HASH
    release = torch.load(repo / RELEASED, map_location="cpu", weights_only=True)
    cfg = release["metadata"]
    assert cfg["td_weight"] == VARIANTS["td"] and cfg["updates"] == 6000
    assert sha256(repo / cfg["source"]) == cfg["source_sha256"]
    data = [inputs / "data" / PureWindowsPath(r["path"]).name for r in cfg["new_data"]]
    assert [sha256(p) for p in data] == [r["sha256"] for r in cfg["new_data"]]
    seen = set(cfg["old_data"]["seeds"]).union(*(r["seeds"] for r in cfg["new_data"]))
    seen.update(range(5420000, 5420106))
    known = read_json(repo / KNOWN)["seeds"]
    assert len(known) == 33 and set(known) == set(cfg["old_data"]["seeds"])
    assert set(FRESH_SEEDS).isdisjoint(seen)
    groups = {"range": list(FRESH_SEEDS), "known": known}

    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(training_seeds=list(TRAIN_SEEDS), variants=VARIANTS, evaluation_groups=groups,
                    primary="range", secondary="known", execution_noise=0.0,
                    updates=cfg["updates"], batch_size=cfg["batch_size_per_group"],
                    shared_configuration={key: cfg[key] for key in
                        ("learning_rate", "old_weight", "margin", "target_sync_interval",
                         "observation", "supervision", "demo_weights")},
                    teacher_sha256=cfg["source_sha256"],
                    data_sha256=[sha256(path) for path in data],
                    old_manifest_sha256=sha256(repo / "docs/experiments/029_training_coverage/evidence/dataset.json"),
                    old_demos_sha256=sha256(repo / cfg["old_data"]["path"]),
                    pytorch=torch.__version__,
                    primary_metric="Within-training-seed success-rate difference on the 100 fresh layouts.")
    protocol_path = out / "protocol.json"
    if protocol_path.exists():
        assert read_json(protocol_path) == protocol
    else:
        save_json(protocol_path, protocol)
    jobs = [(seed, variant) for seed in TRAIN_SEEDS for variant in VARIANTS]
    checkpoints = {
        model_name(seed, variant): out / "checkpoints" / model_name(seed, variant) / "update_06000.pt"
        for seed, variant in jobs
    }
    # Reuse the locally repeated TD checkpoint, with verified matching inputs and settings.
    prior_audit = inputs / "local-repeat-audit.json"
    prior = inputs / "training/update_06000.pt"
    if prior_audit.is_file() and prior.is_file() and read_json(prior_audit)["exact_network_tensors"]:
        checkpoints[model_name(5418, "td")] = prior
    save_json(out / "checkpoint-paths.json", {k: str(v) for k, v in checkpoints.items()})
    with ThreadPoolExecutor(max_workers=args.train_workers) as pool:
        futures = [pool.submit(fit, repo, out, checkpoints[model_name(s, v)], s, v, release, data)
                   for s, v in jobs]
        for future in as_completed(futures):
            future.result()
    if not args.train_only:
        result_path = out / "evaluation.json"
        if not result_path.exists():
            model_flags = [part for k, p in checkpoints.items() for part in ("--model", f"{k}={p}")]
            run(repo, (
                "scripts/evaluate_coordinate_models.py", *model_flags,
                "--seed", FRESH_SEEDS[0], "--episodes", len(FRESH_SEEDS),
                "--known-dataset", "--workers", args.eval_workers, "--output", result_path,
            ))
        count_pairs(read_json(result_path), groups, checkpoints, out)


if __name__ == "__main__":
    main()
