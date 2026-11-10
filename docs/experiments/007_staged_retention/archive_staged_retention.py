"""Retain bounded staged experiment evidence without weights or encoded datasets."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

PHASES = ("pretraining", "margin_only", "td_only", "td_margin")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive(run_dir, destination):
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    if status["status"] == "running":
        raise ValueError("Do not archive a diagnostic that is still running.")
    destination.mkdir(parents=True, exist_ok=False)
    names = ["settings.json", "provenance.json", "run_status.json", "summary.json",
             "demonstrations.json", "dataset.json", "pretraining_reproduction.json",
             "phase_boundaries.json", "reload_checks.json", "retention_diagnostic.png"]
    names += [f"{phase}/{name}" for phase in PHASES for name in ("observations.json", "summary.json")]
    files, coverage = {}, {}
    for name in names:
        source = run_dir / name
        if not source.is_file():
            continue
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if digest(source) != digest(target):
            raise RuntimeError(f"Archived file mismatch: {name}")
        files[name] = {"sha256": digest(target), "bytes": target.stat().st_size}
    for phase in PHASES:
        source = run_dir / phase / "updates.jsonl"
        if not source.is_file():
            continue
        lines = source.read_text(encoding="utf-8").splitlines()
        selected = [line for index, line in enumerate(lines)
                    if index < 10 or index == len(lines) - 1
                    or json.loads(line)["update"] % 50 == 0]
        target = destination / phase / "update_samples.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(selected) + "\n", encoding="utf-8")
        files[target.relative_to(destination).as_posix()] = {
            "sha256": digest(target), "bytes": target.stat().st_size,
        }
        coverage[phase] = {
            "raw_sha256": digest(source), "raw_records": len(lines),
            "archived_records": len(selected), "sampling": "first 10, every 50th, and last update",
        }
    manifest = {
        "source_run": str(run_dir.resolve()), "status": status["status"], "files": files,
        "update_log_coverage": coverage, "archive_script_sha256": digest(Path(__file__)),
        "excluded": "Weights, encoded demonstration arrays and full update logs remain in results.",
        "note": "All scheduled neural rollout action sequences and paths are retained.",
    }
    (destination / "archive_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    print(f"Archived {len(files)} evidence files to {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    archive(args.run_dir, args.destination)


if __name__ == "__main__":
    main()
