"""Retain every gradient diagnostic and verification record without model binaries."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive(run_dir, destination):
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    if status["status"] == "running":
        raise ValueError("Gradient diagnostics are still running.")
    if status["status"] == "completed":
        verification = json.loads((run_dir / "verification.json").read_text(encoding="utf-8"))
        if not verification["passed"] or not (run_dir / "analysis.json").is_file():
            raise ValueError("Complete independent verification and analysis before archiving.")
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        if len(summary["cases"]) != 28:
            raise ValueError("The complete archive must retain all 28 selected cases.")
        for row in summary["cases"]:
            if digest(run_dir / row["file"]) != row["sha256"]:
                raise ValueError("A completed diagnostic case differs from its recorded hash.")
    sources = sorted([*run_dir.rglob("*.json"), *run_dir.glob("*.png")])
    if any(path.stat().st_size > 8 * 1024 * 1024 for path in sources):
        raise ValueError("An evidence file exceeds the archive size limit.")
    destination.mkdir(parents=True, exist_ok=False)
    records = {}
    for source in sources:
        name = source.relative_to(run_dir).as_posix()
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if digest(source) != digest(target):
            raise RuntimeError(f"Archived file differs from source: {name}")
        records[name] = {"sha256": digest(target), "bytes": target.stat().st_size}
    manifest = {
        "source_run": str(run_dir.resolve()), "status": status["status"], "files": records,
        "archive_script_sha256": digest(Path(__file__)),
        "coverage": "Every selected case, every independent one-step ablation, reconstructed actual-state probe, provenance, accounting, analysis and verification; no case sampling.",
        "excluded": "Historical binary models remain at their unchanged source paths; console output remains in ignored results.",
    }
    (destination / "archive_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Archived {len(records)} evidence files and one manifest to {destination}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    archive(args.run_dir.resolve(), args.destination.resolve())


if __name__ == "__main__":
    main()
