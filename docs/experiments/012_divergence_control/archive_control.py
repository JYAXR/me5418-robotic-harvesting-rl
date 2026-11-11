"""Retain diagnosis and controlled-comparison evidence, including every failure."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive(diagnosis_dir, run_dir, destination):
    sources = {"diagnosis": diagnosis_dir, "control": run_dir}
    statuses = {}
    for label, directory in sources.items():
        statuses[label] = json.loads((directory / "run_status.json").read_text(encoding="utf-8"))
        if statuses[label]["status"] == "running":
            raise ValueError(f"The {label} run is still running.")
    destination.mkdir(parents=True, exist_ok=False)
    files, coverage = {}, {}
    for label, directory in sources.items():
        for source in sorted([*directory.rglob("*.json"), *directory.glob("control_outcomes.png")]):
            if source.stat().st_size > 8 * 1024 * 1024:
                raise ValueError(f"Evidence file exceeds the compact archive limit: {source}")
            relative = Path(label) / source.relative_to(directory)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if digest(source) != digest(target):
                raise RuntimeError(f"Evidence copy differs: {relative}")
            files[relative.as_posix()] = {"sha256": digest(target), "bytes": target.stat().st_size}
        for source in sorted(directory.rglob("updates.jsonl")):
            lines = source.read_text(encoding="utf-8").splitlines()
            selected = [line for index, line in enumerate(lines)
                        if index < 10 or index == len(lines) - 1 or json.loads(line)["update"] % 50 == 0]
            relative = Path(label) / source.relative_to(directory).with_name("update_samples.jsonl")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("\n".join(selected) + "\n", encoding="utf-8")
            files[relative.as_posix()] = {"sha256": digest(target), "bytes": target.stat().st_size}
            coverage[(Path(label) / source.relative_to(directory)).as_posix()] = {
                "raw_sha256": digest(source), "raw_records": len(lines),
                "archived_records": len(selected), "sampling": "first 10, every 50th, and last update",
            }
    manifest = {
        "source_runs": {name: str(path.resolve()) for name, path in sources.items()},
        "statuses": statuses, "files": files, "update_log_coverage": coverage,
        "archive_script_sha256": digest(Path(__file__)),
        "excluded": "Models, encoded arrays, console output and full update logs remain in ignored results.",
        "note": "All first-divergence Q analyses, diagnostic replay steps and scheduled policy outcomes are retained.",
    }
    (destination / "archive_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Archived {len(files)} evidence files to {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnosis-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    archive(args.diagnosis_dir.resolve(), args.run_dir.resolve(), args.destination.resolve())


if __name__ == "__main__":
    main()
