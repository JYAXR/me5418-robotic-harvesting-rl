"""Archive compact calibration evidence and update samples, preserving failed phases."""

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
        raise ValueError("The calibration diagnostic is still running.")
    destination.mkdir(parents=True, exist_ok=False)
    files, coverage = {}, {}
    # The experiment writes configuration, observations and summaries as JSON.
    # Model weights, full encoded inputs and complete update logs are excluded.
    sources = sorted([*run_dir.rglob("*.json"), *run_dir.glob("return_calibration.png")])
    for source in sources:
        if source.stat().st_size > 5 * 1024 * 1024:
            raise ValueError(f"Evidence file exceeds the compact archive limit: {source}")
        name = source.relative_to(run_dir).as_posix()
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if digest(source) != digest(target):
            raise RuntimeError(f"Archived file differs: {name}")
        files[name] = {"sha256": digest(target), "bytes": target.stat().st_size}
    for source in sorted(run_dir.rglob("updates.jsonl")):
        lines = source.read_text(encoding="utf-8").splitlines()
        selected = [line for index, line in enumerate(lines)
                    if index < 10 or index == len(lines) - 1
                    or json.loads(line)["update"] % 50 == 0]
        name = source.relative_to(run_dir).with_name("update_samples.jsonl").as_posix()
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(selected) + "\n", encoding="utf-8")
        files[name] = {"sha256": digest(target), "bytes": target.stat().st_size}
        coverage[source.relative_to(run_dir).as_posix()] = {
            "raw_sha256": digest(source), "raw_records": len(lines),
            "archived_records": len(selected), "sampling": "first 10, every 50th, and last update",
        }
    manifest = {
        "source_run": str(run_dir.resolve()), "status": status["status"], "files": files,
        "update_log_coverage": coverage, "archive_script_sha256": digest(Path(__file__)),
        "excluded": "Model weights, encoded arrays and full update logs stay in ignored results.",
        "note": "All scheduled calibration and TD observations, including failures, are retained.",
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
