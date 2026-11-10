"""Archive bounded diagnostic evidence, excluding checkpoints and raw trajectories."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ARMS = ("td_margin", "margin_only", "cross_entropy")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive(run_dir, destination):
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    if status["status"] == "running":
        raise ValueError("Archive only completed, failed or interrupted diagnostics.")
    destination.mkdir(parents=True, exist_ok=False)
    names = ["settings.json", "provenance.json", "run_status.json", "summary.json",
             "demonstrations.json", "dataset.json", "reload_checks.json", "fitting_diagnostic.png",
             "driver_source.json", "prefix_comparison.json"]
    names += [f"{arm}/{name}" for arm in ARMS for name in ("observations.json", "summary.json")]
    files, raw_logs = {}, {}
    for name in names:
        source = run_dir / name
        if source.is_file():
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            assert digest(target) == digest(source)
            files[name] = {"sha256": digest(target), "bytes": target.stat().st_size}
    for arm in ARMS:
        source = run_dir / arm / "updates.jsonl"
        if not source.is_file():
            continue
        lines = source.read_text(encoding="utf-8").splitlines()
        selected = [line for index, line in enumerate(lines)
                    if index == 0 or index == len(lines) - 1
                    or json.loads(line)["update"] % 100 == 0]
        target = destination / arm / "update_samples.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(selected) + "\n", encoding="utf-8")
        name = target.relative_to(destination).as_posix()
        files[name] = {"sha256": digest(target), "bytes": target.stat().st_size}
        raw_logs[arm] = {
            "raw_sha256": digest(source), "raw_records": len(lines),
            "archived_records": len(selected), "sampling": "first, every 100th, and last update",
        }
    manifest = {
        "source_run": str(run_dir.resolve()), "status": status["status"], "files": files,
        "update_log_coverage": raw_logs, "archive_script_sha256": digest(Path(__file__)),
        "excluded": "Checkpoints, full encoded data and full update logs stay in ignored results.",
        "note": "Observation snapshots include greedy robot paths as compact failure evidence.",
    }
    (destination / "archive_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    print(f"Archived {len(files)} files to {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    archive(args.run_dir, args.destination)


if __name__ == "__main__":
    main()
