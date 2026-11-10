"""Archive rate-control evidence, retaining failures while excluding large model files."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ARMS = ("lr_3e_5", "lr_3e_6")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive(run_dir, destination):
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    if status["status"] == "running":
        raise ValueError("The learning-rate experiment is still running.")
    destination.mkdir(parents=True, exist_ok=False)
    names = ["settings.json", "provenance.json", "run_status.json", "summary.json",
             "input_manifest.json", "dataset.json", "historical_td_margin.json",
             "reload_checks.json", "learning_rate_retention.png"]
    for arm in ARMS:
        names += [f"{arm}/{name}" for name in ("settings.json", "phase_boundary.json")]
        names += [f"{arm}/td_margin/{name}" for name in ("observations.json", "summary.json")]
    files, coverage = {}, {}
    for name in names:
        source = run_dir / name
        if not source.is_file():
            continue
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if digest(source) != digest(target):
            raise RuntimeError(f"Archived evidence differs: {name}")
        files[name] = {"sha256": digest(target), "bytes": target.stat().st_size}
    for arm in ARMS:
        source = run_dir / arm / "td_margin/updates.jsonl"
        if not source.is_file():
            continue
        lines = source.read_text(encoding="utf-8").splitlines()
        selected = [line for index, line in enumerate(lines)
                    if index < 10 or index == len(lines) - 1
                    or json.loads(line)["update"] % 50 == 0]
        target = destination / arm / "td_margin/update_samples.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(selected) + "\n", encoding="utf-8")
        files[target.relative_to(destination).as_posix()] = {
            "sha256": digest(target), "bytes": target.stat().st_size,
        }
        coverage[arm] = {
            "raw_sha256": digest(source), "raw_records": len(lines),
            "archived_records": len(selected), "sampling": "first 10, every 50th, and last update",
        }
    manifest = {
        "source_run": str(run_dir.resolve()), "status": status["status"], "files": files,
        "update_log_coverage": coverage, "archive_script_sha256": digest(Path(__file__)),
        "excluded": "Weights, encoded inputs and full update logs remain in ignored results.",
        "note": "All sampled rollouts and the clearly labeled historical baseline are retained.",
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
