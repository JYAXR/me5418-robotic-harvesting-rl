"""Keep all four diagnostic trajectories and verification evidence in Git."""

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
        raise ValueError("The diagnostic is still running.")
    if status["status"] == "completed":
        required = ["summary.json", "verification.json", "single_action_outcomes.png"]
        required += [f"update_{update}_{mode}{suffix}" for update in (2009, 10000)
                     for mode in ("greedy", "counterfactual") for suffix in (".json", "_steps.jsonl")]
        if any(not (run_dir / name).is_file() for name in required):
            raise ValueError("Complete all four traces, verification and figure before archiving.")
        if not json.loads((run_dir / "verification.json").read_text(encoding="utf-8"))["passed"]:
            raise ValueError("Independent verification has not passed.")
    sources = sorted([*run_dir.rglob("*.json"), *run_dir.rglob("*.jsonl"),
                      *run_dir.glob("single_action_outcomes.png")])
    if any(source.stat().st_size > 8 * 1024 * 1024 for source in sources):
        raise ValueError("An evidence file exceeds the compact archive limit.")
    destination.mkdir(parents=True, exist_ok=False)
    files = {}
    for source in sources:
        name = source.relative_to(run_dir).as_posix()
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if digest(source) != digest(target):
            raise RuntimeError(f"Archived file differs: {name}")
        files[name] = {"sha256": digest(target), "bytes": target.stat().st_size}
    manifest = {
        "source_run": str(run_dir.resolve()), "status": status["status"], "files": files,
        "archive_script_sha256": digest(Path(__file__)),
        "coverage": "Every original and corrected episode, every action record, all failures and read-only verification are retained without sampling.",
        "excluded": "Original model/data binaries and console output remain in ignored results.",
    }
    (destination / "archive_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Archived {len(files)} evidence files to {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    archive(args.run_dir.resolve(), args.destination.resolve())


if __name__ == "__main__":
    main()
