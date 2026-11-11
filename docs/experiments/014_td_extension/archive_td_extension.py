"""Archive complete outcomes and compact dense-probe evidence for attempt 014."""

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
        raise ValueError("The extension is still running.")
    if status["status"] == "completed":
        required = ("summary.json", "q_analysis.json", "reload_checks.json",
                    "extension_outcomes.png", "sentinel_gaps.png")
        if any(not (run_dir / name).is_file() for name in required):
            raise ValueError("Complete analysis, verification and figures before archiving.")
        if not json.loads((run_dir / "reload_checks.json").read_text(encoding="utf-8"))["passed"]:
            raise ValueError("Independent verification has not passed.")
    sources = sorted([*run_dir.rglob("*.json"), *run_dir.glob("extension_outcomes.png"),
                      *run_dir.glob("sentinel_gaps.png")])
    if any(source.stat().st_size > 8 * 1024 * 1024 for source in sources):
        raise ValueError("An evidence file exceeds the compact archive limit.")
    destination.mkdir(parents=True, exist_ok=False)
    files, coverage = {}, {}
    for source in sources:
        name = source.relative_to(run_dir).as_posix()
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if digest(source) != digest(target):
            raise RuntimeError(f"Archived file differs: {name}")
        files[name] = {"sha256": digest(target), "bytes": target.stat().st_size}
    for basename, sample_name, first_count, note in (
        ("updates.jsonl", "update_samples.jsonl", 10, "first 10 new updates, every 50th absolute update, last"),
        ("sentinel_q.jsonl", "sentinel_q_samples.jsonl", 11,
         "initial state, first 10 new updates, every 50th absolute update, last"),
    ):
        source = run_dir / "td" / basename
        if not source.exists() and status["status"] != "completed":
            continue
        lines = source.read_text(encoding="utf-8").splitlines()
        expected = 8000 if basename == "updates.jsonl" else 8001
        start = 2001 if basename == "updates.jsonl" else 2000
        if status["status"] == "completed" and (
            len(lines) != expected
            or [json.loads(line)["update"] for line in lines] != list(range(start, 10001))
        ):
            raise ValueError(f"Incomplete raw evidence: {source}")
        selected = [line for index, line in enumerate(lines)
                    if index < first_count or index == len(lines) - 1
                    or json.loads(line)["update"] % 50 == 0]
        name = f"td/{sample_name}"
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(selected) + "\n", encoding="utf-8")
        files[name] = {"sha256": digest(target), "bytes": target.stat().st_size}
        coverage[f"td/{basename}"] = {
            "raw_sha256": digest(source), "raw_records": len(lines),
            "archived_records": len(selected), "sampling": note,
        }
    manifest = {
        "source_run": str(run_dir.resolve()), "status": status["status"], "files": files,
        "raw_log_coverage": coverage, "archive_script_sha256": digest(Path(__file__)),
        "excluded": "Models, encoded arrays, console output and full raw Q/update logs remain in ignored results.",
        "note": "All scheduled outcomes, failures, complete compact Q-gap/action series and verification results are retained.",
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
