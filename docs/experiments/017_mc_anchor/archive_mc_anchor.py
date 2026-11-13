"""Archive complete MC-anchor outcomes and compact evidence without model binaries."""

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
        raise ValueError("The MC-anchor experiment is still running.")
    if status["status"] == "completed":
        required = (
            "summary.json", "reload_checks.json", "q_analysis.json",
            "q_analysis_baseline_dense.json", "q_analysis_treatment_dense.json",
            "mc_anchor_outcomes.png", "neighbor_q_gaps.png",
        )
        if any(not (run_dir / name).is_file() for name in required):
            raise ValueError("Complete verification, analysis and figures before archiving.")
        if not json.loads((run_dir / "reload_checks.json").read_text(encoding="utf-8"))["passed"]:
            raise ValueError("Independent verification has not passed.")
    sources = sorted([*run_dir.rglob("*.json"), *run_dir.glob("mc_anchor_outcomes.png"),
                      *run_dir.glob("neighbor_q_gaps.png")])
    if any(path.stat().st_size > 8 * 1024 * 1024 for path in sources):
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
    for basename, sample, first_count, start in (
        ("updates.jsonl", "update_samples.jsonl", 10, 2001),
        ("sentinel_q.jsonl", "sentinel_q_samples.jsonl", 11, 2000),
    ):
        source = run_dir / "treatment/td" / basename
        if not source.exists() and status["status"] != "completed":
            continue
        lines = source.read_text(encoding="utf-8").splitlines()
        if status["status"] == "completed" and (
            [json.loads(line)["update"] for line in lines] != list(range(start, 10001))
        ):
            raise ValueError(f"Incomplete raw evidence: {source}")
        selected = [line for index, line in enumerate(lines)
                    if index < first_count or index == len(lines) - 1
                    or json.loads(line)["update"] % 50 == 0]
        name = f"treatment/td/{sample}"
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(selected) + "\n", encoding="utf-8")
        files[name] = {"sha256": digest(target), "bytes": target.stat().st_size}
        coverage[f"treatment/td/{basename}"] = {
            "raw_sha256": digest(source), "raw_records": len(lines),
            "archived_records": len(selected),
            "sampling": "Initial probe if present, first ten new updates, every 50th absolute update and last.",
        }
    manifest = {
        "source_run": str(run_dir.resolve()), "status": status["status"],
        "files": files, "raw_log_coverage": coverage,
        "archive_script_sha256": digest(Path(__file__)),
        "historical_control": "Reused attempt-016 baseline; original control evidence remains in its committed archive.",
        "excluded": "Model binaries, full raw logs and console output remain in ignored results.",
        "note": "All measured treatment outcomes and failures, complete compact paired dense traces, and independent verification are retained.",
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
