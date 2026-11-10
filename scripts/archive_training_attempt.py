"""Archive compact evidence from a completed, failed, or unfinished training attempt."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

EVIDENCE_FILES = (
    "settings.json",
    "summary.json",
    "analysis.json",
    "initial_validation.json",
    "pretraining.json",
    "pretraining_metrics.jsonl",
    "pretraining_validation.json",
    "demonstrations.json",
    "provenance.json",
    "run_status.json",
    "source_manifest.json",
    "metrics.jsonl",
    "learning_diagnostics.png",
)
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 32 * 1024 * 1024


def archive_attempt(run_dir: Path, output_dir: Path) -> dict:
    """Copy an explicit evidence allowlist, never checkpoints or console logs.

    An archive is a snapshot, not a claim that training succeeded. A run-status
    record takes precedence over the presence of a summary. Older runs without
    a status record are considered completed only when their summary exists.
    """
    run_dir = Path(run_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing archive: {output_dir}")
    if not (run_dir / "settings.json").is_file():
        raise ValueError("An attempt archive requires settings.json.")
    if not any((run_dir / name).is_file()
               for name in ("run_status.json", "summary.json", "metrics.jsonl")):
        raise ValueError("An attempt archive requires a status, summary, or episode log.")

    # Read and validate before creating the destination, preserving partial runs.
    contents = {}
    for name in EVIDENCE_FILES:
        source = run_dir / name
        if not source.is_file():
            continue
        if source.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"Evidence file exceeds the 16 MiB archive limit: {name}")
        contents[name] = source.read_bytes()
        if len(contents[name]) > MAX_FILE_BYTES:
            raise ValueError(f"Evidence file exceeds the 16 MiB archive limit: {name}")
    if sum(map(len, contents.values())) > MAX_TOTAL_BYTES:
        raise ValueError("Evidence exceeds the 32 MiB compact archive limit.")

    if "run_status.json" in contents:
        state = json.loads(contents["run_status.json"])
        status = state.get("status") if isinstance(state, dict) else None
        if not isinstance(status, str) or not status.strip():
            raise ValueError("run_status.json must contain a nonempty status string.")
        status_source = "run_status.json"
    elif "summary.json" in contents:
        status, status_source = "completed", "summary.json"
    else:
        status, status_source = "incomplete", "no_completion_record"

    manifest = {
        "format_version": 1,
        "archived_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_dir": str(run_dir),
        "status": status,
        "status_source": status_source,
        "files": {
            name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
            for name, data in contents.items()
        },
        "missing_evidence_files": [name for name in EVIDENCE_FILES if name not in contents],
        "notes": [
            "Missing evidence files may be optional or unavailable in older or incomplete runs.",
            "Completed describes execution status, not policy performance or successful learning.",
            "Checkpoints and raw console logs remain in the local run directory.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, data in contents.items():
        (output_dir / name).write_bytes(data)
    (output_dir / "archive_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New evidence directory; existing directories are never overwritten.")
    args = parser.parse_args()
    try:
        manifest = archive_attempt(args.run_dir, args.output_dir)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(manifest, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
