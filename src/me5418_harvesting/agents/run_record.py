"""Local provenance and explicit outcomes for completed, failed, or interrupted runs."""

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(root: Path, *arguments: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *arguments], stderr=subprocess.DEVNULL,
            text=True, timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def begin_run(output: Path, environment: dict, dqn: dict, training: dict) -> None:
    """Hash the actual local inputs, including uncommitted source changes."""
    started = _now()
    (output / "run_status.json").write_text(json.dumps({
        "status": "running", "started_at_utc": started,
    }, indent=2), encoding="utf-8")
    (output / "settings.json").write_text(json.dumps({
        "environment": environment, "dqn": dqn, "training": training,
    }, indent=2, allow_nan=False), encoding="utf-8")
    root = Path(__file__).resolve().parents[3]
    paths = [*root.glob("src/**/*.py"), *root.glob("scripts/*.py"), *root.glob("configs/*.yaml")]
    paths += [root / name for name in ("pyproject.toml", "environment.yml", "requirements.txt")]
    hashes = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths) if path.is_file()
    }
    packages = {}
    for name in ("torch", "numpy", "gymnasium", "PyYAML", "matplotlib"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    status = _git(root, "status", "--porcelain")
    provenance = {
        "started_at_utc": started, "git_head": _git(root, "rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": packages, "argv": sys.argv, "source_sha256": hashes,
        "note": "Git HEAD alone may not describe a dirty run; source hashes identify its inputs.",
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False), encoding="utf-8",
    )


def finish_run(output: Path, status: str, error: BaseException | None = None) -> None:
    record = json.loads((output / "run_status.json").read_text(encoding="utf-8"))
    record.update(status=status, finished_at_utc=_now())
    if error is not None:
        record.update(error_type=type(error).__name__, error_message=str(error))
    record["checkpoints"] = {
        name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "bytes": path.stat().st_size}
        for name in ("best.pt", "latest.pt") if (path := output / name).is_file()
    }
    (output / "run_status.json").write_text(
        json.dumps(record, indent=2, allow_nan=False), encoding="utf-8",
    )
