"""Package source, configuration, tests, and the report without run results."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def submission_files(root: Path) -> list[Path]:
    """Select submission inputs from an explicit list, excluding generated files."""
    candidates = [
        root / name for name in (
            "README.md", "environment.yml", "pyproject.toml",
            ".gitignore", ".gitattributes",
            "report/ME5418_Group18_Environment_Report.pdf",
        )
    ]
    for directory in ("configs", "src", "scripts", "tests"):
        candidates.extend(path for path in (root / directory).rglob("*") if path.is_file())
    excluded = {"__pycache__", ".pytest_cache", ".ruff_cache", ".git", ".venv"}
    selected = sorted({
        path for path in candidates
        if not excluded.intersection(path.relative_to(root).parts)
        and not any(
            part.endswith(".egg-info") or part.startswith(".venv")
            for part in path.relative_to(root).parts
        )
        and path.suffix not in {".pyc", ".pyo"}
    })
    for path in selected:
        if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Missing or out-of-project submission file: {path}")
    return selected


def build_submission(root: Path, output: Path) -> int:
    """Write and CRC-check a clean archive before replacing the output."""
    selected = submission_files(root)
    temporary = output.with_suffix(".building.zip")
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
        for path in selected:
            archive.write(path, arcname=Path(root.name) / path.relative_to(root))
        # Preserve the output directory without bundling results from earlier runs.
        archive.writestr(f"{root.name}/results/.gitkeep", "")
    with ZipFile(temporary) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise ValueError(f"Archive CRC failure: {corrupt}")
    temporary.replace(output)
    return len(selected) + 1


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=root.parent / "ME5418_Group18_Environment_Submission.zip",
        help="Destination ZIP path (parent directory must already exist).",
    )
    args = parser.parse_args()
    count = build_submission(root, args.output)
    print(f"Packaged {count} files, {args.output.stat().st_size:,} bytes")
    print(args.output)


if __name__ == "__main__":
    main()
