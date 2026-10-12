"""Package the runnable environment, current demonstrations, and two-page report."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root.parent / "ME5418_Group18_Environment_Submission.zip"
    candidates = [root / name for name in ("README.md", "environment.yml", "pyproject.toml")]
    for directory in ("configs", "src", "scripts", "tests", "results/comparison"):
        candidates.extend(path for path in (root / directory).rglob("*") if path.is_file())
    candidates.append(root / "report/ME5418_Group18_Environment_Report.pdf")
    demo = root / "results/demo"
    for stem in ("push_seed_0", "detour_seed_0", "random_seed_7", "greedy_seed_0"):
        candidates.extend(path for path in demo.glob(f"{stem}*") if path.is_file())
    excluded = {"__pycache__", ".pytest_cache", ".ruff_cache", ".git", ".venv"}
    selected = sorted({
        path for path in candidates
        if not excluded.intersection(path.relative_to(root).parts)
        and not any(part.endswith(".egg-info") for part in path.relative_to(root).parts)
        and path.suffix not in {".pyc", ".pyo"}
    })
    for path in selected:
        if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Missing or out-of-project submission file: {path}")
    # Only replace the generated submission after writing and CRC-verifying it.
    temporary = output.with_suffix(".building.zip")
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
        for path in selected:
            archive.write(path, arcname=Path(root.name) / path.relative_to(root))
    with ZipFile(temporary) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise ValueError(f"Archive CRC failure: {corrupt}")
    temporary.replace(output)
    print(f"Packaged {len(selected)} files, {output.stat().st_size:,} bytes")
    print(output)


if __name__ == "__main__":
    main()
