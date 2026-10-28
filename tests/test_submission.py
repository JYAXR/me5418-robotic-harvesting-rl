"""Check that submission packaging never includes generated run outputs."""

import importlib.util
from pathlib import Path
from zipfile import ZipFile

import pytest


def load_packager():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_submission.py"
    spec = importlib.util.spec_from_file_location("build_submission", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_project(root):
    for name in (
        "README.md", "environment.yml", "pyproject.toml", ".gitignore", ".gitattributes",
        "report/ME5418_Group18_Environment_Report.pdf", "src/package/__init__.py",
        "scripts/run_demo.py", "configs/environment.yaml", "tests/test_example.py",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")


def test_archive_excludes_stale_results_and_caches(tmp_path):
    root = tmp_path / "project"
    make_project(root)
    for name in (
        "results/demo/old.gif", "results/comparison/episodes.csv",
        "results/complex_demo/trace.json", "report/rendered/page.png",
        "src/package/__pycache__/module.pyc", "src/package/module.pyo",
        "src/package.egg-info/PKG-INFO", "src/.venv-local/ignored.py",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"generated output")
    output = tmp_path / "submission.zip"
    count = load_packager().build_submission(root, output)
    with ZipFile(output) as archive:
        names = set(archive.namelist())
        assert count == len(names)
        assert archive.testzip() is None
        assert {name for name in names if "/results/" in name} == {
            "project/results/.gitkeep"
        }
        assert "project/src/package/__init__.py" in names
        assert "project/report/ME5418_Group18_Environment_Report.pdf" in names
        assert not any("__pycache__" in name or ".egg-info" in name for name in names)
        assert not any(name.endswith((".pyc", ".pyo")) or ".venv" in name for name in names)
        assert not any("/rendered/" in name for name in names)
    assert (root / "results/demo/old.gif").read_bytes() == b"generated output"


def test_missing_report_does_not_replace_existing_archive(tmp_path):
    root = tmp_path / "project"
    make_project(root)
    (root / "report/ME5418_Group18_Environment_Report.pdf").unlink()
    output = tmp_path / "submission.zip"
    output.write_bytes(b"previous archive")
    with pytest.raises(ValueError, match="Missing"):
        load_packager().build_submission(root, output)
    assert output.read_bytes() == b"previous archive"
