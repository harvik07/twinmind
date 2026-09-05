"""PROJECT_ROOT must come from the package file, never the working directory."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from twinmind.utils import paths
from twinmind.utils.paths import PROJECT_ROOT

PRINT_ROOT = "from twinmind.utils.paths import PROJECT_ROOT; print(PROJECT_ROOT)"


def test_project_root_contains_the_repo_markers() -> None:
    assert (PROJECT_ROOT / "pyproject.toml").is_file()
    assert (PROJECT_ROOT / "configs" / "twin.yaml").is_file()
    assert (PROJECT_ROOT / "src" / "twinmind" / "__init__.py").is_file()


def test_project_root_is_independent_of_cwd(tmp_path: Path) -> None:
    """Re-import in a subprocess whose cwd is elsewhere; the answer must not change."""
    result = subprocess.run(
        [sys.executable, "-c", PRINT_ROOT],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
    )
    assert Path(result.stdout.strip()) == PROJECT_ROOT


def test_project_root_resolves_from_inside_tests_directory() -> None:
    """The acceptance-criteria command, run with tests/ as the cwd."""
    tests_dir = PROJECT_ROOT / "tests"
    result = subprocess.run(
        [sys.executable, "-c", PRINT_ROOT],
        cwd=tests_dir,
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
    )
    assert Path(result.stdout.strip()) == PROJECT_ROOT


def test_accessors_create_their_directories() -> None:
    for accessor in (
        paths.raw_dir,
        paths.interim_dir,
        paths.processed_dir,
        paths.models_dir,
        paths.figures_dir,
        paths.rollouts_dir,
    ):
        directory = accessor()
        assert directory.is_dir(), f"{accessor.__name__} did not create {directory}"
        assert PROJECT_ROOT in directory.parents or directory == PROJECT_ROOT
