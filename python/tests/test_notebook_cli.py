"""Notebook CLI reports filesystem failures without overwriting user files."""

from pathlib import Path
import subprocess
import sys

import ntqr


def test_notebook_cli_copies_once_and_reports_existing_destination(tmp_path):
    entrypoint = Path(sys.executable).with_name("ntqr-docs")
    notebook_root = Path(ntqr.__file__).parent / "notebooks"
    notebook = next(notebook_root.rglob("*.ipynb"))
    destination = tmp_path / "ntqr_notebooks"

    first = subprocess.run(
        [sys.executable, str(entrypoint)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert first.returncode == 0, first.stderr
    copied = destination / notebook.relative_to(notebook_root)
    assert copied.read_bytes() == notebook.read_bytes()

    sentinel = destination / "user-notes.txt"
    sentinel.write_text("preserve my edits\n")
    second = subprocess.run(
        [sys.executable, str(entrypoint)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert second.returncode == 1
    assert "Unable to copy notebooks" in second.stderr
    assert sentinel.read_text() == "preserve my edits\n"
    assert copied.read_bytes() == notebook.read_bytes()
