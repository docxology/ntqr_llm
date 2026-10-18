"""Shared helpers for the thin orchestrator scripts.

These scripts are deliberately thin: they import the real ``ntqr`` package
(or a simple baseline), call its public methods, and write results under
``outputs/``. All of the cross-cutting plumbing (locating the repo, making the
package importable, serializing SymPy values to JSON, resolving output dirs)
lives here so each orchestrator stays a few readable lines.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# scripts/_common.py -> scripts/ -> repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
_PACKAGE_SRC = REPO_ROOT / "python" / "src"


def ensure_ntqr_importable() -> None:
    """Use this checkout's NTQR 0.9 plus study extensions consistently."""
    if str(_PACKAGE_SRC) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_SRC))
    import ntqr

    expected = (_PACKAGE_SRC / "ntqr" / "__init__.py").resolve()
    if Path(ntqr.__file__).resolve() != expected or ntqr.__version__ != "0.9":
        raise RuntimeError("run with this checkout's NTQR 0.9: uv sync --extra dev")


def output_dir(category: str, name: str) -> Path:
    """Resolve (and create) ``outputs/<category>/<name>/`` for an orchestrator."""
    target = OUTPUTS_DIR / category / name
    target.mkdir(parents=True, exist_ok=True)
    return target


def _jsonable(obj: Any) -> Any:
    """Recursively convert a result tree into something ``json`` can encode.

    Handles the two shapes ntqr returns that stdlib json cannot: SymPy
    numbers/expressions, and dicts keyed by tuples (e.g. vote patterns).
    """
    ensure_ntqr_importable()
    from ntqr.llm.artifacts import jsonable

    return jsonable(obj)


def _key(key: Any) -> str:
    if isinstance(key, tuple):
        return ",".join(str(part) for part in key)
    return str(key)


def save_json(directory: Path, filename: str, payload: Any) -> Path:
    ensure_ntqr_importable()
    from ntqr.llm.artifacts import save_record

    return save_record(directory, filename, payload)


def save_text(directory: Path, filename: str, text: str) -> Path:
    ensure_ntqr_importable()
    from ntqr.llm.artifacts import atomic_write_text

    return atomic_write_text(
        directory / filename, text if text.endswith("\n") else text + "\n"
    )


def banner(title: str) -> None:
    print(f"\n=== {title} ===")
