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


def _sidecar_provenance(
    filename: str, result: dict[str, Any], generated_at: str
) -> dict[str, Any]:
    metadata = result.get("metadata")
    meta = metadata if isinstance(metadata, dict) else {}
    provenance: dict[str, Any] = {
        "source_file": filename,
    }
    for key in (
        "generated_utc",
        "study",
        "deterministic",
        "model",
        "n_scenarios",
        "n_boot",
        "seed",
    ):
        if key in meta:
            provenance[key] = meta[key]
    if isinstance(meta.get("experiment"), dict):
        provenance["experiment"] = meta["experiment"]
    if "generated_utc" not in provenance:
        provenance["generated_at"] = generated_at
    return provenance


def save_json(directory: Path, filename: str, payload: Any) -> Path:
    """Write ``payload`` as pretty JSON, wrapped with a small provenance header."""
    result = _jsonable(payload)
    from ntqr.llm.artifacts import generated_at, save_json as atomic_save_json
    from ntqr import __version__
    record = {
        "generated_at": generated_at(),
        "schema_version": 2,
        "ntqr_version": __version__,
        "result": result,
    }
    if (
        filename.endswith(".data.json")
        and isinstance(result, dict)
        and ("figure" in result or "caption" in result)
    ):
        if result.get("caption"):
            record["caption"] = result["caption"]
        if isinstance(result.get("metadata"), dict):
            record["metadata"] = result["metadata"]
        provenance = result.get("provenance")
        record["provenance"] = (
            provenance
            if isinstance(provenance, dict)
            else _sidecar_provenance(filename, result, record["generated_at"])
        )
    path = directory / filename
    return atomic_save_json(path, record)


def save_text(directory: Path, filename: str, text: str) -> Path:
    ensure_ntqr_importable()
    from ntqr.llm.artifacts import atomic_write_text
    return atomic_write_text(directory / filename, text if text.endswith("\n") else text + "\n")


def banner(title: str) -> None:
    print(f"\n=== {title} ===")
