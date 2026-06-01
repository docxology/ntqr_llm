"""Shared helpers for the thin orchestrator scripts.

These scripts are deliberately thin: they import the real ``ntqr`` package
(or a simple baseline), call its public methods, and write results under
``outputs/``. All of the cross-cutting plumbing (locating the repo, making the
package importable, serializing SymPy values to JSON, resolving output dirs)
lives here so each orchestrator stays a few readable lines.
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any

# scripts/_common.py -> scripts/ -> repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
_PACKAGE_SRC = REPO_ROOT / "python" / "src"


def ensure_ntqr_importable() -> None:
    """Make ``import ntqr`` work whether or not the package is pip-installed.

    Preferred setup is ``pip install -e python`` inside a virtualenv. As a
    fallback (e.g. a fresh checkout with no install), we add the in-repo source
    tree to ``sys.path`` so the orchestrators still run against local source.
    """
    try:
        import ntqr  # noqa: F401
    except ModuleNotFoundError:
        if _PACKAGE_SRC.is_dir():
            sys.path.insert(0, str(_PACKAGE_SRC))
        import ntqr  # noqa: F401  (re-raise if still missing)


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
    # SymPy objects expose .evalf(); render both exact and float forms.
    if hasattr(obj, "free_symbols") or obj.__class__.__module__.startswith(
        "sympy"
    ):
        try:
            return {"exact": str(obj), "float": float(obj)}
        except (TypeError, ValueError):
            return {"exact": str(obj)}
    if isinstance(obj, dict):
        return {_key(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _key(key: Any) -> str:
    if isinstance(key, tuple):
        return ",".join(str(part) for part in key)
    return str(key)


def _sidecar_provenance(
    filename: str, result: dict[str, Any]
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
        provenance["generated_at"] = _dt.datetime.now().isoformat(
            timespec="seconds"
        )
    return provenance


def save_json(directory: Path, filename: str, payload: Any) -> Path:
    """Write ``payload`` as pretty JSON, wrapped with a small provenance header."""
    result = _jsonable(payload)
    record = {
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
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
            else _sidecar_provenance(filename, result)
        )
    path = directory / filename
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path


def save_text(directory: Path, filename: str, text: str) -> Path:
    path = directory / filename
    path.write_text(text if text.endswith("\n") else text + "\n")
    return path


def banner(title: str) -> None:
    print(f"\n=== {title} ===")
