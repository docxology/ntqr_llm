"""Strict JSON and atomic UTF-8 artifacts with reproducible UTC provenance."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import stat
import tempfile


def generated_at() -> str:
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    when = (
        datetime.fromtimestamp(int(epoch), timezone.utc)
        if epoch is not None
        else datetime.now(timezone.utc)
    )
    return when.isoformat(timespec="seconds")


def encode_key(key) -> str:
    return ",".join(str(part) for part in key) if isinstance(key, tuple) else str(key)


def jsonable(obj):
    """Encode exact algebra explicitly; missing/nonfinite numeric values become null."""
    if hasattr(obj, "free_symbols") or obj.__class__.__module__.startswith("sympy"):
        result = {"exact": str(obj)}
        try:
            numeric = float(obj)
            result["float"] = numeric if math.isfinite(numeric) else None
        except (TypeError, ValueError, OverflowError):
            pass
        return result
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            encoded = encode_key(key)
            if encoded in result:
                raise ValueError(f"JSON key collision after encoding: {encoded!r}")
            result[encoded] = jsonable(value)
        return result
    if isinstance(obj, (set, frozenset)):
        return sorted(
            (jsonable(value) for value in obj),
            key=lambda value: json.dumps(value, sort_keys=True, allow_nan=False),
        )
    if isinstance(obj, (list, tuple)):
        return [jsonable(value) for value in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    if isinstance(obj, Integral):
        return int(obj)
    if isinstance(obj, Real):
        value = float(obj)
        return value if math.isfinite(value) else None
    if hasattr(obj, "tolist"):
        return jsonable(obj.tolist())
    return str(obj)


def atomic_write_bytes(path: Path, content: bytes) -> Path:
    """Replace only a complete artifact; preserve the prior file on write failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            os.chmod(temporary, mode)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def atomic_write_text(path: Path, content: str) -> Path:
    return atomic_write_bytes(path, content.encode("utf-8"))


def save_json(path: Path, record) -> Path:
    encoded = (
        json.dumps(jsonable(record), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
    )
    return atomic_write_text(path, encoded)


from typing import Any


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


def save_record(directory: Path, filename: str, payload: Any) -> Path:
    """Write ``payload`` as pretty JSON, wrapped with a small provenance header."""
    result = jsonable(payload)
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
    return save_json(path, record)
