"""Atomic artifact writes, deterministic encoding and publication preflight use real files."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import sympy

from ntqr import __version__
from ntqr.llm.artifacts import save_json, save_record


def test_json_is_strict_and_retains_exact_symbolic_values(tmp_path):
    path = tmp_path / "nested/result.json"
    save_json(
        path,
        {
            "array": np.array([1, math.nan]),
            "x": sympy.nan,
            "rational": sympy.Rational(1, 3),
            "unicode": "α",
        },
    )
    result = json.loads(
        path.read_text(), parse_constant=lambda token: pytest.fail(token)
    )
    assert result["array"] == [1, None]
    assert result["x"] == {"exact": "nan", "float": None}
    assert result["rational"]["exact"] == "1/3"
    assert result["unicode"] == "α"


def test_save_record_matches_normalized_json_for_nested_values(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "0")
    payload = {
        ("outer", "tuple"): {
            ("inner", "tuple"): {
                "set": {"gamma", "alpha", "beta"},
                "array": np.array([[1, 2], [3, 4]]),
                "nonfinite_array": np.array([math.nan, math.inf]),
                "rational": sympy.Rational(1, 3),
                "undefined": sympy.nan,
            }
        }
    }
    expected = {
        "generated_at": "1970-01-01T00:00:00+00:00",
        "schema_version": 2,
        "ntqr_version": __version__,
        "result": {
            "outer,tuple": {
                "inner,tuple": {
                    "set": ["alpha", "beta", "gamma"],
                    "array": [[1, 2], [3, 4]],
                    "nonfinite_array": [None, None],
                    "rational": {"exact": "1/3", "float": 1 / 3},
                    "undefined": {"exact": "nan", "float": None},
                }
            }
        },
    }

    actual = save_record(tmp_path, "record.json", payload)
    reference = save_json(tmp_path / "expected.json", expected)
    assert actual.read_bytes() == reference.read_bytes()


def test_invalid_encoding_preserves_previous_artifact_and_file_mode(tmp_path):
    path = tmp_path / "result.json"
    save_json(path, {"previous": True})
    path.chmod(0o640)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="collision"):
        save_json(path, {("a", "b"): 1, "a,b": 2})
    assert path.read_bytes() == before
    save_json(path, {"updated": True})
    assert path.stat().st_mode & 0o777 == 0o640
    assert list(tmp_path.iterdir()) == [path]


def test_label_vote_counts_retain_unwrapped_schema(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import _config

    path = tmp_path / "nested/votes.json"
    counts = {"a": {("yes", "no"): 3}, "b": {("no", "yes"): 1}}
    assert _config.save_label_vote_counts(path, counts) == path
    assert json.loads(path.read_text()) == {
        "a": {"yes,no": 3},
        "b": {"no,yes": 1},
    }
    assert _config.load_label_vote_counts_file(path) == counts


@pytest.mark.parametrize("invalid_count", [math.nan, math.inf, -math.inf])
def test_label_vote_counts_serialization_failure_preserves_file_and_mode(
    tmp_path, invalid_count
):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import _config

    path = tmp_path / "votes.json"
    _config.save_label_vote_counts(path, {"a": {("0", "1"): 7}})
    path.chmod(0o640)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        _config.save_label_vote_counts(
            path, {"a": {("0", "1"): invalid_count}}
        )
    assert path.read_bytes() == before
    assert path.stat().st_mode & 0o777 == 0o640
    assert list(tmp_path.iterdir()) == [path]


def test_set_serialization_is_stable_across_hash_seeds():
    code = "import json;from ntqr.llm.artifacts import jsonable,generated_at;print(json.dumps([jsonable({'alpha','beta','gamma'}),generated_at()]))"
    outputs = [
        subprocess.check_output(
            [sys.executable, "-c", code],
            text=True,
            env={
                **os.environ,
                "PYTHONHASHSEED": str(seed),
                "SOURCE_DATE_EPOCH": "0",
            },
        )
        for seed in (1, 999)
    ]
    assert outputs[0] == outputs[1]
    assert "1970-01-01T00:00:00+00:00" in outputs[0]


def test_manuscript_preflight_reads_every_input_before_returning(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import z_generate_manuscript_variables as bridge

    image, sidecar = tmp_path / "image.png", tmp_path / "image.data.json"
    image.write_bytes(b"artifact image bytes")
    sidecar.write_text('{"figure": "image.png"}')
    before = {
        path: path.read_bytes()
        for path in bridge.OUTPUT_FIGURES.glob("*")
        if path.is_file()
    }
    assert len(bridge.figure_payloads(((image, sidecar),))) == 2
    with pytest.raises(FileNotFoundError):
        bridge.figure_payloads(
            ((image, sidecar), (tmp_path / "missing.png", None))
        )
    assert {path: path.read_bytes() for path in before} == before


def test_bridge_check_does_not_write_rendered_files():
    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "output/data/manuscript_variables.json",
        root / "output/manuscript/02_methods.md",
    ]
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths
    }
    result = subprocess.run(
        [
            sys.executable,
            "scripts/z_generate_manuscript_variables.py",
            "--check",
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths
    } == before
