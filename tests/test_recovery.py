"""Tests for ntqr.llm.recovery_scoring -- the shared recovery-scoring core.

Pure arithmetic over the ntqr solution dict shape; no ntqr algebra import
needed. These pin the behaviour that ntqr.llm.analysis, ntqr.llm.simulation,
and ntqr.llm.bootstrap all depend on.

    .venv/bin/python -m pytest tests/test_recovery.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "python" / "src"))

from ntqr.llm import recovery_scoring as R  # noqa: E402


def _solution(accs, prev=0.6):
    """Build a minimal ntqr-shaped solution dict from 3 (a, b) accuracy pairs."""
    return {
        "prevalence": {"a": prev, "b": 1 - prev},
        "accuracy": [{"a": a, "b": b} for a, b in accs],
    }


def test_safe_float_guards_complex_and_nonfinite():
    assert R.safe_float(0.5) == 0.5
    assert R.safe_float(complex(0, 1)) is None
    assert R.safe_float(None) is None
    assert R.safe_float(float("nan")) is None
    assert R.safe_float(float("inf")) is None
    assert R.safe_float(float("-inf")) is None


def test_mean_accuracy_averages_six_terms():
    sol = _solution([(0.9, 0.8), (0.7, 1.0), (0.6, 0.5)])
    assert R.mean_accuracy(sol) == pytest.approx(
        (0.9 + 0.8 + 0.7 + 1.0 + 0.6 + 0.5) / 6
    )


def test_mean_accuracy_degenerate_is_negative():
    # all-None accuracies -> sorts strictly below any real solution
    sol = _solution([(None, None), (None, None), (None, None)])
    assert R.mean_accuracy(sol) == -1.0


def test_select_best_prefers_better_than_random():
    good = _solution([(0.9, 0.9), (0.9, 0.9), (0.9, 0.9)])
    mirror = _solution([(0.1, 0.1), (0.1, 0.1), (0.1, 0.1)])
    assert R.select_best([good, mirror]) is good
    assert R.select_best([mirror, good]) is good


def test_select_best_skips_degenerate():
    good = _solution([(0.8, 0.8), (0.8, 0.8), (0.8, 0.8)])
    degen = _solution([(None, None), (None, None), (None, None)])
    assert R.select_best([degen, good]) is good


def test_accuracy_abs_errors_matches_truth():
    sol = _solution([(0.9, 0.8), (0.7, 1.0), (0.6, 0.5)])
    truth = [{"a": 0.8, "b": 0.8}, {"a": 0.7, "b": 0.9}, {"a": 0.6, "b": 0.6}]
    errs = R.accuracy_abs_errors(sol, truth)
    assert len(errs) == 6
    assert errs == pytest.approx([0.1, 0.0, 0.0, 0.1, 0.0, 0.1])


def test_accuracy_abs_errors_skips_nonfinite_terms():
    sol = _solution([(0.9, float("nan")), (0.7, 1.0), (0.6, 0.5)])
    truth = [{"a": 0.8, "b": 0.8}, {"a": 0.7, "b": 0.9}, {"a": 0.6, "b": 0.6}]
    errs = R.accuracy_abs_errors(sol, truth)
    assert len(errs) == 5  # the NaN term is dropped, not counted as 0
