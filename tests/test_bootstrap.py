"""Tests for scripts/llm/_bootstrap.py -- the scenario-bootstrap recovery CI.

No mocks: resamples drive the real ntqr solve via the shared _simulation path.
Records are built from a seeded synthetic ensemble so the test is deterministic
and needs no cached LLM data.

    .venv/bin/python -m pytest tests/test_bootstrap.py -q
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "llm"))
sys.path.insert(0, str(REPO / "python" / "src"))

import _bootstrap as B  # noqa: E402
import _simulation as sim  # noqa: E402


def _records(n=256, seed=1):
    """A deterministic list of (truth_label, pattern) records."""
    ens = sim.Ensemble(0.6, ((0.9, 0.85), (0.85, 0.8), (0.8, 0.9)))
    counts = sim.generate_label_counts(ens, n, random.Random(seed))
    records = []
    for truth, patterns in counts.items():
        for pattern, k in patterns.items():
            records.extend([(truth, pattern)] * k)
    return records


def test_label_counts_from_records_conserves():
    recs = _records(120, seed=3)
    lc = B.label_counts_from_records(recs)
    assert sum(lc["a"].values()) + sum(lc["b"].values()) == len(recs)


def test_point_estimate_matches_full_sample_solve():
    recs = _records(256, seed=5)
    out = B.bootstrap_recovery(recs, n_boot=50, base_seed=100)
    direct = sim.solve_recovery(B.label_counts_from_records(recs))
    assert out["point_recovery_mae"] == pytest.approx(direct["accuracy_mae"])


def test_ci_brackets_and_rates_well_formed():
    recs = _records(256, seed=7)
    out = B.bootstrap_recovery(recs, n_boot=80, base_seed=200)
    mae = out["recovery_mae"]
    assert mae["ci_lo"] <= mae["median"] <= mae["ci_hi"]
    assert 0.0 <= out["degenerate_rate"] <= 1.0
    assert mae["n"] + out["degenerate_count"] == out["n_boot"]
    # raw samples returned for plotting, one per non-degenerate resample
    assert len(out["mae_samples"]) == mae["n"]


def test_bootstrap_is_reproducible():
    recs = _records(200, seed=9)
    a = B.bootstrap_recovery(recs, n_boot=40, base_seed=42)
    b = B.bootstrap_recovery(recs, n_boot=40, base_seed=42)
    assert a["recovery_mae"] == b["recovery_mae"]
    assert a["degenerate_count"] == b["degenerate_count"]


def test_different_seed_changes_samples_not_point():
    recs = _records(200, seed=11)
    a = B.bootstrap_recovery(recs, n_boot=40, base_seed=1)
    b = B.bootstrap_recovery(recs, n_boot=40, base_seed=2)
    # point estimate is the full sample (seed-independent) ...
    assert a["point_recovery_mae"] == pytest.approx(b["point_recovery_mae"])
    # ... but the resample distribution differs
    assert a["mae_samples"] != b["mae_samples"]
