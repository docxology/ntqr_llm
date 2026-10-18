"""Tests for the statistical validation report.

The report is an offline audit layer over generated artifacts: it does not run
models or simulations, it verifies that the manuscript's load-bearing claims are
backed by the JSON outputs and writes a small validation matrix figure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "validation"))

import statistical_validation_report as report  # noqa: E402


def test_validation_report_checks_all_load_bearing_claims():
    summary = report.build_report(REPO)
    checks = {row["id"]: row for row in summary["checks"]}
    expected = {
        "diagnostic_matches_solve",
        "real_recovery_within_noise_floor",
        "bootstrap_ci_within_noise_floor",
        "sqrt_q_convergence",
        "alarm_false_positive_control",
        "common_mode_blind_spot",
        "tie_break_boundary",
        "ei_mv_tradeoff",
        "figure_sidecars_complete",
    }
    assert set(checks) == expected
    assert all(row["passed"] for row in checks.values())
    count = len(report.manuscript_variables.FIGURES)
    assert f"{count}/{count} sidecars" in checks["figure_sidecars_complete"]["evidence"]
    assert summary["n_passed"] == summary["n_checks"] == len(expected)


def test_validation_report_writes_summary_sidecar_and_figure(tmp_path):
    out_dir = tmp_path / "validation"
    summary_path, sidecar_path, figure_path = report.write_report(
        REPO, out_dir
    )
    for path in (summary_path, sidecar_path, figure_path):
        assert path.exists(), path
    summary = json.loads(summary_path.read_text())["result"]
    sidecar = json.loads(sidecar_path.read_text())["result"]
    assert summary["n_failed"] == 0
    assert sidecar["figure"] == figure_path.name
    assert sidecar["caption"]
    assert sidecar["metadata"]["figure_inventory_count"] == len(report.manuscript_variables.FIGURES)
    assert len(sidecar["checks"]) == summary["n_checks"]
