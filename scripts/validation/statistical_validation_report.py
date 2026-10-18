"""Offline statistical validation matrix for the NTQR LLM-personas manuscript.

This report reads generated JSON artifacts and checks the paper's load-bearing
claims: label-free evaluability, recovery scale, bootstrap uncertainty,
synthetic convergence, alarm behavior, tie-break limits, EI-vs-MV trade-off, and
figure sidecar completeness. It does not call LLMs or rerun simulations.

Run:
    .venv/bin/python scripts/validation/statistical_validation_report.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _common  # noqa: E402
import z_generate_manuscript_variables as manuscript_variables  # noqa: E402

LLM_OUT = Path("outputs/llm/multi_llm_personas_evaluation")
SIM_OUT = Path("outputs/sim/synthetic_recovery_study")
BOOT_OUT = Path("outputs/sim/bootstrap_recovery_ci")
VALIDATION_SIDECAR = Path(
    "outputs/validation/statistical_validation/validation_matrix.data.json"
)


from ntqr.llm.supporting_figures import plot_validation_matrix as _plot


def _load(path: Path):
    doc = json.loads(path.read_text())
    return doc.get("result", doc) if isinstance(doc, dict) else doc


def _check(check_id: str, label: str, passed: bool, evidence: str) -> dict:
    return {
        "id": check_id,
        "label": label,
        "passed": bool(passed),
        "evidence": evidence,
    }


def _source_sidecars_exist(repo: Path) -> tuple[bool, str]:
    repo = repo.resolve()
    sidecars = [
        sidecar.resolve().relative_to(repo)
        for _, sidecar in manuscript_variables.FIGURES
        if sidecar is not None
    ]
    missing = []
    invalid = []
    generated_by_report = 0
    parsed = 0
    for path in sidecars:
        full = repo / path
        if not full.exists():
            if path == VALIDATION_SIDECAR:
                generated_by_report += 1
                continue
            missing.append(str(path))
            continue
        try:
            _load(full)
        except (json.JSONDecodeError, OSError) as exc:
            invalid.append(f"{path}: {exc}")
            continue
        parsed += 1
    ok = not missing and not invalid
    accounted = parsed + generated_by_report
    if generated_by_report:
        evidence = (
            f"{accounted}/{len(sidecars)} sidecars accounted for "
            f"({parsed} inputs parse; validation sidecar generated here)"
        )
    else:
        evidence = f"{parsed}/{len(sidecars)} sidecars parse"
    return ok, evidence


def build_report(repo: Path | str = _common.REPO_ROOT) -> dict:
    """Build the validation matrix from generated artifacts under ``repo``."""
    repo = Path(repo)
    llm = _load(repo / LLM_OUT / "summary.json")
    sim = _load(repo / SIM_OUT / "summary.json")
    boot = _load(repo / BOOT_OUT / "summary.json")

    cms = llm["cross_model_statistics"]
    diag = llm["evaluability_diagnostic"]
    conv = sim["convergence"]
    corr = sim["correlation"]
    tie = sim["tie_break"]
    mvei = sim["mv_vs_ei"]

    fit = conv["sqrt_q_fit_accuracy"]
    boot_hi = boot["recovery_mae"]["ci_hi"]
    noise_floor = boot["sampling_noise_floor"]
    independent = next(r for r in corr["rows"] if r["label"] == "independent")
    sidecars_ok, sidecars_evidence = _source_sidecars_exist(repo)

    checks = [
        _check(
            "diagnostic_matches_solve",
            "Label-free diagnostic matches solver outcome",
            diag["diagnostic_matched_outcome"]
            and sorted(diag["predicted_evaluable"]) == sorted(llm["evaluated_models"]),
            (
                f"predicted {len(diag['predicted_evaluable'])}; "
                f"evaluated {len(llm['evaluated_models'])}"
            ),
        ),
        _check(
            "real_recovery_within_noise_floor",
            "Recorded recovery is below the binomial SE reference",
            llm["mean_recovery_mae"] <= cms["sampling_noise_floor"],
            (
                f"MAE {llm['mean_recovery_mae']:.3f} <= "
                f"SE reference {cms['sampling_noise_floor']:.3f}"
            ),
        ),
        _check(
            "bootstrap_ci_within_noise_floor",
            "Recorded bootstrap interval is below the binomial SE reference",
            boot_hi <= noise_floor,
            f"CI_hi {boot_hi:.3f} <= SE reference {noise_floor:.3f}",
        ),
        _check(
            "sqrt_q_convergence",
            "Synthetic recovery follows 1/sqrt(Q)",
            abs(fit["slope"] + 0.5) <= 0.12 and fit["r_squared"] >= 0.95,
            f"slope {fit['slope']:.3f}; R2 {fit['r_squared']:.3f}",
        ),
        _check(
            "alarm_false_positive_control",
            "No false alarms observed in the selected independent runs",
            corr["independent_false_alarm_rate"] == 0.0,
            f"false alarm {corr['independent_false_alarm_rate']:.0%}",
        ),
        _check(
            "common_mode_blind_spot",
            "Common-mode dependence is reported as a blind spot",
            corr["common_mode_max_alarm_rate"] == 0.0
            and corr["common_mode_max_silent_error"]
            > independent["silent_prevalence_abs_error_mean"],
            (
                f"alarm {corr['common_mode_max_alarm_rate']:.0%}; "
                f"silent error {corr['common_mode_max_silent_error']:.3f}"
            ),
        ),
        _check(
            "tie_break_boundary",
            "Better-than-random tie-break boundary is quantified",
            tie["safe_accuracy"] >= 0.8 and tie["max_catastrophic_rate"] >= 0.5,
            (
                f"safe to {tie['safe_accuracy']:.2f}; "
                f"max inversion {tie['max_catastrophic_rate']:.0%}"
            ),
        ),
        _check(
            "ei_mv_tradeoff",
            "EI-vs-MV trade-off has both regimes",
            mvei["ei_best_cell"]["ratio_mv_over_ei"] > 1.0
            and mvei["ei_best_cell"]["ei_win_rate"] > 0.5
            and mvei["mv_best_cell"]["ei_win_rate"] < 0.5
            and mvei["mv_best_cell"]["ei_mae_mean"]
            > mvei["mv_best_cell"]["mv_mae_mean"],
            (
                f"EI win {mvei['ei_best_cell']['ei_win_rate']:.0%}; "
                f"danger-zone EI win {mvei['mv_best_cell']['ei_win_rate']:.0%}"
            ),
        ),
        _check(
            "figure_sidecars_complete",
            "Every source data figure has a parseable sidecar",
            sidecars_ok,
            sidecars_evidence,
        ),
    ]
    n_passed = sum(row["passed"] for row in checks)
    return {
        "study": "statistical_validation",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_checks": len(checks),
        "n_passed": n_passed,
        "n_failed": len(checks) - n_passed,
        "checks": checks,
    }


def write_report(
    repo: Path | str = _common.REPO_ROOT,
    out_dir: Path | None = None,
) -> tuple[Path, Path, Path]:
    repo = Path(repo)
    out_dir = out_dir or _common.output_dir("validation", "statistical_validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_report(repo)
    summary_path = _common.save_json(out_dir, "summary.json", summary)
    figure_path = _plot(summary, out_dir)
    figure_inventory_count = len(
        [sidecar for _, sidecar in manuscript_variables.FIGURES if sidecar is not None]
    )
    sidecar_path = _common.save_json(
        out_dir,
        "validation_matrix.data.json",
        {
            "figure": figure_path.name,
            "title": "Statistical validation matrix",
            "caption": (
                "Offline validation matrix checking the manuscript's "
                "load-bearing claims against generated artifacts, including "
                f"the complete {figure_inventory_count}-figure sidecar "
                "inventory."
            ),
            "checks": summary["checks"],
            "thresholds": [
                {
                    "label": "pass threshold",
                    "value": "all checks pass",
                    "meaning": (
                        "The report is considered green only when every "
                        "load-bearing claim check passes."
                    ),
                }
            ],
            "metadata": {
                "generated_utc": summary["generated_utc"],
                "n_checks": summary["n_checks"],
                "n_passed": summary["n_passed"],
                "n_failed": summary["n_failed"],
                "figure_inventory_count": figure_inventory_count,
            },
        },
    )
    return summary_path, sidecar_path, figure_path


def main() -> int:
    _common.banner("Statistical validation report")
    summary_path, sidecar_path, figure_path = write_report()
    summary = _load(summary_path)
    print(
        f"{summary['n_passed']}/{summary['n_checks']} validation checks passed "
        f"({summary['n_failed']} failed)"
    )
    for path in (summary_path, sidecar_path, figure_path):
        print(str(path))
    return 1 if summary["n_failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
