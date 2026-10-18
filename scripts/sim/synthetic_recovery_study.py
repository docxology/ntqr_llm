"""Synthetic-ensemble validation of the NTQR error-independent solve.

A thin orchestrator: it imports the pure harness in ``ntqr.llm.simulation``,
runs two deterministic studies through the *real* ntqr solver, writes the data
and figures under ``outputs/sim/synthetic_recovery_study/``, and prints the
output paths for the pipeline manifest.

Unlike the LLM sweep, this study needs **no network and no Ollama** -- it is a
seeded, fully reproducible companion that generalises the finite real-model
ranking to a controlled family of ensembles:

1. **Convergence** -- recovery error vs test size ``Q`` over many independent
   error-independent ensembles; the fitted log-log slope shows the error falls
   like sampling noise (~1/sqrt(Q)).
2. **Alarm characterisation** -- how often the imaginary-prevalence alarm fires
   under independence (false-alarm control), positive common-mode correlation
   (its blind spot), and an anti-correlated pair (which trips it).

Run:
    .venv/bin/python scripts/sim/synthetic_recovery_study.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _common  # noqa: E402

_common.ensure_ntqr_importable()

from ntqr.llm import simulation as sim  # noqa: E402

# --- canonical, frozen study parameters (change here, never in prose) ---
ENSEMBLE = sim.Ensemble(
    prevalence_a=0.55,
    accuracies=((0.90, 0.80), (0.85, 0.75), (0.70, 0.95)),
)
Q_VALUES = [32, 64, 128, 256, 512, 1024, 2048, 4096]
N_TRIALS = 200
CONVERGENCE_SEED = 2024
CORRELATION_Q = 1024
CORRELATION_SEED = 5000
CORRELATION_CONFIGS = [
    sim.CorrelationConfig("independent"),
    sim.CorrelationConfig("common-mode 0.1", common_mode=0.1),
    sim.CorrelationConfig("common-mode 0.2", common_mode=0.2),
    sim.CorrelationConfig("common-mode 0.3", common_mode=0.3),
    sim.CorrelationConfig("common-mode 0.4", common_mode=0.4),
    sim.CorrelationConfig("anti-pair 0.2", anti_pair=(0, 1), anti_strength=0.2),
    sim.CorrelationConfig("anti-pair 0.3", anti_pair=(0, 1), anti_strength=0.3),
    sim.CorrelationConfig("anti-pair 0.4", anti_pair=(0, 1), anti_strength=0.4),
]

# Tie-break sweep: judges held symmetric so mean accuracy is the only varying
# factor; we watch the better-than-random tie-break fail as accuracy -> random.
TIEBREAK_PREVALENCE = 0.60
TIEBREAK_ACCURACIES = [0.90, 0.82, 0.76, 0.70, 0.62, 0.56, 0.52]
TIEBREAK_Q = 256
TIEBREAK_SEED = 4000

# Robustness: confirm the ~1/sqrt(Q) slope across diverse ensembles, not just one.
ROBUSTNESS_ENSEMBLES = [
    (
        "balanced p=0.50",
        sim.Ensemble(0.50, ((0.88, 0.82), (0.80, 0.86), (0.74, 0.78))),
    ),
    (
        "tilted p=0.70",
        sim.Ensemble(0.70, ((0.92, 0.78), (0.83, 0.80), (0.71, 0.90))),
    ),
    (
        "strong p=0.55",
        sim.Ensemble(0.55, ((0.95, 0.90), (0.90, 0.93), (0.88, 0.85))),
    ),
    (
        "modest p=0.65",
        sim.Ensemble(0.65, ((0.80, 0.72), (0.78, 0.75), (0.73, 0.82))),
    ),
]
ROBUSTNESS_Q_VALUES = [64, 128, 256, 512, 1024, 2048]
ROBUSTNESS_N_TRIALS = 120
ROBUSTNESS_SEED = 7000

# EI vs majority-voting across a prevalence × accuracy grid: exactness vs robustness.
MVEI_PREVALENCES = [0.55, 0.70, 0.85]
MVEI_ACCURACIES = [0.90, 0.82, 0.74, 0.66, 0.58]
MVEI_Q = 512
MVEI_N_TRIALS = 150
MVEI_SEED = 8000


from ntqr.llm.supporting_figures import (
    plot_convergence as _plot_convergence,
    plot_alarm as _plot_alarm,
    plot_tiebreak as _plot_tiebreak,
    plot_mv_vs_ei as _plot_mv_vs_ei,
)


def _ensemble_block() -> dict:
    return {
        "prevalence_a": ENSEMBLE.prevalence_a,
        "accuracies": [
            {"judge": i, "alpha": a, "beta": b}
            for i, (a, b) in enumerate(ENSEMBLE.accuracies)
        ],
    }


def run_study() -> dict:
    """Run both sweeps and assemble the summary payload (no I/O)."""
    convergence_rows = sim.convergence_sweep(
        ENSEMBLE, Q_VALUES, N_TRIALS, CONVERGENCE_SEED
    )
    fit_acc = sim.sqrt_q_fit(convergence_rows, "accuracy_mae_mean")
    fit_prev = sim.sqrt_q_fit(convergence_rows, "prevalence_abs_error_mean")
    correlation_rows = sim.correlation_sweep(
        ENSEMBLE,
        CORRELATION_Q,
        CORRELATION_CONFIGS,
        N_TRIALS,
        CORRELATION_SEED,
    )
    tiebreak_rows = sim.tie_break_sweep(
        TIEBREAK_PREVALENCE,
        TIEBREAK_ACCURACIES,
        TIEBREAK_Q,
        N_TRIALS,
        TIEBREAK_SEED,
    )
    robustness_rows = sim.ensemble_robustness_sweep(
        ROBUSTNESS_ENSEMBLES,
        ROBUSTNESS_Q_VALUES,
        ROBUSTNESS_N_TRIALS,
        ROBUSTNESS_SEED,
    )
    mvei_rows = sim.mv_vs_ei_sweep(
        MVEI_PREVALENCES,
        MVEI_ACCURACIES,
        MVEI_Q,
        MVEI_N_TRIALS,
        MVEI_SEED,
    )

    independent = next(r for r in correlation_rows if r["label"] == "independent")
    common = [r for r in correlation_rows if r["label"].startswith("common")]
    anti = [r for r in correlation_rows if r["label"].startswith("anti")]
    best_row = min(
        (r for r in convergence_rows if r["accuracy_mae_mean"] is not None),
        key=lambda r: r["q"],
    )
    largest_row = max(
        (r for r in convergence_rows if r["accuracy_mae_mean"] is not None),
        key=lambda r: r["q"],
    )

    return {
        "study": "synthetic_recovery_study",
        "deterministic": True,
        "ensemble": _ensemble_block(),
        "convergence": {
            "n_trials": N_TRIALS,
            "seed": CONVERGENCE_SEED,
            "rows": convergence_rows,
            "sqrt_q_fit_accuracy": fit_acc,
            "sqrt_q_fit_prevalence": fit_prev,
            "q_min": min(Q_VALUES),
            "q_max": max(Q_VALUES),
            "smallest_q_acc_mae": best_row["accuracy_mae_mean"],
            "largest_q_acc_mae": largest_row["accuracy_mae_mean"],
        },
        "correlation": {
            "q": CORRELATION_Q,
            "n_trials": N_TRIALS,
            "seed": CORRELATION_SEED,
            "rows": correlation_rows,
            "independent_false_alarm_rate": independent["alarm_rate"],
            "common_mode_max_alarm_rate": max(r["alarm_rate"] for r in common),
            "common_mode_max_silent_error": max(
                r["silent_prevalence_abs_error_mean"]
                for r in common
                if r["silent_prevalence_abs_error_mean"] is not None
            ),
            "anti_max_alarm_rate": max(r["alarm_rate"] for r in anti),
        },
        "tie_break": {
            "prevalence_a": TIEBREAK_PREVALENCE,
            "q": TIEBREAK_Q,
            "n_trials": N_TRIALS,
            "seed": TIEBREAK_SEED,
            "rows": tiebreak_rows,
            # the lowest accuracy that still resolves cleanly (rate < 1%) ...
            "safe_accuracy": min(
                (r["accuracy"] for r in tiebreak_rows if r["catastrophic_rate"] < 0.01),
                default=None,
            ),
            # ... and the worst observed inversion rate as judges approach random
            "max_catastrophic_rate": max(r["catastrophic_rate"] for r in tiebreak_rows),
        },
        "robustness": {
            "q_values": ROBUSTNESS_Q_VALUES,
            "n_trials": ROBUSTNESS_N_TRIALS,
            "seed": ROBUSTNESS_SEED,
            "rows": robustness_rows,
            "slope_min": min(r["slope"] for r in robustness_rows),
            "slope_max": max(r["slope"] for r in robustness_rows),
            "n_ensembles": len(robustness_rows),
        },
        "mv_vs_ei": _mv_vs_ei_block(mvei_rows),
    }


def _mv_vs_ei_block(rows: list[dict]) -> dict:
    """Summarise the EI-vs-MV grid: the regime EI wins and the regime MV wins."""
    solved = [r for r in rows if r["ei_win_rate"] is not None]
    # Where EI helps most: highest MV/EI MAE ratio (EI clearly more accurate).
    best_ei = max(solved, key=lambda r: (r["mv_mae_mean"] / r["ei_mae_mean"]))
    # Where MV helps most: lowest EI win-rate (the tie-break danger zone).
    best_mv = min(solved, key=lambda r: r["ei_win_rate"])
    return {
        "q": MVEI_Q,
        "n_trials": MVEI_N_TRIALS,
        "seed": MVEI_SEED,
        "prevalences": MVEI_PREVALENCES,
        "accuracies": MVEI_ACCURACIES,
        "rows": rows,
        "ei_best_cell": {
            "prevalence_a": best_ei["prevalence_a"],
            "accuracy": best_ei["accuracy"],
            "ei_mae_mean": best_ei["ei_mae_mean"],
            "mv_mae_mean": best_ei["mv_mae_mean"],
            "ei_win_rate": best_ei["ei_win_rate"],
            "ratio_mv_over_ei": best_ei["mv_mae_mean"] / best_ei["ei_mae_mean"],
        },
        "mv_best_cell": {
            "prevalence_a": best_mv["prevalence_a"],
            "accuracy": best_mv["accuracy"],
            "ei_mae_mean": best_mv["ei_mae_mean"],
            "mv_mae_mean": best_mv["mv_mae_mean"],
            "ei_win_rate": best_mv["ei_win_rate"],
        },
    }


def _provenance(summary: dict) -> str:
    e = summary["ensemble"]
    accs = ", ".join(
        f"J{j['judge']}({j['alpha']:.2f},{j['beta']:.2f})" for j in e["accuracies"]
    )
    return (
        f"synthetic NTQR study | p_a={e['prevalence_a']:.2f} | {accs} | "
        f"{summary['convergence']['n_trials']} trials/point | "
        f"{datetime.now(timezone.utc).date()}"
    )


def _annotate_sidecars(summary: dict) -> None:
    """Append figure documentation fields without changing existing data keys."""
    generated = datetime.now(timezone.utc).isoformat()
    common_metadata = {
        "generated_utc": generated,
        "study": summary["study"],
        "deterministic": summary["deterministic"],
        "ensemble": summary["ensemble"],
    }
    summary["convergence"].update(
        {
            "figure": "synthetic_convergence.png",
            "title": "Synthetic convergence of NTQR recovery error",
            "caption": (
                "Synthetic error-independent ensembles show accuracy MAE "
                "falling close to the expected 1/sqrt(Q) sampling-noise rate."
            ),
            "x_label": "Test size Q (log scale)",
            "y_label": "Unsupervised recovery error (log scale)",
            "thresholds": [
                {
                    "label": "sampling-noise slope",
                    "value": -0.5,
                    "meaning": "ideal log-log slope for 1/sqrt(Q) error decay",
                }
            ],
            "metadata": {
                **common_metadata,
                "n_trials_per_q": summary["convergence"]["n_trials"],
                "seed": summary["convergence"]["seed"],
            },
        }
    )
    summary["correlation"].update(
        {
            "figure": "synthetic_alarm.png",
            "title": "Imaginary-root alarm and common-mode blind spot",
            "caption": (
                "The alarm has zero false positives under independence and "
                "fires on anti-correlated errors, while positive common-mode "
                "dependence remains a documented blind spot."
            ),
            "x_label": "Dependence condition",
            "y_label": "Alarm rate and silent prevalence error",
            "thresholds": [
                {
                    "label": "imaginary-root alarm",
                    "value": "D < 0",
                    "meaning": (
                        "negative prevalence-quadratic discriminant triggers "
                        "the logical alarm"
                    ),
                }
            ],
            "metadata": {
                **common_metadata,
                "q": summary["correlation"]["q"],
                "n_trials": summary["correlation"]["n_trials"],
                "seed": summary["correlation"]["seed"],
            },
        }
    )
    summary["tie_break"].update(
        {
            "figure": "synthetic_tiebreak.png",
            "title": "Better-than-random branch-selection boundary",
            "caption": (
                "The unsupervised branch rule is stable for strong judges and "
                "fails as symmetric judge accuracy approaches random guessing."
            ),
            "x_label": "Per-label judge accuracy",
            "y_label": "Catastrophic mirror-selection rate",
            "thresholds": [
                {
                    "label": "random guessing",
                    "value": 0.5,
                    "meaning": (
                        "symmetric judges at 0.5 accuracy provide no "
                        "better-than-random branch signal"
                    ),
                }
            ],
            "metadata": {
                **common_metadata,
                "q": summary["tie_break"]["q"],
                "n_trials": summary["tie_break"]["n_trials"],
                "seed": summary["tie_break"]["seed"],
            },
        }
    )
    summary["robustness"].update(
        {
            "title": "Synthetic convergence robustness sweep",
            "caption": (
                "Additional ensembles confirm that the convergence slope is "
                "not an artifact of a single synthetic parameter setting."
            ),
            "metadata": {
                **common_metadata,
                "n_trials": summary["robustness"]["n_trials"],
                "seed": summary["robustness"]["seed"],
            },
        }
    )
    summary["mv_vs_ei"].update(
        {
            "figure": "synthetic_mv_vs_ei.png",
            "title": "Error-independent versus majority-voting recovery",
            "caption": (
                "The error-independent estimator is more accurate when its "
                "assumption holds, while majority voting is more robust in the "
                "weak-judge, extreme-prevalence danger zone."
            ),
            "x_label": "Per-label judge accuracy",
            "y_label": "Recovery MAE (log scale)",
            "thresholds": [
                {
                    "label": "EI win rate",
                    "value": "reported per grid cell",
                    "meaning": (
                        "fraction of trials in which error-independent "
                        "recovery has lower MAE than majority voting"
                    ),
                }
            ],
            "metadata": {
                **common_metadata,
                "q": summary["mv_vs_ei"]["q"],
                "n_trials": summary["mv_vs_ei"]["n_trials"],
                "seed": summary["mv_vs_ei"]["seed"],
            },
        }
    )


def main() -> int:
    out_dir = _common.output_dir("sim", "synthetic_recovery_study")
    _common.banner("Synthetic NTQR recovery study (deterministic)")

    summary = run_study()
    _annotate_sidecars(summary)
    footer = _provenance(summary)

    summary_path = _common.save_json(out_dir, "summary.json", summary)
    conv_path = _common.save_json(
        out_dir, "convergence.data.json", summary["convergence"]
    )
    corr_path = _common.save_json(
        out_dir, "correlation.data.json", summary["correlation"]
    )
    tb_path = _common.save_json(out_dir, "tiebreak.data.json", summary["tie_break"])
    rob_path = _common.save_json(out_dir, "robustness.data.json", summary["robustness"])
    mvei_path = _common.save_json(out_dir, "mv_vs_ei.data.json", summary["mv_vs_ei"])
    fig1 = _plot_convergence(summary, out_dir, footer)
    fig2 = _plot_alarm(summary, out_dir, footer)
    fig3 = _plot_tiebreak(summary, out_dir, footer)
    fig4 = _plot_mv_vs_ei(summary, out_dir, footer)

    fit = summary["convergence"]["sqrt_q_fit_accuracy"]
    print(
        f"convergence: acc-MAE slope {fit['slope']:.3f} "
        f"(R^2={fit['r_squared']:.4f}); "
        f"Q={summary['convergence']['q_min']} -> "
        f"{summary['convergence']['smallest_q_acc_mae']:.4f}, "
        f"Q={summary['convergence']['q_max']} -> "
        f"{summary['convergence']['largest_q_acc_mae']:.4f}"
    )
    print(
        f"alarm: independent false-alarm "
        f"{summary['correlation']['independent_false_alarm_rate']*100:.1f}%, "
        f"common-mode max {summary['correlation']['common_mode_max_alarm_rate']*100:.1f}%, "
        f"anti-pair max {summary['correlation']['anti_max_alarm_rate']*100:.1f}%"
    )
    rob = summary["robustness"]
    print(
        f"robustness: {rob['n_ensembles']} ensembles, "
        f"slope band [{rob['slope_min']:.3f}, {rob['slope_max']:.3f}]; "
        f"tie-break clean to acc={summary['tie_break']['safe_accuracy']}, "
        f"max inversion {summary['tie_break']['max_catastrophic_rate']*100:.0f}%"
    )
    mvei = summary["mv_vs_ei"]
    ei_c, mv_c = mvei["ei_best_cell"], mvei["mv_best_cell"]
    print(
        f"mv_vs_ei: EI best @ p={ei_c['prevalence_a']:.2f}/acc={ei_c['accuracy']:.2f} "
        f"(MV/EI {ei_c['ratio_mv_over_ei']:.2f}x, EI wins {ei_c['ei_win_rate']*100:.0f}%); "
        f"MV best @ p={mv_c['prevalence_a']:.2f}/acc={mv_c['accuracy']:.2f} "
        f"(EI wins {mv_c['ei_win_rate']*100:.0f}%)"
    )
    for path in (
        summary_path,
        conv_path,
        corr_path,
        tb_path,
        rob_path,
        mvei_path,
        fig1,
        fig2,
        fig3,
        fig4,
    ):
        print(str(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
