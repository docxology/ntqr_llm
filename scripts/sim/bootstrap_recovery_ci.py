"""Scenario-bootstrap confidence interval for the real evaluable model.

Reads the cached votes of the evaluable model (from the LLM sweep), resamples the
authored scenarios with replacement, and re-runs the same ntqr error-independent
solve on each resample — giving the headline recovery MAE a confidence interval
and a degenerate-rate, with **no Ollama and no new model calls**. It reads only
committed artifacts, so it is deterministic and reproducible offline.

Run:
    .venv/bin/python scripts/sim/bootstrap_recovery_ci.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _common  # noqa: E402

_common.ensure_ntqr_importable()

from ntqr.llm import bootstrap  # noqa: E402
from ntqr.llm import personas  # noqa: E402

N_BOOT = 2000
SEED = 9000
LLM_OUT = _common.OUTPUTS_DIR / "llm" / "multi_llm_personas_evaluation"


from ntqr.llm.supporting_figures import plot_bootstrap as _plot


def _load_result(path: Path) -> dict:
    doc = json.loads(path.read_text())
    return doc.get("result", doc) if isinstance(doc, dict) else doc


def _model_dir(model: str) -> Path:
    return LLM_OUT / "models" / model.replace(":", "_").replace("/", "_")


def reconstruct_records(model: str) -> list[tuple[str, tuple]]:
    """Rebuild ``(truth_label, vote_pattern)`` records from cached votes + truth.

    Uses the authored truth in ``personas.SCENARIOS`` and the model's
    ``by_scenario`` votes; the persona order matches ``personas.PERSONAS``.
    """
    votes = _load_result(_model_dir(model) / "votes.json")["by_scenario"]
    truth = {s["id"]: personas.CALL_TO_LABEL[s["truth"]] for s in personas.SCENARIOS}
    order = [p["key"] for p in personas.PERSONAS]
    records = []
    for sid, calls in votes.items():
        pattern = tuple(personas.CALL_TO_LABEL[calls[k]] for k in order)
        records.append((truth[sid], pattern))
    return records


def main() -> int:
    out_dir = _common.output_dir("sim", "bootstrap_recovery_ci")
    _common.banner("Scenario-bootstrap recovery CI (deterministic, offline)")

    sweep = _load_result(LLM_OUT / "summary.json")
    model = sweep["best_model"]
    floor = sweep["cross_model_statistics"]["sampling_noise_floor"]

    records = reconstruct_records(model)
    summary = bootstrap.bootstrap_recovery(records, N_BOOT, SEED)
    samples = summary.pop("mae_samples")
    summary["model"] = model
    summary["sampling_noise_floor"] = floor

    footer = (
        f"scenario bootstrap | model {model} | {N_BOOT} resamples | "
        f"{datetime.now(timezone.utc).date()}"
    )
    summary_path = _common.save_json(out_dir, "summary.json", summary)

    plot_summary = dict(summary)
    plot_summary["_mae_samples"] = samples
    fig = _plot(plot_summary, out_dir, footer)
    sidecar = _common.save_json(
        out_dir,
        "bootstrap_recovery_mae.data.json",
        {
            "figure": "bootstrap_recovery_mae.png",
            "title": "Recovery MAE under scenario resampling",
            "caption": (
                "Scenario bootstrap distribution for the headline recovery "
                "MAE; the confidence interval remains below the conservative "
                "sampling-noise floor."
            ),
            "x_label": "Unsupervised recovery MAE",
            "y_label": "Bootstrap resamples",
            "samples": samples,
            "point_recovery_mae": summary["point_recovery_mae"],
            "recovery_mae": summary["recovery_mae"],
            "prevalence_abs_error": summary["prevalence_abs_error"],
            "degenerate_count": summary["degenerate_count"],
            "degenerate_rate": summary["degenerate_rate"],
            "thresholds": [
                {
                    "value": floor,
                    "label": "sampling-noise floor",
                    "meaning": (
                        "worst-case binomial standard error of the "
                        "supervised per-label accuracy estimate"
                    ),
                }
            ],
            "metadata": {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "model": model,
                "n_scenarios": summary["n_scenarios"],
                "n_boot": N_BOOT,
                "seed": SEED,
            },
        },
    )

    mae = summary["recovery_mae"]
    print(
        f"model {model}: point MAE {summary['point_recovery_mae']:.4f}; "
        f"95% CI [{mae['ci_lo']:.4f}, {mae['ci_hi']:.4f}]; "
        f"degenerate {summary['degenerate_rate'] * 100:.1f}%"
    )
    for path in (summary_path, sidecar, fig):
        print(str(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
