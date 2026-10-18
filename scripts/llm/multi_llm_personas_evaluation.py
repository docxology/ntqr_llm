"""Method module: the trader-persona ntqr demo across SEVERAL real LLMs.

``trader_personas_evaluation.py`` runs the three trader personas through one
model. This module runs the same three-persona / many-scenario grid through a
*configurable list of models* and asks a sharper question:

    Across different LLMs -- strong and weak -- how well does ntqr's
    *unsupervised* algebraic evaluation recover what each persona's accuracy
    *actually* was (which we know only because the scenarios are authored)?

For every model we:
  1. collect the trio's votes (cached, shared cache across models),
  2. recover persona accuracies with ``ErrorIndependentEvaluation`` -- no
     answer key,
  3. compute the supervised truth as a check,
  4. score the recovery: the mean absolute error between recovered and true
     per-persona accuracies (the "recovery MAE"), and the prevalence error.

Then we compare models head to head: a ranking table in ``summary.json`` and
two plots -- recovery MAE per model, and a recovered-vs-true scatter that
should hug the ``y = x`` line if the unsupervised math is tracking truth.

A model whose votes are degenerate (e.g. a weak model that answers the same
way every time) can make the algebra unsolvable; that model is recorded with
its error and excluded from the ranking and plots rather than crashing the run.

Config (``--set k=v`` / ``--config FILE.json``):
    backend       "ollama" (default) | "anthropic"
    models        JSON list of model names, e.g.
                  --set models='["granite4.1:3b","mistral:latest"]'
    temperature   sampling temperature (default 0.0)
    refresh       true to ignore the response cache
    timeout       per-call timeout in seconds (default 300, for cold loads)

A model that times out or is otherwise unreachable is isolated like a failed
evaluation -- recorded under ``failed_models`` and skipped -- so one slow model
never aborts the sweep. Partial votes are cached, so a rerun resumes.

Run:
    python scripts/llm/multi_llm_personas_evaluation.py
    python scripts/run_all.py --include llm
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _common  # noqa: E402
import _config  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

_common.ensure_ntqr_importable()

import _analysis  # noqa: E402  (thin collect-votes glue)
import _llm  # noqa: E402
from ntqr.llm import analysis  # noqa: E402
from ntqr.llm import personas  # noqa: E402

from ntqr.r2.datasketches import TrioLabelVoteCounts  # noqa: E402
from ntqr.r2.evaluators import (  # noqa: E402
    ErrorIndependentEvaluation,
    MajorityVotingEvaluation,
    SupervisedEvaluation,
)

REQUIRES = ["ollama"]

# A spread of locally-runnable chat models, small to large, so the sweep spans
# a range of capabilities. Votes are never fabricated: a model that emits an
# unparseable reply on any scenario is reported non-compliant (see
# evaluate_model), so every solved row rests on real, parsed model responses.
from ntqr.llm.visualization import (
    BASELINE_SUMMARY,
    DEFAULT_MODELS,
    plot_models as _plot,
    rank_models as _rank,
)


def _model_dirname(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


def _ollama_has(available: list[str], model: str) -> bool:
    return model in available or any(n.startswith(model) for n in available)


def _select_models(backend: str, models: list[str]):
    """Return (present, missing): which requested models can actually run."""
    if backend != "ollama":
        return list(models), []
    available = _llm.ollama_tags()
    present = [m for m in models if _ollama_has(available, m)]
    missing = [m for m in models if not _ollama_has(available, m)]
    return present, missing


def _save_cache(cache_path: Path, cache: dict) -> None:
    from ntqr.llm.artifacts import save_json

    save_json(cache_path, cache)


def _result_body(path: Path) -> dict:
    payload = json.loads(path.read_text())
    return payload.get("result", payload)


def _preserve_temperature_baseline(out: Path, temperature: float) -> None:
    """Save the pre-rerun t0.7 summary before writing the new headline."""
    summary_path = out / "summary.json"
    baseline_path = out / BASELINE_SUMMARY
    if not summary_path.exists() or baseline_path.exists():
        return
    prior = _result_body(summary_path)
    if prior.get("temperature") == temperature:
        return
    try:
        prior_temp = float(prior.get("temperature", -1))
    except (TypeError, ValueError):
        prior_temp = -1
    if prior_temp == 0.7:
        shutil.copy2(summary_path, baseline_path)


def evaluate_model(client, cache, refresh, out_dir, cache_path=None):
    """Run the full persona->ntqr pipeline for one model; persist + return row.

    Returns a dict row for the cross-model table. On an unsolvable evaluation
    the row carries an ``error`` key instead of statistics (and is dropped from
    the ranking/plots), so one bad model never sinks the whole run.
    """

    def _persist(updated):
        if cache_path is not None:
            _save_cache(cache_path, updated)

    votes, unparsed, cache = _analysis.collect_votes(
        client,
        cache,
        refresh,
        on_error="tolerate",
        on_cache_update=_persist,
    )
    records = analysis.response_records(client, cache)
    prompt_health = analysis.prompt_health(votes, records)
    label_counts = analysis.build_label_counts(votes)
    _config.save_label_vote_counts(out_dir / "label_vote_counts.json", label_counts)
    _common.save_json(
        out_dir,
        "votes.json",
        {
            "backend": client.backend,
            "model": client.model,
            "prompt_version": personas.PROMPT_VERSION,
            "prompt_hash": personas.prompt_hash(),
            "personas": [p["name"] for p in personas.PERSONAS],
            "unparsed_responses": unparsed,
            "by_scenario": votes,
            "response_records": records,
            "prompt_health": prompt_health,
        },
    )
    _common.save_json(out_dir, "prompt_health.json", prompt_health)

    # Label-free evaluability diagnostic: computed for EVERY model (even ones
    # that will fail the solve) because its whole purpose is to PREDICT the
    # failure from the votes alone, with no answer key.
    diagnostic = analysis.evaluability_diagnostic(votes)
    _common.save_json(out_dir, "evaluability_diagnostic.json", diagnostic)
    base = {
        "model": client.model,
        "unparsed_responses": unparsed,
        "predicted_evaluable": diagnostic["predicted_evaluable"],
        "constant_classifiers": diagnostic["constant_classifiers"],
        "mean_pairwise_disagreement": (diagnostic["mean_pairwise_disagreement"]),
        "prompt_health": prompt_health,
    }

    # No fabrication: a model that left any cell unparseable is reported
    # as non-compliant rather than solved on invented votes. Its votes and
    # label-free diagnostic are still saved above, so it still feeds the
    # disagreement comparison.
    if unparsed:
        return {
            **base,
            "error": (
                f"non-compliant: {unparsed} unparseable response(s) "
                "(votes are never fabricated)"
            ),
        }

    trio_label = TrioLabelVoteCounts(label_counts)
    trio_votes = trio_label.to_TrioVoteCounts()
    try:
        error_independent = ErrorIndependentEvaluation(trio_votes)
        supervised = SupervisedEvaluation(trio_label)
        comparison = analysis.build_comparison(
            supervised, error_independent, physical_only=True
        )
        stats = analysis.recovery_stats(comparison)
    except Exception as exc:  # invalid or singular votes -> unavailable EI evaluation
        return {**base, "error": repr(exc)}
    # A baseline failure does not erase a complete physical EI recovery.
    try:
        majority = MajorityVotingEvaluation(trio_votes)
        mv_comparison = analysis.build_comparison(
            supervised, majority, physical_only=True
        )
        mv_stats = analysis.recovery_stats(mv_comparison)
    except ValueError as exc:
        mv_comparison = {"error": str(exc)}
        mv_stats = {"recovery_mae": None, "error": str(exc)}

    _common.save_json(
        out_dir,
        "evaluation_error_independent.json",
        {
            "exact": error_independent.evaluation_exact,
            "float": error_independent.evaluation_float,
        },
    )
    _common.save_json(
        out_dir,
        "evaluation_majority_voting.json",
        {
            "exact": majority.evaluation_exact,
            "float": majority.evaluation_float,
        },
    )
    _common.save_json(
        out_dir,
        "evaluation_supervised.json",
        {
            "exact": supervised.evaluation_exact,
            "float": supervised.evaluation_float,
        },
    )
    _common.save_json(out_dir, "comparison.json", comparison)
    _common.save_json(out_dir, "recovery_stats.json", stats)
    _common.save_json(out_dir, "comparison_majority_voting.json", mv_comparison)
    _common.save_json(out_dir, "recovery_stats_majority_voting.json", mv_stats)

    if stats["recovery_mae"] is None:
        # All recovered accuracies were non-finite: the algebra was unsolvable,
        # almost always because a persona voted one-sidedly (constant
        # classifier). Report it as a named failure, not a NaN statistic.
        consts = analysis.constant_personas(label_counts)
        reason = "degenerate (unsolvable / NaN solution)"
        if consts:
            reason += "; constant classifier(s): " + ", ".join(consts)
        return {**base, "error": reason}

    return {
        **base,
        "true_prevalence_a": comparison["true_prevalence_a"],
        "recovered_prevalence_a": comparison["recovered_prevalence_a"],
        "prevalence_abs_error": stats["prevalence_abs_error"],
        "recovery_mae": stats["recovery_mae"],
        "recovery_mae_by_label": stats["recovery_mae_by_label"],
        "max_abs_error": stats["max_abs_error"],
        "worst_term": stats["worst_term"],
        "mv_recovery_mae": mv_stats["recovery_mae"],
        "comparison": comparison,  # kept in-memory for plotting; not serialized
    }


def main(argv: list[str] | None = None) -> None:
    config = _config.build_config(
        "llm",
        "multi_llm_personas_evaluation",
        default_params={
            "backend": "ollama",
            "models": DEFAULT_MODELS,
            "temperature": 0.1,
            "refresh": False,
            "timeout": 300.0,
        },
        argv=argv,
    )
    out = _common.output_dir(config.category, config.name)
    backend = config.get("backend", "ollama")
    models = config.get("models", DEFAULT_MODELS)
    temperature = config.get("temperature", 0.0)
    refresh = bool(config.get("refresh", False))
    timeout = float(config.get("timeout", 300.0))

    present, missing = _select_models(backend, models)
    if not present:
        raise _llm.LLMUnavailable(
            f"none of the requested models are available: {models}"
        )

    cache_path = out / "cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    _preserve_temperature_baseline(out, temperature)

    rows = []
    for model in present:
        client = _llm.LLMClient(
            backend=backend,
            model=model,
            temperature=temperature,
            timeout=timeout,
        )
        model_dir = out / "models" / _model_dirname(model)
        model_dir.mkdir(parents=True, exist_ok=True)
        try:
            client.preflight()
            row = evaluate_model(client, cache, refresh, model_dir, cache_path)
        except Exception as exc:  # unreachable/slow model -> isolate, continue
            row = {"model": model, "error": repr(exc)}
        finally:
            # Persist whatever calls succeeded so a rerun resumes from here.
            _save_cache(cache_path, cache)
        rows.append(row)

    ranked = _rank(rows)
    maes = [r["recovery_mae"] for r in ranked]
    failed_rows = [r for r in rows if "error" in r]
    # Drop the in-memory comparison blob before serializing the summary. Keep
    # the JSON field true to its name: evaluable rows are ordered by recovery
    # MAE, followed by failed/unevaluable rows for diagnostics.
    table = [
        {k: v for k, v in r.items() if k != "comparison"}
        for r in [*ranked, *failed_rows]
    ]

    # Did the label-free diagnostic predict which models would be evaluable?
    evaluated = {r["model"] for r in ranked}
    predicted = {r["model"] for r in rows if r.get("predicted_evaluable")}
    # Only models we actually got votes for carry a prediction; a model that
    # was unreachable (no votes) has neither, and is excluded from the check.
    diagnosed = {r["model"] for r in rows if "predicted_evaluable" in r}

    # Cross-model statistics: spread of the recovery MAEs, and -- the headline
    # contrast -- the disagreement of EVALUABLE vs UNEVALUABLE models. If
    # evaluability tracked disagreement, the evaluable group would disagree
    # more; in practice it disagrees LESS, which is the non-obvious finding.
    dis_eval = [
        r["mean_pairwise_disagreement"]
        for r in rows
        if r["model"] in evaluated and r.get("mean_pairwise_disagreement") is not None
    ]
    dis_uneval = [
        r["mean_pairwise_disagreement"]
        for r in rows
        if "predicted_evaluable" in r
        and r["model"] not in evaluated
        and r.get("mean_pairwise_disagreement") is not None
    ]
    disagreement_gates = (
        bool(dis_eval) and bool(dis_uneval) and (min(dis_eval) >= max(dis_uneval))
    )
    if disagreement_gates:
        disagreement_note = (
            "In this run, aggregate disagreement also separates the groups "
            "because the unevaluable model(s) are constant-classifier "
            "collapses with low/no disagreement. The structural gate remains "
            "per-judge variation and complete votes, not a universal "
            "ensemble-disagreement threshold."
        )
    else:
        disagreement_note = (
            "In this run, disagreement does not separate evaluable and "
            "unevaluable models. Evaluability is gated by per-judge variation "
            "and complete votes, not by ensemble disagreement."
        )
    cross_model_statistics = {
        "recovery_mae": analysis.descriptive_stats(maes),
        "mv_recovery_mae": analysis.descriptive_stats(
            [r.get("mv_recovery_mae") for r in ranked]
        ),
        "prevalence_abs_error": analysis.descriptive_stats(
            [r.get("prevalence_abs_error") for r in ranked]
        ),
        "sampling_noise_floor": analysis.sampling_noise_floor(
            len(personas.SCENARIOS),
            min_label_count=analysis.min_truth_label_count(),
        ),
        "disagreement_evaluable": analysis.descriptive_stats(dis_eval),
        "disagreement_unevaluable": analysis.descriptive_stats(dis_uneval),
        "disagreement_gates_evaluability": disagreement_gates,
        "note": disagreement_note,
    }
    summary = {
        "backend": backend,
        "temperature": temperature,
        "prompt_version": personas.PROMPT_VERSION,
        "prompt_hash": personas.prompt_hash(),
        "prompt_payload": personas.prompt_payload(),
        "requested_models": models,
        "evaluated_models": [r["model"] for r in ranked],
        "skipped_unavailable": missing,
        "failed_models": [r["model"] for r in rows if "error" in r],
        "n_scenarios": len(personas.SCENARIOS),
        "best_model": ranked[0]["model"] if ranked else None,
        "worst_model": ranked[-1]["model"] if ranked else None,
        "mean_recovery_mae": (sum(maes) / len(maes)) if maes else None,
        "cross_model_statistics": cross_model_statistics,
        "evaluability_diagnostic": {
            "note": (
                "predicted_evaluable is derived from votes alone (no answer "
                "key): an ensemble is evaluable iff every persona varied. "
                "diagnostic_matched_outcome is True when that label-free "
                "prediction agrees with which models the ntqr solve actually "
                "evaluated."
            ),
            "predicted_evaluable": sorted(predicted),
            "actually_evaluated": sorted(evaluated & diagnosed),
            "diagnostic_matched_outcome": (predicted == (evaluated & diagnosed)),
        },
        "ranking_by_recovery_mae": table,
    }
    _common.save_json(out, "summary.json", summary)
    plot_ctx = {
        "backend": backend,
        "temperature": temperature,
        "n_personas": len(personas.PERSONAS),
        "n_scenarios": len(personas.SCENARIOS),
        "n_labels": len(personas.LABELS),
        "n_models_requested": len(models),
        "n_models_evaluated": len(ranked),
        "prompt_version": personas.PROMPT_VERSION,
        "prompt_hash": personas.prompt_hash(),
        "baseline_summary": (
            BASELINE_SUMMARY if (out / BASELINE_SUMMARY).exists() else None
        ),
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }
    plots = _plot(rows, out, plot_ctx)

    _common.banner(
        f"multi-LLM personas via {backend} "
        f"({len(present)} models, {len(personas.SCENARIOS)} scenarios)"
    )
    if missing:
        print(f"skipped (not pulled): {', '.join(missing)}")
    for r in rows:
        disagree = r.get("mean_pairwise_disagreement")
        tail = f"  disagreement={disagree:.2f}" if disagree is not None else ""
        if "error" in r:
            print(f"  {r['model']:<32} FAILED ({r['error']}){tail}")
        else:
            print(
                f"  {r['model']:<32} "
                f"recovery MAE={r['recovery_mae']:.3f}  "
                f"prevalence err={r['prevalence_abs_error']:.3f}  "
                f"unparsed={r['unparsed_responses']}{tail}"
            )
    if ranked:
        print(f"best: {ranked[0]['model']} (MAE {ranked[0]['recovery_mae']:.3f})")
    for p in plots:
        print(f"-> plot {p.name}")
    print(f"-> wrote results to {out}")


if __name__ == "__main__":
    main()
