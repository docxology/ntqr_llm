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
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _common  # noqa: E402
import _config  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _analysis  # noqa: E402
import _llm  # noqa: E402
import _personas  # noqa: E402

_common.ensure_ntqr_importable()

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
DEFAULT_MODELS = [
    "granite4.1:3b",
    "mistral:latest",
    "gemma2:2b",
    "gemma3:4b",
    "gemma4:latest",
    "smollm2:135m-instruct-q4_K_S",
]

BASELINE_SUMMARY = "baseline_t0.7_summary.json"


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
    cache_path.write_text(json.dumps(cache, indent=2) + "\n")


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
    records = _analysis.response_records(client, cache)
    prompt_health = _analysis.prompt_health(votes, records)
    label_counts = _analysis.build_label_counts(votes)
    _config.save_label_vote_counts(
        out_dir / "label_vote_counts.json", label_counts
    )
    _common.save_json(
        out_dir,
        "votes.json",
        {
            "backend": client.backend,
            "model": client.model,
            "prompt_version": _personas.PROMPT_VERSION,
            "prompt_hash": _personas.prompt_hash(),
            "personas": [p["name"] for p in _personas.PERSONAS],
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
    diagnostic = _analysis.evaluability_diagnostic(votes)
    _common.save_json(out_dir, "evaluability_diagnostic.json", diagnostic)
    base = {
        "model": client.model,
        "unparsed_responses": unparsed,
        "predicted_evaluable": diagnostic["predicted_evaluable"],
        "constant_classifiers": diagnostic["constant_classifiers"],
        "mean_pairwise_disagreement": (
            diagnostic["mean_pairwise_disagreement"]
        ),
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
        majority = MajorityVotingEvaluation(trio_votes)
        supervised = SupervisedEvaluation(trio_label)
        comparison = _analysis.build_comparison(supervised, error_independent)
        stats = _analysis.recovery_stats(comparison)
        # Head-to-head: how well does the simpler majority-voting evaluator
        # recover the same truth? Both return two solutions, so build_comparison
        # selects each one's better-than-random branch identically.
        mv_comparison = _analysis.build_comparison(supervised, majority)
        mv_stats = _analysis.recovery_stats(mv_comparison)
    except Exception as exc:  # degenerate votes -> unsolvable algebra
        return {**base, "error": repr(exc)}

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
    _common.save_json(
        out_dir, "comparison_majority_voting.json", mv_comparison
    )
    _common.save_json(out_dir, "recovery_stats_majority_voting.json", mv_stats)

    if stats["recovery_mae"] is None:
        # All recovered accuracies were non-finite: the algebra was unsolvable,
        # almost always because a persona voted one-sidedly (constant
        # classifier). Report it as a named failure, not a NaN statistic.
        consts = _analysis.constant_personas(label_counts)
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


def _rank(rows):
    """Models that evaluated, sorted by recovery MAE (best first)."""
    scored = [
        r
        for r in rows
        if "recovery_mae" in r and r["recovery_mae"] is not None
    ]
    return sorted(scored, key=lambda r: r["recovery_mae"])


def _fig_metadata(extra=None, ctx=None):
    """Shared provenance block stamped into every figure data sidecar."""
    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_scenarios": len(_personas.SCENARIOS),
        "n_personas": len(_personas.PERSONAS),
        "personas": [p["name"] for p in _personas.PERSONAS],
        "labels": list(_personas.LABELS),
    }
    if ctx:
        meta["experiment"] = dict(ctx)
    if extra:
        meta.update(extra)
    return meta


def _provenance_line(ctx):
    """One-line experiment provenance footer drawn on every figure."""
    if not ctx:
        return ""
    bits = [
        f"backend={ctx.get('backend')}",
        f"T={ctx.get('temperature')}",
        f"N={ctx.get('n_personas')}",
        f"Q={ctx.get('n_scenarios')}",
        f"R={ctx.get('n_labels')}",
        f"{ctx.get('n_models_evaluated')}/{ctx.get('n_models_requested')} "
        "evaluable",
        "ntqr EI",
        str(ctx.get("date")),
    ]
    return "  •  ".join(b for b in bits if b)


def _plot(rows, out, ctx=None):
    """Render the cross-model plots and a data+metadata sidecar each.

    Every PNG is paired with a ``<stem>.data.json`` holding the exact series
    behind the figure, the interpretation thresholds drawn on it, and
    provenance metadata -- so the figures are reproducible and machine-readable,
    not just pixels. Imports matplotlib lazily.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ntqr.plots flips usetex on at import; force it off so we need no TeX.
    # Bump every default size so the figures read well at publication scale.
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.size": 13,
            "axes.titlesize": 17,
            "axes.titleweight": "bold",
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11.5,
            "figure.titlesize": 18,
        }
    )
    note_fs = 11  # in-axes explanatory annotations
    val_fs = 11  # bar value / point labels

    ranked = _rank(rows)
    if not ranked:
        return []
    written = []
    cmap = plt.get_cmap("tab10")
    n_scen = len(_personas.SCENARIOS)
    # Below this recovery MAE, the label-free recovery is within the binomial
    # sampling noise of the supervised accuracy it is compared against. The deck
    # is imbalanced, so the minority label backs the weakest estimate and sets
    # the conservative floor.
    min_label = _analysis.min_truth_label_count()
    noise_floor = _analysis.sampling_noise_floor(
        n_scen, min_label_count=min_label
    )
    excellent = _analysis.RECOVERY_MAE_EXCELLENT
    provenance = _provenance_line(ctx)
    evaluable_models = {x["model"] for x in ranked}
    n_evaluable = len(evaluable_models)
    n_total = len(rows)

    def _save(stem, fig, data, caption="", bottom=0.14, top=1.0):
        # Reserve bottom space for a descriptive caption + provenance footer,
        # then stamp both so every figure is self-documenting.
        fig.tight_layout(rect=[0, bottom, 1, top])
        if caption:
            fig.text(
                0.5,
                bottom * 0.62,
                caption,
                ha="center",
                va="center",
                fontsize=11.5,
                style="italic",
                color="#222222",
                wrap=True,
            )
        if provenance:
            fig.text(
                0.5,
                0.012,
                provenance,
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#777777",
            )
        png = out / f"{stem}.png"
        fig.savefig(png, dpi=150)
        plt.close(fig)
        data.setdefault("caption", caption)
        _common.save_json(out, f"{stem}.data.json", data)
        written.append(png)

    # M0. Methods overview: the data boundary is the whole point. This figure is
    # intentionally a flow diagram, not a statistical plot: it shows which
    # objects enter the unsupervised NTQR solve and where authored truth appears
    # only as a validation check.
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(13.2, 5.8))
    ax.set_axis_off()
    stages = [
        {
            "id": "scenario_facts",
            "label": "Scenario facts\n(authored truth hidden)",
            "xy": (0.04, 0.64),
            "color": "#DBEAFE",
        },
        {
            "id": "binary_prompt",
            "label": "v3 binary schema\nmodel must choose one label",
            "xy": (0.29, 0.64),
            "color": "#E0F2FE",
        },
        {
            "id": "parse_repair",
            "label": "Parse + one repair retry\nno local vote fabrication",
            "xy": (0.54, 0.64),
            "color": "#DCFCE7",
        },
        {
            "id": "vote_matrix",
            "label": "Complete vote matrix\n3 personas x Q scenarios",
            "xy": (0.79, 0.64),
            "color": "#FDE68A",
        },
        {
            "id": "trio_counts",
            "label": "Unlabeled TrioVoteCounts\nagreement sketch only",
            "xy": (0.54, 0.26),
            "color": "#FEF3C7",
        },
        {
            "id": "ntqr_solve",
            "label": "NTQR EI solve\ntwo roots or alarm",
            "xy": (0.29, 0.26),
            "color": "#EDE9FE",
        },
        {
            "id": "validation",
            "label": "Branch selection + validation\nMAE scored after solve",
            "xy": (0.04, 0.26),
            "color": "#FCE7F3",
        },
    ]
    box_w, box_h = 0.17, 0.18
    for stage in stages:
        x, y = stage["xy"]
        box = patches.FancyBboxPatch(
            (x, y),
            box_w,
            box_h,
            boxstyle="round,pad=0.018,rounding_size=0.02",
            facecolor=stage["color"],
            edgecolor="#334155",
            linewidth=1.2,
        )
        ax.add_patch(box)
        ax.text(
            x + box_w / 2,
            y + box_h / 2,
            stage["label"],
            ha="center",
            va="center",
            fontsize=10.5,
            color="#111827",
        )

    by_id = {s["id"]: s for s in stages}

    def arrow(start_id, end_id):
        sx, sy = by_id[start_id]["xy"]
        ex, ey = by_id[end_id]["xy"]
        if abs(ey - sy) <= 0.04 and ex > sx:
            start = (sx + box_w, sy + box_h / 2)
            end = (ex, ey + box_h / 2)
        elif abs(ey - sy) <= 0.04:
            # Bottom-row arrows travel leftward in the whitespace between
            # boxes, high enough to avoid the centered two-line labels.
            lane = 0.84
            start = (sx, sy + box_h * lane)
            end = (ex + box_w, ey + box_h * lane)
        elif ey < sy - 0.05:
            start = (sx + box_w / 2, sy)
            end = (ex + box_w / 2, ey + box_h)
        else:
            start = (sx + box_w / 2, sy + box_h)
            end = (ex + box_w / 2, ey)
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={
                "arrowstyle": "-|>",
                "lw": 1.4,
                "color": "#475569",
                "shrinkA": 4,
                "shrinkB": 4,
            },
        )

    for start, end in (
        ("scenario_facts", "binary_prompt"),
        ("binary_prompt", "parse_repair"),
        ("parse_repair", "vote_matrix"),
        ("vote_matrix", "trio_counts"),
        ("trio_counts", "ntqr_solve"),
        ("ntqr_solve", "validation"),
    ):
        arrow(start, end)
    ax.annotate(
        "",
        xy=(0.125, 0.44),
        xytext=(0.125, 0.64),
        arrowprops={
            "arrowstyle": "-|>",
            "lw": 1.2,
            "color": "#BE185D",
            "linestyle": "--",
        },
    )
    ax.text(
        0.17,
        0.49,
        "authored truth is held out\nuntil validation",
        ha="left",
        va="center",
        fontsize=9,
        color="#9D174D",
    )
    ax.set_title(
        "What enters the unsupervised solve, and what is validation-only",
        fontsize=16,
        fontweight="bold",
        pad=14,
    )
    caption = (
        "The NTQR solve sees only parsed votes summarized as unlabeled agreement "
        "counts. Authored truth is held out until after branch selection, where "
        "it scores recovery MAE for this controlled experiment."
    )
    _save(
        "evaluation_pipeline",
        fig,
        {
            "figure": "evaluation_pipeline.png",
            "title": (
                "What enters the unsupervised solve, and what is "
                "validation-only"
            ),
            "stages": [
                {"id": s["id"], "label": s["label"].replace("\n", " -> ")}
                for s in stages
            ],
            "edges": [
                ["scenario_facts", "binary_prompt"],
                ["binary_prompt", "parse_repair"],
                ["parse_repair", "vote_matrix"],
                ["vote_matrix", "trio_counts"],
                ["trio_counts", "ntqr_solve"],
                ["ntqr_solve", "validation"],
            ],
            "validation_only": [
                "authored truth",
                "SupervisedEvaluation",
                "recovery MAE",
            ],
            "metadata": _fig_metadata(ctx=ctx),
        },
        caption=caption,
        bottom=0.22,
    )

    # 0. THE HEADLINE: disagreement vs per-judge concentration. Each model is a
    # point; x = its most-concentrated judge's modal-vote fraction (1.0 = a
    # constant classifier exists), y = ensemble mean pairwise disagreement.
    # Evaluability is gated by the VERTICAL line at x = 1.0 (per-judge), not by
    # any horizontal disagreement threshold -- so evaluable points sit strictly
    # left of 1.0 while unevaluable ones pile up on it, at every y.
    concentration = []
    for r in rows:
        diag_path = (
            out
            / "models"
            / _model_dirname(r["model"])
            / "evaluability_diagnostic.json"
        )
        if (
            not diag_path.exists()
            or r.get("mean_pairwise_disagreement") is None
        ):
            continue
        payload = json.loads(diag_path.read_text())
        diag = payload.get("result", payload)
        max_modal = max(
            vv["modal_fraction"] for vv in diag["vote_variation"].values()
        )
        concentration.append(
            {
                "model": r["model"],
                "max_modal_fraction": max_modal,
                "mean_pairwise_disagreement": r["mean_pairwise_disagreement"],
                "evaluable": r["model"] in evaluable_models,
                "constant_classifiers": diag["constant_classifiers"],
            }
        )
    if concentration:
        fig, ax = plt.subplots(figsize=(11.2, 6.8))
        max_y = max(p["mean_pairwise_disagreement"] for p in concentration)
        y_top = max(0.28, min(1.02, max_y * 1.35))
        # Shade the unevaluable half-plane (a constant judge exists at x = 1.0).
        ax.axvspan(0.995, 1.18, color="#DC2626", alpha=0.07, zorder=0)
        ax.axvline(
            1.0,
            color="#DC2626",
            ls="--",
            lw=1.8,
        )
        # Stagger labels of points that collide near x = 1.0 so each stays
        # readable: sort the on-the-line failures by y and fan their labels out.
        on_line = sorted(
            (p for p in concentration if p["max_modal_fraction"] >= 0.999),
            key=lambda p: p["mean_pairwise_disagreement"],
        )
        fan = {id(p): (12, 10 + 18 * i) for i, p in enumerate(on_line)}
        free_points = [
            p
            for p in sorted(concentration, key=lambda q: q["model"])
            if id(p) not in fan
        ]
        free_offsets = {
            id(p): off
            for p, off in zip(
                free_points,
                [
                    (-16, 12),
                    (12, 14),
                    (-26, -16),
                    (14, -18),
                    (-18, 28),
                    (16, 28),
                ],
                strict=False,
            )
        }
        for pt in concentration:
            color = "#16A34A" if pt["evaluable"] else "#DC2626"
            marker = "o" if pt["evaluable"] else "X"
            ax.scatter(
                pt["max_modal_fraction"],
                pt["mean_pairwise_disagreement"],
                s=200,
                color=color,
                marker=marker,
                edgecolor="white",
                linewidth=1.4,
                zorder=3,
            )
            if id(pt) in fan:
                # Failures stacked on x = 1.0: label to the right, fanned in y,
                # with a thin leader line back to the marker.
                ax.annotate(
                    pt["model"],
                    (
                        pt["max_modal_fraction"],
                        pt["mean_pairwise_disagreement"],
                    ),
                    textcoords="offset points",
                    xytext=fan[id(pt)],
                    ha="left",
                    fontsize=note_fs,
                    color="#222222",
                    arrowprops=dict(
                        arrowstyle="-", color="#BBBBBB", lw=0.8, shrinkB=6
                    ),
                )
            else:
                offset = free_offsets.get(id(pt), (-12, 8))
                ha = "right" if offset[0] < 0 else "left"
                ax.annotate(
                    pt["model"],
                    (
                        pt["max_modal_fraction"],
                        pt["mean_pairwise_disagreement"],
                    ),
                    textcoords="offset points",
                    xytext=offset,
                    ha=ha,
                    fontsize=note_fs,
                    color="#222222",
                )
        ax.set_xlim(0.55, 1.18)
        ax.set_ylim(0.0, y_top)
        ax.set_xlabel(
            "per-judge concentration  (max modal-vote fraction; "
            "1.0 = a constant judge)"
        )
        ax.set_ylabel("ensemble mean pairwise disagreement")
        ax.set_title(
            "Disagreement does not gate evaluability;\nper-judge variation does"
        )
        ax.text(
            0.57,
            y_top * 0.95,
            "evaluable region\n(all judges vary)",
            fontsize=note_fs,
            color="#16A34A",
            va="top",
            weight="bold",
        )
        ax.text(
            1.005,
            y_top * 0.84,
            "constant-judge boundary\nx = 1.0 -> unevaluable",
            fontsize=note_fs,
            color="#DC2626",
            va="top",
            ha="left",
            weight="bold",
        )
        eval_phrase = (
            "one model is evaluable"
            if n_evaluable == 1
            else f"{n_evaluable}/{n_total} models are evaluable"
        )
        caption = (
            "Each marker is one model. Green circles are evaluable; red crosses "
            f"failed; {eval_phrase}. The split is purely horizontal: any model "
            "at x = 1.0 has at least one constant judge, regardless of its "
            "ensemble disagreement (y)."
        )
        _save(
            "disagreement_vs_evaluability",
            fig,
            {
                "figure": "disagreement_vs_evaluability.png",
                "title": (
                    "Disagreement does not gate evaluability -- "
                    "per-judge variation does"
                ),
                "x_label": (
                    "per-judge concentration (max modal-vote fraction; "
                    "1.0 = constant judge)"
                ),
                "y_label": "ensemble mean pairwise disagreement",
                "thresholds": [
                    {
                        "value": 1.0,
                        "label": "constant-judge boundary",
                        "meaning": (
                            "at x = 1.0 some judge never varied; the "
                            "error-independent algebra is unsolvable -> "
                            "unevaluable, at any disagreement level"
                        ),
                    }
                ],
                "points": concentration,
                "metadata": _fig_metadata(ctx=ctx),
            },
            caption=caption,
            bottom=0.20,
        )

    # 0b. Prompt compliance: parsed vs abstained cells by model. This separates
    # algebraic failure (constant classifier) from prompt/interface failure
    # (unparseable or unavailable responses).
    compliance = []
    for r in rows:
        model = r["model"]
        health = r.get("prompt_health") or {}
        parsed = sum(v.get("n_parsed", 0) for v in health.values())
        unparsed = sum(v.get("n_unparsed", 0) for v in health.values())
        primary_valid = round(
            sum(
                v.get("primary_parse_rate", 0.0) * v.get("n_total", 0)
                for v in health.values()
            )
        )
        repaired = sum(v.get("n_repaired", 0) for v in health.values())
        unavailable = sum(v.get("n_unavailable", 0) for v in health.values())
        final_invalid = max(unparsed - unavailable, 0)
        repair_needed = sum(
            v.get("n_repair_needed", 0) for v in health.values()
        )
        invalid_label = sum(
            v.get("n_invalid_label", 0) for v in health.values()
        )
        total = parsed + unparsed
        methods = {}
        for v in health.values():
            for method, count in v.get("parse_methods", {}).items():
                methods[method] = methods.get(method, 0) + count
        if total:
            compliance.append(
                {
                    "model": model,
                    "parsed": parsed,
                    "unparsed": unparsed,
                    "primary_valid": primary_valid,
                    "repaired_valid": repaired,
                    "final_invalid": final_invalid,
                    "unavailable": unavailable,
                    "repair_needed": repair_needed,
                    "invalid_label": invalid_label,
                    "total": total,
                    "parse_rate": parsed / total,
                    "parse_methods": methods,
                    "evaluable": model in evaluable_models,
                    "error": r.get("error"),
                }
            )
    if compliance:
        fig, ax = plt.subplots(figsize=(1.75 * len(compliance) + 3.5, 6.2))
        xs = list(range(len(compliance)))
        primary = [p["primary_valid"] for p in compliance]
        repaired = [p["repaired_valid"] for p in compliance]
        final_invalid = [p["final_invalid"] for p in compliance]
        unavailable = [p["unavailable"] for p in compliance]
        ax.bar(xs, primary, color="#16A34A", label="primary valid")
        ax.bar(
            xs,
            repaired,
            bottom=primary,
            color="#65A30D",
            label="valid after repair retry",
        )
        invalid_bottom = [a + b for a, b in zip(primary, repaired)]
        ax.bar(
            xs,
            final_invalid,
            bottom=invalid_bottom,
            color="#DC2626",
            label="final invalid",
        )
        unavailable_bottom = [
            a + b for a, b in zip(invalid_bottom, final_invalid)
        ]
        ax.bar(
            xs,
            unavailable,
            bottom=unavailable_bottom,
            color="#6B7280",
            label="unavailable",
        )
        for x, point in zip(xs, compliance):
            ax.text(
                x,
                point["total"] + 4,
                f"{point['parse_rate']:.0%}",
                ha="center",
                va="bottom",
                fontsize=val_fs,
                weight="bold",
            )
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [p["model"] for p in compliance],
            rotation=20,
            ha="right",
            fontsize=note_fs,
        )
        ax.set_ylabel("persona-scenario cells")
        ax.set_ylim(0, max(p["total"] for p in compliance) * 1.33)
        ax.set_title("Prompt compliance: schema-valid, repaired, or invalid")
        ax.legend(loc="upper center", ncol=4, frameon=False)
        caption = (
            "Each model is asked for the same persona-scenario grid under a "
            "binary JSON schema. Dark green cells were valid on the first "
            "attempt; light green cells became valid after one model-driven "
            "repair retry. Red and grey cells remain honest missing votes; no "
            "invalid response is converted into a fabricated vote."
        )
        _save(
            "prompt_compliance",
            fig,
            {
                "figure": "prompt_compliance.png",
                "title": "Prompt compliance: schema-valid, repaired, or invalid",
                "y_label": "persona-scenario cells",
                "series": compliance,
                "metadata": _fig_metadata(ctx=ctx),
            },
            caption=caption,
            bottom=0.20,
            top=0.94,
        )

    # 0b2. Temperature/schema comparison: preserve the old t0.7 prompt record
    # as a stress-test baseline and show the new t0.1/schema contract as the
    # headline collection path.
    baseline_path = out / BASELINE_SUMMARY
    if compliance and baseline_path.exists():
        baseline = _result_body(baseline_path)

        def _series_from_summary(summary, label):
            out_rows = {}
            for row in summary.get("ranking_by_recovery_mae", []):
                health = row.get("prompt_health") or {}
                parsed = sum(v.get("n_parsed", 0) for v in health.values())
                unparsed = sum(v.get("n_unparsed", 0) for v in health.values())
                total = parsed + unparsed
                out_rows[row["model"]] = {
                    "model": row["model"],
                    "run": label,
                    "temperature": summary.get("temperature"),
                    "prompt_version": summary.get("prompt_version"),
                    "parsed": parsed,
                    "unparsed": unparsed,
                    "total": total,
                    "parse_rate": parsed / total if total else 0.0,
                    "evaluable": row["model"]
                    in set(summary.get("evaluated_models", [])),
                }
            return out_rows

        base_series = _series_from_summary(baseline, "baseline")
        current_summary = {
            "temperature": ctx.get("temperature") if ctx else None,
            "prompt_version": ctx.get("prompt_version") if ctx else None,
            "evaluated_models": [r["model"] for r in ranked],
            "ranking_by_recovery_mae": rows,
        }
        cur_series = _series_from_summary(current_summary, "headline")
        model_names = [
            m for m in DEFAULT_MODELS if m in base_series or m in cur_series
        ]
        fig, ax = plt.subplots(figsize=(1.75 * len(model_names) + 3.6, 6.0))
        xs = list(range(len(model_names)))
        bw = 0.36
        base_rates = [
            base_series.get(m, {}).get("parse_rate", 0) for m in model_names
        ]
        cur_rates = [
            cur_series.get(m, {}).get("parse_rate", 0) for m in model_names
        ]
        ax.bar(
            [x - bw / 2 for x in xs],
            base_rates,
            width=bw,
            color="#F97316",
            label="t0.7 / v2 baseline",
        )
        ax.bar(
            [x + bw / 2 for x in xs],
            cur_rates,
            width=bw,
            color="#2563EB",
            label="t0.1 / v3 schema",
        )
        for x, model in zip(xs, model_names):
            if base_series.get(model, {}).get("evaluable"):
                ax.text(x - bw / 2, base_rates[x] + 0.035, "E", ha="center")
            if cur_series.get(model, {}).get("evaluable"):
                ax.text(x + bw / 2, cur_rates[x] + 0.035, "E", ha="center")
        ax.set_xticks(xs)
        ax.set_xticklabels(
            model_names, rotation=20, ha="right", fontsize=note_fs
        )
        ax.set_ylim(0, 1.22)
        ax.set_ylabel("final parse rate")
        ax.set_title(
            "Structured output turns parseability into a controlled interface"
        )
        ax.legend(loc="upper center", ncol=2, frameon=False)
        caption = (
            "The old temperature-0.7 prompt is retained as a stress-test "
            "baseline. The new temperature-0.1 schema run is the headline "
            "experiment: parseability should approach 100%, leaving genuine "
            "vote-matrix evaluability as the object of study. E marks an "
            "ensemble that the ntqr solve could evaluate."
        )
        _save(
            "temperature_schema_comparison",
            fig,
            {
                "figure": "temperature_schema_comparison.png",
                "title": (
                    "Temperature/schema comparison: parseability and "
                    "evaluability"
                ),
                "y_label": "final parse rate",
                "baseline": {
                    "source": BASELINE_SUMMARY,
                    "series": [
                        base_series[m] for m in model_names if m in base_series
                    ],
                },
                "headline": {
                    "series": [
                        cur_series[m] for m in model_names if m in cur_series
                    ],
                },
                "metadata": _fig_metadata(ctx=ctx),
            },
            caption=caption,
            bottom=0.22,
            top=0.94,
        )

    # 0c. Scenario hardness: across all parsed model/persona votes, how split
    # was each scenario? This makes the operational meaning of bullish/bearish
    # visible and shows which authored cases actually induce disagreement.
    scenario_points = []
    for scenario in _personas.SCENARIOS:
        calls = []
        for r in rows:
            votes_path = (
                out / "models" / _model_dirname(r["model"]) / "votes.json"
            )
            if not votes_path.exists():
                continue
            payload = json.loads(votes_path.read_text())
            body = payload.get("result", payload)
            by_scenario = body.get("by_scenario", {})
            for call in by_scenario.get(scenario["id"], {}).values():
                if call in _personas.CALL_TO_LABEL:
                    calls.append(call)
        counts = Counter(calls)
        total = sum(counts.values())
        bull = counts.get("bullish", 0)
        p_bull = bull / total if total else None
        entropy = _analysis._entropy(counts) if total else None
        scenario_points.append(
            {
                "id": scenario["id"],
                "truth": scenario["truth"],
                "n_parsed": total,
                "bullish_votes": bull,
                "bearish_votes": counts.get("bearish", 0),
                "fraction_bullish": p_bull,
                "entropy_bits": entropy,
                "text": scenario["text"],
            }
        )
    if scenario_points and any(
        p["fraction_bullish"] is not None for p in scenario_points
    ):
        fig, ax = plt.subplots(figsize=(14.5, 6.2))
        xs = list(range(1, len(scenario_points) + 1))
        ys = [
            p["fraction_bullish"] if p["fraction_bullish"] is not None else 0
            for p in scenario_points
        ]
        colors = [
            "#2563EB" if p["truth"] == "bullish" else "#F97316"
            for p in scenario_points
        ]
        ax.scatter(xs, ys, c=colors, s=55, alpha=0.9, edgecolor="white")
        hard_x = [
            x
            for x, p in zip(xs, scenario_points)
            if p["fraction_bullish"] is not None
            and 0.35 <= p["fraction_bullish"] <= 0.65
        ]
        hard_y = [
            p["fraction_bullish"]
            for p in scenario_points
            if p["fraction_bullish"] is not None
            and 0.35 <= p["fraction_bullish"] <= 0.65
        ]
        if hard_x:
            ax.scatter(
                hard_x,
                hard_y,
                facecolors="none",
                edgecolors="#111827",
                s=135,
                linewidth=1.4,
                label="near-split scenario",
            )
        ax.axhline(0.5, color="#555555", ls="--", lw=1.2)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, len(scenario_points) + 1)
        ax.set_xlabel("scenario id")
        ax.set_ylabel("fraction of parsed votes that were bullish")
        ax.set_title("Scenario-level vote split and authored direction")
        from matplotlib.lines import Line2D

        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="authored bullish",
                markerfacecolor="#2563EB",
                markersize=8,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="authored bearish",
                markerfacecolor="#F97316",
                markersize=8,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="#111827",
                label="near split",
                markerfacecolor="none",
                markersize=9,
                linestyle="None",
            ),
        ]
        ax.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=False,
        )
        caption = (
            "Each point is one authored market scenario. The y-value is the "
            "fraction of parsed persona/model calls that were bullish; marker "
            "color is the authored operational truth. Points near 0.5 are the "
            "hard or intentionally mixed cases that create useful variation."
        )
        _save(
            "scenario_hardness",
            fig,
            {
                "figure": "scenario_hardness.png",
                "title": "Scenario-level vote split and authored direction",
                "x_label": "scenario id",
                "y_label": "fraction of parsed votes that were bullish",
                "points": scenario_points,
                "metadata": _fig_metadata(ctx=ctx),
            },
            caption=caption,
            bottom=0.18,
            top=0.92,
        )

    # 1. Recovery MAE per model (lower = unsupervised recovery closer to truth).
    fig, ax = plt.subplots(figsize=(9, 0.7 * len(ranked) + 3.0))
    names = [r["model"] for r in ranked][::-1]
    maes = [r["recovery_mae"] for r in ranked][::-1]
    ax.barh(names, maes, color="#3B82F6")
    for y, v in enumerate(maes):
        ax.text(v, y, f" {v:.3f}", va="center", fontsize=val_fs, weight="bold")
    ax.axvline(excellent, color="#16A34A", ls=":", lw=1.6)
    ax.text(
        excellent,
        len(names) - 0.4,
        f"  {excellent:.2f}  excellent",
        color="#16A34A",
        fontsize=note_fs,
        va="top",
    )
    if noise_floor is not None:
        ax.axvline(noise_floor, color="#DC2626", ls="--", lw=1.6)
        ax.text(
            noise_floor + 0.002,
            len(names) - 0.4,
            f"{noise_floor:.2f}  sampling-noise floor (Q={n_scen})",
            color="#DC2626",
            fontsize=note_fs,
            va="top",
        )
    ax.set_xlabel(
        "recovery MAE  (mean |recovered - true| accuracy; 0 = perfect)"
    )
    ax.set_title("Unsupervised accuracy recovery error, by model")
    x_max = max(noise_floor or 0, max(maes, default=0)) * 1.32
    ax.set_xlim(0, max(x_max, 0.08))
    caption = (
        "Each bar is one model's mean |recovered - true| persona accuracy, "
        "recovered with no answer key. Left of the red line the error is "
        f"within the conservative binomial sampling-noise reference for the "
        f"minority label ({min_label} of {n_scen} questions)."
    )
    _save(
        "recovery_mae_by_model",
        fig,
        {
            "figure": "recovery_mae_by_model.png",
            "title": "Unsupervised accuracy recovery error, by model",
            "x_label": "recovery MAE (mean |recovered - true| accuracy)",
            "thresholds": [
                {
                    "value": excellent,
                    "label": "excellent",
                    "meaning": (
                        "recovery error below 0.05 accuracy points -- a tight "
                        "agreement bar between label-free recovery and truth"
                    ),
                },
                {
                    "value": noise_floor,
                    "label": "sampling-noise floor",
                    "meaning": (
                        "worst-case binomial SE of a per-label accuracy from "
                        f"the minority label's {min_label} questions; at or "
                        "below this, recovery error is inside the conservative "
                        "finite-deck noise reference"
                    ),
                },
            ],
            "series": [
                {
                    "model": r["model"],
                    "recovery_mae": r["recovery_mae"],
                    "recovery_mae_by_label": r.get("recovery_mae_by_label"),
                    "mv_recovery_mae": r.get("mv_recovery_mae"),
                    "prevalence_abs_error": r.get("prevalence_abs_error"),
                }
                for r in ranked
            ],
            "metadata": _fig_metadata(ctx=ctx),
        },
        caption=caption,
        bottom=0.20,
    )

    # 2. Recovered vs true accuracy; points should hug y = x.
    fig, ax = plt.subplots(figsize=(8.8, 7.4))
    ax.plot(
        [0, 1],
        [0, 1],
        color="#999999",
        ls="--",
        lw=1.3,
        label="y = x (perfect)",
    )
    if noise_floor is not None:
        ax.fill_between(
            [0, 1],
            [0 - noise_floor, 1 - noise_floor],
            [0 + noise_floor, 1 + noise_floor],
            color="#9CA3AF",
            alpha=0.18,
            label=f"+/- sampling noise ({noise_floor:.2f})",
        )
    points = []
    for idx, r in enumerate(ranked):
        xs, ys = [], []
        for row in r["comparison"]["per_persona"]:
            for label in ("a", "b"):
                tv = row["true_accuracy"][label]
                rv = row["recovered_accuracy_unsupervised"][label]
                if tv is not None and rv is not None:
                    xs.append(tv)
                    ys.append(rv)
                    points.append(
                        {
                            "model": r["model"],
                            "persona": row["persona"],
                            "label": label,
                            "true": tv,
                            "recovered": rv,
                        }
                    )
        ax.scatter(
            xs, ys, color=cmap(idx % 10), label=r["model"], s=70, alpha=0.85
        )
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("true accuracy (supervised)")
    ax.set_ylabel("recovered accuracy (unsupervised ntqr)")
    ax.set_title("Recovered vs true persona accuracy")
    ax.text(
        0.03,
        0.97,
        "above line:\nrecovery OVER-estimates",
        transform=ax.transAxes,
        fontsize=note_fs,
        color="#444444",
        va="top",
    )
    ax.text(
        0.97,
        0.08,
        "below line:\nrecovery UNDER-estimates",
        transform=ax.transAxes,
        fontsize=note_fs,
        color="#444444",
        ha="right",
        va="bottom",
    )
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    caption = (
        "One point per persona/label accuracy term. Points on the dashed "
        "line are perfectly recovered; inside the grey band they are within "
        "one-question sampling noise of the supervised truth."
    )
    _save(
        "recovered_vs_true_scatter",
        fig,
        {
            "figure": "recovered_vs_true_scatter.png",
            "title": "Recovered vs true persona accuracy",
            "x_label": "true accuracy (supervised)",
            "y_label": "recovered accuracy (unsupervised ntqr)",
            "reference_line": "y = x (perfect recovery)",
            "thresholds": [
                {
                    "value": noise_floor,
                    "label": "sampling-noise band",
                    "meaning": (
                        "shaded +/- band around y=x; points inside are within "
                        "the supervised estimate's own sampling error"
                    ),
                }
            ],
            "points": points,
            "metadata": _fig_metadata(ctx=ctx),
        },
        caption=caption,
        bottom=0.16,
        top=0.98,
    )

    # 2b. Validation-only accuracy landscape: once the unsupervised solve has
    # selected a branch, the authored labels let us inspect what was actually
    # measured. This is deliberately validation-only, not an input to the solve.
    heat_terms = []
    col_labels = []
    for persona in _personas.PERSONAS:
        for label in ("a", "b"):
            col_labels.append(f"{persona['name'][:3]}\n{label}")
            heat_terms.append((persona["name"], label))
    true_matrix = []
    err_matrix = []
    heat_rows = []
    for r in ranked:
        by_persona = {
            row["persona"]: row for row in r["comparison"]["per_persona"]
        }
        true_row = []
        err_row = []
        row_terms = []
        for persona, label in heat_terms:
            row = by_persona[persona]
            tv = row["true_accuracy"][label]
            rv = row["recovered_accuracy_unsupervised"][label]
            err = abs(rv - tv) if tv is not None and rv is not None else None
            true_row.append(tv if tv is not None else 0.0)
            err_row.append(err if err is not None else 0.0)
            row_terms.append(
                {
                    "persona": persona,
                    "label": label,
                    "true_accuracy": tv,
                    "recovered_accuracy": rv,
                    "abs_error": err,
                }
            )
        true_matrix.append(true_row)
        err_matrix.append(err_row)
        heat_rows.append(
            {
                "model": r["model"],
                "recovery_mae": r["recovery_mae"],
                "terms": row_terms,
            }
        )
    if heat_rows:
        fig, axes = plt.subplots(1, 2, figsize=(15.2, 6.7), sharey=True)
        row_labels = [r["model"] for r in heat_rows]
        true_img = axes[0].imshow(
            true_matrix, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto"
        )
        err_img = axes[1].imshow(
            err_matrix,
            vmin=0,
            vmax=max(noise_floor or 0.08, 0.08),
            cmap="YlOrRd",
            aspect="auto",
        )
        for ax, matrix, title, fmt in (
            (
                axes[0],
                true_matrix,
                "true accuracy (validation only)",
                "{:.2f}",
            ),
            (axes[1], err_matrix, "|recovered - true|", "{:.2f}"),
        ):
            ax.set_title(title)
            ax.set_xticks(range(len(col_labels)))
            ax.set_xticklabels(
                col_labels, rotation=35, ha="right", fontsize=10
            )
            ax.set_yticks(range(len(row_labels)))
            ax.set_yticklabels(row_labels, fontsize=10)
            for yi, values in enumerate(matrix):
                for xi, value in enumerate(values):
                    high_contrast = (ax is axes[0] and value >= 0.72) or (
                        ax is axes[1] and value >= 0.055
                    )
                    ax.text(
                        xi,
                        yi,
                        fmt.format(value),
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="white" if high_contrast else "#111827",
                    )
        axes[0].set_ylabel("evaluable model", fontsize=12)
        cbar_true = fig.colorbar(
            true_img, ax=axes[0], fraction=0.046, pad=0.03
        )
        cbar_true.set_label("supervised true accuracy")
        cbar_err = fig.colorbar(err_img, ax=axes[1], fraction=0.046, pad=0.03)
        cbar_err.set_label("absolute recovery error")
        fig.suptitle(
            "Validation-only accuracy landscape across evaluable models",
            y=0.98,
        )
        caption = (
            "Left panel: supervised accuracies computed only after the "
            "unlabeled solve. Right panel: absolute recovery error for the same "
            "model/persona/label terms. The heatmap shows whether a model is "
            "useful because its judges are genuinely accurate, because NTQR can "
            "recover their weaknesses, or both."
        )
        _save(
            "validation_accuracy_heatmap",
            fig,
            {
                "figure": "validation_accuracy_heatmap.png",
                "title": (
                    "Validation-only accuracy landscape across evaluable models"
                ),
                "x_label": "persona / label accuracy term",
                "y_label": "evaluable model ranked by recovery MAE",
                "columns": [
                    {"persona": persona, "label": label}
                    for persona, label in heat_terms
                ],
                "rows": heat_rows,
                "thresholds": [
                    {
                        "value": noise_floor,
                        "label": "sampling-noise floor",
                        "meaning": (
                            "right-panel color scale is capped at this "
                            "finite-deck uncertainty reference"
                        ),
                    }
                ],
                "metadata": _fig_metadata(ctx=ctx),
            },
            caption=caption,
            bottom=0.24,
            top=0.90,
        )

    # 3. Evaluability diagnostic: per-persona vote concentration per model.
    # A bar at 1.0 means that persona never varied (a constant classifier),
    # which is exactly what makes the unsupervised algebra unsolvable -- so this
    # plot reads the failures off the votes, with no answer key.
    diagnosed = []
    for r in rows:
        diag_path = (
            out
            / "models"
            / _model_dirname(r["model"])
            / "evaluability_diagnostic.json"
        )
        if not diag_path.exists():
            continue
        payload = json.loads(diag_path.read_text())
        diag = payload.get("result", payload)
        diagnosed.append((r["model"], diag))
    if diagnosed:
        persona_names = [p["name"] for p in _personas.PERSONAS]
        n_models = len(diagnosed)
        n_personas = len(persona_names)
        width = 0.8 / n_personas
        fig, ax = plt.subplots(figsize=(max(12.5, 2.05 * n_models + 3.0), 6.2))
        for pi, pname in enumerate(persona_names):
            xs = [
                mi + (pi - (n_personas - 1) / 2) * width
                for mi in range(n_models)
            ]
            fracs = [
                diag["vote_variation"][pname]["modal_fraction"]
                for _, diag in diagnosed
            ]
            colors = [
                (
                    "#DC2626"
                    if diag["vote_variation"][pname]["is_constant"]
                    else cmap(pi % 10)
                )
                for _, diag in diagnosed
            ]
            ax.bar(xs, fracs, width=width, color=colors, label=pname)
        ax.axhline(
            1.0,
            color="#DC2626",
            ls="--",
            lw=1.6,
            label="constant boundary",
        )
        ax.axhline(
            0.5,
            color="#16A34A",
            ls=":",
            lw=1.6,
            label="balanced reference",
        )
        ax.axhspan(0.95, 1.05, color="#DC2626", alpha=0.08)
        # Mark each constant classifier so the unevaluable judges are obvious.
        for mi, (_, diag) in enumerate(diagnosed):
            for pi, pname in enumerate(persona_names):
                if diag["vote_variation"][pname]["is_constant"]:
                    x = mi + (pi - (n_personas - 1) / 2) * width
                    ax.text(
                        x,
                        0.985,
                        "constant",
                        rotation=90,
                        ha="center",
                        va="top",
                        fontsize=note_fs - 1,
                        color="white",
                        weight="bold",
                    )
        ax.set_xticks(range(n_models))
        ax.set_xticklabels(
            [m for m, _ in diagnosed],
            rotation=20,
            ha="right",
            fontsize=note_fs,
        )
        ax.set_ylim(0.45, 1.10)
        ax.set_ylabel("modal vote fraction")
        ax.set_title(
            "Evaluability diagnostic: vote concentration by persona and model"
        )
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=True,
            borderaxespad=0.0,
        )
        caption = (
            "A judge at 1.0 is a constant classifier: it casts the same vote "
            "on every parseable call, carrying zero error signal. That removes "
            "a degree of freedom from the 3-judge system, so the "
            "error-independent algebra becomes unsolvable and the model is "
            "excluded -- read off the votes alone, with no answer key."
        )
        _save(
            "evaluability_diagnostic",
            fig,
            {
                "figure": "evaluability_diagnostic.png",
                "title": (
                    "Evaluability diagnostic: vote concentration by persona "
                    "and model"
                ),
                "y_label": "modal vote fraction (0.5 balanced, 1.0 constant)",
                "thresholds": [
                    {
                        "value": 1.0,
                        "label": "constant classifier",
                        "meaning": (
                            "persona never varied its vote; carries no error "
                            "signal; makes the error-independent algebra "
                            "unsolvable -> model is unevaluable"
                        ),
                    },
                    {
                        "value": 0.5,
                        "label": "balanced",
                        "meaning": (
                            "evenly split votes -- the most informative case "
                            "for recovering accuracy"
                        ),
                    },
                ],
                "series": [
                    {
                        "model": model,
                        "predicted_evaluable": diag["predicted_evaluable"],
                        "constant_classifiers": diag["constant_classifiers"],
                        "mean_pairwise_disagreement": diag[
                            "mean_pairwise_disagreement"
                        ],
                        "vote_variation": diag["vote_variation"],
                        "pairwise_disagreement": diag["pairwise_disagreement"],
                    }
                    for model, diag in diagnosed
                ],
                "metadata": _fig_metadata(ctx=ctx),
            },
            caption=caption,
            bottom=0.20,
            top=0.94,
        )

    # 4. Best model: true vs recovered accuracy per persona/label (grouped).
    best = ranked[0]
    cats, trues, recs, terms = [], [], [], []
    for row in best["comparison"]["per_persona"]:
        short = row["persona"].split(" (")[0]
        for label in ("a", "b"):
            tv = row["true_accuracy"][label]
            rv = row["recovered_accuracy_unsupervised"][label]
            if tv is None or rv is None:
                continue
            cats.append(f"{short}\n{label}")
            trues.append(tv)
            recs.append(rv)
            terms.append(
                {
                    "persona": row["persona"],
                    "label": label,
                    "true": tv,
                    "recovered": rv,
                    "abs_error": abs(rv - tv),
                }
            )
    if cats:
        xs = list(range(len(cats)))
        bw = 0.4
        fig, ax = plt.subplots(figsize=(1.55 * len(cats) + 3.8, 6.0))
        ax.bar(
            [x - bw / 2 for x in xs],
            trues,
            width=bw,
            color="#9CA3AF",
            label="true (supervised)",
        )
        ax.bar(
            [x + bw / 2 for x in xs],
            recs,
            width=bw,
            color="#3B82F6",
            label="recovered (unsupervised)",
        )
        for x, t in zip(xs, terms):
            ax.text(
                x,
                max(t["true"], t["recovered"]) + 0.02,
                f"d={t['abs_error']:.2f}",
                ha="center",
                fontsize=val_fs,
                weight="bold",
                color="#444444",
            )
        worst = best.get("worst_term")
        if noise_floor is not None:
            ax.axhline(
                noise_floor,
                color="#DC2626",
                ls="--",
                lw=1.3,
                label=f"sampling-noise floor ({noise_floor:.2f})",
            )
        ax.set_xticks(xs)
        ax.set_xticklabels(cats, fontsize=note_fs)
        ax.set_ylim(0, 1.18)
        ax.set_ylabel("accuracy")
        subtitle = f"recovery MAE {best['recovery_mae']:.3f}"
        ax.set_title(
            f"True vs recovered persona accuracy -- {best['model']} "
            f"({subtitle})"
        )
        caption = (
            "d = |recovered - true| per term. With no answer key the algebra "
            "tracks all six per-persona accuracies closely"
        )
        if worst:
            caption += (
                f", worst term {worst['persona']} / {worst['label']} off by "
                f"{worst['abs_error']:.2f}"
            )
        caption += " -- including a deliberately bad judge."
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=True,
            borderaxespad=0.0,
        )
        _save(
            "best_model_recovery",
            fig,
            {
                "figure": "best_model_recovery.png",
                "title": (
                    f"True vs recovered persona accuracy -- {best['model']}"
                ),
                "y_label": "accuracy",
                "model": best["model"],
                "recovery_mae": best["recovery_mae"],
                "recovery_mae_by_label": best.get("recovery_mae_by_label"),
                "terms": terms,
                "thresholds": [
                    {
                        "value": noise_floor,
                        "label": "sampling-noise floor",
                        "meaning": (
                            "per-term errors (d) at or below this are within "
                            "the supervised estimate's own sampling error"
                        ),
                    }
                ],
                "metadata": _fig_metadata({"worst_term": worst}, ctx=ctx),
            },
            caption=caption,
            bottom=0.18,
        )
    return written


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
        if r["model"] in evaluated
        and r.get("mean_pairwise_disagreement") is not None
    ]
    dis_uneval = [
        r["mean_pairwise_disagreement"]
        for r in rows
        if "predicted_evaluable" in r
        and r["model"] not in evaluated
        and r.get("mean_pairwise_disagreement") is not None
    ]
    disagreement_gates = (
        bool(dis_eval)
        and bool(dis_uneval)
        and (min(dis_eval) >= max(dis_uneval))
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
        "recovery_mae": _analysis.descriptive_stats(maes),
        "mv_recovery_mae": _analysis.descriptive_stats(
            [r.get("mv_recovery_mae") for r in ranked]
        ),
        "prevalence_abs_error": _analysis.descriptive_stats(
            [r.get("prevalence_abs_error") for r in ranked]
        ),
        "sampling_noise_floor": _analysis.sampling_noise_floor(
            len(_personas.SCENARIOS),
            min_label_count=_analysis.min_truth_label_count(),
        ),
        "disagreement_evaluable": _analysis.descriptive_stats(dis_eval),
        "disagreement_unevaluable": _analysis.descriptive_stats(dis_uneval),
        "disagreement_gates_evaluability": disagreement_gates,
        "note": disagreement_note,
    }
    summary = {
        "backend": backend,
        "temperature": temperature,
        "prompt_version": _personas.PROMPT_VERSION,
        "prompt_hash": _personas.prompt_hash(),
        "prompt_payload": _personas.prompt_payload(),
        "requested_models": models,
        "evaluated_models": [r["model"] for r in ranked],
        "skipped_unavailable": missing,
        "failed_models": [r["model"] for r in rows if "error" in r],
        "n_scenarios": len(_personas.SCENARIOS),
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
            "diagnostic_matched_outcome": (
                predicted == (evaluated & diagnosed)
            ),
        },
        "ranking_by_recovery_mae": table,
    }
    _common.save_json(out, "summary.json", summary)
    plot_ctx = {
        "backend": backend,
        "temperature": temperature,
        "n_personas": len(_personas.PERSONAS),
        "n_scenarios": len(_personas.SCENARIOS),
        "n_labels": len(_personas.LABELS),
        "n_models_requested": len(models),
        "n_models_evaluated": len(ranked),
        "prompt_version": _personas.PROMPT_VERSION,
        "prompt_hash": _personas.prompt_hash(),
        "baseline_summary": (
            BASELINE_SUMMARY if (out / BASELINE_SUMMARY).exists() else None
        ),
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }
    plots = _plot(rows, out, plot_ctx)

    _common.banner(
        f"multi-LLM personas via {backend} "
        f"({len(present)} models, {len(_personas.SCENARIOS)} scenarios)"
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
        print(
            f"best: {ranked[0]['model']} (MAE {ranked[0]['recovery_mae']:.3f})"
        )
    for p in plots:
        print(f"-> plot {p.name}")
    print(f"-> wrote results to {out}")


if __name__ == "__main__":
    main()
