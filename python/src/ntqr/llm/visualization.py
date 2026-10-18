"""Composable publication figures built from recorded model results.

Each builder consumes a shared context; plotting requires no LLM client or server.
The public plot_models API can select individual figures without running a study.
"""

from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field
from functools import cached_property
import json
from pathlib import Path
from . import analysis, personas
from .artifacts import generated_at, save_record
from .figure_export import export_figure, publication_figure

DEFAULT_MODELS = [
    "granite4.1:3b",
    "mistral:latest",
    "gemma2:2b",
    "gemma3:4b",
    "gemma4:latest",
    "smollm2:135m-instruct-q4_K_S",
]
BASELINE_SUMMARY = "baseline_t0.7_summary.json"


def _model_dirname(model):
    return model.replace(":", "_").replace("/", "_")


def _result_body(path):
    record = json.loads(path.read_text(encoding="utf-8"))
    return record.get("result", record)


def rank_models(rows):
    return sorted(
        (row for row in rows if row.get("recovery_mae") is not None),
        key=lambda row: row["recovery_mae"],
    )


def _fig_metadata(extra=None, ctx=None):
    """Shared provenance block stamped into every figure data sidecar."""
    meta = {
        "generated_utc": generated_at(),
        "n_scenarios": len(personas.SCENARIOS),
        "n_personas": len(personas.PERSONAS),
        "personas": [p["name"] for p in personas.PERSONAS],
        "labels": list(personas.LABELS),
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
        f"{ctx.get('n_models_evaluated')}/{ctx.get('n_models_requested')} " "evaluable",
        "ntqr EI",
        str(ctx.get("date")),
    ]
    return "  •  ".join(b for b in bits if b)


@dataclass
class FigureContext:
    rows: list
    out: Path
    ctx: dict | None = None
    output_dir: Path | None = None
    written: list = field(default_factory=list)
    note_fs: int = 11
    val_fs: int = 11

    @cached_property
    def ranked(self):
        return rank_models(self.rows)

    @cached_property
    def cmap(self):
        from matplotlib.colors import ListedColormap

        return ListedColormap(
            ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]
        )

    @property
    def n_scen(self):
        return len(personas.SCENARIOS)

    @property
    def min_label(self):
        return analysis.min_truth_label_count()

    @property
    def noise_floor(self):
        return analysis.sampling_noise_floor(
            self.n_scen, min_label_count=self.min_label
        )

    @property
    def excellent(self):
        return analysis.RECOVERY_MAE_EXCELLENT

    @property
    def provenance(self):
        return _provenance_line(self.ctx)

    @property
    def evaluable_models(self):
        return {row["model"] for row in self.ranked}

    @property
    def n_evaluable(self):
        return len(self.evaluable_models)

    @property
    def n_total(self):
        return len(self.rows)

    def save(self, stem, fig, data, caption="", bottom=0.14, top=1.0):
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
        if self.provenance:
            fig.text(
                0.5,
                0.012,
                self.provenance,
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#555555",
            )
        png = (self.output_dir or self.out) / f"{stem}.png"
        export_figure(fig, png, description=caption or data.get("title", stem))
        data.setdefault("caption", caption)
        data["exports"] = {"raster": png.name, "vector": png.with_suffix(".svg").name}
        save_record(self.output_dir or self.out, f"{stem}.data.json", data)
        self.written.append(png)


@publication_figure
def plot_pipeline(context: FigureContext):
    import matplotlib.pyplot as plt

    _save = context.save
    ctx = context.ctx
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
                "What enters the unsupervised solve, and what is " "validation-only"
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


@publication_figure
def plot_concentration(context: FigureContext):
    import matplotlib.pyplot as plt

    _save = context.save
    ctx = context.ctx
    evaluable_models = context.evaluable_models
    n_evaluable = context.n_evaluable
    n_total = context.n_total
    note_fs = context.note_fs
    out = context.out
    rows = context.rows
    # 0. THE HEADLINE: disagreement vs per-judge concentration. Each model is a
    # point; x = its most-concentrated judge's modal-vote fraction (1.0 = a
    # constant classifier exists), y = ensemble mean pairwise disagreement.
    # A constant judge gives a nonidentifying empirical system. Points left of
    # the boundary only pass a necessary variation screen; they need a solve.
    concentration = []
    for r in rows:
        diag_path = (
            out / "models" / _model_dirname(r["model"]) / "evaluability_diagnostic.json"
        )
        if not diag_path.exists() or r.get("mean_pairwise_disagreement") is None:
            continue
        payload = json.loads(diag_path.read_text())
        diag = payload.get("result", payload)
        max_modal = max(vv["modal_fraction"] for vv in diag["vote_variation"].values())
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
                    arrowprops=dict(arrowstyle="-", color="#BBBBBB", lw=0.8, shrinkB=6),
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
        ax.set_title("Variation is necessary;\nphysical roots still need verification")
        ax.text(
            1.005,
            y_top * 0.84,
            "constant-judge boundary\nx = 1.0: nonidentifying",
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
            f"failed; {eval_phrase}. Passing the variation screen is not sufficient. A model "
            "at x = 1.0 has at least one constant judge, regardless of its "
            "ensemble disagreement (y)."
        )
        _save(
            "disagreement_vs_evaluability",
            fig,
            {
                "figure": "disagreement_vs_evaluability.png",
                "title": (
                    "Variation is a necessary screen; physical roots need verification"
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
                            "implemented prevalence quadratic is nonidentifying; "
                            "passing variation still needs a physical solve"
                        ),
                    }
                ],
                "points": concentration,
                "metadata": _fig_metadata(ctx=ctx),
            },
            caption=caption,
            bottom=0.20,
        )


@publication_figure
def plot_compliance(context: FigureContext):
    import matplotlib.pyplot as plt

    _save = context.save
    ctx = context.ctx
    evaluable_models = context.evaluable_models
    note_fs = context.note_fs
    out = context.out
    ranked = context.ranked
    rows = context.rows
    val_fs = context.val_fs
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
        repair_needed = sum(v.get("n_repair_needed", 0) for v in health.values())
        invalid_label = sum(v.get("n_invalid_label", 0) for v in health.values())
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
        unavailable_bottom = [a + b for a, b in zip(invalid_bottom, final_invalid)]
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
        model_names = [m for m in DEFAULT_MODELS if m in base_series or m in cur_series]
        fig, ax = plt.subplots(figsize=(1.75 * len(model_names) + 3.6, 6.0))
        xs = list(range(len(model_names)))
        bw = 0.36
        base_rates = [base_series.get(m, {}).get("parse_rate", 0) for m in model_names]
        cur_rates = [cur_series.get(m, {}).get("parse_rate", 0) for m in model_names]
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
        ax.set_xticklabels(model_names, rotation=20, ha="right", fontsize=note_fs)
        ax.set_ylim(0, 1.22)
        ax.set_ylabel("final parse rate")
        ax.set_title("Structured output turns parseability into a controlled interface")
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
                    "Temperature/schema comparison: parseability and " "evaluability"
                ),
                "y_label": "final parse rate",
                "baseline": {
                    "source": BASELINE_SUMMARY,
                    "series": [base_series[m] for m in model_names if m in base_series],
                },
                "headline": {
                    "series": [cur_series[m] for m in model_names if m in cur_series],
                },
                "metadata": _fig_metadata(ctx=ctx),
            },
            caption=caption,
            bottom=0.22,
            top=0.94,
        )


@publication_figure
def plot_hardness(context: FigureContext):
    import matplotlib.pyplot as plt

    _save = context.save
    ctx = context.ctx
    out = context.out
    rows = context.rows
    # 0c. Scenario hardness: across all parsed model/persona votes, how split
    # was each scenario? This makes the operational meaning of bullish/bearish
    # visible and shows which authored cases actually induce disagreement.
    scenario_points = []
    for scenario in personas.SCENARIOS:
        calls = []
        for r in rows:
            votes_path = out / "models" / _model_dirname(r["model"]) / "votes.json"
            if not votes_path.exists():
                continue
            payload = json.loads(votes_path.read_text())
            body = payload.get("result", payload)
            by_scenario = body.get("by_scenario", {})
            for call in by_scenario.get(scenario["id"], {}).values():
                if call in personas.CALL_TO_LABEL:
                    calls.append(call)
        counts = Counter(calls)
        total = sum(counts.values())
        bull = counts.get("bullish", 0)
        p_bull = bull / total if total else None
        entropy = analysis._entropy(counts) if total else None
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
            "#2563EB" if p["truth"] == "bullish" else "#F97316" for p in scenario_points
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


@publication_figure
def plot_recovery(context: FigureContext):
    import matplotlib.pyplot as plt

    _save = context.save
    ctx = context.ctx
    excellent = context.excellent
    min_label = context.min_label
    n_scen = context.n_scen
    noise_floor = context.noise_floor
    note_fs = context.note_fs
    ranked = context.ranked
    val_fs = context.val_fs
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
            f"{noise_floor:.2f}  binomial SE reference (Q={n_scen})",
            color="#DC2626",
            fontsize=note_fs,
            va="top",
        )
    ax.set_xlabel("recovery MAE  (mean |recovered - true| accuracy; 0 = perfect)")
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
                    "label": "binomial SE reference",
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


@publication_figure
def plot_scatter(context: FigureContext):
    import matplotlib.pyplot as plt

    _save = context.save
    cmap = context.cmap
    ctx = context.ctx
    noise_floor = context.noise_floor
    note_fs = context.note_fs
    ranked = context.ranked
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
        ax.scatter(xs, ys, color=cmap(idx % 10), label=r["model"], s=70, alpha=0.85)
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


@publication_figure
def plot_accuracy_landscape(context: FigureContext):
    import matplotlib.pyplot as plt

    _save = context.save
    ctx = context.ctx
    noise_floor = context.noise_floor
    ranked = context.ranked
    # 2b. Validation-only accuracy landscape: once the unsupervised solve has
    # selected a branch, the authored labels let us inspect what was actually
    # measured. This is deliberately validation-only, not an input to the solve.
    heat_terms = []
    col_labels = []
    for persona in personas.PERSONAS:
        for label in ("a", "b"):
            col_labels.append(f"{persona['name'][:3]}\n{label}")
            heat_terms.append((persona["name"], label))
    true_matrix = []
    err_matrix = []
    heat_rows = []
    for r in ranked:
        by_persona = {row["persona"]: row for row in r["comparison"]["per_persona"]}
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
            ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=10)
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
        cbar_true = fig.colorbar(true_img, ax=axes[0], fraction=0.046, pad=0.03)
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
                "title": ("Validation-only accuracy landscape across evaluable models"),
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
                        "label": "binomial SE reference",
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


@publication_figure
def plot_variation(context: FigureContext):
    cmap = context.cmap
    import matplotlib.pyplot as plt

    _save = context.save
    ctx = context.ctx
    note_fs = context.note_fs
    out = context.out
    rows = context.rows
    # 3. Evaluability diagnostic: per-persona vote concentration per model.
    # A bar at 1.0 means that persona never varied (a constant classifier),
    # which is exactly what makes the unsupervised algebra unsolvable -- so this
    # plot reads the failures off the votes, with no answer key.
    diagnosed = []
    for r in rows:
        diag_path = (
            out / "models" / _model_dirname(r["model"]) / "evaluability_diagnostic.json"
        )
        if not diag_path.exists():
            continue
        payload = json.loads(diag_path.read_text())
        diag = payload.get("result", payload)
        diagnosed.append((r["model"], diag))
    if diagnosed:
        persona_names = [p["name"] for p in personas.PERSONAS]
        n_models = len(diagnosed)
        n_personas = len(persona_names)
        width = 0.8 / n_personas
        fig, ax = plt.subplots(figsize=(max(12.5, 2.05 * n_models + 3.0), 6.2))
        for pi, pname in enumerate(persona_names):
            xs = [mi + (pi - (n_personas - 1) / 2) * width for mi in range(n_models)]
            fracs = [
                diag["vote_variation"][pname]["modal_fraction"] for _, diag in diagnosed
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
        ax.set_title("Evaluability diagnostic: vote concentration by persona and model")
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=True,
            borderaxespad=0.0,
        )
        caption = (
            "A judge at 1.0 is a constant classifier: it casts the same vote "
            "on every parseable call. The implemented prevalence quadratic is "
            "then nonidentifying. Variation is necessary; passing this screen "
            "does not guarantee a physical recovery. No answer key enters this screen."
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
                            "evenly split votes; neither identifiability nor "
                            "physical recovery follows from balance alone"
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


@publication_figure
def plot_best_model(context: FigureContext):
    import matplotlib.pyplot as plt

    _save = context.save
    ctx = context.ctx
    noise_floor = context.noise_floor
    note_fs = context.note_fs
    ranked = context.ranked
    val_fs = context.val_fs
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
                label=f"binomial SE reference ({noise_floor:.2f})",
            )
        ax.set_xticks(xs)
        ax.set_xticklabels(cats, fontsize=note_fs)
        ax.set_ylim(0, 1.18)
        ax.set_ylabel("accuracy")
        subtitle = f"recovery MAE {best['recovery_mae']:.3f}"
        ax.set_title(
            f"True vs recovered persona accuracy -- {best['model']} " f"({subtitle})"
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
                "title": (f"True vs recovered persona accuracy -- {best['model']}"),
                "y_label": "accuracy",
                "model": best["model"],
                "recovery_mae": best["recovery_mae"],
                "recovery_mae_by_label": best.get("recovery_mae_by_label"),
                "terms": terms,
                "thresholds": [
                    {
                        "value": noise_floor,
                        "label": "binomial SE reference",
                        "meaning": (
                            "worst-case Bernoulli proportion SE reference; "
                            "not a lower bound or confidence interval for recovery MAE"
                        ),
                    }
                ],
                "metadata": _fig_metadata({"worst_term": worst}, ctx=ctx),
            },
            caption=caption,
            bottom=0.18,
        )


BUILDERS = {
    "pipeline": plot_pipeline,
    "concentration": plot_concentration,
    "compliance": plot_compliance,
    "hardness": plot_hardness,
    "recovery": plot_recovery,
    "scatter": plot_scatter,
    "accuracy_landscape": plot_accuracy_landscape,
    "variation": plot_variation,
    "best_model": plot_best_model,
}


def plot_models(rows, out, ctx=None, *, figures=None, source_dir=None):
    """Render selected named figures from records; preserve the caller's rcParams."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = list(BUILDERS) if figures is None else list(figures)
    unknown = set(selected) - set(BUILDERS)
    if unknown:
        raise ValueError(f"unknown figures: {sorted(unknown)}")
    context = FigureContext(
        list(rows), Path(source_dir or out), ctx, output_dir=Path(out)
    )
    if not context.ranked:
        return []
    context.output_dir.mkdir(parents=True, exist_ok=True)
    existing = set(plt.get_fignums())
    try:
        with matplotlib.rc_context(
            {
                "text.usetex": False,
                "svg.fonttype": "none",
                "svg.hashsalt": "ntqr-0.9",
                "font.size": 13,
                "axes.titlesize": 17,
                "axes.titleweight": "bold",
                "axes.labelsize": 14,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                "legend.fontsize": 11.5,
                "figure.titlesize": 18,
            }
        ):
            for name in selected:
                BUILDERS[name](context)
    finally:
        for number in set(plt.get_fignums()) - existing:
            plt.close(number)
    return context.written
