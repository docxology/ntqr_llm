"""Single source of truth for every number in the manuscript.

The manuscript and supplemental are *generated* from templates: every numeric
claim is a ``{{token}}`` filled from this module, which reads the generated LLM,
simulation, bootstrap, and validation JSON artifacts. Nothing here is hand-typed
from a prior run -- so the prose can never drift from the data the way it did
before this pipeline existed.

``compute_values(out_dir)`` returns a flat ``dict[str, str]`` of ready-to-insert
strings (scalars formatted for prose, and whole markdown tables). The renderer
(``render.py``) substitutes them and fails loudly on any missing or leftover
token, which is the determinism guarantee.
"""

from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

NTQR_SRC = Path(__file__).resolve().parents[2] / "python" / "src"
sys.path.insert(0, str(NTQR_SRC))
from ntqr.llm import personas  # noqa: E402

# Persona display order, matching ntqr.llm.personas.PERSONAS.
PERSONA_ORDER = ["Optimist", "Neutral", "Pessimist"]
LABELS = ("a", "b")
LABEL_CALL = {"a": "bullish", "b": "bearish"}


def _load(path: Path) -> dict | None:
    """Read a provenance-wrapped JSON artifact, returning its ``result`` body."""
    if not path.exists():
        return None
    doc = json.loads(path.read_text())
    return doc.get("result", doc) if isinstance(doc, dict) else doc


def _model_dirname(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


def f3(x: float | None) -> str:
    """Three-decimal fixed (the manuscript's accuracy/MAE convention)."""
    return "n/a" if x is None else f"{x:.3f}"


def f2(x: float | None) -> str:
    """Two-decimal fixed (disagreement and coarse prose)."""
    return "n/a" if x is None else f"{x:.2f}"


def f4(x: float | None) -> str:
    """Four-decimal fixed (supplemental's exact-recovery tables)."""
    return "n/a" if x is None else f"{x:.4f}"


def frac(x: float | None, max_den: int = 64) -> str:
    """Exact fraction string for a float that came from rational ntqr algebra."""
    if x is None:
        return "n/a"
    fr = Fraction(x).limit_denominator(max_den)
    return (
        str(fr.numerator) if fr.denominator == 1 else f"{fr.numerator}/{fr.denominator}"
    )


def _model_dir(out: Path, model: str) -> Path:
    return out / "models" / _model_dirname(model)


def _diag(out: Path, model: str) -> dict | None:
    return _load(_model_dir(out, model) / "evaluability_diagnostic.json")


def _pattern_table(label_counts: dict) -> str:
    """Render an S2-style vote-pattern table from {truth: {pattern: count}}."""
    rows = ["| Truth | Vote pattern | Count |", "|---|---|---|"]
    first_a, first_b = True, True
    for truth in ("a", "b"):
        for pattern, count in label_counts.get(truth, {}).items():
            if truth == "a" and first_a:
                tlabel, first_a = "a (bullish)", False
            elif truth == "b" and first_b:
                tlabel, first_b = "b (bearish)", False
            else:
                tlabel = truth
            rows.append(f"| {tlabel} | {pattern} | {count} |")
    return "\n".join(rows)


def _per_judge_summary(diag: dict) -> str:
    """e.g. 'Optimist a×21/b×11, Neutral a×20/b×12, Pessimist b×29/a×3'."""
    bits = []
    for name in PERSONA_ORDER:
        cc = diag["vote_variation"][name]["call_counts"]
        a = cc.get("bullish", 0)
        b = cc.get("bearish", 0)
        if a >= b:
            bits.append(f"{name} a×{a}/b×{b}")
        else:
            bits.append(f"{name} b×{b}/a×{a}")
    return ", ".join(bits)


def compute_values(out: Path) -> dict[str, str]:
    """Compute every manuscript token from the JSON artifacts under ``out``."""
    summary = _load(out / "summary.json")
    if summary is None:
        raise FileNotFoundError(f"missing summary.json under {out}")

    cms = summary["cross_model_statistics"]
    diag_block = summary["evaluability_diagnostic"]
    evaluated = summary["evaluated_models"]
    failed = summary["failed_models"]
    requested = summary["requested_models"]
    best_model = summary["best_model"]

    n_scen = summary["n_scenarios"]
    floor = cms["sampling_noise_floor"]
    de = cms["disagreement_evaluable"]
    du = cms["disagreement_unevaluable"]

    # Authored deck composition, read from the best model's label counts (the
    # outer keys are the authored truth, so their sums are the deck split).
    best_lc = _load(_model_dir(out, best_model) / "label_vote_counts.json")
    n_a = sum(best_lc["a"].values())
    n_b = sum(best_lc["b"].values())
    min_label = min(n_a, n_b)

    comp = _load(_model_dir(out, best_model) / "comparison.json")
    stats = _load(_model_dir(out, best_model) / "recovery_stats.json")
    mv_stats = _load(
        _model_dir(out, best_model) / "recovery_stats_majority_voting.json"
    )

    v: dict[str, str] = {}

    # --- experiment dimensions ---
    v["N"] = "3"
    v["T"] = "1"
    v["Q"] = str(n_scen)
    v["R"] = "2"
    # Decoding temperature: injected from the run, never hard-coded in prose.
    _temp = summary.get("temperature", 0.0)
    v["TEMP"] = f"{_temp:g}"
    v["PROMPT_VERSION"] = str(summary.get("prompt_version", "unknown"))
    v["PROMPT_HASH"] = str(summary.get("prompt_hash", "unknown"))
    v["DECODING"] = "deterministic" if _temp == 0 else "stochastic"
    if _temp == 0:
        v["SAMPLING_LIMITATION"] = (
            "one deterministic sweep. The recovery result is one strong data "
            "point, not a population estimate; the recovery MAE has no sampling "
            "distribution here because decoding is deterministic. A stronger "
            "design would vary scenarios, repeat at non-zero temperature, and "
            "report a distribution of recovery MAE."
        )
    else:
        v["SAMPLING_LIMITATION"] = (
            f"a single stochastic draw at temperature {v['TEMP']}. The recovery "
            "result is one sample, not a population estimate; because decoding "
            "is stochastic, repeating the sweep would now yield a distribution "
            "of recovery MAE (a fixed temperature-0 run could not). A stronger "
            "design would repeat the draw many times and report that "
            "distribution, and also vary the scenario set."
        )
    v["N_MODELS"] = str(len(requested))
    v["N_EVALUABLE"] = str(len(evaluated))
    v["N_UNEVALUABLE"] = str(len(failed))
    v["EVAL_SET"] = ", ".join(f"`{m}`" for m in evaluated)
    v["FAILED_SET"] = ", ".join(f"`{m}`" for m in failed)
    v["N_PERSONAS"] = "3"
    v["N_TERMS"] = str(stats["n_accuracy_terms"])
    # Figures emitted by the orchestrator (counted from disk so prose tracks it).
    v["N_FIGURES"] = str(len(sorted(out.glob("*.png"))) or 5)

    # Spelled-out variants for in-sentence prose (counts <= twelve read better
    # as words than digits). Tables and parenthetical stats use the digit forms.
    _words = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
        11: "eleven",
        12: "twelve",
    }

    def _word(n: int) -> str:
        return _words.get(n, str(n))

    v["N_MODELS_W"] = _word(len(requested))
    v["N_EVALUABLE_W"] = _word(len(evaluated))
    v["N_UNEVALUABLE_W"] = _word(len(failed))
    v["N_TERMS_W"] = _word(stats["n_accuracy_terms"])
    v["N_FIGURES_W"] = _word(len(sorted(out.glob("*.png"))) or 5)
    if len(failed) == 1:
        v["UNEVALUABLE_COUNT_SENTENCE"] = (
            f"one of {v['N_MODELS_W']} models was not evaluable, and a "
            "label-free diagnostic predicted it in advance"
        )
    else:
        v["UNEVALUABLE_COUNT_SENTENCE"] = (
            f"{v['N_UNEVALUABLE_W']} of {v['N_MODELS_W']} models were not "
            "evaluable, and a label-free diagnostic predicted all "
            f"{v['N_UNEVALUABLE_W']} in advance"
        )

    # --- deck / prevalence ---
    v["N_BULLISH"] = str(n_a)
    v["N_BEARISH"] = str(n_b)
    v["MIN_LABEL_COUNT"] = str(min_label)
    v["TRUE_PREV_A"] = f3(comp["true_prevalence_a"])
    v["REC_PREV_A"] = f3(comp["recovered_prevalence_a"])
    v["PREV_ABS_ERR"] = f3(stats["prevalence_abs_error"])
    v["TRUE_PREV_A4"] = f4(comp["true_prevalence_a"])
    v["REC_PREV_A4"] = f4(comp["recovered_prevalence_a"])
    v["PREV_ABS_ERR4"] = f4(stats["prevalence_abs_error"])
    v["REC_PREV_FRAC"] = frac(comp["recovered_prevalence_a"])

    # --- sampling-noise floor ---
    v["FLOOR"] = f3(floor)
    v["FLOOR2"] = f2(floor)
    v["EXCELLENT"] = "0.05"

    # --- best-model recovery headline ---
    v["BEST_MODEL"] = best_model
    v["BEST_MODEL_DIR"] = _model_dirname(best_model)
    v["REC_MAE"] = f3(stats["recovery_mae"])
    v["REC_MAE4"] = f4(stats["recovery_mae"])
    v["REC_MAE_A"] = f3(stats["recovery_mae_by_label"]["a"])
    v["REC_MAE_B"] = f3(stats["recovery_mae_by_label"]["b"])
    v["REC_MAE_A4"] = f4(stats["recovery_mae_by_label"]["a"])
    v["REC_MAE_B4"] = f4(stats["recovery_mae_by_label"]["b"])
    v["MV_REC_MAE"] = f3(mv_stats["recovery_mae"])
    v["MV_REC_MAE4"] = f4(mv_stats["recovery_mae"])
    v["MAX_ABS_ERR"] = f3(stats["max_abs_error"])

    # Pessimist's poor bullish accuracy -- the "quantifies a bad judge" hook.
    pess = next(r for r in comp["per_persona"] if r["persona"] == "Pessimist")
    v["PESS_TRUE_A"] = f2(pess["true_accuracy"]["a"])
    v["PESS_REC_A"] = f2(pess["recovered_accuracy_unsupervised"]["a"])
    v["PESS_ABS_A"] = f3(
        abs(pess["recovered_accuracy_unsupervised"]["a"] - pess["true_accuracy"]["a"])
    )

    # --- recovery-quality prose (tracks the actual MAE: exact vs approximate)
    # so the adjectives can never contradict the injected numbers, at any
    # temperature.
    mae = stats["recovery_mae"]
    within = ""
    if mae is not None and floor is not None and 0 < mae <= floor:
        within = f", within the {v['FLOOR']} sampling-noise floor"
    if mae is not None and mae == 0:
        v["RECOVERY_VERB"] = "reproduced the authored truth exactly"
        v["BYLABEL_SUFFIX"] = f" -- exact on all {v['N_TERMS_W']} accuracy terms"
        v["PESS_RECOVERED_PHRASE"] = f"as exactly {v['PESS_REC_A']}"
        v["EXACTNESS_NOTE"] = (
            f" The recovery is *exact* here (recovery MAE {v['REC_MAE']}) "
            "because this particular vote pattern is almost perfectly "
            "error-independent; §3.5 and the limitations frame exactness as a "
            "property of this data, not a general guarantee."
        )
        v["ABSTRACT_RECOVERY"] = (
            f"recovered every persona accuracy **exactly** -- a mean absolute "
            f"error of **{v['REC_MAE']}** across all {v['N_TERMS_W']} per-label "
            f"accuracy terms -- including a persona's genuinely poor bullish "
            f"accuracy of **{v['PESS_TRUE_A']}**, recovered exactly, with no "
            "labels"
        )
        v["S3_RECOVERY_QUALIFIER"] = "exact recovery"
        v["S3_RECOVERY_INTERPRETATION"] = (
            "The recovered values match the supervised truth exactly because "
            f"this vote pattern is (to the resolution of {n_scen} questions) "
            "error-independent."
        )
        v["DISCUSSION_RECOVERY_CLAIM"] = (
            "purely algebraic, answer-key-free evaluation recovered per-judge "
            "accuracies exactly -- including correctly identifying and "
            "quantifying a *bad* judge"
        )
    else:
        v["RECOVERY_VERB"] = (
            f"tracked the authored truth to a mean absolute error of "
            f"{v['REC_MAE']}{within}"
        )
        v["BYLABEL_SUFFIX"] = f" across {v['N_TERMS_W']} accuracy terms"
        v["PESS_RECOVERED_PHRASE"] = (
            f"as {v['PESS_REC_A']} (against a true {v['PESS_TRUE_A']})"
        )
        v["EXACTNESS_NOTE"] = (
            f" At temperature {v['TEMP']} the recovery is close but not exact "
            f"(recovery MAE {v['REC_MAE']}{within}); §3.5 and the limitations "
            "frame this as one stochastic draw, not a general guarantee."
        )
        v["ABSTRACT_RECOVERY"] = (
            f"recovered persona accuracies to a mean absolute error of "
            f"**{v['REC_MAE']}**{within} across all {v['N_TERMS_W']} per-label "
            f"accuracy terms, with no labels -- including a persona's genuinely "
            f"poor bullish accuracy of **{v['PESS_TRUE_A']}**, recovered as "
            f"**{v['PESS_REC_A']}**"
        )
        v["S3_RECOVERY_QUALIFIER"] = (
            f"close recovery, within the {v['FLOOR']} sampling-noise floor"
            if floor is not None and mae is not None and mae <= floor
            else "close recovery, not exact"
        )
        v["S3_RECOVERY_INTERPRETATION"] = (
            "The recovered values are close to the supervised truth, not exact; "
            "the residual error is the finite-test recovery error reported "
            f"above and remains {within.lstrip(', ') or 'visible'}."
        )
        v["DISCUSSION_RECOVERY_CLAIM"] = (
            "purely algebraic, answer-key-free evaluation recovered per-judge "
            f"accuracies closely (recovery MAE {v['REC_MAE']}{within}) -- "
            "including correctly identifying and quantifying a *bad* judge"
        )

    # --- evaluable-count phrasing (singular vs plural) ---
    n_eval = len(evaluated)
    if n_eval == 1:
        v["EVAL_MODEL_PHRASE"] = "the one model whose three judges all varied"
        v["EVAL_NOUN"] = "the one evaluable model"
        v["BEST_MODEL_INTRO"] = (
            f"{best_model} was the one model whose three judges all varied."
        )
        v["REAL_MODEL_GENERALIZATION"] = (
            "the one real model that cleared the evaluability bar"
        )
        v["SYNTHETIC_VALIDATION_SCOPE"] = (
            "The one real model that cleared the evaluability bar is one data point."
        )
        v["EVALUABLE_SCOPE_HEADING"] = "One evaluable model."
        v["EVALUABLE_SCOPE_BODY"] = (
            f"Only {v['EVAL_SET']} cleared the evaluability bar, so the "
            "head-to-head recovery ranking has a single entrant. The "
            "recovery-quality claim rests on that one real model; the "
            f"*failure-mode* and *diagnostic* claims rest on all "
            f"{v['N_MODELS_W']}."
        )
    else:
        v["EVAL_MODEL_PHRASE"] = (
            f"the {v['N_EVALUABLE_W']} models whose three judges all varied"
        )
        v["EVAL_NOUN"] = f"the {v['N_EVALUABLE_W']} evaluable models"
        v["BEST_MODEL_INTRO"] = (
            f"Among {v['N_EVALUABLE_W']} evaluable models, {best_model} "
            "recovered the authored truth most closely."
        )
        v["REAL_MODEL_GENERALIZATION"] = "the finite set of real evaluable models"
        v["SYNTHETIC_VALIDATION_SCOPE"] = (
            f"The {v['N_EVALUABLE_W']} evaluable real models give a finite "
            "empirical ranking, not a population estimate."
        )
        v["EVALUABLE_SCOPE_HEADING"] = (
            f"{v['N_EVALUABLE_W'].capitalize()} evaluable models."
        )
        v["EVALUABLE_SCOPE_BODY"] = (
            f"{v['EVAL_SET']} cleared the evaluability bar, so the "
            f"head-to-head recovery ranking compares {v['N_EVALUABLE_W']} "
            "real models. The recovery-quality claim rests on those finite "
            f"real vote matrices; the *failure-mode* and *diagnostic* claims "
            f"rest on all {v['N_MODELS_W']}."
        )

    # --- disagreement contrast ---
    v["DIS_EVAL_MEAN"] = f2(de["mean"])
    v["DIS_UNEVAL_MIN"] = f2(du["min"])
    v["DIS_UNEVAL_MAX"] = f2(du["max"])
    v["DIS_UNEVAL_MEAN"] = f2(du["mean"])
    v["DIS_EVAL_MEAN3"] = f3(de["mean"])
    v["DIS_GATES"] = str(cms["disagreement_gates_evaluability"]).lower()

    # Data-driven §3.3 narrative: do NOT hard-code which model is lowest or "the
    # one". Rank every model by disagreement and describe the actual ordering so
    # the headline can never contradict the run (critical once temperature > 0).
    ordered = _ordered_models(out, requested, evaluated)  # ascending disagree
    eval_disagrees = [d for d, m, _ in ordered if m in set(evaluated)]
    uneval = [(d, m) for d, m, _ in ordered if m not in set(evaluated)]
    n_eval = len(evaluated)
    v["DISAGREEMENT_FIG_CAPTION"] = (
        "Per-judge modal-vote concentration (x, low = every judge varies) "
        "against ensemble mean pairwise disagreement (y). Evaluable models "
        "avoid the constant-classifier boundary; unevaluable models hit it. "
        "The y-axis is descriptive, while the x-axis contains the solvability "
        "condition."
    )
    if eval_disagrees and uneval:
        min_eval_d = min(eval_disagrees)
        max_eval_d = max(eval_disagrees)
        min_uneval_d = min(d for d, _ in uneval)
        max_uneval_d = max(d for d, _ in uneval)
        eval_span = (
            f2(min_eval_d)
            if min_eval_d == max_eval_d
            else f"{f2(min_eval_d)}–{f2(max_eval_d)}"
        )
        uneval_span = (
            f2(min_uneval_d)
            if min_uneval_d == max_uneval_d
            else f"{f2(min_uneval_d)}–{f2(max_uneval_d)}"
        )
        uneval_noun = "model" if len(uneval) == 1 else "models"
        all_eval_below = max_eval_d < min_uneval_d
        all_eval_above = min_eval_d >= max_uneval_d
        top_d, top_m = max(uneval, key=lambda t: t[0])
        if all_eval_above:
            v["DISAGREEMENT_INTRO"] = (
                'The intuitive story -- "you need the judges to disagree" -- '
                "is incomplete. In this low-temperature schema run, aggregate "
                f"disagreement happens to separate the groups: the "
                f"{_word(len(uneval))} unevaluable {uneval_noun} "
                f"sat at {uneval_span}, while {v['EVAL_NOUN']} spanned "
                f"{eval_span}. That separation is a symptom of the failed "
                "model's constant-classifier collapse, not the algebraic gate."
            )
            v["DISAGREEMENT_NARRATIVE"] = (
                f"`{top_m}` is unevaluable because one or more personas never "
                "varied, not because its ensemble-disagreement statistic fell "
                "below a universal threshold."
            )
        elif all_eval_below:
            v["DISAGREEMENT_INTRO"] = (
                'The intuitive story -- "you need the judges to disagree" -- '
                "is incomplete and, on this evidence, misleading. The "
                f"{_word(len(uneval))} unevaluable {uneval_noun} all had "
                f"*higher* mean pairwise disagreement ({uneval_span}) than "
                f"{v['EVAL_NOUN']} ({eval_span})."
            )
            ratio = (top_d / min_eval_d) if min_eval_d > 0 else None
            if ratio is not None and ratio >= 1.4:
                mult = (
                    "nearly twice"
                    if 1.7 <= ratio <= 2.3
                    else ("more than twice" if ratio > 2.3 else f"about {ratio:.1f}×")
                )
                contrast = (
                    f"`{top_m}` disagreed {mult} as much ({f2(top_d)}) as "
                    "the evaluable model(s) yet could not be evaluated."
                )
            else:
                contrast = (
                    f"`{top_m}` ({f2(top_d)}) disagreed more than the "
                    "evaluable model(s) yet could not be evaluated."
                )
            v["DISAGREEMENT_NARRATIVE"] = contrast
        else:
            v["DISAGREEMENT_INTRO"] = (
                'The intuitive story -- "you need the judges to disagree" -- '
                "is incomplete. In this run, disagreement did not cleanly "
                f"order evaluability: {v['EVAL_NOUN']} spanned {eval_span}, "
                f"while the unevaluable {uneval_noun} spanned {uneval_span}."
            )
            v["DISAGREEMENT_NARRATIVE"] = (
                "The group-level disagreement statistic is therefore only a "
                "description of the vote matrix, not the solvability condition."
            )
    else:
        v["DISAGREEMENT_INTRO"] = (
            'The intuitive story -- "you need the judges to disagree" -- is '
            "incomplete; the algebraic condition is per-judge variation."
        )
        v["DISAGREEMENT_NARRATIVE"] = (
            "Disagreement and evaluability do not co-vary in this run "
            f"(`disagreement_gates_evaluability = {v['DIS_GATES']}`)."
        )

    # --- §3.5 majority-voting vs error-independent (do/don't coincide) ---
    mv_mae = mv_stats["recovery_mae"]
    ei_mae = stats["recovery_mae"]
    coincide = mv_mae is not None and ei_mae is not None and abs(mv_mae - ei_mae) < 5e-4
    if coincide:
        v["MV_EI_HEADING"] = (
            "Majority voting and error-independent recovery coincided here"
        )
        v["MV_EI_BODY"] = (
            f"For {best_model}, `MajorityVotingEvaluation` and "
            "`ErrorIndependentEvaluation` returned the same selected solution, "
            f"so their recovery MAE was identical ({v['MV_REC_MAE']}). This is a "
            "property of this particular, near-error-independent vote pattern "
            "rather than a general theorem; we report both so the comparison is "
            "visible per run."
        )
    else:
        better = "error-independent" if ei_mae < mv_mae else "majority-voting"
        v["MV_EI_HEADING"] = (
            "Majority voting and error-independent recovery differ here"
        )
        v["MV_EI_BODY"] = (
            f"For {best_model}, `ErrorIndependentEvaluation` (recovery MAE "
            f"{v['REC_MAE']}) and the simpler `MajorityVotingEvaluation` "
            f"(recovery MAE {v['MV_REC_MAE']}) selected different solutions; the "
            f"{better} estimate was closer to the supervised truth. We report "
            "both so the comparison is visible per run."
        )

    # --- §3.6 figure-4 scorecard sentence (single vs many; how far below floor)
    n_eval = len(evaluated)
    eval_word = "single" if n_eval == 1 else _word(n_eval)
    if ei_mae is not None and floor is not None:
        if ei_mae == 0:
            rel = f"far below the {v['FLOOR']} sampling-noise floor"
        elif ei_mae <= floor:
            rel = f"within the {v['FLOOR']} sampling-noise floor"
        else:
            rel = f"above the {v['FLOOR']} sampling-noise floor"
    else:
        rel = "scored against the sampling-noise floor"
    plural = "model sits" if n_eval == 1 else "models sit"
    v["FIG4_SENTENCE"] = (
        f"the {eval_word} evaluable {plural} at recovery MAE {v['REC_MAE']}, " f"{rel}"
    )

    # --- diagnostic ---
    v["DIAG_MATCHED"] = str(diag_block["diagnostic_matched_outcome"]).lower()
    v["PRED_EVAL_SET"] = ", ".join(f"`{m}`" for m in diag_block["predicted_evaluable"])
    v["EVAL_SET"] = ", ".join(f"`{m}`" for m in evaluated)
    v["FAILED_SET"] = ", ".join(f"`{m}`" for m in failed)

    # --- failure-mode breakdown (honest, data-driven; no fabrication) ---
    # Each failed model fell into exactly one bucket: a constant classifier
    # (a judge that never varied) or non-compliance (≥1 unparseable response;
    # votes are never invented, so such a model cannot be solved). We read the
    # bucket straight from each ranking row, so the prose tracks the run.
    rows = summary["ranking_by_recovery_mae"]
    by_model = {r["model"]: r for r in rows}
    constant_fail = []
    noncompliant = []
    for m in failed:
        row = by_model.get(m, {})
        if row.get("unparsed_responses"):
            noncompliant.append((m, row["unparsed_responses"]))
        elif row.get("constant_classifiers"):
            constant_fail.append(m)
        else:
            # Defensive: any other failure (e.g. unreachable) is reported too.
            noncompliant.append((m, row.get("unparsed_responses") or 0))
    v["N_CONSTANT_FAIL"] = str(len(constant_fail))
    v["N_NONCOMPLIANT"] = str(len(noncompliant))
    total_calls = n_scen * 3
    if noncompliant:
        parts = ", ".join(f"`{m}` ({u}/{total_calls})" for m, u in noncompliant)
        v["NONCOMPLIANCE_NOTE"] = (
            f"At temperature {v['TEMP']}, {_word(len(noncompliant))} "
            f"model(s) emitted at least one unparseable response and were "
            f"reported non-compliant rather than solved on invented votes: "
            f"{parts} (persona calls that could not be parsed into a "
            f"bullish/bearish vote). Votes are never fabricated, so a "
            f"non-compliant model is excluded from the unsupervised solve "
            f"while still contributing its parsed votes to the disagreement "
            f"comparison."
        )
    else:
        v["NONCOMPLIANCE_NOTE"] = (
            f"At temperature {v['TEMP']}, every model returned a parseable "
            f"vote on all {total_calls} persona calls -- no responses were "
            f"unparseable, so no model was dropped for non-compliance."
        )
    if constant_fail and not noncompliant:
        if v["DIS_GATES"] == "true":
            v["WARNING_NOT_DISAGREE_CLAIM"] = (
                '*not* "did the ensemble disagree enough?"; the failing '
                "model sits at low disagreement because its personas collapsed "
                "into constant classifiers."
            )
            v["S6_DISAGREEMENT_TEXT"] = (
                "In this low-temperature schema run, disagreement happens to "
                "sort the models because the only unevaluable model is a "
                "zero-disagreement constant-classifier collapse. That is not "
                "the NTQR solvability rule; it is a visible symptom of the "
                "per-judge rule. The diagnostic therefore keys on per-judge "
                "variation, not on ensemble disagreement."
            )
        else:
            v["WARNING_NOT_DISAGREE_CLAIM"] = (
                '*not* "they didn\'t disagree enough" -- the failing '
                "ensembles disagreed more."
            )
            v["S6_DISAGREEMENT_TEXT"] = (
                "Disagreement does not order with evaluability; the presence "
                "of a constant classifier does. This is why the diagnostic "
                "keys on per-judge variation, not on ensemble disagreement."
            )
    elif noncompliant and not constant_fail:
        v["WARNING_NOT_DISAGREE_CLAIM"] = (
            '*not* "they did not disagree enough" -- the blocker was missing '
            "valid binary votes, which the headline schema-constrained run is "
            "designed to isolate."
        )
        v["S6_DISAGREEMENT_TEXT"] = (
            "Disagreement is computed only from observed votes. A missing or "
            "unparseable vote is an abstention, so the diagnostic treats it as "
            "a matrix-completeness failure rather than inventing a label."
        )
    else:
        v["WARNING_NOT_DISAGREE_CLAIM"] = "*not* an ensemble-disagreement threshold."
        v["S6_DISAGREEMENT_TEXT"] = (
            "Disagreement is descriptive; evaluability is determined by "
            "per-judge variation and complete parseable votes."
        )

    current_parsed, current_total = _parse_totals(summary)
    baseline = _load(out / "baseline_t0.7_summary.json")
    if baseline:
        base_parsed, base_total = _parse_totals(baseline)
        v["BASELINE_STRESS_NOTE"] = (
            f"The previous temperature {baseline.get('temperature', 0.7):g} "
            f"prompt contract is retained as a stress-test baseline: it parsed "
            f"{base_parsed}/{base_total} cells, whereas the current "
            f"temperature {v['TEMP']} schema-constrained headline parsed "
            f"{current_parsed}/{current_total}. The comparison isolates "
            f"interface compliance from the algebraic evaluability question."
        )
    else:
        v["BASELINE_STRESS_NOTE"] = (
            "No prior temperature-0.7 baseline sidecar is present, so this "
            "render reports only the current schema-constrained headline run."
        )

    # Data-driven abstract disagreement sentence (the headline contrast), built
    # from the actual per-model disagreement ranking so it can never assert a
    # "lowest"/"the one" relation the run does not support.
    _ord = _ordered_models(out, requested, evaluated)
    _eval_d = [d for d, m, _ in _ord if m in set(evaluated)]
    _uneval_d = [d for d, m, _ in _ord if m not in set(evaluated)]
    if _eval_d and _uneval_d:
        lo_un, hi_un = min(_uneval_d), max(_uneval_d)
        lo_ev, hi_ev = min(_eval_d), max(_eval_d)
        eval_span = f2(lo_ev) if len(_eval_d) == 1 else f"{f2(lo_ev)}–{f2(hi_ev)}"
        uneval_span = f2(lo_un) if lo_un == hi_un else f"{f2(lo_un)}–{f2(hi_un)}"
        uneval_noun = "model" if len(_uneval_d) == 1 else "models"
        if hi_ev < lo_un:
            v["ABSTRACT_DISAGREEMENT"] = (
                f"The {_word(len(_uneval_d))} unevaluable {uneval_noun} "
                "had "
                f"*higher* mean pairwise disagreement "
                f"({uneval_span}) than {v['EVAL_NOUN']} ({eval_span})"
            )
        elif lo_ev >= hi_un:
            v["ABSTRACT_DISAGREEMENT"] = (
                "Aggregate disagreement separated this run only because the "
                f"unevaluable model(s) collapsed to {uneval_span}; "
                f"{v['EVAL_NOUN']} spanned {eval_span}"
            )
        else:
            v["ABSTRACT_DISAGREEMENT"] = (
                f"Disagreement did not separate the groups — the evaluable "
                f"model(s) ranged {eval_span} and the unevaluable "
                f"{uneval_span}, overlapping"
            )
    else:
        v["ABSTRACT_DISAGREEMENT"] = (
            "disagreement and evaluability did not co-vary in this run"
        )

    # Data-driven abstract failure sentence (constant vs unparseable buckets).
    if n_fail := len(failed):
        if constant_fail and not noncompliant:
            subject = (
                "The other model collapsed"
                if n_fail == 1
                else f"The other {_word(n_fail)} models each collapsed"
            )
            v["ABSTRACT_FAILURE"] = (
                f"{subject} at least one "
                f"persona into a **constant classifier** (a judge that voted "
                f"one way on all {n_scen} scenarios), which makes the "
                f"error-independent algebra unsolvable."
            )
        elif noncompliant and not constant_fail:
            subject = (
                "The other model was not evaluable"
                if n_fail == 1
                else f"The other {_word(n_fail)} models were not evaluable"
            )
            v["ABSTRACT_FAILURE"] = (
                f"{subject} because "
                f"at least one persona returned an unparseable response; we "
                f"never fabricate a vote, so those models are excluded from the "
                f"solve."
            )
        else:
            v["ABSTRACT_FAILURE"] = (
                f"The other {_word(n_fail)} models were not evaluable: "
                f"{_word(len(constant_fail))} collapsed a persona into a "
                f"**constant classifier** and {_word(len(noncompliant))} "
                f"returned an unparseable response (no votes are fabricated)."
            )
    else:
        v["ABSTRACT_FAILURE"] = "Every model was evaluable on this run."

    # Data-driven §3.2 opening: state precisely why each failed model failed,
    # since at temperature > 0 the buckets can be a mix (constant vs unparseable)
    # rather than all-constant.
    n_fail = len(failed)
    if n_fail == 0:
        v["FAILURE_MODE_INTRO"] = (
            "Every model was evaluable: each judge varied and produced a "
            "parseable vote on every scenario."
        )
    elif constant_fail and not noncompliant:
        subject = (
            "The other model collapsed"
            if n_fail == 1
            else f"The other {_word(n_fail)} models each collapsed"
        )
        v["FAILURE_MODE_INTRO"] = (
            f"{subject} at least one " "persona into a constant classifier:"
        )
    elif noncompliant and not constant_fail:
        subject = (
            "The other model was not evaluable"
            if n_fail == 1
            else f"The other {_word(n_fail)} models were not evaluable"
        )
        v["FAILURE_MODE_INTRO"] = (
            f"{subject} because at "
            "least one persona returned an unparseable response (see the "
            "non-compliance note below):"
        )
    else:
        v["FAILURE_MODE_INTRO"] = (
            f"The other {_word(n_fail)} models were not evaluable: "
            f"{_word(len(constant_fail))} collapsed a persona into a constant "
            f"classifier and {_word(len(noncompliant))} returned at least one "
            "unparseable response. The per-model breakdown:"
        )

    # --- tie-break mean accuracies (selected vs mirror) ---
    eei = _load(_model_dir(out, best_model) / "evaluation_error_independent.json")
    sols = eei["float"]

    def _mean_acc(sol):
        vals = [
            sol["accuracy"][i][lab]["float"] for i in range(3) for lab in ("a", "b")
        ]
        return sum(vals) / len(vals)

    means = sorted((_mean_acc(s) for s in sols), reverse=True)
    v["SEL_MEAN_ACC"] = f2(means[0])
    v["MIRROR_MEAN_ACC"] = f2(means[-1])

    # --- exact fractions for S3 ---
    opt = next(r for r in comp["per_persona"] if r["persona"] == "Optimist")
    v["OPT_B_FRAC"] = frac(opt["recovered_accuracy_unsupervised"]["b"])
    v["PESS_A_FRAC"] = frac(pess["recovered_accuracy_unsupervised"]["a"])

    # --- test counts (parsed from the suites, not hard-coded) ---
    v["N_SCRIPT_TESTS"] = _count_script_tests()
    v["N_LIB_TESTS"] = _count_lib_tests()

    # ============ TABLE BLOCKS ============
    v.update(_recovery_distribution_values(summary))
    v["TABLE_S3_RECOVERY"] = _s3_recovery_table(comp)
    v["TABLE_RECOVERY_RANKING"] = _recovery_ranking_table(summary)
    v["TABLE_CONSTANTS"] = _constants_table(out, requested, evaluated)
    v["TABLE_DISAGREEMENT"] = _disagreement_table(out, requested, evaluated)
    v["TABLE_S6_DISAGREEMENT"] = _s6_disagreement_table(out, requested, evaluated)
    v["TABLE_EVALUATION_OBJECTS"] = _evaluation_objects_table()
    v["TABLE_SCENARIO_EXAMPLES"] = _scenario_examples_table()
    v["TABLE_SCENARIOS_FULL"] = _scenario_table()
    v["TABLE_PERSONA_PROMPTS"] = _persona_prompt_table()
    v["TABLE_MODEL_LINEUP"] = _model_lineup_table(summary)
    v["USER_PROMPT"] = personas.USER_TEMPLATE.format(
        scenario="[scenario text]"
    ).replace("\n", "<br>")
    v["TABLE_PROMPT_HEALTH"] = _prompt_health_table(summary)

    # Per-model S2 vote-pattern sections.
    v.update(_s2_sections(out, requested, evaluated, failed))

    # Worked example: the prevalence algebra run on the best model's REAL votes.
    v.update(_worked_example(out, best_model, comp))

    # Synthetic validation study (deterministic; complements the single real
    # evaluable model). out = outputs/llm/<study>; the sim lives at outputs/sim/.
    sim_root = out.parents[1] / "sim"
    v.update(_simulation_values(sim_root / "synthetic_recovery_study"))
    # Scenario-bootstrap CI on the real evaluable model's recovery.
    v.update(_bootstrap_values(sim_root / "bootstrap_recovery_ci"))
    validation_root = out.parents[1] / "validation" / "statistical_validation"
    v.update(_validation_values(validation_root))

    embedded_figures = (
        len(sorted(out.glob("*.png")))
        + len(sorted((sim_root / "synthetic_recovery_study").glob("*.png")))
        + len(sorted((sim_root / "bootstrap_recovery_ci").glob("*.png")))
        + len(sorted(validation_root.glob("*.png")))
        + len(
            sorted((validation_root.parent / "recovery_validity_report").glob("*.png"))
        )
    )
    v["N_EMBEDDED_FIGURES"] = str(embedded_figures)
    v["N_EMBEDDED_FIGURES_W"] = _word(embedded_figures)

    return v


def _evaluation_objects_table() -> str:
    rows = [
        "| Object | Observed without labels? | Validation-only? | Meaning in this study |",
        "|---|---:|---:|---|",
        "| Scenario text | yes | no | A finite authored question containing only the market facts the model may use. |",
        "| Persona vote | yes | no | One parsed `bullish` or `bearish` response, or an abstention if parsing fails. |",
        "| Bullish / bearish labels | yes | no | Operational response labels: expected next-quarter direction up or down under the stated facts. |",
        "| Authored truth | no | yes | The defensible answer used after the unsupervised solve to check recovery accuracy. |",
        "| Prevalence | recovered | checked | Fraction of scenarios whose authored truth is `bullish` (`a`). |",
        "| Persona accuracy | recovered | checked | Per-persona, per-label correctness rates inferred from vote-pattern structure. |",
        "| Best predictor | recovered convention | checked | The higher-mean-accuracy algebraic branch selected without seeing truth. |",
        "| Task/world model | no | no | Not estimated here; the algebra evaluates judges on a finite test rather than learning market semantics. |",
    ]
    return "\n".join(rows)


_SCENARIO_RATIONALES = {
    "s01": "Growth, estimate beat, and raised guidance all point to upside.",
    "s02": "Repeated misses plus a dividend cut indicate deteriorating fundamentals.",
    "s19": "Mixed but positive: the earnings beat is the operational tie-break despite full valuation.",
    "s20": "Mixed but negative: margin compression and weak macro outweigh sales growth.",
    "s23": "Mixed but positive: cost cuts plus reinvestment create a turnaround path.",
    "s24": "Mixed but negative: low valuation is outweighed by structural decline.",
}


def _scenario_examples_table() -> str:
    rows = [
        "| ID | Operational truth | Why this is the authored answer | Scenario |",
        "|---|---|---|---|",
    ]
    for sid in ("s01", "s02", "s19", "s20", "s23", "s24"):
        scenario = next(s for s in personas.SCENARIOS if s["id"] == sid)
        rows.append(
            f"| {sid} | {scenario['truth']} | "
            f"{_SCENARIO_RATIONALES[sid]} | {scenario['text']} |"
        )
    return "\n".join(rows)


def _scenario_table() -> str:
    rows = [
        "| ID | Truth | Scenario |",
        "|---|---|---|",
    ]
    ambiguous = {"s19", "s20", "s21", "s22", "s23", "s24"}
    for scenario in personas.SCENARIOS:
        marker = " *(mixed)*" if scenario["id"] in ambiguous else ""
        rows.append(
            f"| {scenario['id']} | {scenario['truth']} | "
            f"{scenario['text']}{marker} |"
        )
    return "\n".join(rows)


def _persona_prompt_table() -> str:
    rows = ["| Persona | System prompt |", "|---|---|"]
    for persona in personas.PERSONAS:
        rows.append(f"| {persona['name']} | {persona['system']} |")
    return "\n".join(rows)


def _model_lineup_table(summary: dict) -> str:
    """Template-ready model provenance and status table."""
    ranking = {row["model"]: row for row in summary["ranking_by_recovery_mae"]}
    evaluated = set(summary["evaluated_models"])
    best = summary["best_model"]
    rows = [
        "| Local Ollama tag | Public family citation | Run status | What the status means here |",
        "|---|---|---|---|",
    ]
    for model in summary["requested_models"]:
        row = ranking.get(model, {})
        if model in evaluated:
            mae = row.get("recovery_mae")
            role = "best recovered model" if model == best else "evaluable"
            status = f"{role}; recovery MAE {f3(mae)}"
            meaning = (
                "All three personas varied, every cell parsed, and the "
                "error-independent solve returned a selected branch."
            )
        elif row.get("unparsed_responses"):
            status = f"not solved; {row['unparsed_responses']} unparsed cells"
            meaning = (
                "The matrix was incomplete; no local vote was fabricated for "
                "the unsupervised solve."
            )
        elif row.get("constant_classifiers"):
            personas = ", ".join(row["constant_classifiers"])
            status = f"not solved; constant persona(s): {personas}"
            meaning = (
                "At least one judge never varied, so the trio carries no "
                "recoverable error signal for that judge."
            )
        else:
            status = "not solved"
            meaning = "The run recorded no selected unsupervised solution."
        rows.append(
            f"| `{model}` | {_model_family_citation(model)} | {status} | "
            f"{meaning} |"
        )
    return "\n".join(rows)


def _model_family_citation(model: str) -> str:
    if model.startswith("mistral"):
        return "Mistral family [@jiang_2023_mistral]"
    if model.startswith("gemma2"):
        return "Gemma and Gemma 2 family [@gemma_team_2024; @gemma2_team_2024]"
    if model.startswith("gemma3") or model.startswith("gemma4"):
        return "Gemma family reports [@gemma_team_2024; @gemma3_team_2025]"
    if model.startswith("granite"):
        return "IBM Granite 4.1 [@ibm_granite_4_1_2026]"
    if model.startswith("smollm2"):
        return "SmolLM2 family [@ben_allal_2025_smollm2]"
    return "local Ollama model tag"


def _recovery_distribution_values(summary: dict) -> dict[str, str]:
    rows = [
        row
        for row in summary["ranking_by_recovery_mae"]
        if row.get("recovery_mae") is not None
    ]
    maes = sorted(row["recovery_mae"] for row in rows)
    if not maes:
        return {
            "REC_MAE_MIN": "n/a",
            "REC_MAE_MEDIAN": "n/a",
            "REC_MAE_MAX": "n/a",
            "REC_MAE_SPREAD": "n/a",
        }
    mid = len(maes) // 2
    median = maes[mid] if len(maes) % 2 else (maes[mid - 1] + maes[mid]) / 2
    return {
        "REC_MAE_MIN": f3(maes[0]),
        "REC_MAE_MEDIAN": f3(median),
        "REC_MAE_MAX": f3(maes[-1]),
        "REC_MAE_SPREAD": f3(maes[-1] - maes[0]),
    }


def _recovery_ranking_table(summary: dict) -> str:
    rows = [
        "| Rank | Model | Recovery MAE | Max term error | Prevalence error | MV MAE | Validation role |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    rank = 1
    for row in summary["ranking_by_recovery_mae"]:
        if row.get("recovery_mae") is None:
            continue
        role = (
            "best recovered matrix"
            if row["model"] == summary["best_model"]
            else "evaluable comparison"
        )
        rows.append(
            f"| {rank} | `{row['model']}` | {f3(row['recovery_mae'])} | "
            f"{f3(row.get('max_abs_error'))} | "
            f"{f3(row.get('prevalence_abs_error'))} | "
            f"{f3(row.get('mv_recovery_mae'))} | {role} |"
        )
        rank += 1
    return "\n".join(rows)


def _parse_totals(summary: dict) -> tuple[int, int]:
    parsed = 0
    total = 0
    for row in summary.get("ranking_by_recovery_mae", []):
        for health in (row.get("prompt_health") or {}).values():
            parsed += int(health.get("n_parsed", 0))
            total += int(health.get("n_total", 0))
    return parsed, total


def _prompt_health_table(summary: dict) -> str:
    rows = [
        "| Model | Persona | Primary | Repaired | Parsed | Invalid | Modal | Entropy | Parser |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model_row in summary["ranking_by_recovery_mae"]:
        health = model_row.get("prompt_health") or {}
        for persona in PERSONA_ORDER:
            row = health.get(persona)
            if not row:
                continue
            methods = ", ".join(
                f"{_parser_label(k)}:{v}"
                for k, v in row.get("parse_methods", {}).items()
            )
            rows.append(
                f"| {_model_table_label(model_row['model'])} | {persona} | "
                f"{f3(row.get('primary_parse_rate'))} | "
                f"{row.get('n_repaired', 0)} | "
                f"{row.get('n_parsed', 0)} | {row.get('n_unparsed', 0)} | "
                f"{f3(row.get('modal_fraction'))} | "
                f"{f3(row.get('entropy_bits'))} | {methods or 'n/a'} |"
            )
    return "\n".join(rows)


def _model_table_label(model: str) -> str:
    """Compact label for dense PDF tables; full model names stay in JSON."""
    labels = {
        "granite4.1:3b": "granite",
        "mistral:latest": "mistral",
        "gemma2:2b": "gemma2",
        "gemma3:4b": "gemma3",
        "gemma4:latest": "gemma4",
        "smollm2:135m-instruct-q4_K_S": "smollm2",
    }
    return labels.get(model, model)


def _parser_label(method: str) -> str:
    labels = {
        "schema__json": "schema",
        "schema_json": "schema",
        "json": "json",
        "regex": "regex",
    }
    return labels.get(method, method)


def _validation_values(validation_dir: Path) -> dict[str, str]:
    """Tokens for the statistical validation matrix."""
    summary = _load(validation_dir / "summary.json")
    if summary is None:
        raise FileNotFoundError(
            f"missing validation summary.json under {validation_dir}; "
            "run: .venv/bin/python scripts/validation/statistical_validation_report.py"
        )
    rows = [
        "| Check | Result | Evidence |",
        "|---|---|---|",
    ]
    for check in summary["checks"]:
        result = "PASS" if check["passed"] else "FAIL"
        rows.append(f"| {check['label']} | {result} | {check['evidence']} |")
    return {
        "VALIDATION_N_CHECKS": str(summary["n_checks"]),
        "VALIDATION_N_PASSED": str(summary["n_passed"]),
        "VALIDATION_N_FAILED": str(summary["n_failed"]),
        "VALIDATION_TABLE": "\n".join(rows),
    }


def _bootstrap_values(boot_dir: Path) -> dict[str, str]:
    """Tokens for the scenario-bootstrap recovery CI (§3.1) on the real model.

    Reads the deterministic summary from
    ``scripts/sim/bootstrap_recovery_ci.py``. Fails loudly if absent: like the
    synthetic study it is fully reproducible offline from committed votes.
    """
    summary = _load(boot_dir / "summary.json")
    if summary is None:
        raise FileNotFoundError(
            f"missing bootstrap summary.json under {boot_dir}; "
            "run: .venv/bin/python scripts/sim/bootstrap_recovery_ci.py"
        )
    mae = summary["recovery_mae"]
    prev = summary["prevalence_abs_error"]
    return {
        "BOOT_N": str(summary["n_boot"]),
        "BOOT_POINT_MAE": f3(summary["point_recovery_mae"]),
        "BOOT_MAE_CI_LO": f3(mae["ci_lo"]),
        "BOOT_MAE_CI_HI": f3(mae["ci_hi"]),
        "BOOT_PREV_CI_LO": f3(prev["ci_lo"]),
        "BOOT_PREV_CI_HI": f3(prev["ci_hi"]),
        "BOOT_DEGENERATE": f"{summary['degenerate_rate'] * 100:.0f}%",
    }


def _simulation_values(sim_dir: Path) -> dict[str, str]:
    """Tokens for the synthetic NTQR validation study (§3.7).

    Reads the deterministic summary written by
    ``scripts/sim/synthetic_recovery_study.py`` and formats its convergence fit,
    alarm-rate characterisation, and two markdown tables. Fails loudly if the
    summary is absent: the study is fully reproducible with no network, so a
    missing artifact means the generator simply has not been run yet.
    """
    summary = _load(sim_dir / "summary.json")
    if summary is None:
        raise FileNotFoundError(
            f"missing synthetic-study summary.json under {sim_dir}; "
            "run: .venv/bin/python scripts/sim/synthetic_recovery_study.py"
        )
    conv = summary["convergence"]
    corr = summary["correlation"]
    ens = summary["ensemble"]
    fit = conv["sqrt_q_fit_accuracy"]
    fit_prev = conv["sqrt_q_fit_prevalence"]

    def _pct(x: float | None) -> str:
        return "n/a" if x is None else f"{x * 100:.0f}%"

    accs = ", ".join(f"({j['alpha']:.2f}, {j['beta']:.2f})" for j in ens["accuracies"])

    conv_tab = [
        "| Q | Accuracy MAE | Prevalence error | Trials solved |",
        "|---|---|---|---|",
    ]
    for r in conv["rows"]:
        conv_tab.append(
            f"| {r['q']} | {f3(r['accuracy_mae_mean'])} | "
            f"{f3(r['prevalence_abs_error_mean'])} | "
            f"{r['n_solved']}/{r['n_trials']} |"
        )

    corr_tab = [
        "| Error structure | Alarm fires | Mean $D$ | Recovery error (alarm silent) |",
        "|---|---|---|---|",
    ]
    for r in corr["rows"]:
        silent = r["silent_prevalence_abs_error_mean"]
        corr_tab.append(
            f"| {r['label']} | {_pct(r['alarm_rate'])} | "
            f"{f4(r['discriminant_mean'])} | "
            f"{f3(silent) if silent is not None else 'n/a'} |"
        )

    indep = next(r for r in corr["rows"] if r["label"].startswith("independent"))

    tie = summary["tie_break"]
    rob = summary["robustness"]
    tie_tab = [
        "| Judge accuracy | Catastrophic mirror-selection | Mean prevalence error |",
        "|---|---|---|",
    ]
    for r in tie["rows"]:
        tie_tab.append(
            f"| {f2(r['accuracy'])} | {_pct(r['catastrophic_rate'])} | "
            f"{f3(r['prevalence_abs_error_mean'])} |"
        )
    rob_tab = [
        "| Ensemble | Prevalence | Fitted slope | $R^2$ |",
        "|---|---|---|---|",
    ]
    for r in rob["rows"]:
        rob_tab.append(
            f"| {r['label']} | {f2(r['prevalence_a'])} | {f2(r['slope'])} | "
            f"{f3(r['r_squared'])} |"
        )

    mvei = summary["mv_vs_ei"]
    ei_c, mv_c = mvei["ei_best_cell"], mvei["mv_best_cell"]
    mvei_tab = [
        "| Prevalence | Judge accuracy | EI MAE | MV MAE | EI wins |",
        "|---|---|---|---|---|",
    ]
    for r in mvei["rows"]:
        mvei_tab.append(
            f"| {f2(r['prevalence_a'])} | {f2(r['accuracy'])} | "
            f"{f3(r['ei_mae_mean'])} | {f3(r['mv_mae_mean'])} | "
            f"{_pct(r['ei_win_rate'])} |"
        )

    return {
        "SIM_N_TRIALS": str(conv["n_trials"]),
        "SIM_Q_MIN": str(conv["q_min"]),
        "SIM_Q_MAX": str(conv["q_max"]),
        "SIM_CORR_Q": str(corr["q"]),
        "SIM_ACC_MAE_QMIN": f3(conv["smallest_q_acc_mae"]),
        "SIM_ACC_MAE_QMAX": f3(conv["largest_q_acc_mae"]),
        "SIM_SLOPE": f2(fit["slope"]),
        "SIM_R2": f3(fit["r_squared"]),
        "SIM_PREV_SLOPE": f2(fit_prev["slope"]),
        "SIM_PREVALENCE": f2(ens["prevalence_a"]),
        "SIM_ACCS": accs,
        "SIM_FALSE_ALARM": _pct(corr["independent_false_alarm_rate"]),
        "SIM_COMMON_ALARM": _pct(corr["common_mode_max_alarm_rate"]),
        "SIM_ANTI_ALARM": _pct(corr["anti_max_alarm_rate"]),
        "SIM_COMMON_SILENT_ERR": f3(corr["common_mode_max_silent_error"]),
        "SIM_INDEP_SILENT_ERR": f3(indep["silent_prevalence_abs_error_mean"]),
        "SIM_TABLE_CONVERGENCE": "\n".join(conv_tab),
        "SIM_TABLE_CORRELATION": "\n".join(corr_tab),
        # robustness across diverse ensembles
        "SIM_ROBUST_N": str(rob["n_ensembles"]),
        "SIM_ROBUST_SLOPE_LO": f2(rob["slope_min"]),
        "SIM_ROBUST_SLOPE_HI": f2(rob["slope_max"]),
        "SIM_TABLE_ROBUSTNESS": "\n".join(rob_tab),
        # tie-break safe region
        "SIM_TIEBREAK_Q": str(tie["q"]),
        "SIM_TIEBREAK_PREV": f2(tie["prevalence_a"]),
        "SIM_TIEBREAK_SAFE_ACC": f2(tie["safe_accuracy"]),
        "SIM_TIEBREAK_MAX_CAT": _pct(tie["max_catastrophic_rate"]),
        "SIM_TABLE_TIEBREAK": "\n".join(tie_tab),
        # EI vs majority-voting: exactness vs robustness
        "SIM_MVEI_Q": str(mvei["q"]),
        "SIM_MVEI_EI_PREV": f2(ei_c["prevalence_a"]),
        "SIM_MVEI_EI_ACC": f2(ei_c["accuracy"]),
        "SIM_MVEI_EI_EIMAE": f3(ei_c["ei_mae_mean"]),
        "SIM_MVEI_EI_MVMAE": f3(ei_c["mv_mae_mean"]),
        "SIM_MVEI_EI_RATIO": f"{ei_c['ratio_mv_over_ei']:.2f}",
        "SIM_MVEI_EI_WIN": _pct(ei_c["ei_win_rate"]),
        "SIM_MVEI_MV_PREV": f2(mv_c["prevalence_a"]),
        "SIM_MVEI_MV_ACC": f2(mv_c["accuracy"]),
        "SIM_MVEI_MV_EIMAE": f3(mv_c["ei_mae_mean"]),
        "SIM_MVEI_MV_MVMAE": f3(mv_c["mv_mae_mean"]),
        "SIM_MVEI_MV_EIWIN": _pct(mv_c["ei_win_rate"]),
        "SIM_TABLE_MVEI": "\n".join(mvei_tab),
    }


def _worked_example(out: Path, best_model: str, comp: dict) -> dict[str, str]:
    """Run the prevalence algebra on the best model's REAL votes.

    Reconstructs the unsupervised evaluator from the on-disk label-vote counts
    and extracts the exact data-sketch moments, the prevalence quadratic, its
    discriminant, and the two prevalence solutions -- so the formalism in §2.5
    is shown operating on real data, not a toy. Every value is computed from the
    votes, never hand-typed. Guarded: any failure yields ``n/a`` tokens so the
    render never breaks on an environment without the ntqr library.
    """
    keys = [
        "WE_F_BBB",
        "WE_TRIO_MOMENT",
        "WE_DELTA",
        "WE_PAIR_01",
        "WE_PAIR_02",
        "WE_PAIR_12",
        "WE_PROD_PAIRS",
        "WE_DISCRIMINANT",
        "WE_HALF_SQRT",
        "WE_ROOT_LO",
        "WE_ROOT_HI",
        "WE_SELECTED_PREV",
        "WE_TRUE_PREV",
        "WE_PREV_ERR",
    ]
    blank = {k: "n/a" for k in keys}
    try:
        from ntqr.r2.datasketches import TrioLabelVoteCounts
        from ntqr.r2.evaluators import ErrorIndependentEvaluation

        raw = _load(_model_dir(out, best_model) / "label_vote_counts.json")
        if not raw:
            return blank
        # JSON serializes the tuple vote-patterns as "a,a,b" strings; the ntqr
        # data class wants tuples, so convert back on read.
        lc = {
            truth: {tuple(k.split(",")): n for k, n in pats.items()}
            for truth, pats in raw.items()
        }
        tvc = TrioLabelVoteCounts(lc).to_TrioVoteCounts()
        freqs = tvc.to_frequencies_exact()
        f_bbb = float(freqs[("b", "b", "b")])
        m = float(tvc.trio_frequency_moment())
        pm = {
            p: float(val) for p, val in tvc.label_pairs_frequency_moments("b").items()
        }
        prod = pm[(0, 1)] * pm[(0, 2)] * pm[(1, 2)]
        ei = ErrorIndependentEvaluation(tvc)
        coeffs = ei.alpha_prevalence_quadratic_terms()
        a_coeff, c_coeff = float(coeffs[2]), float(coeffs[0])
        disc = 1 - 4 * c_coeff / a_coeff if a_coeff else float("nan")
        half_sqrt = (disc**0.5) / 2 if disc >= 0 else float("nan")
        roots = sorted(float(r) for r in ei.alpha_prevalence_estimates())
        sel = comp.get("recovered_prevalence_a")
        true_p = comp.get("true_prevalence_a")
        return {
            "WE_F_BBB": f4(f_bbb),
            "WE_TRIO_MOMENT": f4(m),
            "WE_DELTA": f4(f_bbb - m),
            "WE_PAIR_01": f4(pm[(0, 1)]),
            "WE_PAIR_02": f4(pm[(0, 2)]),
            "WE_PAIR_12": f4(pm[(1, 2)]),
            "WE_PROD_PAIRS": f"{prod:.2e}",
            "WE_DISCRIMINANT": f4(disc),
            "WE_HALF_SQRT": f4(half_sqrt),
            "WE_ROOT_LO": f4(roots[0]),
            "WE_ROOT_HI": f4(roots[-1]),
            "WE_SELECTED_PREV": f4(sel) if sel is not None else "n/a",
            "WE_TRUE_PREV": f4(true_p) if true_p is not None else "n/a",
            "WE_PREV_ERR": (
                f4(abs(sel - true_p))
                if sel is not None and true_p is not None
                else "n/a"
            ),
        }
    except (
        KeyError,
        ValueError,
        TypeError,
        ZeroDivisionError,
        ImportError,
        OSError,
    ):
        return blank


def _s3_recovery_table(comp: dict) -> str:
    rows = [
        "| Persona | Label | True | Recovered | Abs. error |",
        "|---|---|---|---|---|",
    ]
    for r in comp["per_persona"]:
        for lab in LABELS:
            tv = r["true_accuracy"][lab]
            rv = r["recovered_accuracy_unsupervised"][lab]
            err = abs(rv - tv) if tv is not None and rv is not None else None
            rows.append(f"| {r['persona']} | {lab} | {f4(tv)} | {f4(rv)} | {f4(err)} |")
    return "\n".join(rows)


def _ordered_models(out: Path, requested, evaluated):
    """Models sorted by mean pairwise disagreement (ascending)."""
    scored = []
    for m in requested:
        d = _diag(out, m)
        if d is None:
            continue
        scored.append((d["mean_pairwise_disagreement"], m, d))
    return sorted(scored, key=lambda t: t[0])


def _constants_table(out: Path, requested, evaluated) -> str:
    rows = ["| Model | Constant classifier(s) | Evaluable? |", "|---|---|---|"]
    # Evaluable first, then the rest in requested order.
    ordered = [m for m in requested if m in evaluated] + [
        m for m in requested if m not in evaluated
    ]
    for m in ordered:
        d = _diag(out, m)
        if d is None:
            continue
        consts = d["constant_classifiers"]
        cstr = ", ".join(consts) if consts else "—"
        ev = "**yes**" if m in evaluated else "no"
        rows.append(f"| `{m}` | {cstr} | {ev} |")
    return "\n".join(rows)


def _disagreement_table(out: Path, requested, evaluated) -> str:
    rows = [
        "| Model | Mean pairwise disagreement | Evaluable? |",
        "|---|---|---|",
    ]
    for dis, m, d in _ordered_models(out, requested, evaluated):
        ev = "**yes**" if m in evaluated else "no"
        cell = f"**{f2(dis)}**" if m in evaluated else f2(dis)
        rows.append(f"| `{m}` | {cell} | {ev} |")
    return "\n".join(rows)


def _s6_disagreement_table(out: Path, requested, evaluated) -> str:
    rows = [
        "| Model | Mean pairwise disagreement | Constant classifier? | Evaluable |",
        "|---|---|---|---|",
    ]
    for dis, m, d in _ordered_models(out, requested, evaluated):
        consts = d["constant_classifiers"]
        if consts:
            cc = "yes (" + ", ".join(consts) + ")"
        else:
            cc = "no"
        ev = "**yes**" if m in evaluated else "no"
        rows.append(f"| {m} | {f3(dis)} | {cc} | {ev} |")
    return "\n".join(rows)


def _s2_sections(out: Path, requested, evaluated, failed) -> dict[str, str]:
    """One ``S2_<i>`` block per model: header line + vote-pattern table."""
    blocks: dict[str, str] = {}
    ordered = [m for m in requested if m in evaluated] + [
        m for m in requested if m not in evaluated
    ]
    for i, m in enumerate(ordered, start=1):
        d = _diag(out, m)
        lc = _load(_model_dir(out, m) / "label_vote_counts.json")
        if d is None or lc is None:
            continue
        consts = d["constant_classifiers"]
        n_unparsed = d.get("n_unparsed", 0)
        if n_unparsed:
            head = (
                f"**`{m}`** — non-compliant "
                f"({n_unparsed} unparseable response(s); no votes fabricated):"
            )
            foot = (
                f"\nPer-judge: {_per_judge_summary(d)}; "
                f"{n_unparsed} cell(s) abstained, so the model was not solved."
            )
        elif not consts:
            head = f"**`{m}`** — evaluable (every judge varies):"
            foot = (
                f"\nPer-judge: {_per_judge_summary(d)} → "
                f"**no constant classifier**."
            )
        else:
            kind = "constant" if len(consts) == 1 else "two constant classifiers"
            head = (
                f"**`{m}`** — "
                + " *and* ".join(consts)
                + (" constant:" if len(consts) == 1 else " constant:")
            )
            named = ", ".join(consts)
            foot = (
                f"\n{named} never varied → "
                f"**{'constant classifier' if len(consts)==1 else kind} "
                f"({named})**."
            )
        blocks[f"S2_{i}"] = f"{head}\n\n{_pattern_table(lc)}\n{foot}"
    # Combined token: every model's block joined in order, so the supplemental
    # template needs ONE slot regardless of how many models the run contained
    # (the per-model S2_<i> tokens are retained for backward compatibility).
    blocks["S2_ALL"] = "\n\n".join(
        blocks[f"S2_{i}"] for i in range(1, len(ordered) + 1) if f"S2_{i}" in blocks
    )
    return blocks


def _count_script_tests() -> str:
    """Count test_* functions across the project test suite (``tests/``)."""
    root = Path(__file__).resolve().parents[2] / "tests"
    if not root.is_dir():
        return "?"
    n = 0
    for py in root.rglob("test_*.py"):
        n += sum(
            1
            for line in py.read_text().splitlines()
            if line.lstrip().startswith("def test_")
        )
    return str(n)


def _count_lib_tests() -> str:
    """Count test_* functions across the ntqr library test tree."""
    root = Path(__file__).resolve().parents[2] / "python" / "tests"
    if not root.is_dir():
        return "?"
    n = 0
    for py in root.rglob("test_*.py"):
        n += sum(
            1
            for line in py.read_text().splitlines()
            if line.lstrip().startswith("def test_")
        )
    return str(n)
