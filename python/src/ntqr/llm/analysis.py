"""Pure analysis of LLM persona votes: response parsing, vote diagnostics, and
recovery scoring.

Consumed by the ``scripts/llm`` orchestrators:

* ``trader_personas_evaluation.py`` -- one model, one trio of personas.
* ``multi_llm_personas_evaluation.py`` -- the same trio run across several
  models, with cross-model statistics.

Everything here is deliberately free of I/O beyond the duck-typed LLM client
passed in (the network client itself lives in ``scripts/llm/_llm.py``), and
imports no matplotlib, so the numeric core can be unit-tested with no network
and no display (see ``tests/test_llm_analysis.py``).
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter

from ntqr.llm import personas
from ntqr.llm.recovery_scoring import safe_float, select_best

# Interpretation thresholds for the recovery MAE, reused by the orchestrator's
# figures and data sidecars so the plots and the JSON agree on what the numbers
# mean. With Q questions split across two labels, an accuracy estimated from one
# label rests on ~Q/2 questions; its binomial standard error is at most
# sqrt(0.25 / k) for the minority label's k questions -- the sampling-noise
# floor below which a recovery error is within the conservative sampling
# uncertainty of the supervised comparison. The 0.05 mark is a tighter
# "excellent" bar.
RECOVERY_MAE_EXCELLENT = 0.05


def sampling_noise_floor(n_questions, n_labels=2, min_label_count=None):
    """Worst-case binomial SE of a per-label accuracy estimate.

    A persona's per-label accuracy is estimated from the subset of questions
    carrying that label. The standard error of a proportion peaks at p=0.5,
    giving ``sqrt(0.25 / k)`` where ``k`` is the number of questions backing the
    estimate. A recovery MAE at or below this is within the noise of the
    supervised estimate it is being compared to, so no method (supervised or
    not) could reliably do better on this much data.

    For an evenly split deck ``k`` is ``n_questions / n_labels``. When the deck
    is imbalanced, the *minority* label backs the weakest estimate, so pass its
    count as ``min_label_count`` to get the conservative (largest) floor rather
    than the optimistic balanced one.
    """
    if min_label_count is not None:
        per_label = min_label_count
    else:
        per_label = n_questions / n_labels if n_labels else n_questions
    if per_label <= 0:
        return None
    return math.sqrt(0.25 / per_label)


def min_truth_label_count():
    """Smallest per-label count in the authored answer key.

    The minority label backs the weakest per-label accuracy estimate, so this
    is what sets the conservative sampling-noise floor for an imbalanced deck.
    """
    counts: dict[str, int] = {}
    for scenario in personas.SCENARIOS:
        label = personas.CALL_TO_LABEL[scenario["truth"]]
        counts[label] = counts.get(label, 0) + 1
    return min(counts.values()) if counts else 0


def descriptive_stats(values):
    """n / min / max / mean / median / std / spread of a list of numbers.

    Population standard deviation (ddof=0). ``None`` entries are dropped first;
    an empty list yields all-``None`` so callers can serialize it unconditionally.
    """
    xs = sorted(v for v in values if v is not None)
    n = len(xs)
    if n == 0:
        return {
            "n": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
            "spread": None,
        }
    mean = sum(xs) / n
    mid = n // 2
    median = xs[mid] if n % 2 else (xs[mid - 1] + xs[mid]) / 2
    std = math.sqrt(sum((x - mean) ** 2 for x in xs) / n)
    return {
        "n": n,
        "min": xs[0],
        "max": xs[-1],
        "mean": mean,
        "median": median,
        "std": std,
        "spread": xs[-1] - xs[0],
    }


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model response."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Strip code fences / surrounding prose: grab the outermost {...}.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"no JSON object found in model response: {text!r}")


def _validate_schema_call(obj) -> tuple[str | None, str | None]:
    if not isinstance(obj, dict):
        return None, "response is not a JSON object"
    extra = sorted(set(obj) - {"call"})
    if extra:
        return None, "unexpected key(s): " + ", ".join(extra)
    if "call" not in obj:
        return None, "missing required key: call"
    call = obj["call"]
    if not isinstance(call, str):
        return None, "call is not a string"
    if call not in personas.CALL_TO_LABEL:
        return None, f"invalid call: {call!r}"
    return call, None


def parse_call_record(
    raw: str | None, schema_strict: bool = False
) -> dict[str, object]:
    """Extract a call plus the parser route used for diagnostics."""
    if raw is None:
        return {
            "call": None,
            "parse_method": "unavailable",
            "schema_valid": False,
            "invalid_reason": "model unavailable",
        }
    if schema_strict:
        try:
            obj = json.loads(raw.strip())
        except (AttributeError, json.JSONDecodeError):
            return {
                "call": None,
                "parse_method": "none",
                "schema_valid": False,
                "invalid_reason": "response is not exactly one JSON object",
            }
        call, reason = _validate_schema_call(obj)
        return {
            "call": call,
            "parse_method": "schema_json",
            "schema_valid": reason is None,
            "invalid_reason": reason,
        }

    try:
        obj = extract_json(raw)
        call = str(obj.get("call", "")).strip().lower()
        if call in personas.CALL_TO_LABEL:
            return {
                "call": call,
                "parse_method": "json",
                "schema_valid": True,
                "invalid_reason": None,
            }
    except (ValueError, json.JSONDecodeError):
        pass
    low = raw.lower()
    has_bull, has_bear = "bull" in low, "bear" in low
    if has_bull and not has_bear:
        return {
            "call": "bullish",
            "parse_method": "keyword",
            "schema_valid": True,
            "invalid_reason": None,
        }
    if has_bear and not has_bull:
        return {
            "call": "bearish",
            "parse_method": "keyword",
            "schema_valid": True,
            "invalid_reason": None,
        }
    return {
        "call": None,
        "parse_method": "none",
        "schema_valid": False,
        "invalid_reason": "no valid bullish/bearish call found",
    }


def parse_call(raw: str) -> str | None:
    """Extract 'bullish'/'bearish' from a model response, robustly."""
    return parse_call_record(raw)["call"]


def cache_key(client, persona_key: str, scenario_id: str) -> str:
    """Prompt-versioned cache key for one model/persona/scenario cell."""
    return (
        f"{client.backend}|{client.model}|t{client.temperature}|"
        f"p{personas.PROMPT_VERSION}|h{personas.prompt_hash()}|"
        f"{persona_key}|{scenario_id}"
    )


def normalize_cache_record(record) -> dict[str, object]:
    """Return a backward-compatible cache record with diagnostic fields."""
    if isinstance(record, dict):
        call = record.get("call")
        unparsed = bool(record.get("unparsed", call is None))
        attempts = record.get("attempts") or []
        first_attempt = attempts[0] if attempts else {}
        return {
            "call": call if call in personas.CALL_TO_LABEL else None,
            "unparsed": unparsed,
            "raw": record.get("raw"),
            "parse_method": record.get(
                "parse_method", "none" if unparsed else "legacy"
            ),
            "schema_valid": bool(record.get("schema_valid", not unparsed)),
            "invalid_reason": record.get("invalid_reason"),
            "attempts": attempts,
            "repair_used": bool(record.get("repair_used", False)),
            "n_attempts": int(record.get("n_attempts", len(attempts) or 1)),
            "primary_parse_valid": bool(
                record.get(
                    "primary_parse_valid",
                    first_attempt.get("schema_valid", not unparsed),
                )
            ),
            "final_parse_valid": bool(
                record.get("final_parse_valid", not unparsed)
            ),
            "prompt_version": record.get("prompt_version"),
            "prompt_hash": record.get("prompt_hash"),
        }
    call = record if record in personas.CALL_TO_LABEL else None
    return {
        "call": call,
        "unparsed": call is None,
        "raw": None,
        "parse_method": "legacy",
        "schema_valid": call is not None,
        "invalid_reason": None if call is not None else "legacy unparsed",
        "attempts": [],
        "repair_used": False,
        "n_attempts": 1,
        "primary_parse_valid": call is not None,
        "final_parse_valid": call is not None,
        "prompt_version": None,
        "prompt_hash": None,
    }


def response_records(client, cache):
    """Return normalized cache diagnostics for this prompt/model run."""
    out: dict[str, dict[str, dict[str, object]]] = {}
    for scenario in personas.SCENARIOS:
        sid = scenario["id"]
        out[sid] = {}
        for persona in personas.PERSONAS:
            key = cache_key(client, persona["key"], sid)
            if key in cache:
                out[sid][persona["key"]] = normalize_cache_record(cache[key])
    return out


def _entropy(counts):
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum(
        (n / total) * math.log2(n / total) for n in counts.values() if n
    )


def prompt_health(votes, records=None):
    """Prompt compliance and label-variation summary by persona."""
    out = {}
    for persona in personas.PERSONAS:
        key = persona["key"]
        calls = [votes[s["id"]][key] for s in personas.SCENARIOS]
        real = [c for c in calls if c is not None]
        counts = Counter(real)
        methods = Counter()
        n_primary_valid = 0
        n_repair_needed = 0
        n_repaired = 0
        n_invalid_label = 0
        n_unavailable = 0
        if records:
            for scenario in personas.SCENARIOS:
                rec = records.get(scenario["id"], {}).get(key, {})
                methods[str(rec.get("parse_method", "missing"))] += 1
                attempts = rec.get("attempts") or []
                if attempts:
                    first = attempts[0]
                    if first.get("schema_valid"):
                        n_primary_valid += 1
                    if len(attempts) > 1 and not first.get("schema_valid"):
                        n_repair_needed += 1
                        n_repaired += int(not bool(rec.get("unparsed")))
                    for attempt in attempts:
                        reason = str(attempt.get("invalid_reason") or "")
                        if reason.startswith("invalid call"):
                            n_invalid_label += 1
                        if attempt.get("parse_method") == "unavailable":
                            n_unavailable += 1
                elif rec.get("unparsed") is False:
                    n_primary_valid += 1
        n_total = len(calls)
        n_parsed = len(real)
        modal = max(counts.values()) if counts else 0
        out[persona["name"]] = {
            "n_total": n_total,
            "n_parsed": n_parsed,
            "n_unparsed": n_total - n_parsed,
            "parse_rate": n_parsed / n_total if n_total else 0.0,
            "primary_parse_rate": (
                n_primary_valid / n_total if n_total else 0.0
            ),
            "final_parse_rate": n_parsed / n_total if n_total else 0.0,
            "n_repair_needed": n_repair_needed,
            "n_repaired": n_repaired,
            "n_invalid_label": n_invalid_label,
            "n_unavailable": n_unavailable,
            "parse_methods": dict(sorted(methods.items())),
            "call_counts": dict(sorted(counts.items())),
            "distinct_calls": sorted(counts),
            "modal_fraction": modal / n_parsed if n_parsed else 0.0,
            "entropy_bits": _entropy(counts),
            "is_constant": len(counts) < 2,
        }
    return out


def build_label_counts(votes):
    """Votes -> ntqr labeled vote counts {truth_label: {vote_tuple: count}}.

    Scenarios where any persona abstained (a ``None`` vote, i.e. an unparseable
    reply) are skipped: a partial trio cannot form an R=2 vote pattern. The
    orchestrator only reaches this for a fully-compliant model, so in practice
    nothing is dropped; the guard keeps the function total and honest.
    """
    order = [p["key"] for p in personas.PERSONAS]
    label_counts: dict[str, dict[tuple, int]] = {"a": {}, "b": {}}
    for scenario in personas.SCENARIOS:
        calls = [votes[scenario["id"]][key] for key in order]
        if any(c is None for c in calls):
            continue
        truth_label = personas.CALL_TO_LABEL[scenario["truth"]]
        pattern = tuple(personas.CALL_TO_LABEL[c] for c in calls)
        bucket = label_counts[truth_label]
        bucket[pattern] = bucket.get(pattern, 0) + 1
    return label_counts


def vote_variation(votes):
    """Per-persona vote diagnostics computed from raw votes -- no answer key.

    For each persona we report, over its non-abstained calls, the distinct calls
    used, the fraction on its single most common call (``modal_fraction``; 1.0
    means it never varied), ``is_constant``, and the number of abstentions
    (``n_unparsed``). A constant classifier carries no error signal and an
    abstention is a missing vote -- both block a clean solve, so this is the
    label-free predictor of evaluability, available before any ntqr solve and
    without the authored truth.
    """
    out = {}
    for persona in personas.PERSONAS:
        key = persona["key"]
        calls = [votes[s["id"]][key] for s in personas.SCENARIOS]
        real = [c for c in calls if c is not None]
        counts: dict[str, int] = {}
        for call in real:
            counts[call] = counts.get(call, 0) + 1
        n = len(real)
        modal = max(counts.values()) if counts else 0
        out[persona["name"]] = {
            "n": n,
            "n_unparsed": len(calls) - n,
            "distinct_calls": sorted(counts),
            "call_counts": counts,
            "modal_fraction": modal / n if n else 0.0,
            "is_constant": len(counts) < 2,
        }
    return out


def pairwise_disagreement(votes):
    """Fraction of scenarios on which each persona pair disagrees.

    Error-independent evaluation needs judges whose mistakes are not identical;
    the pairwise disagreement rate is the simplest label-free proxy for that.
    Only scenarios where BOTH judges produced a real vote are compared (an
    abstention is not a disagreement). Returns
    ``{"pairs": {"A vs B": frac, ...}, "mean": mean_over_pairs}``. This
    complements ``vote_variation``: a usable ensemble needs BOTH per-judge
    variation AND disagreement across judges.
    """
    panel = personas.PERSONAS
    pairs: dict[str, float] = {}
    for i in range(len(panel)):
        for j in range(i + 1, len(panel)):
            ki, kj = panel[i]["key"], panel[j]["key"]
            compared = 0
            diff = 0
            for s in personas.SCENARIOS:
                vi = votes[s["id"]][ki]
                vj = votes[s["id"]][kj]
                if vi is None or vj is None:
                    continue
                compared += 1
                if vi != vj:
                    diff += 1
            label = f"{panel[i]['name']} vs {panel[j]['name']}"
            pairs[label] = diff / compared if compared else 0.0
    mean = sum(pairs.values()) / len(pairs) if pairs else 0.0
    return {"pairs": pairs, "mean": mean}


def evaluability_diagnostic(votes):
    """Label-free verdict on whether the ensemble is ntqr-evaluable.

    Fuses ``vote_variation`` and ``pairwise_disagreement`` into one prediction
    that uses NO answer key. The error-independent algebra is solvable only if
    every judge varied (no constant classifier) AND every judge produced a real
    vote on every scenario (no abstentions): a constant judge carries no error
    signal, and an abstention is a missing equation. When the ensemble is not
    evaluable the blocking personas are named, and the abstention count is
    reported -- all derived straight from the votes.
    """
    variation = vote_variation(votes)
    disagreement = pairwise_disagreement(votes)
    constant = [n for n, v in variation.items() if v["is_constant"]]
    n_unparsed = sum(v["n_unparsed"] for v in variation.values())
    return {
        "predicted_evaluable": (not constant) and n_unparsed == 0,
        "constant_classifiers": constant,
        "n_unparsed": n_unparsed,
        "mean_pairwise_disagreement": disagreement["mean"],
        "vote_variation": variation,
        "pairwise_disagreement": disagreement,
    }


def constant_personas(label_counts):
    """Names of personas whose vote never varied (a constant classifier).

    A constant classifier carries no error signal, so it makes the
    error-independent evaluation unsolvable. Reported to explain a degenerate
    recovery.
    """
    names = [p["name"] for p in personas.PERSONAS]
    seen = [set() for _ in names]
    for by_votes in label_counts.values():
        for pattern in by_votes:
            for i, vote in enumerate(pattern):
                seen[i].add(vote)
    return [names[i] for i, labels in enumerate(seen) if len(labels) < 2]


def build_comparison(supervised, error_independent):
    """Pair each persona's true accuracy with the unsupervised recovery."""
    true_prev_a = safe_float(supervised.evaluation_float["prevalence"]["a"])
    best = select_best(error_independent.evaluation_float)
    order = [p["name"] for p in personas.PERSONAS]
    per_persona = []
    for i, name in enumerate(order):
        true_acc = supervised.evaluation_float["accuracy"][i]
        rec_acc = best["accuracy"][i]
        per_persona.append(
            {
                "persona": name,
                "true_accuracy": {
                    "a": safe_float(true_acc["a"]),
                    "b": safe_float(true_acc["b"]),
                },
                "recovered_accuracy_unsupervised": {
                    "a": safe_float(rec_acc["a"]),
                    "b": safe_float(rec_acc["b"]),
                },
            }
        )
    return {
        "true_prevalence_a": true_prev_a,
        "recovered_prevalence_a": safe_float(best["prevalence"]["a"]),
        "note": (
            "recovered_* comes from ntqr's ErrorIndependentEvaluation using "
            "ONLY the personas' votes -- no answer key. Of its two algebraic "
            "solutions we keep the higher-mean-accuracy one (the "
            "better-than-random assumption). true_* uses the authored ground "
            "truth, shown only as a check."
        ),
        "per_persona": per_persona,
    }


def recovery_stats(comparison):
    """Quantify how well the unsupervised recovery matched the truth.

    From a ``build_comparison`` result, compute the absolute error between each
    recovered and true accuracy (per persona, per label), their mean (the
    recovery MAE -- the headline number), and the prevalence absolute error.
    Pairs where either value is missing (None) are skipped, so a degenerate
    solution does not poison the statistic.
    """
    abs_errors = []
    by_label: dict[str, list[float]] = {"a": [], "b": []}
    per_persona = []
    worst_term = None  # (persona, label, abs_error) with the largest error
    for row in comparison["per_persona"]:
        row_errors = {}
        for label in ("a", "b"):
            true_v = row["true_accuracy"][label]
            rec_v = row["recovered_accuracy_unsupervised"][label]
            if true_v is None or rec_v is None:
                row_errors[label] = None
                continue
            err = abs(rec_v - true_v)
            row_errors[label] = err
            abs_errors.append(err)
            by_label[label].append(err)
            if worst_term is None or err > worst_term["abs_error"]:
                worst_term = {
                    "persona": row["persona"],
                    "label": label,
                    "abs_error": err,
                }
        per_persona.append(
            {"persona": row["persona"], "abs_error": row_errors}
        )

    true_prev = comparison["true_prevalence_a"]
    rec_prev = comparison["recovered_prevalence_a"]
    prevalence_abs_error = (
        abs(rec_prev - true_prev)
        if true_prev is not None and rec_prev is not None
        else None
    )
    return {
        "recovery_mae": (
            sum(abs_errors) / len(abs_errors) if abs_errors else None
        ),
        "recovery_mae_by_label": {
            label: (sum(errs) / len(errs) if errs else None)
            for label, errs in by_label.items()
        },
        "max_abs_error": max(abs_errors) if abs_errors else None,
        "worst_term": worst_term,
        "prevalence_abs_error": prevalence_abs_error,
        "n_accuracy_terms": len(abs_errors),
        "per_persona": per_persona,
    }
