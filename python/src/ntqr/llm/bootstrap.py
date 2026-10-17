"""Scenario bootstrap: a confidence interval on the unsupervised recovery error.

The recovery MAE reported for the evaluable model is a point estimate from one
finite test — the authored scenarios. Those scenarios are themselves a sample,
so we can quantify how stable the recovery is to *which* scenarios we happened to
write, **without any new model calls**, by resampling the per-scenario vote
records with replacement and re-running the same ntqr solve on each resample.

This directly answers the manuscript's own limitation ("the recovery MAE has no
sampling distribution here because decoding is deterministic"): decoding is
fixed, but the test is a finite draw, and the bootstrap gives that draw a
distribution. Each resample is scored by the shared
``ntqr.llm.simulation.solve_recovery`` (the identical
``ErrorIndependentEvaluation`` path), so the point estimate here
equals the headline recovery MAE by construction.

A record is ``(truth_label, vote_pattern)`` for one scenario, e.g.
``("a", ("a", "a", "b"))``. Pure: seeded ``random.Random``, no network.
"""

from __future__ import annotations

from ntqr.llm.simulation import solve_recovery


def label_counts_from_records(records):
    """Tally ``(truth_label, pattern)`` records into ntqr label-vote counts."""
    counts: dict[str, dict[tuple, int]] = {"a": {}, "b": {}}
    for truth, pattern in records:
        bucket = counts[truth]
        bucket[pattern] = bucket.get(pattern, 0) + 1
    return counts


def _percentile(sorted_xs: list[float], p: float) -> float:
    """Lower-rank percentile of an already-sorted list (p in [0, 1])."""
    if not sorted_xs:
        return float("nan")
    idx = int(p * (len(sorted_xs) - 1))
    return sorted_xs[idx]


def _summary(values: list[float]) -> dict:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "ci_lo": None,
            "ci_hi": None,
        }
    xs = sorted(values)
    n = len(xs)
    mean = sum(xs) / n
    mid = n // 2
    median = xs[mid] if n % 2 else (xs[mid - 1] + xs[mid]) / 2
    return {
        "n": n,
        "mean": mean,
        "median": median,
        "ci_lo": _percentile(xs, 0.025),
        "ci_hi": _percentile(xs, 0.975),
    }


def bootstrap_recovery(records, n_boot: int, base_seed: int) -> dict:
    """Bootstrap the recovery MAE and prevalence error over scenario resamples.

    Returns the point estimate (the full sample), the 95% percentile CIs for the
    recovery MAE and the prevalence absolute error over ``n_boot`` resamples, and
    the fraction of resamples that turned degenerate (a persona collapsed to a
    constant classifier in the draw, so the solve was unavailable) — itself a
    measure of how fragile evaluability is to the scenario sample.
    """
    import random

    if type(n_boot) is not int or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    if not records:
        raise ValueError("records must be nonempty")
    point = solve_recovery(label_counts_from_records(records))
    n = len(records)
    maes: list[float] = []
    prev_errs: list[float] = []
    degenerate = 0
    for b in range(n_boot):
        rng = random.Random(base_seed + b)
        resample = [records[rng.randrange(n)] for _ in range(n)]
        result = solve_recovery(label_counts_from_records(resample))
        if result["alarm"] or result["accuracy_mae"] is None:
            degenerate += 1
            continue
        maes.append(result["accuracy_mae"])
        if result["prevalence_abs_error"] is not None:
            prev_errs.append(result["prevalence_abs_error"])

    return {
        "n_scenarios": n,
        "n_boot": n_boot,
        "seed": base_seed,
        "point_recovery_mae": point["accuracy_mae"],
        "point_prevalence_abs_error": point["prevalence_abs_error"],
        "degenerate_count": degenerate,
        "degenerate_rate": degenerate / n_boot if n_boot else 0.0,
        "recovery_mae": _summary(maes),
        "prevalence_abs_error": _summary(prev_errs),
        # Raw per-resample MAE samples (for a histogram); callers that persist a
        # compact summary should pop this before writing.
        "mae_samples": maes,
    }
