"""Pure recovery-scoring core shared across the evaluation paths.

Turning an ``ntqr`` two-solution result into scored numbers is the same math
whether the votes came from real LLMs (``ntqr.llm.analysis``), a synthetic
generator (``ntqr.llm.simulation``), or a scenario resample
(``ntqr.llm.bootstrap``). That math lives here,
once: the finite-float guard, a solution's mean accuracy, the better-than-random
tie-break between the two logically-consistent solutions, and the per-term
absolute errors against a known (or supervised) truth.

No I/O, no ``ntqr`` import, no personas — just arithmetic over the plain dict
shape ntqr's evaluators expose (``{"prevalence": {...}, "accuracy": [{...}*3]}``),
so it is trivially unit-testable and reusable.
"""

from __future__ import annotations

import math


def safe_float(val) -> float | None:
    """Finite float, or ``None`` for complex/irrational/None/NaN/inf.

    A degenerate evaluation (e.g. a persona that voted one-sidedly, leaving a
    constant classifier) makes ntqr's algebra unsolvable and yields SymPy
    ``nan`` or a complex root. ``float(nan)`` succeeds, so non-finite values are
    rejected explicitly; callers then treat the whole solution as degenerate
    rather than averaging NaNs into a meaningless statistic.
    """
    try:
        out = float(val)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def mean_accuracy(solution) -> float:
    """Average of every per-classifier, per-label accuracy in a solution.

    Returns ``-1.0`` for a fully-degenerate solution (no finite terms), which
    sorts strictly below any real accuracy so :func:`select_best` never picks it.
    """
    vals = [
        safe_float(acc[label])
        for acc in solution["accuracy"]
        for label in ("a", "b")
    ]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else -1.0


def select_best(solutions):
    """Pick the higher-mean-accuracy solution (better-than-random tie-break).

    Error-independent evaluation returns two logically consistent solutions that
    are algebraically indistinguishable: the labelling we want and its mirror
    image (accuracies near zero). We break the tie with the standard,
    answer-key-free assumption that the experts are better than random. Choosing
    by closeness to the true prevalence would fail when the two prevalences are
    symmetric about 0.5, and it would peek at the ground truth.

    NOTE: this assumption *is* the failure mode quantified in the synthetic
    tie-break sweep — as the ensemble approaches random, the mirror can win by
    chance and recovery inverts.
    """
    return max(solutions, key=mean_accuracy)


def accuracy_abs_errors(solution, truth_accuracies) -> list[float]:
    """Per-term ``|recovered - true|`` over the six (classifier, label) accuracies.

    ``solution`` is one ntqr evaluation dict; ``truth_accuracies`` is a list of
    three ``{"a": .., "b": ..}`` dicts (sympy or float). Terms where either side
    is non-finite are skipped, so a degenerate solution yields fewer than six
    errors rather than poisoning the mean — callers can check the count.
    """
    errors = []
    for i, truth in enumerate(truth_accuracies):
        for label in ("a", "b"):
            rec = safe_float(solution["accuracy"][i][label])
            tru = safe_float(truth[label])
            if rec is not None and tru is not None:
                errors.append(abs(rec - tru))
    return errors
