"""Synthetic-ensemble validation harness for the NTQR error-independent solve.

The real-LLM study (``multi_llm_personas_evaluation.py``) may yield only a small
number of evaluable ensembles: a model must return a complete binary response
matrix and every judge must vary. This module complements that real result with
a controlled, fully-deterministic study on *synthetic* judges, where the ground
truth is set by construction:

* We draw votes for a trio of binary judges with known per-label accuracies on a
  test of known prevalence, under a configurable error-dependence structure.
* We feed the resulting vote counts through the **identical** ntqr solver the
  LLM study uses (``ErrorIndependentEvaluation``), and compare the unsupervised
  recovery to the sample ground truth.

It answers two questions the single real model cannot:

1. **Does recovery track the truth across many ensembles, and how does its error
   scale with test size Q?** (``convergence_sweep``) -- the recovery MAE should
   fall like sampling noise, ~1/sqrt(Q), with no persistent bias.
2. **When does the imaginary-prevalence alarm fire?** (``correlation_sweep``) --
   the alarm is an *iron-clad sufficient* signal of violated error-independence,
   but this study characterises its blind spot: positive common-mode correlation
   (the realistic shared-training-data failure for LLMs) can evade it, while an
   anti-correlated pair (negative pairwise vote-moment product) trips it.

Everything here is pure (seeded ``random.Random``, no network, no matplotlib, no
TeX); the orchestrator (``scripts/sim/synthetic_recovery_study.py``) handles I/O
and figures. Ground truth is computed directly from the generated counts (cheap
integer arithmetic), never via the heavy ``SupervisedEvaluation`` sympy path, so
the sweeps stay fast enough to average over many trials.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from ntqr.llm.recovery_scoring import accuracy_abs_errors, safe_float, select_best

# A judge's accuracy is (alpha, beta): P(correct | true label a), P(correct | b).
JudgeAccuracy = tuple[float, float]


@dataclass(frozen=True)
class Ensemble:
    """A synthetic trio: a test prevalence and three judges' per-label accuracy.

    ``prevalence_a`` is the fraction of test items whose true label is ``a``.
    ``accuracies`` holds three ``(alpha, beta)`` pairs. Every accuracy must lie
    strictly inside ``(0, 1)`` so that each judge genuinely varies on both labels
    (no constant classifier, which the algebra cannot solve).
    """

    prevalence_a: float
    accuracies: tuple[JudgeAccuracy, ...]

    def __post_init__(self) -> None:
        if not 0.0 < self.prevalence_a < 1.0:
            raise ValueError("prevalence_a must be in (0, 1)")
        if len(self.accuracies) != 3:
            raise ValueError("exactly three judges are required (R=2 trio)")
        for alpha, beta in self.accuracies:
            if not (0.0 < alpha < 1.0 and 0.0 < beta < 1.0):
                raise ValueError("each accuracy must be in (0, 1)")


@dataclass
class CorrelationConfig:
    """How the synthetic judges' errors are coupled, beyond independence.

    * ``common_mode`` -- probability that a shared latent event forces *all*
      three judges to err together on an item (positive error correlation, the
      shared-training-data analogue for LLMs).
    * ``anti_pair`` -- an optional ``(i, j)`` judge-index pair to anti-correlate.
    * ``anti_strength`` -- probability that judge ``j`` mirrors the *opposite* of
      judge ``i``'s correctness (driving their pairwise vote-moment negative).

    ``CorrelationConfig()`` with all defaults is exact error-independence.
    """

    label: str = "independent"
    common_mode: float = 0.0
    anti_pair: tuple[int, int] | None = None
    anti_strength: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.common_mode) or not 0 <= self.common_mode <= 1:
            raise ValueError("common_mode must be in [0, 1]")
        if not math.isfinite(self.anti_strength) or not 0 <= self.anti_strength <= 1:
            raise ValueError("anti_strength must be in [0, 1]")
        if self.anti_pair is not None and (len(self.anti_pair) != 2 or
                any(type(i) is not int or i not in range(3) for i in self.anti_pair) or
                self.anti_pair[0] == self.anti_pair[1]):
            raise ValueError("anti_pair must contain two distinct judge indices from 0 to 2")
        if self.anti_pair is None and self.anti_strength != 0:
            raise ValueError("anti_strength requires an anti_pair")


def _wrong_vote(true_label: str) -> str:
    return "b" if true_label == "a" else "a"


def generate_label_counts(
    ensemble: Ensemble,
    q: int,
    rng: random.Random,
    config: CorrelationConfig | None = None,
) -> dict[str, dict[tuple[str, str, str], int]]:
    """Draw ``q`` items and tally vote patterns by true label.

    Returns ``{"a": {pattern: count}, "b": {...}}`` ready for ntqr's
    ``TrioLabelVoteCounts``. ``Q_a = round(prevalence_a * Q)`` items carry truth
    ``a``; the rest carry ``b``. For each item every judge votes correctly with
    its per-label accuracy, then the optional correlation structure is applied.
    No vote is ever fabricated or dropped -- each item contributes exactly one
    length-3 pattern, so the counts sum to ``q``.
    """
    if type(q) is not int or q < 1:
        raise ValueError("q must be a positive integer")
    config = config or CorrelationConfig()
    q_a = round(ensemble.prevalence_a * q)
    truths = ["a"] * q_a + ["b"] * (q - q_a)
    counts: dict[str, dict[tuple[str, str, str], int]] = {"a": {}, "b": {}}

    for truth in truths:
        shared_error = (
            config.common_mode > 0 and rng.random() < config.common_mode
        )
        correct = []
        for idx, (alpha, beta) in enumerate(ensemble.accuracies):
            acc = alpha if truth == "a" else beta
            is_correct = rng.random() < acc
            if shared_error:
                is_correct = False
            correct.append(is_correct)

        if config.anti_pair is not None and config.anti_strength > 0:
            i, j = config.anti_pair
            if rng.random() < config.anti_strength:
                correct[j] = not correct[i]

        pattern = tuple(truth if ok else _wrong_vote(truth) for ok in correct)
        bucket = counts[truth]
        bucket[pattern] = bucket.get(pattern, 0) + 1

    return counts


def sample_truth(label_counts: dict) -> dict:
    """Exact sample prevalence and per-judge per-label accuracy from counts.

    This is the ground truth the unsupervised recovery is scored against -- the
    *realised* accuracies on the very items the judges took, identical to what
    ntqr's ``SupervisedEvaluation`` would compute, but via plain integer counting
    (no sympy). Comparing recovery to the sample (not the population parameter)
    isolates the algebraic recovery error from the irreducible sampling spread.
    """
    if set(label_counts) != {"a", "b"}:
        raise ValueError("label counts must provide both binary label strata")
    for counts in label_counts.values():
        for pattern, count in counts.items():
            if len(pattern) != 3 or any(label not in ("a", "b") for label in pattern):
                raise ValueError("vote patterns must contain three binary labels")
            if type(count) is not int or count < 0:
                raise ValueError("vote counts must be nonnegative integers")
    totals = {lab: sum(label_counts[lab].values()) for lab in ("a", "b")}
    q = totals["a"] + totals["b"]
    if q == 0:
        raise ValueError("vote counts must contain at least one observation")
    prevalence_a = totals["a"] / q
    accuracies = []
    for idx in range(3):
        row = {}
        for lab in ("a", "b"):
            denom = totals[lab]
            correct = sum(
                n for pat, n in label_counts[lab].items() if pat[idx] == lab
            )
            row[lab] = correct / denom if denom else None
        accuracies.append(row)
    return {"prevalence_a": prevalence_a, "accuracies": accuracies}


def prevalence_discriminant(evaluator) -> float:
    """The alarm quantity ``D = 1 - 4c/a`` of the prevalence quadratic.

    ``ErrorIndependentEvaluation`` solves ``P_a = 1/2 +/- sqrt(D)/2``. ``D >= 0``
    gives two real, logically-consistent prevalences; ``D < 0`` makes the root
    imaginary -- the iron-clad, answer-key-free signal that the error-independence
    assumption is violated on this test. Returns ``nan`` if the leading
    coefficient is zero (a degenerate sketch).
    """
    coeffs = evaluator.alpha_prevalence_quadratic_terms()
    a_coeff, c_coeff = float(coeffs[2]), float(coeffs[0])
    if a_coeff == 0:
        return float("nan")
    return 1 - 4 * c_coeff / a_coeff


def solve_recovery(label_counts: dict) -> dict:
    """Run the real ntqr error-independent solve and score it against truth.

    Builds the same ``TrioLabelVoteCounts -> ErrorIndependentEvaluation`` chain
    the LLM study uses, selects the better-than-random branch of the two
    solutions (shared ``_recovery.select_best``), and reports the recovery error
    against ``sample_truth``. When the prevalence root is imaginary (``alarm``)
    the recovery is undefined and the error fields are ``None`` -- the honest
    outcome, never a fabricated number.
    """
    from ntqr.r2.datasketches import TrioLabelVoteCounts
    from ntqr.r2.evaluators import ErrorIndependentEvaluation

    truth = sample_truth(label_counts)
    tvc = TrioLabelVoteCounts(label_counts).to_TrioVoteCounts()
    evaluator = ErrorIndependentEvaluation(tvc)
    discriminant = prevalence_discriminant(evaluator)

    solutions = evaluator.evaluation_float
    best = select_best(solutions)
    rec_prev = safe_float(best["prevalence"]["a"])
    alarm = (
        not math.isfinite(discriminant) or discriminant < 0 or rec_prev is None
    )

    roots = sorted(
        v
        for v in (safe_float(s["prevalence"]["a"]) for s in solutions)
        if v is not None
    )

    prevalence_abs_error = (
        abs(rec_prev - truth["prevalence_a"]) if rec_prev is not None else None
    )

    abs_errors = accuracy_abs_errors(best, truth["accuracies"])
    accuracy_mae = sum(abs_errors) / len(abs_errors) if len(abs_errors) == 6 else None

    return {
        "discriminant": discriminant,
        "alarm": alarm,
        "prevalence_real": not alarm,
        "true_prevalence_a": truth["prevalence_a"],
        "recovered_prevalence_a": rec_prev,
        "prevalence_abs_error": prevalence_abs_error,
        "accuracy_mae": accuracy_mae,
        "roots": roots,
    }


def _mean_sd(values: list[float]) -> tuple[float | None, float | None]:
    xs = [v for v in values if v is not None]
    if not xs:
        return None, None
    mean = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / len(xs))
    return mean, sd


def convergence_sweep(
    ensemble: Ensemble,
    q_values: list[int],
    n_trials: int,
    base_seed: int,
) -> list[dict]:
    """Mean +/- sd recovery error vs test size ``Q`` over independent replicas.

    For each ``Q`` we draw ``n_trials`` independent error-independent tests
    (distinct seeds, fully reproducible) and average the prevalence error and the
    accuracy MAE. The expected signature of an unbiased estimator limited only by
    sampling noise is ``error ~ C / sqrt(Q)`` -- see :func:`sqrt_q_fit`.
    """
    rows = []
    for q in q_values:
        prev_errs, acc_maes = [], []
        for trial in range(n_trials):
            rng = random.Random(base_seed + q * 100_003 + trial)
            counts = generate_label_counts(ensemble, q, rng)
            result = solve_recovery(counts)
            if result["alarm"]:
                continue
            prev_errs.append(result["prevalence_abs_error"])
            acc_maes.append(result["accuracy_mae"])
        prev_mean, prev_sd = _mean_sd(prev_errs)
        acc_mean, acc_sd = _mean_sd(acc_maes)
        rows.append(
            {
                "q": q,
                "n_trials": n_trials,
                "n_solved": len(acc_maes),
                "prevalence_abs_error_mean": prev_mean,
                "prevalence_abs_error_sd": prev_sd,
                "accuracy_mae_mean": acc_mean,
                "accuracy_mae_sd": acc_sd,
            }
        )
    return rows


def sqrt_q_fit(rows: list[dict], key: str = "accuracy_mae_mean") -> dict:
    """Least-squares slope of ``log(error)`` vs ``log(Q)``.

    A slope near ``-0.5`` confirms the recovery error falls like sampling noise
    (``error ~ Q**slope``). Returns the slope, intercept, the implied exponent,
    and ``r_squared`` of the log-log fit. Needs at least two solved rows.
    """
    pts = [
        (math.log(r["q"]), math.log(r[key]))
        for r in rows
        if r.get(key) and r[key] > 0
    ]
    if len(pts) < 2:
        return {
            "slope": None,
            "intercept": None,
            "r_squared": None,
            "n": len(pts),
        }
    n = len(pts)
    sx = sum(x for x, _ in pts)
    sy = sum(y for _, y in pts)
    sxx = sum(x * x for x, _ in pts)
    sxy = sum(x * y for x, y in pts)
    denom = n * sxx - sx * sx
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    mean_y = sy / n
    ss_tot = sum((y - mean_y) ** 2 for _, y in pts)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in pts)
    r_squared = 1 - ss_res / ss_tot if ss_tot else 1.0
    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "n": n,
    }


def correlation_sweep(
    ensemble: Ensemble,
    q: int,
    configs: list[CorrelationConfig],
    n_trials: int,
    base_seed: int,
) -> list[dict]:
    """Alarm-firing rate per error-dependence structure, at fixed ``Q``.

    For each config we draw ``n_trials`` reproducible tests and record how often
    the imaginary-prevalence alarm fires, the discriminant distribution, and --
    among the silent (non-alarm) trials -- the residual recovery error. The
    independent config doubles as a false-alarm control: a correct alarm never
    fires on a genuinely error-independent ensemble.
    """
    rows = []
    for cfg_index, config in enumerate(configs):
        alarms = 0
        discs, silent_prev_errs = [], []
        for trial in range(n_trials):
            rng = random.Random(base_seed + cfg_index * 1_000_003 + trial)
            counts = generate_label_counts(ensemble, q, rng, config)
            result = solve_recovery(counts)
            disc = result["discriminant"]
            if math.isfinite(disc):
                discs.append(disc)
            if result["alarm"]:
                alarms += 1
            elif result["prevalence_abs_error"] is not None:
                silent_prev_errs.append(result["prevalence_abs_error"])
        disc_mean, _ = _mean_sd(discs)
        silent_mean, _ = _mean_sd(silent_prev_errs)
        rows.append(
            {
                "label": config.label,
                "common_mode": config.common_mode,
                "anti_pair": (
                    list(config.anti_pair) if config.anti_pair else None
                ),
                "anti_strength": config.anti_strength,
                "q": q,
                "n_trials": n_trials,
                "alarm_count": alarms,
                "alarm_rate": alarms / n_trials if n_trials else 0.0,
                "discriminant_mean": disc_mean,
                "discriminant_min": min(discs) if discs else None,
                "discriminant_max": max(discs) if discs else None,
                "silent_prevalence_abs_error_mean": silent_mean,
            }
        )
    return rows


def uniform_ensemble(prevalence_a: float, accuracy: float) -> Ensemble:
    """A symmetric trio: all three judges at the same per-label ``accuracy``."""
    return Ensemble(prevalence_a, ((accuracy, accuracy),) * 3)


def tie_break_sweep(
    prevalence_a: float,
    accuracies: list[float],
    q: int,
    n_trials: int,
    base_seed: int,
    catastrophe_threshold: float = 0.25,
) -> list[dict]:
    """Where the better-than-random tie-break breaks, as judges approach random.

    ``select_best`` resolves the two-solution ambiguity by assuming the judges
    are better than random. That assumption weakens as accuracy falls toward 0.5:
    finite-sample noise can make the *mirror* solution score higher, and recovery
    catastrophically inverts. For each mean per-label ``accuracy`` (judges held
    symmetric so accuracy is the only varying factor) we draw ``n_trials`` tests
    and report the catastrophic-inversion rate (``prevalence_abs_error`` above
    ``catastrophe_threshold``), the mean prevalence error, and the degenerate
    rate. This quantifies the §S1 tie-break assumption rather than asserting it.
    """
    rows = []
    for acc_index, accuracy in enumerate(accuracies):
        ensemble = uniform_ensemble(prevalence_a, accuracy)
        catastrophic = 0
        degenerate = 0
        prev_errs = []
        for trial in range(n_trials):
            rng = random.Random(base_seed + acc_index * 100_003 + trial)
            counts = generate_label_counts(ensemble, q, rng)
            result = solve_recovery(counts)
            if result["alarm"] or result["prevalence_abs_error"] is None:
                degenerate += 1
                continue
            err = result["prevalence_abs_error"]
            prev_errs.append(err)
            if err > catastrophe_threshold:
                catastrophic += 1
        prev_mean, _ = _mean_sd(prev_errs)
        rows.append(
            {
                "accuracy": accuracy,
                "q": q,
                "n_trials": n_trials,
                "catastrophic_count": catastrophic,
                "catastrophic_rate": (
                    catastrophic / n_trials if n_trials else 0.0
                ),
                "degenerate_count": degenerate,
                "degenerate_rate": degenerate / n_trials if n_trials else 0.0,
                "prevalence_abs_error_mean": prev_mean,
            }
        )
    return rows


def ensemble_robustness_sweep(
    ensembles: list[tuple[str, Ensemble]],
    q_values: list[int],
    n_trials: int,
    base_seed: int,
) -> list[dict]:
    """Confirm the ~1/sqrt(Q) convergence holds across *diverse* ensembles.

    The headline convergence fit (``convergence_sweep`` + ``sqrt_q_fit``) is run
    on a single ensemble; this checks the slope is not an artifact of that choice
    by repeating it across several labelled ensembles (varied prevalence and
    accuracy spread) and reporting each fitted slope and R². A tight band of
    slopes near -0.5 is the robustness evidence.
    """
    rows = []
    for ens_index, (label, ensemble) in enumerate(ensembles):
        conv = convergence_sweep(
            ensemble, q_values, n_trials, base_seed + ens_index * 10_000_019
        )
        fit = sqrt_q_fit(conv, "accuracy_mae_mean")
        rows.append(
            {
                "label": label,
                "prevalence_a": ensemble.prevalence_a,
                "accuracies": [list(a) for a in ensemble.accuracies],
                "slope": fit["slope"],
                "r_squared": fit["r_squared"],
                "smallest_q_acc_mae": conv[0]["accuracy_mae_mean"],
                "largest_q_acc_mae": conv[-1]["accuracy_mae_mean"],
            }
        )
    return rows


def _solution_mae(best, truth_accuracies) -> float | None:
    """Recovery MAE of a single solution vs the sample truth (None if degenerate)."""
    errors = accuracy_abs_errors(best, truth_accuracies)
    return sum(errors) / len(errors) if len(errors) == 6 else None


def compare_evaluators(label_counts: dict) -> dict:
    """Recovery MAE of error-independent vs majority-voting evaluation, same counts.

    Both ntqr evaluators return two logically-consistent solutions; each is
    resolved by the shared better-than-random tie-break (``select_best``) and
    scored against the sample truth. ``ErrorIndependentEvaluation`` is *exact*
    under error-independence; ``MajorityVotingEvaluation`` is biased (it assumes
    the majority vote is the truth) but rock-solid (no imaginary roots, no mirror
    inversion). Returns both MAEs (``None`` for a degenerate solve).
    """
    from ntqr.r2.datasketches import TrioLabelVoteCounts
    from ntqr.r2.evaluators import (
        ErrorIndependentEvaluation,
        MajorityVotingEvaluation,
    )

    truth = sample_truth(label_counts)
    tvc = TrioLabelVoteCounts(label_counts).to_TrioVoteCounts()
    ei_best = select_best(ErrorIndependentEvaluation(tvc).evaluation_float)
    mv_best = select_best(MajorityVotingEvaluation(tvc).evaluation_float)
    return {
        "ei_mae": _solution_mae(ei_best, truth["accuracies"]),
        "mv_mae": _solution_mae(mv_best, truth["accuracies"]),
    }


def mv_vs_ei_sweep(
    prevalences: list[float],
    accuracies: list[float],
    q: int,
    n_trials: int,
    base_seed: int,
) -> list[dict]:
    """Error-independent vs majority-voting recovery across a prevalence×accuracy grid.

    For each (prevalence, symmetric judge accuracy) cell we draw ``n_trials``
    error-independent tests and compare the two evaluators' recovery MAE, reporting
    each mean and the fraction of trials on which the exact algebra (EI) beat
    majority voting (MV). The expected picture: EI wins where its assumption holds
    (clearly-better-than-random judges), MV wins in the tie-break danger zone
    (extreme prevalence + weak judges, where EI's mirror inverts).
    """
    rows = []
    for p_index, prevalence in enumerate(prevalences):
        for a_index, accuracy in enumerate(accuracies):
            ensemble = uniform_ensemble(prevalence, accuracy)
            ei_maes, mv_maes = [], []
            ei_wins = 0
            for trial in range(n_trials):
                seed = base_seed + (p_index * 1000 + a_index) * 100_003 + trial
                counts = generate_label_counts(
                    ensemble, q, random.Random(seed)
                )
                cmp = compare_evaluators(counts)
                if cmp["ei_mae"] is None or cmp["mv_mae"] is None:
                    continue
                ei_maes.append(cmp["ei_mae"])
                mv_maes.append(cmp["mv_mae"])
                if cmp["ei_mae"] < cmp["mv_mae"]:
                    ei_wins += 1
            ei_mean, _ = _mean_sd(ei_maes)
            mv_mean, _ = _mean_sd(mv_maes)
            n_solved = len(ei_maes)
            rows.append(
                {
                    "prevalence_a": prevalence,
                    "accuracy": accuracy,
                    "q": q,
                    "n_trials": n_trials,
                    "n_solved": n_solved,
                    "ei_mae_mean": ei_mean,
                    "mv_mae_mean": mv_mean,
                    "ei_win_rate": ei_wins / n_solved if n_solved else None,
                }
            )
    return rows
