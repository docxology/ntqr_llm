"""Validate and bound the legacy enumerative alarm used by offline examples."""

from __future__ import annotations

import ntqr
from ntqr.alarms import SingleClassifierAxiomsAlarm


def run_alarm(spec, axioms_cls, evals_cls):
    """Evaluate real label-response counts with an explicit enumeration budget."""
    labels = tuple(spec["labels"])
    classifiers = tuple(spec["classifiers"])
    q = spec["Q"]
    max_q = spec.get("max_q", 30 if len(labels) == 2 else 6)
    if type(q) is not int or q < 1 or type(max_q) is not int or not 1 <= q <= max_q:
        raise ValueError(
            "Q must be positive and <= max_q; raise max_q explicitly for expensive enumeration"
        )
    if len(labels) not in (2, 3) or len(set(labels)) != len(labels):
        raise ValueError("alarm demo requires two or three distinct labels")
    if not classifiers or len(set(classifiers)) != len(classifiers):
        raise ValueError("classifiers must be nonempty and distinct")
    factors = spec["factors"]
    if len(factors) != len(labels) or any(type(f) is not int or f < 1 for f in factors):
        raise ValueError("one positive integer safety factor is required per label")
    responses = [tuple(row) for row in spec["responses"]]
    if len(responses) != len(classifiers) or any(
        len(row) != len(labels)
        or any(type(n) is not int or n < 0 for n in row)
        or sum(row) != q
        for row in responses
    ):
        raise ValueError(
            "one nonnegative response-count row summing to Q is required per classifier"
        )
    axioms = [axioms_cls(labels, c) for c in classifiers]
    alarm = SingleClassifierAxiomsAlarm(q, axioms, evals_cls)
    alarm.set_safety_specification(factors)
    return {
        "labels": list(labels),
        "Q": q,
        "safety_factors": list(factors),
        "responses": [list(row) for row in responses],
        "are_misaligned": alarm.are_misaligned(responses),
        "metadata": {
            "ntqr_version": ntqr.__version__,
            "max_q": max_q,
            "algorithm": "legacy exact enumeration",
        },
    }
