"""Bounded, reproducible possible/consistent-set example for NTQR 0.9.

NTQR 0.9 yields sparse HashablePoint rows. Reservoir sampling over its exact
finite generator avoids the global, unseeded Numba random stream and does not
promise independent draws from the upstream MCMC sampler.
"""

from __future__ import annotations

from collections import Counter
import math
import random

import ntqr
from ntqr.evaluations import ConsistentSet, PossibleSet


def simulate_counts(labels, classifiers, ql, seed):
    """Simulate a labeled test whose true-label counts are exactly ql."""
    rng = random.Random(seed)
    truth = [label for label, count in zip(labels, ql) for _ in range(count)]
    rng.shuffle(truth)
    accuracies = [
        {label: rng.uniform(0.45, 0.9) for label in labels} for _ in classifiers
    ]
    return Counter(
        tuple(
            (
                label
                if rng.random() < accuracies[j][label]
                else rng.choice([other for other in labels if other != label])
            )
            for j in range(len(classifiers))
        )
        for label in truth
    )


def summarize_sets(
    labels, classifiers, ql, seed=0, n_samples=3, max_allocations=1_000_000
):
    """Count an exact finite set and take a seeded sample without replacement.

    The product of per-event label allocations bounds work before enumerating;
    callers must explicitly raise the budget to explore larger examples.
    """
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError("labels must contain at least two distinct values")
    if not classifiers or len(set(classifiers)) != len(classifiers):
        raise ValueError("classifiers must be nonempty and distinct")
    if (
        len(ql) != len(labels)
        or any(type(q) is not int or q < 0 for q in ql)
        or sum(ql) < 1
    ):
        raise ValueError(
            "ql must contain nonnegative integer counts for every label and a positive total"
        )
    if type(n_samples) is not int or n_samples < 0:
        raise ValueError("n_samples must be a nonnegative integer")
    if type(max_allocations) is not int or max_allocations < 1:
        raise ValueError("max_allocations must be a positive integer")
    counts = simulate_counts(labels, classifiers, ql, seed)
    allocation_bound = math.prod(
        math.comb(n + len(labels) - 1, len(labels) - 1) for n in counts.values()
    )
    if allocation_bound > max_allocations:
        raise ValueError(
            f"enumeration bound {allocation_bound} exceeds max_allocations={max_allocations}"
        )
    possible = PossibleSet(labels, classifiers)
    consistent = ConsistentSet(labels, classifiers, counts)
    rng = random.Random(seed)
    reservoir = []
    count = 0
    for count, point in enumerate(consistent.set_generator(ql), 1):
        valid, reason = consistent.is_valid_point(point, ql)
        if not valid:
            raise ValueError(f"NTQR returned an inconsistent evaluation: {reason}")
        dense = [row.toarray().ravel().tolist() for row in point]
        if len(reservoir) < n_samples:
            reservoir.append(dense)
        else:
            index = rng.randrange(count)
            if index < n_samples:
                reservoir[index] = dense
    marginals = consistent.correct_cuboid_marginal_matrices()
    return {
        "labels": list(labels),
        "classifiers": list(classifiers),
        "ql": list(ql),
        "observed_counts": counts,
        "possible_set_count": int(possible.set_count_at_ql(ql)),
        "consistent_set_count": count,
        "consistent_random_points": reservoir,
        "correct_cuboid_points": [
            [(matrix @ row).tolist() for matrix, row in zip(marginals, point)]
            for point in reservoir
        ],
        "metadata": {
            "ntqr_version": ntqr.__version__,
            "seed": seed,
            "sampling": "exact-set reservoir without replacement",
            "allocation_bound": allocation_bound,
        },
    }
