"""Orchestrate ntqr's PossibleSet / ConsistentSet computation for an R=3,
N=3 test.

The possible set counts every group evaluation allowed by the simplex and
marginalization axioms at a given answer-key point; the consistent set filters
that down to evaluations compatible with observed agreement/disagreement counts.

Configurable through ``params`` (labels, classifiers, ql answer-key point,
RNG seed, number of sampled points). Defaults reproduce the canonical R=3,
N=3, Q=20 example.

Real ntqr methods exercised:
    ntqr.evaluations.PossibleSet.set_count_at_ql
    ntqr.evaluations.ConsistentSet.set_generator / random_points /
        random_correct_cuboid_points
"""

from __future__ import annotations

import collections
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _common  # noqa: E402
import _config  # noqa: E402

_common.ensure_ntqr_importable()

import ntqr.evaluations  # noqa: E402

DEFAULTS = {
    "labels": ["a", "b", "c"],
    "classifiers": ["i", "j", "k"],
    "ql": [6, 8, 6],  # answer-key point, Q = 20
    "seed": 0,
    "n_samples": 3,
}


def simulate_counts(labels, classifiers, ql, seed):
    """Build an observable count sketch from a small synthetic labeled test."""
    rng = random.Random(seed)
    answer_key = [rng.choice(labels) for _ in range(sum(ql))]
    accuracies = [
        {label: rng.uniform(0.45, 0.9) for label in labels}
        for _ in classifiers
    ]
    test_results = [
        tuple(
            (
                true
                if accuracies[j][true] > rng.random()
                else rng.choice([o for o in labels if o != true])
            )
            for j in range(len(classifiers))
        )
        for true in answer_key
    ]
    return collections.Counter(test_results)


def main(argv: list[str] | None = None) -> None:
    config = _config.build_config(
        "ntqr",
        "possible_and_consistent_sets",
        default_params=DEFAULTS,
        argv=argv,
    )
    out = _common.output_dir(config.category, config.name)

    labels = tuple(config.get("labels", DEFAULTS["labels"]))
    classifiers = tuple(config.get("classifiers", DEFAULTS["classifiers"]))
    ql = tuple(config.get("ql", DEFAULTS["ql"]))
    seed = config.get("seed", DEFAULTS["seed"])
    n_samples = config.get("n_samples", DEFAULTS["n_samples"])

    counts = simulate_counts(labels, classifiers, ql, seed)
    p_set = ntqr.evaluations.PossibleSet(labels, classifiers)
    c_set = ntqr.evaluations.ConsistentSet(labels, classifiers, counts)

    possible_count = int(p_set.set_count_at_ql(ql))
    consistent_count = sum(1 for _ in c_set.set_generator(ql))

    summary = {
        "labels": list(labels),
        "classifiers": list(classifiers),
        "ql": list(ql),
        "observed_counts": counts,
        "possible_set_count": possible_count,
        "consistent_set_count": consistent_count,
        "consistent_random_points": [
            list(point) for point in c_set.random_points(ql, n_samples)
        ],
        "correct_cuboid_points": [
            [list(label_pt) for label_pt in point]
            for point in c_set.random_correct_cuboid_points(ql, n_samples)
        ],
    }
    _common.save_json(out, "summary.json", summary)

    _common.banner(
        f"PossibleSet / ConsistentSet (R={len(labels)}, "
        f"N={len(classifiers)}, Q={sum(ql)})"
    )
    print(f"possible set size:   {possible_count}")
    print(f"consistent set size: {consistent_count}")
    print(f"-> wrote results to {out}")


if __name__ == "__main__":
    main()
