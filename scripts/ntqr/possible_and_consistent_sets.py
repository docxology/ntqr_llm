"""Orchestrate ntqr's PossibleSet / ConsistentSet computation for an R=3,
N=3 test.

The possible set counts every group evaluation allowed by the simplex and
marginalization axioms at a given answer-key point; the consistent set filters
that down to evaluations compatible with observed agreement/disagreement counts.

Configurable through ``params`` (labels, classifiers, ql answer-key point,
RNG seed, number of sampled points). Defaults reproduce the canonical R=3,
N=3, Q=6 example.

Real ntqr methods exercised:
    ntqr.evaluations.PossibleSet.set_count_at_ql
    ntqr.evaluations.ConsistentSet.set_generator / is_valid_point
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _common  # noqa: E402
import _config  # noqa: E402

_common.ensure_ntqr_importable()

from ntqr.llm.consistency import summarize_sets  # noqa: E402

DEFAULTS = {
    "labels": ["a", "b", "c"],
    "classifiers": ["i", "j", "k"],
    "ql": [2, 2, 2],  # answer-key point, Q = 6
    "seed": 0,
    "n_samples": 3,
    "max_allocations": 1_000_000,
    "max_events": 4096,
}


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

    summary = summarize_sets(
        labels, classifiers, ql, seed, n_samples,
        config.get("max_allocations", DEFAULTS["max_allocations"]),
        config.get("max_events", DEFAULTS["max_events"]),
    )
    _common.save_json(out, "summary.json", summary)

    _common.banner(
        f"PossibleSet / ConsistentSet (R={len(labels)}, "
        f"N={len(classifiers)}, Q={sum(ql)})"
    )
    print(f"possible set size:   {summary['possible_set_count']}")
    print(f"consistent set size: {summary['consistent_set_count']}")
    print(f"-> wrote results to {out}")


if __name__ == "__main__":
    main()
