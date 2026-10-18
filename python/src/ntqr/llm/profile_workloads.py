"""Bounded real algebra and enumeration; seeded synthetic inputs, no model calls."""

from __future__ import annotations
import hashlib
import json
import random
from .runtime_profile import Workload


def workloads():
    from .simulation import Ensemble, generate_label_counts, solve_recovery
    from .consistency import summarize_sets
    from .validity_audit import audit_votes

    counts = generate_label_counts(
        Ensemble(0.6, ((0.9, 0.8), (0.85, 0.75), (0.8, 0.9))), 96, random.Random(104)
    )
    observed = {
        pattern: sum(bucket.get(pattern, 0) for bucket in counts.values())
        for pattern in counts["a"]
    }
    observations = [pattern for pattern, n in observed.items() for _ in range(n)]
    votes = list(zip(*observations))
    canonical = {
        label: {"".join(pattern): n for pattern, n in bucket.items()}
        for label, bucket in counts.items()
    }
    specs = [
        (
            "recovery",
            lambda: solve_recovery(counts),
            "Real error-independent recovery from 96 seeded synthetic observations",
            canonical,
        ),
        (
            "enumeration",
            lambda: summarize_sets(("a", "b"), ("0", "1"), (3, 3), seed=9, n_samples=0),
            "Exact consistent-set enumeration, counting without retaining sample points",
            {"ql": [3, 3], "seed": 9, "n_samples": 0},
        ),
        (
            "validity",
            lambda: audit_votes(votes),
            "Answer-key-free physical audit of a seeded three-judge vote matrix",
            votes,
        ),
    ]
    return [
        Workload(
            name,
            run,
            description,
            hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(),
        )
        for name, run, description, inputs in specs
    ]
