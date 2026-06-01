"""Baseline (NOT ntqr): pairwise agreement rates among the three classifiers.

Pairwise agreement is the statistic many practitioners use as a proxy for
"how good is this ensemble" when they have no labels. It is a baseline, not an
evaluation: two classifiers that are wrong in the same way agree perfectly.
This script reports it precisely so the contrast with ntqr's logical alarm
(which can flag exactly that failure) is concrete. Dataset-agnostic via
``--dataset`` / ``--dataset-file``; defaults to UCI Adult.
"""

from __future__ import annotations

import collections
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _common  # noqa: E402
import _config  # noqa: E402

_common.ensure_ntqr_importable()


def main(argv: list[str] | None = None) -> None:
    config = _config.build_config(
        "baselines", "naive_agreement_accuracy", argv=argv
    )
    out = _common.output_dir(config.category, config.name)

    vote_totals: collections.Counter = collections.Counter()
    for by_votes in config.load_label_vote_counts().values():
        for votes, count in by_votes.items():
            vote_totals[votes] += count

    total = sum(vote_totals.values())
    pairs = list(itertools.combinations(range(3), 2))
    agreement = {pair: 0 for pair in pairs}
    for votes, count in vote_totals.items():
        for a, b in pairs:
            if votes[a] == votes[b]:
                agreement[(a, b)] += count

    summary = {
        "method": "pairwise agreement rate (baseline, non-ntqr)",
        "dataset": config.data_source,
        "total_instances": total,
        "pairwise_agreement": {
            f"{a}-{b}": agreement[(a, b)] / total for a, b in pairs
        },
    }
    _common.save_json(out, "summary.json", summary)

    _common.banner(f"pairwise agreement (baseline, {config.data_source})")
    for a, b in pairs:
        print(f"classifiers {a}-{b}: {agreement[(a, b)] / total:.4f}")
    print(f"-> wrote results to {out}")


if __name__ == "__main__":
    main()
