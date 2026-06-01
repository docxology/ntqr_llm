"""Baseline (NOT ntqr): plain majority-vote consensus on a trio's votes.

This is the naive thing practitioners reach for first: take the three
classifiers' votes per instance, call the majority the "answer", and report
the consensus label distribution and how often each classifier agrees with it.
It uses no algebra and no logic of evaluation -- it is here purely as a
comparison point for the ntqr evaluators, which can estimate accuracy WITHOUT
assuming the crowd is right. Dataset-agnostic via ``--dataset`` /
``--dataset-file``; defaults to UCI Adult.
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _common  # noqa: E402
import _config  # noqa: E402

_common.ensure_ntqr_importable()


def main(argv: list[str] | None = None) -> None:
    config = _config.build_config("baselines", "raw_majority_vote", argv=argv)
    out = _common.output_dir(config.category, config.name)

    # Flatten the labeled data into (votes -> count), discarding true labels
    # to mimic the unlabeled setting a majority vote actually operates in.
    vote_totals: collections.Counter = collections.Counter()
    for by_votes in config.load_label_vote_counts().values():
        for votes, count in by_votes.items():
            vote_totals[votes] += count

    total = sum(vote_totals.values())
    consensus = collections.Counter()
    agreement = [0, 0, 0]
    for votes, count in vote_totals.items():
        winner = collections.Counter(votes).most_common(1)[0][0]
        consensus[winner] += count
        for i, v in enumerate(votes):
            if v == winner:
                agreement[i] += count

    summary = {
        "method": "raw majority vote (baseline, non-ntqr)",
        "dataset": config.data_source,
        "total_instances": total,
        "consensus_label_fraction": {
            label: count / total for label, count in consensus.items()
        },
        "classifier_agreement_with_majority": [a / total for a in agreement],
    }
    _common.save_json(out, "summary.json", summary)

    _common.banner(f"raw majority vote (baseline, {config.data_source})")
    for label, count in consensus.items():
        print(f"consensus '{label}': {count / total:.4f}")
    print(f"-> wrote results to {out}")


if __name__ == "__main__":
    main()
