"""Orchestrate ntqr's majority-voting evaluation of three binary classifiers.

Majority voting returns two logically consistent evaluations: one assuming the
crowd is always right, one assuming it is always wrong (the safety-relevant
a-priori assumption). Dataset-agnostic via ``--dataset`` / ``--dataset-file``.

Real ntqr methods exercised:
    ntqr.r2.datasketches.TrioLabelVoteCounts.to_TrioVoteCounts
    ntqr.r2.evaluators.MajorityVotingEvaluation
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _common  # noqa: E402
import _config  # noqa: E402

_common.ensure_ntqr_importable()

from ntqr.r2.datasketches import TrioLabelVoteCounts  # noqa: E402
from ntqr.r2.evaluators import MajorityVotingEvaluation  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    config = _config.build_config(
        "ntqr", "majority_voting_evaluation", argv=argv
    )
    out = _common.output_dir(config.category, config.name)

    data_sketch = TrioLabelVoteCounts(
        config.load_label_vote_counts()
    ).to_TrioVoteCounts()
    evaluation = MajorityVotingEvaluation(data_sketch)

    _common.save_json(
        out, "evaluation_exact.json", evaluation.evaluation_exact
    )
    _common.save_json(
        out, "evaluation_float.json", evaluation.evaluation_float
    )

    _common.banner(f"MajorityVotingEvaluation ({config.dataset})")
    names = ("crowd-always-right", "crowd-always-wrong")
    for name, solution in zip(names, evaluation.evaluation_float):
        print(f"{name}: a-prevalence = {solution['prevalence']['a']:.4f}")
    print(f"-> wrote results to {out}")


if __name__ == "__main__":
    main()
