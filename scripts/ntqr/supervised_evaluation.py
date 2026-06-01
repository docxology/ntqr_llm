"""Orchestrate ntqr's supervised (ground-truth) evaluation of three binary
classifiers.

This is the labeled-data reference point the unsupervised algebraic evaluators
are compared against. Dataset-agnostic via ``--dataset`` / ``--dataset-file``;
defaults to the labeled UCI Adult example.

Real ntqr methods exercised:
    ntqr.r2.datasketches.TrioLabelVoteCounts
    ntqr.r2.evaluators.SupervisedEvaluation
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _common  # noqa: E402
import _config  # noqa: E402

_common.ensure_ntqr_importable()

from ntqr.r2.datasketches import TrioLabelVoteCounts  # noqa: E402
from ntqr.r2.evaluators import SupervisedEvaluation  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    config = _config.build_config("ntqr", "supervised_evaluation", argv=argv)
    out = _common.output_dir(config.category, config.name)

    label_counts = TrioLabelVoteCounts(config.load_label_vote_counts())
    evaluation = SupervisedEvaluation(label_counts)

    _common.save_json(
        out, "evaluation_exact.json", evaluation.evaluation_exact
    )
    _common.save_json(
        out, "evaluation_float.json", evaluation.evaluation_float
    )

    _common.banner(
        f"SupervisedEvaluation ({config.data_source}, ground truth)"
    )
    prev = evaluation.evaluation_float["prevalence"]
    print(f"true prevalence: a = {prev['a']:.4f}, b = {prev['b']:.4f}")
    for i, acc in enumerate(evaluation.evaluation_float["accuracy"]):
        print(
            f"classifier {i}: a-acc = {acc['a']:.4f}, b-acc = {acc['b']:.4f}"
        )
    print(f"-> wrote results to {out}")


if __name__ == "__main__":
    main()
