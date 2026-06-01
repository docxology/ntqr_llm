"""Single-model trader-persona ntqr demo (one model, three personas).

A focused companion to ``multi_llm_personas_evaluation.py``: it runs the three
trader personas through ONE model and writes the recovered-vs-true comparison,
so you can see the unsupervised algebra reproduce a single model's per-persona
accuracies without an answer key. The multi-model module reuses the same
``_analysis`` core to sweep several models and compare them.

Votes are never fabricated: if the model emits a reply that cannot be parsed
into a ``bullish``/``bearish`` call, that cell is an abstention (``None``) and
the run reports it. A model that abstains on any scenario cannot form a complete
R=2 vote pattern, so the demo stops with an actionable message rather than
solving on invented votes.

Config (``--set k=v`` / ``--config FILE.json``):
    backend       "ollama" (default) | "anthropic"
    model         model name (default "granite4.1:3b")
    temperature   sampling temperature (default 0.0)
    refresh       true to ignore the response cache

Run:
    python scripts/llm/trader_personas_evaluation.py
    python scripts/llm/trader_personas_evaluation.py --set model=mistral:latest
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _common  # noqa: E402
import _config  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _analysis  # noqa: E402
import _llm  # noqa: E402
import _personas  # noqa: E402

_common.ensure_ntqr_importable()

from ntqr.r2.datasketches import TrioLabelVoteCounts  # noqa: E402
from ntqr.r2.evaluators import (  # noqa: E402
    ErrorIndependentEvaluation,
    SupervisedEvaluation,
)

REQUIRES = ["ollama"]


def main(argv: list[str] | None = None) -> None:
    config = _config.build_config(
        "llm",
        "trader_personas_evaluation",
        default_params={
            "backend": "ollama",
            "model": "granite4.1:3b",
            "temperature": 0.0,
            "refresh": False,
        },
        argv=argv,
    )
    out_dir = _common.output_dir(config.category, config.name)
    client = _llm.LLMClient(
        backend=config.get("backend", "ollama"),
        model=config.get("model", "granite4.1:3b"),
        temperature=config.get("temperature", 0.0),
    )

    cache_path = out_dir / "cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    votes, unparsed, cache = _analysis.collect_votes(
        client,
        cache,
        config.get("refresh", False),
        on_error="tolerate",
    )
    cache_path.write_text(json.dumps(cache, indent=2) + "\n")

    label_counts = _analysis.build_label_counts(votes)
    _config.save_label_vote_counts(
        out_dir / "label_vote_counts.json", label_counts
    )
    _common.save_json(
        out_dir,
        "votes.json",
        {
            "backend": client.backend,
            "model": client.model,
            "personas": [p["name"] for p in _personas.PERSONAS],
            "unparsed_responses": unparsed,
            "by_scenario": votes,
        },
    )

    # No fabrication: an abstention leaves an incomplete vote pattern, so the
    # algebra cannot run on real data. Report it honestly and stop.
    if unparsed:
        _common.banner(
            f"trader personas via {client.backend}:{client.model} "
            f"({len(_personas.SCENARIOS)} scenarios)"
        )
        print(
            f"non-compliant: {unparsed} unparseable response(s); votes are "
            "never fabricated, so no unsupervised solve was attempted. "
            "Inspect votes.json, then retry (e.g. a different model or "
            "temperature)."
        )
        return

    trio_label = TrioLabelVoteCounts(label_counts)
    trio_votes = trio_label.to_TrioVoteCounts()
    error_independent = ErrorIndependentEvaluation(trio_votes)
    supervised = SupervisedEvaluation(trio_label)
    comparison = _analysis.build_comparison(supervised, error_independent)
    stats = _analysis.recovery_stats(comparison)

    _common.save_json(
        out_dir,
        "evaluation_supervised.json",
        {
            "exact": supervised.evaluation_exact,
            "float": supervised.evaluation_float,
        },
    )
    _common.save_json(
        out_dir,
        "evaluation_error_independent.json",
        {
            "exact": error_independent.evaluation_exact,
            "float": error_independent.evaluation_float,
        },
    )
    _common.save_json(out_dir, "comparison.json", comparison)
    _common.save_json(out_dir, "recovery_stats.json", stats)

    _common.banner(
        f"trader personas via {client.backend}:{client.model} "
        f"({len(_personas.SCENARIOS)} scenarios)"
    )
    print(
        f"true a-prevalence = {comparison['true_prevalence_a']:.3f}  "
        f"recovered (unsupervised) = "
        f"{comparison['recovered_prevalence_a']:.3f}"
    )
    for row in comparison["per_persona"]:
        true_acc = row["true_accuracy"]
        rec_acc = row["recovered_accuracy_unsupervised"]
        print(
            f"  {row['persona']:<24} "
            f"true(a/b)={true_acc['a']:.3f}/{true_acc['b']:.3f}  "
            f"recovered(a/b)={rec_acc['a']:.3f}/{rec_acc['b']:.3f}"
        )
    print(f"recovery MAE = {stats['recovery_mae']:.3f}")
    print(f"-> wrote results to {out_dir}")


if __name__ == "__main__":
    main()
