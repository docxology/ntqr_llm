"""Regression tests for multi-model evaluation failure isolation and paths."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "llm"))

import multi_llm_personas_evaluation as multi  # noqa: E402
from ntqr.llm import analysis, personas  # noqa: E402
from ntqr.llm.study_records import ensure_unique_model_dirnames  # noqa: E402
from ntqr.r2.datasketches import TrioLabelVoteCounts  # noqa: E402
from ntqr.r2.evaluators import (  # noqa: E402
    MajorityVotingEvaluation,
    SupervisedEvaluation,
)

SOURCE = (
    ROOT / "outputs/llm/multi_llm_personas_evaluation/models/mistral_latest"
)


class _CachedClient:
    backend = "ollama"
    model = "mistral:latest"
    temperature = 0.1

    def __init__(self):
        self.calls = 0

    def complete(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError(
            "the recorded response fixture should avoid model calls"
        )


def _result(path: Path) -> dict:
    payload = json.loads(path.read_text())
    return payload.get("result", payload)


def _recorded_cache(client):
    recorded_votes = _result(SOURCE / "votes.json")["by_scenario"]
    cache = {}
    for scenario in personas.SCENARIOS:
        for persona in personas.PERSONAS:
            call = recorded_votes[scenario["id"]][persona["key"]]
            assert call is not None
            cache[
                analysis.cache_key(client, persona["key"], scenario["id"])
            ] = {
                "call": call,
                "unparsed": False,
                "raw": json.dumps({"call": call}),
                "parse_method": "schema_json",
                "schema_valid": True,
            }
    return cache


def test_successful_ei_result_preserves_real_majority_baseline(tmp_path):
    """The real recorded fixture keeps both successful evaluator results."""
    client = _CachedClient()
    row = multi.evaluate_model(
        client, _recorded_cache(client), False, tmp_path
    )

    expected_mae = _result(SOURCE / "recovery_stats.json")["recovery_mae"]
    expected_mv_mae = _result(SOURCE / "recovery_stats_majority_voting.json")[
        "recovery_mae"
    ]
    assert client.calls == 0
    assert "error" not in row
    assert row["recovery_mae"] == pytest.approx(expected_mae)
    assert row["mv_recovery_mae"] == pytest.approx(expected_mv_mae)
    assert _result(tmp_path / "recovery_stats.json")[
        "recovery_mae"
    ] == pytest.approx(expected_mae)
    assert _result(tmp_path / "evaluation_error_independent.json")["float"]
    assert _result(tmp_path / "evaluation_majority_voting.json")["float"]
    assert _result(tmp_path / "recovery_stats_majority_voting.json")[
        "recovery_mae"
    ] == pytest.approx(expected_mv_mae)


def test_real_majority_failure_has_no_physical_solution():
    """A unanimous majority pattern has no complete physical comparison."""
    counts = {"a": {("a", "a", "a"): 1}, "b": {}}
    labeled = TrioLabelVoteCounts(counts)
    majority = MajorityVotingEvaluation(labeled.to_TrioVoteCounts())
    supervised = SupervisedEvaluation(labeled)

    with pytest.raises(ValueError, match="no complete physical solution"):
        analysis.build_comparison(supervised, majority, physical_only=True)


def test_directory_collision_is_rejected_by_real_preflight():
    """Distinct model IDs cannot share the real historical artifact directory."""
    with pytest.raises(ValueError) as exc_info:
        ensure_unique_model_dirnames(["team/alpha:one", "team/alpha/one"])

    message = str(exc_info.value)
    assert "model directory collision" in message
    assert "team_alpha_one" in message
    assert "team/alpha:one" in message
    assert "team/alpha/one" in message
