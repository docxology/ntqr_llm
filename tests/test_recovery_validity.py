"""Real sparse/dense vote sketches and persisted model votes exercise the validity audit."""

from itertools import product
import json
import math
from pathlib import Path

import pytest

from ntqr.llm import recovery_scoring as scoring
from ntqr.llm import simulation
from ntqr.llm.validity_audit import audit_votes, audit_persona_records
from ntqr.llm.bootstrap import bootstrap_recovery


def test_varying_independent_vote_columns_are_not_identifiable():
    observations = list(product("ab", repeat=3))
    result = audit_votes(list(zip(*observations)))
    assert result["variation_screen_passed"]
    assert result["status"] == "singular_pair_moments"
    assert not result["physical_solution_available"]
    assert result["uses_answer_key"] is False


def test_recorded_model_audit_needs_no_scenario_truth():
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "outputs/llm/multi_llm_personas_evaluation/models/mistral_latest/votes.json"
    )
    records = json.loads(path.read_text())["result"]
    first = audit_persona_records(records)
    changed_truth = {
        **records,
        "truth": {key: "deliberately irrelevant" for key in records["by_scenario"]},
    }
    assert first == audit_persona_records(changed_truth)
    assert first["status"] == "physical_solution"
    assert all(entry["physical"] for entry in first["solutions"])


def test_constant_and_missing_votes_have_distinct_statuses():
    assert (
        audit_votes([list("aaaa"), list("abba"), list("baab")])["status"]
        == "constant_classifier"
    )
    assert (
        audit_votes([[None, "b"], ["a", "b"], ["b", "a"]])["status"]
        == "incomplete_votes"
    )
    with pytest.raises(ValueError, match="aligned"):
        audit_votes([["a"], ["b", "a"], ["a"]])
    with pytest.raises(ValueError, match="votes"):
        audit_votes([["c"], ["b"], ["a"]])


def solution(a=0.8):
    return {
        "prevalence": {"a": 0.6, "b": 0.4},
        "accuracy": [{"a": a, "b": a} for _ in range(3)],
    }


def test_real_out_of_range_roots_are_not_probabilities():
    good, impossible = solution(), solution(1.2)
    assert scoring.mean_accuracy(impossible) > scoring.mean_accuracy(good)
    assert scoring.select_physical([impossible, good]) is good
    assert scoring.select_physical([impossible]) is None
    assert "probability_out_of_range" in scoring.solution_validity(impossible)["issues"]


def test_partial_recovery_does_not_outrank_a_complete_solution():
    partial, good = solution(1), solution(0.7)
    partial["accuracy"][0]["a"] = None
    assert scoring.mean_accuracy(partial) == -1
    assert scoring.select_best([partial, good]) is good
    assert not scoring.solution_validity(partial)["physical"]
    with pytest.raises(ValueError, match="same number"):
        scoring.accuracy_abs_errors(good, [{"a": 1, "b": 1}])
    assert scoring.safe_float(10**500) is None


@pytest.mark.parametrize(
    "updates",
    [
        {"common_mode": math.nan},
        {"anti_strength": 2},
        {"anti_pair": (0, 0)},
        {"anti_pair": (-1, 2)},
        {"anti_strength": 0.1},
    ],
)
def test_correlation_configuration_rejects_unrealizable_parameters(updates):
    with pytest.raises(ValueError):
        simulation.CorrelationConfig(**updates)


def test_empty_bootstrap_and_invalid_replicates_fail_clearly():
    with pytest.raises(ValueError, match="records"):
        bootstrap_recovery([], 10, 0)
    with pytest.raises(ValueError, match="n_boot"):
        bootstrap_recovery([("a", ("a", "a", "a"))], 0, 0)


def test_constant_classifier_does_not_rule_out_all_interior_physical_models():
    from ntqr.r2.datasketches import TrioVoteCounts
    from ntqr.r2.evaluators import ErrorIndependentEvaluation

    # Both true-label strata have a constant first judge and independent fair
    # second/third judges. This is a physical latent model with p=1/2.
    patterns = [("a", second, third) for second, third in product("ab", repeat=2)]
    labeled = {label: dict.fromkeys(patterns, 1) for label in "ab"}
    truth = simulation.sample_truth(labeled)
    physical = {
        "prevalence": {"a": truth["prevalence_a"], "b": 1 - truth["prevalence_a"]},
        "accuracy": truth["accuracies"],
    }
    assert scoring.solution_validity(physical)["physical"]
    observed = {pattern: 2 for pattern in patterns}
    observed.update(
        {pattern: 0 for pattern in product("ab", repeat=3) if pattern not in observed}
    )
    evaluator = ErrorIndependentEvaluation(TrioVoteCounts(observed))
    assert all(term == 0 for term in evaluator.alpha_prevalence_quadratic_terms().values())


@pytest.mark.parametrize(
    "counts",
    [
        {"a": {}, "b": {}},
        {"a": {("a", "b", "a"): -1}, "b": {}},
        {"a": {("a", "b"): 2}, "b": {}},
    ],
)
def test_invalid_labeled_counts_are_not_scored_as_real_observations(counts):
    with pytest.raises(ValueError):
        simulation.solve_recovery(counts)
