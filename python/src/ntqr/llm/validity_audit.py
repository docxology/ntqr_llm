"""Answer-key-free audit of binary-trio identifiability and physical recovery.

Passing a variation screen is necessary but insufficient. This audit checks
pair moments and every probability returned by the real NTQR solver, without
access to scenario truth or to any supervised evaluation.
"""

from __future__ import annotations

from itertools import combinations, product

from ntqr.r2.datasketches import TrioVoteCounts
from ntqr.r2.evaluators import ErrorIndependentEvaluation
from ntqr.llm.recovery_scoring import select_physical, solution_validity, safe_float


def audit_votes(votes_by_judge) -> dict:
    if len(votes_by_judge) != 3:
        raise ValueError("the audit requires exactly three judges")
    q = len(votes_by_judge[0])
    if q == 0 or any(len(row) != q for row in votes_by_judge):
        raise ValueError("vote rows must be nonempty and aligned")
    if any(value not in ("a", "b", None) for row in votes_by_judge for value in row):
        raise ValueError("votes must be a, b, or None for an abstention")
    missing = sum(value is None for row in votes_by_judge for value in row)
    constants = [
        index for index, row in enumerate(votes_by_judge) if len(set(row) - {None}) < 2
    ]
    result = {
        "n_items": q,
        "n_missing_votes": missing,
        "constant_classifiers": constants,
        "variation_screen_passed": not missing and not constants,
        "physical_solution_available": False,
        "uses_answer_key": False,
        "status": None,
        "pair_moment_numerators": {},
        "solutions": [],
        "selected_solution": None,
    }
    if missing:
        return {**result, "status": "incomplete_votes"}
    bits = [[int(value == "a") for value in row] for row in votes_by_judge]
    numerators = {
        f"{i},{j}": q * sum(x * y for x, y in zip(bits[i], bits[j]))
        - sum(bits[i]) * sum(bits[j])
        for i, j in combinations(range(3), 2)
    }
    result["pair_moment_numerators"] = numerators
    if constants:
        return {**result, "status": "constant_classifier"}
    if any(value == 0 for value in numerators.values()):
        return {**result, "status": "singular_pair_moments"}
    counts = dict.fromkeys(product("ab", repeat=3), 0)
    for pattern in zip(*votes_by_judge):
        counts[pattern] += 1
    evaluator = ErrorIndependentEvaluation(TrioVoteCounts(counts))
    solutions = evaluator.evaluation_float
    validity = [solution_validity(solution) for solution in solutions]
    selected = select_physical(solutions)
    result["solutions"] = validity
    if selected is None:
        finite = any(entry["finite"] for entry in validity)
        return {
            **result,
            "status": "nonphysical_roots" if finite else "nonfinite_or_complex_roots",
        }
    result["selected_solution"] = {
        "prevalence": {
            label: safe_float(value) for label, value in selected["prevalence"].items()
        },
        "accuracy": [
            {label: safe_float(value) for label, value in row.items()}
            for row in selected["accuracy"]
        ],
    }
    return {
        **result,
        "status": "physical_solution",
        "physical_solution_available": True,
    }


def audit_persona_records(records: dict) -> dict:
    """Read only recorded votes; ignore any truth field accompanying the records."""
    from ntqr.llm.personas import PERSONAS, CALL_TO_LABEL

    scenarios = records["by_scenario"]
    if not scenarios:
        raise ValueError("by_scenario must be nonempty")
    rows = [[] for _ in PERSONAS]
    for scenario_id, scenario in scenarios.items():
        for index, persona in enumerate(PERSONAS):
            if persona["key"] not in scenario:
                raise ValueError(
                    f"missing persona {persona['key']} in scenario {scenario_id}"
                )
            call = scenario[persona["key"]]
            if call is not None and call not in CALL_TO_LABEL:
                raise ValueError(
                    f"invalid recorded vote in scenario {scenario_id}: {call!r}"
                )
            rows[index].append(CALL_TO_LABEL[call] if call is not None else None)
    return {"model": records.get("model"), **audit_votes(rows)}
