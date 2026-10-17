"""Real NTQR 0.9 sparse-set/API regression and reproducibility checks."""

from itertools import product

import numpy as np
import pytest
import ntqr
from ntqr.evaluations import ConsistentSet
from ntqr.llm.consistency import summarize_sets
from ntqr.statistics import ResponseVariables


def test_release_version():
    assert ntqr.__version__ == "0.9"


@pytest.mark.parametrize("ql", [(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)])
def test_sparse_generator_matches_exhaustive_allocations(ql):
    counts = {("a",): 3, ("b",): 1}
    cs = ConsistentSet(("a", "b"), ("i",), counts)
    actual = set()
    for point in cs.set_generator(ql):
        assert cs.is_valid_point(point, ql)[0]
        actual.add(tuple(tuple(int(v) for v in row.toarray().ravel()) for row in point))
    expected = {
        ((a, b), (3 - a, 1 - b))
        for a, b in product(range(4), range(2))
        if a + b == ql[0]
    }
    assert actual == expected


def test_exact_correct_cuboid_handles_sparse_rows():
    cs = ConsistentSet(("a", "b"), ("i",), {("a",): 3, ("b",): 1})
    assert set(cs.correct_cuboid_generator((2, 2))) == {((1,), (0,)), ((2,), (1,))}


def test_09_response_mappings_use_observation_values():
    rv = ResponseVariables(("a", "b"), ("i",))
    counts = {("a",): 3, ("b",): 1}
    obs = rv.label_response_to_observable(counts)
    qs = rv.label_response_to_q((2, 2))
    for label in rv.labels:
        for event, symbol in rv.label_responses[label].items():
            assert obs[symbol] == counts[event]
            assert qs[symbol] == 2


def test_seeded_sparse_example_is_repeatable_and_conserves_counts():
    args = (("a", "b", "c"), ("i", "j", "k"), (2, 2, 2))
    first = summarize_sets(*args, seed=7)
    assert first == summarize_sets(*args, seed=7)
    assert 0 < first["consistent_set_count"] <= first["possible_set_count"]
    assert first["metadata"]["ntqr_version"] == "0.9"
    expected = ConsistentSet(
        *args[:2], first["observed_counts"]
    ).get_expected_event_counts()
    for point in first["consistent_random_points"]:
        assert [sum(row) for row in point] == [2, 2, 2]
        np.testing.assert_array_equal(np.asarray(point).sum(axis=0), expected)


def test_sampling_count_and_budget_are_bounded():
    args = (("a", "b"), ("i",), (1, 0))
    assert len(summarize_sets(*args, n_samples=99)["consistent_random_points"]) == 1
    assert summarize_sets(*args, n_samples=0)["consistent_random_points"] == []
    with pytest.raises(ValueError, match="max_allocations"):
        summarize_sets(*args, max_allocations=1)
    with pytest.raises(ValueError, match="ql"):
        summarize_sets(("a", "b"), ("i",), (-1, 2))


def test_alarm_demo_guards_expensive_and_malformed_inputs():
    from ntqr.llm.alarm_demo import run_alarm
    from ntqr.r2.raxioms import SingleClassifierAxioms
    from ntqr.r2.evaluations import SingleClassifierEvaluations

    spec = {
        "labels": ["a", "b"],
        "classifiers": ["i"],
        "Q": 2,
        "factors": [2, 2],
        "responses": [[1, 1]],
    }
    result = run_alarm(spec, SingleClassifierAxioms, SingleClassifierEvaluations)
    assert isinstance(result["are_misaligned"], bool)
    for changes in ({"Q": 31}, {"responses": [[1, 0]]}, {"factors": [0, 2]}):
        with pytest.raises(ValueError):
            run_alarm(
                {**spec, **changes}, SingleClassifierAxioms, SingleClassifierEvaluations
            )


@pytest.mark.parametrize("ql", [(1,), (-1, 5), (1, 1)])
def test_sparse_generator_rejects_invalid_label_totals(ql):
    cs = ConsistentSet(("a", "b"), ("i",), {("a",): 3, ("b",): 1})
    with pytest.raises(ValueError, match="ql"):
        list(cs.set_generator(ql))


def test_hydration_preserves_directory_guides():
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "output" / folder / filename
        for folder in ("figures", "manuscript")
        for filename in ("README.md", "AGENTS.md")
    ]
    before = {path: path.read_bytes() for path in paths}
    subprocess.run(
        [sys.executable, "scripts/z_generate_manuscript_variables.py"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert {path: path.read_bytes() for path in paths} == before


def test_hashable_points_have_value_semantics_across_storage_formats():
    from ntqr.evaluations import HashablePoint
    from scipy.sparse import csr_array
    import numpy as np
    original = np.array([[1, 0, 2]], dtype=np.int32)
    dense = HashablePoint(original)
    numeric = HashablePoint(original.astype(float))
    sparse = HashablePoint(csr_array(original))
    assert dense == numeric == sparse
    assert len({dense, numeric, sparse}) == 1
    original[0, 0] = 99
    assert dense == sparse
    assert sparse in {dense}
    assert dense.sum() == 3
    with pytest.raises(ValueError):
        dense._obj[0, 0] = 99


def test_consistency_budget_precedes_large_simulation_or_symbol_construction():
    from ntqr.llm.consistency import summarize_sets
    with pytest.raises(ValueError, match="max_events"):
        summarize_sets(("a", "b"), tuple(str(i) for i in range(30)), (1, 1))
    with pytest.raises(ValueError, match="max_allocations"):
        summarize_sets(("a", "b"), ("i",), (10 ** 9, 0))
