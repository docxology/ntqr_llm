"""Tests for ntqr.llm.simulation -- the synthetic NTQR validation harness.

No mocks: every test drives the *real* ntqr solver on seeded, deterministic
synthetic vote counts, exercising convergence, the 1/sqrt(Q) scaling, physical
solution boundaries, and the imaginary-prevalence alarm (including its
common-mode blind spot).

    .venv/bin/python -m pytest tests/test_simulation.py -q
"""

from __future__ import annotations

import json
import random
import runpy
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "python" / "src"))

from ntqr.llm import simulation as sim  # noqa: E402

ENS = sim.Ensemble(
    prevalence_a=0.55,
    accuracies=((0.90, 0.80), (0.85, 0.75), (0.70, 0.95)),
)


# --------------------------------------------------------------------------- #
# Ensemble validation
# --------------------------------------------------------------------------- #
def test_ensemble_rejects_bad_prevalence():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            sim.Ensemble(bad, ((0.9, 0.8), (0.9, 0.8), (0.9, 0.8)))


def test_ensemble_rejects_wrong_judge_count():
    with pytest.raises(ValueError):
        sim.Ensemble(0.5, ((0.9, 0.8), (0.9, 0.8)))


def test_ensemble_rejects_constant_classifier_accuracy():
    # accuracy of exactly 1.0 or 0.0 means the judge never varies on that label
    with pytest.raises(ValueError):
        sim.Ensemble(0.5, ((1.0, 0.8), (0.9, 0.8), (0.9, 0.8)))


# --------------------------------------------------------------------------- #
# Generation: conservation, reproducibility
# --------------------------------------------------------------------------- #
def test_generate_conserves_q_and_split():
    q = 200
    counts = sim.generate_label_counts(ENS, q, random.Random(0))
    total = sum(counts["a"].values()) + sum(counts["b"].values())
    assert total == q
    assert sum(counts["a"].values()) == round(ENS.prevalence_a * q)
    # every pattern is a length-3 tuple over {a, b}
    for bucket in counts.values():
        for pattern in bucket:
            assert len(pattern) == 3
            assert set(pattern) <= {"a", "b"}


def test_generate_is_reproducible():
    a = sim.generate_label_counts(ENS, 300, random.Random(42))
    b = sim.generate_label_counts(ENS, 300, random.Random(42))
    assert a == b
    c = sim.generate_label_counts(ENS, 300, random.Random(43))
    assert a != c  # a different seed gives different draws


# --------------------------------------------------------------------------- #
# Sample truth: exact integer ground truth
# --------------------------------------------------------------------------- #
def test_sample_truth_exact():
    # 4 items truth a: judge0 votes a on 3 of them; judge1 always a; judge2 never
    # 2 items truth b: judge0 votes b on both
    counts = {
        "a": {("a", "a", "b"): 3, ("b", "a", "b"): 1},
        "b": {("b", "b", "a"): 2},
    }
    truth = sim.sample_truth(counts)
    assert truth["prevalence_a"] == pytest.approx(4 / 6)
    # judge0 on label a: correct (votes a) on 3/4
    assert truth["accuracies"][0]["a"] == pytest.approx(3 / 4)
    # judge1 on label a: always a -> 4/4
    assert truth["accuracies"][1]["a"] == pytest.approx(1.0)
    # judge2 on label a: never a -> 0/4
    assert truth["accuracies"][2]["a"] == pytest.approx(0.0)
    # judge0 on label b: votes b on 2/2
    assert truth["accuracies"][0]["b"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Recovery on error-independent data
# --------------------------------------------------------------------------- #
def test_solve_recovers_independent_ensemble():
    counts = sim.generate_label_counts(ENS, 4000, random.Random(7))
    result = sim.solve_recovery(counts)
    assert not result["alarm"]
    assert result["prevalence_real"]
    # at Q=4000 the recovery is within a few percent of the sample truth
    assert result["accuracy_mae"] < 0.05
    assert result["prevalence_abs_error"] < 0.05


def test_two_prevalence_roots_are_mirror_images():
    counts = sim.generate_label_counts(ENS, 2000, random.Random(11))
    result = sim.solve_recovery(counts)
    assert len(result["roots"]) == 2
    # the two logically-consistent prevalences are P_a and 1 - P_a
    assert result["roots"][0] + result["roots"][1] == pytest.approx(
        1.0, abs=1e-9
    )


def test_prevalence_discriminant_matches_quadratic():
    from ntqr.r2.datasketches import TrioLabelVoteCounts
    from ntqr.r2.evaluators import ErrorIndependentEvaluation

    counts = sim.generate_label_counts(ENS, 1000, random.Random(3))
    tvc = TrioLabelVoteCounts(counts).to_TrioVoteCounts()
    ev = ErrorIndependentEvaluation(tvc)
    coeffs = ev.alpha_prevalence_quadratic_terms()
    expected = 1 - 4 * float(coeffs[0]) / float(coeffs[2])
    assert sim.prevalence_discriminant(ev) == pytest.approx(expected)


def test_raw_recovery_mode_preserves_finite_nonphysical_roots():
    counts = {
        "a": {("a", "a", "a"): 3},
        "b": {("a", "b", "b"): 1, ("b", "b", "b"): 1, ("b", "a", "b"): 1},
    }
    result = sim.solve_recovery(counts, physical_only=False)
    assert not result["alarm"]
    assert result["prevalence_real"]
    assert result["recovered_prevalence_a"] == pytest.approx(1 / 3)
    assert result["prevalence_abs_error"] == pytest.approx(1 / 6)
    assert result["accuracy_mae"] is not None


# --------------------------------------------------------------------------- #
# Convergence and the 1/sqrt(Q) scaling
# --------------------------------------------------------------------------- #
def test_convergence_error_decreases_with_q():
    rows = sim.convergence_sweep(ENS, [64, 1024], n_trials=20, base_seed=2024)
    small_q, large_q = rows[0], rows[1]
    assert large_q["accuracy_mae_mean"] < small_q["accuracy_mae_mean"]
    assert (
        large_q["prevalence_abs_error_mean"]
        < small_q["prevalence_abs_error_mean"]
    )
    # Physical filtering can reject small-Q draws; both means use the remaining
    # positive, bounded set of complete recoveries.
    for row in rows:
        assert 0 < row["n_solved"] <= row["n_trials"]
        assert row["prevalence_abs_error_mean"] is not None
        assert row["accuracy_mae_mean"] is not None
        assert row["prevalence_abs_error_sd"] is not None
        assert row["accuracy_mae_sd"] is not None


def test_convergence_counts_only_complete_recoveries():
    # Trial 1 has real prevalence roots but no complete physical solution. It
    # must contribute to neither mean nor to the solved count.
    row = sim.convergence_sweep(ENS, [8], n_trials=2, base_seed=7000)[0]
    assert row["n_solved"] == 1
    assert row["prevalence_abs_error_mean"] == pytest.approx(0.125)
    assert row["accuracy_mae_mean"] == pytest.approx(0.0861111111111111)
    assert row["prevalence_abs_error_sd"] == 0
    assert row["accuracy_mae_sd"] == 0


def test_sqrt_q_fit_slope_near_minus_half():
    rows = sim.convergence_sweep(
        ENS, [64, 256, 1024], n_trials=30, base_seed=2024
    )
    fit = sim.sqrt_q_fit(rows, "accuracy_mae_mean")
    # sampling-noise-limited error scales like Q**-0.5
    assert fit["slope"] == pytest.approx(-0.5, abs=0.2)
    assert fit["r_squared"] > 0.9


def test_sqrt_q_fit_needs_two_points():
    assert (
        sim.sqrt_q_fit([{"q": 64, "accuracy_mae_mean": 0.1}])["slope"] is None
    )


# --------------------------------------------------------------------------- #
# The imaginary-prevalence alarm
# --------------------------------------------------------------------------- #
def test_independent_effective_accuracies_equal_configuration():
    expected = [{"a": alpha, "b": beta} for alpha, beta in ENS.accuracies]
    config = sim.CorrelationConfig("independent")
    assert sim.effective_accuracies(ENS, config) == expected


@pytest.mark.parametrize(
    "config, expected",
    [
        (
            sim.CorrelationConfig("common", common_mode=0.4),
            [
                {"a": 0.54, "b": 0.48},
                {"a": 0.51, "b": 0.45},
                {"a": 0.42, "b": 0.57},
            ],
        ),
        (
            sim.CorrelationConfig(
                "anti+common",
                common_mode=0.25,
                anti_pair=(0, 1),
                anti_strength=0.7,
            ),
            [
                {"a": 0.675, "b": 0.6},
                {"a": 0.41875, "b": 0.44875},
                {"a": 0.525, "b": 0.7125},
            ],
        ),
    ],
    ids=["common_mode", "anti_pair_with_common_mode"],
)
def test_effective_marginals_match_formula_and_realized_generation(
    config, expected
):
    effective = sim.effective_accuracies(ENS, config)
    for expected_row, effective_row in zip(expected, effective):
        assert effective_row["a"] == pytest.approx(expected_row["a"])
        assert effective_row["b"] == pytest.approx(expected_row["b"])

    counts = sim.generate_label_counts(
        ENS, 20_000, random.Random(9182), config
    )
    realized = sim.sample_truth(counts)["accuracies"]
    for effective_row, realized_row in zip(effective, realized):
        assert realized_row["a"] == pytest.approx(effective_row["a"], abs=0.02)
        assert realized_row["b"] == pytest.approx(effective_row["b"], abs=0.02)


def test_no_false_alarm_on_independent():
    rows = sim.correlation_sweep(
        ENS,
        1024,
        [sim.CorrelationConfig("independent")],
        n_trials=25,
        base_seed=5000,
    )
    assert rows[0]["alarm_rate"] == 0.0
    assert rows[0]["discriminant_min"] > 0  # discriminant stays positive


def test_alarm_fires_on_anticorrelated_pair():
    cfg = sim.CorrelationConfig("anti", anti_pair=(0, 1), anti_strength=0.4)
    rows = sim.correlation_sweep(ENS, 1024, [cfg], n_trials=25, base_seed=5000)
    # a strongly anti-correlated pair drives the prevalence root imaginary often
    assert rows[0]["alarm_rate"] > 0.5
    assert rows[0]["discriminant_min"] < 0


def test_common_mode_is_a_blind_spot():
    independent = sim.CorrelationConfig("independent")
    common = sim.CorrelationConfig("common", common_mode=0.4)
    rows = sim.correlation_sweep(
        ENS, 1024, [independent, common], n_trials=25, base_seed=5000
    )
    indep_row, common_row = rows
    # positive common-mode correlation never trips the alarm ...
    assert common_row["alarm_rate"] == 0.0
    # ... yet it shrinks the discriminant toward the threshold ...
    assert common_row["discriminant_mean"] < indep_row["discriminant_mean"]
    # ... and silently worsens the recovery it failed to warn about.
    assert (
        common_row["silent_prevalence_abs_error_mean"]
        > indep_row["silent_prevalence_abs_error_mean"]
    )


# --------------------------------------------------------------------------- #
# Tie-break failure as judges approach random
# --------------------------------------------------------------------------- #
def test_uniform_ensemble_is_symmetric():
    ens = sim.uniform_ensemble(0.6, 0.8)
    assert ens.prevalence_a == 0.6
    assert ens.accuracies == ((0.8, 0.8), (0.8, 0.8), (0.8, 0.8))


def test_tie_break_safe_for_strong_judges_fails_near_random():
    rows = sim.tie_break_sweep(
        0.6, [0.90, 0.55], q=256, n_trials=40, base_seed=4000
    )
    strong, near_random = rows
    # strong judges resolve the mirror cleanly ...
    assert strong["catastrophic_rate"] < 0.05
    # ... while near-random judges invert far more often
    assert near_random["catastrophic_rate"] > strong["catastrophic_rate"] + 0.3


# --------------------------------------------------------------------------- #
# Convergence robustness across diverse ensembles
# --------------------------------------------------------------------------- #
def test_ensemble_robustness_slopes_near_minus_half():
    ensembles = [
        (
            "balanced",
            sim.Ensemble(0.50, ((0.88, 0.82), (0.80, 0.86), (0.74, 0.78))),
        ),
        (
            "tilted",
            sim.Ensemble(0.70, ((0.92, 0.78), (0.83, 0.80), (0.71, 0.90))),
        ),
    ]
    rows = sim.ensemble_robustness_sweep(
        ensembles, [64, 256, 1024], n_trials=20, base_seed=7000
    )
    assert len(rows) == 2
    for r in rows:
        # every ensemble shows the sampling-noise-limited 1/sqrt(Q) signature
        assert r["slope"] == pytest.approx(-0.5, abs=0.2)
        assert r["r_squared"] > 0.85
        assert r["largest_q_acc_mae"] < r["smallest_q_acc_mae"]


# --------------------------------------------------------------------------- #
# Error-independent vs majority-voting evaluation
# --------------------------------------------------------------------------- #
def test_compare_evaluators_returns_both_maes():
    counts = sim.generate_label_counts(ENS, 512, random.Random(8))
    out = sim.compare_evaluators(counts)
    assert out["ei_mae"] is not None and out["mv_mae"] is not None
    assert out["ei_mae"] >= 0 and out["mv_mae"] >= 0


def test_ei_beats_mv_for_strong_judges():
    # strong, clearly-better-than-random judges: the exact algebra should win
    rows = sim.mv_vs_ei_sweep(
        [0.55], [0.88], q=512, n_trials=60, base_seed=8000
    )
    cell = rows[0]
    assert cell["ei_mae_mean"] < cell["mv_mae_mean"]
    assert cell["ei_win_rate"] > 0.5


def test_mv_more_robust_in_tiebreak_danger_zone():
    # extreme prevalence + weak judges: EI's mirror inverts, MV stays robust
    rows = sim.mv_vs_ei_sweep(
        [0.85], [0.66], q=512, n_trials=60, base_seed=8500
    )
    cell = rows[0]
    # MV's recovery is markedly better here, so EI wins well under half the time
    assert cell["ei_win_rate"] < 0.5
    assert cell["mv_mae_mean"] < cell["ei_mae_mean"]


def test_recorded_artwork_uses_each_experiments_own_provenance(tmp_path):
    """Generator and cached builders render real summaries, never new trials."""
    source = (
        REPO / "outputs/sim/synthetic_recovery_study/summary.json"
    ).read_text(encoding="utf-8")
    record = json.loads(source)
    summary = record.get("result", record)
    generator = runpy.run_path(
        str(REPO / "scripts/sim/synthetic_recovery_study.py")
    )
    generated_record = json.loads(source)
    generated_summary = generated_record.get("result", generated_record)
    generator["_annotate_sidecars"](generated_summary)
    for study, field in (
        ("tie_break", "prevalence_a"),
        ("mv_vs_ei", "prevalences"),
    ):
        metadata = generated_summary[study]["metadata"]
        assert metadata[field] == summary[study][field]
        assert "ensemble" not in metadata

    generated_dir = tmp_path / "generator"
    cached_dir = tmp_path / "cached"
    generated_dir.mkdir()
    cached_dir.mkdir()
    generated = generator["render_figures"](generated_summary, generated_dir)
    cached_renderer = runpy.run_path(
        str(REPO / "scripts/validation/render_supporting_figures.py")
    )
    assert cached_renderer["main"](["--output-dir", str(cached_dir)]) == 0
    studies = (
        ("convergence", "attempts/Q"),
        ("correlation", "trials/condition"),
        ("tie_break", "trials/accuracy"),
        ("mv_vs_ei", "trials/cell"),
    )
    artwork = []
    for path, (study, unit) in zip(generated, studies, strict=True):
        artwork.append((path, study, unit))
        artwork.append((cached_dir / path.name, study, unit))
    for path, study, unit in artwork:
        assert path.is_file()
        vector = path.with_suffix(".svg")
        text = " ".join(
            " ".join(ET.parse(vector).getroot().itertext()).split()
        )
        block = summary[study]
        assert f"{block['n_trials']} {unit}" in text
        assert f"seed={block['seed']}" in text
        if "q" in block:
            assert f"Q={block['q']}" in text
        if study == "mv_vs_ei":
            prevalences = ", ".join(f"{p:.2f}" for p in block["prevalences"])
            assert f"p_a={{{prevalences}}}" in text
            assert "symmetric judges" in text
            assert "J0(" not in text
        elif study == "tie_break":
            assert f"p_a={block['prevalence_a']:.2f}" in text
            assert "symmetric judges" in text
            assert "J0(" not in text
        else:
            assert f"p_a={summary['ensemble']['prevalence_a']:.2f}" in text
        if study == "convergence":
            assert "complete physical / attempted" in text
            assert "remaining attempts unavailable" in text
            for row in block["rows"]:
                assert f"{row['n_solved']}/{row['n_trials']}" in text
