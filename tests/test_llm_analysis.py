"""Pure unit tests for the persona-analysis core (``ntqr.llm.analysis``,
``ntqr.llm.recovery_scoring``) and the thin collect-votes glue
(``scripts/llm/analysis.py``) -- no network, no ntqr algebra solve, no TeX.

These cover the numeric core of the persona evaluation (vote parsing, solution
selection, and the recovery statistics) so the LLM method modules can be
validated without a running Ollama. Runs under pytest *and* standalone:

    pytest tests/test_llmanalysis.py
    python tests/test_llmanalysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "llm"))
sys.path.insert(0, str(SCRIPTS.parent / "python" / "src"))

import _analysis  # noqa: E402  (thin collect-votes glue)
import _llm  # noqa: E402
from ntqr.llm import analysis  # noqa: E402
from ntqr.llm import personas as _personas  # noqa: E402
from ntqr.llm import recovery_scoring as R  # noqa: E402


class _RaisingClient:
    backend = "ollama"
    model = "fake"
    temperature = 0.0

    def complete(self, system, user, json_mode=False, schema=None):
        raise _llm.LLMUnavailable("simulated hang")


def test_collect_votes_raises_by_default():
    n_calls = 0

    class _Client(_RaisingClient):
        def complete(self, *a, **k):
            nonlocal n_calls
            n_calls += 1
            return super().complete(*a, **k)

    try:
        _analysis.collect_votes(_Client(), {}, refresh=True)
        raise AssertionError("expected LLMUnavailable to propagate")
    except _llm.LLMUnavailable:
        pass
    assert n_calls == 1  # fails fast on the first call


def test_collect_votes_tolerate_records_abstention():
    votes, unparsed, cache = _analysis.collect_votes(
        _RaisingClient(), {}, refresh=True, on_error="tolerate"
    )
    n_calls = len(_personas.SCENARIOS) * len(_personas.PERSONAS)
    assert unparsed == n_calls  # every unreachable call -> abstention
    assert len(cache) == n_calls
    # No fabrication: an unreachable cell is recorded as None, not a vote.
    some = _personas.SCENARIOS[0]["id"]
    assert votes[some]["pessimist"] is None
    sample = next(iter(cache.values()))
    assert sample["call"] is None
    assert sample["unparsed"] is True
    assert sample["parse_method"] == "unavailable"
    assert sample["prompt_version"] == _personas.PROMPT_VERSION


def test_cache_preserves_unparsed_on_replay():
    """A cached abstention stays an abstention on replay (no silent recount).

    Regression for the bug where the cache stored only the post-fallback call
    string, so re-running from cache reported 0 unparsed and silently erased a
    material caveat (how many votes were unparseable model output).
    """
    _, unparsed1, cache = _analysis.collect_votes(
        _RaisingClient(), {}, refresh=True, on_error="tolerate"
    )
    n_calls = len(_personas.SCENARIOS) * len(_personas.PERSONAS)
    assert unparsed1 == n_calls

    class _WouldSucceed(_RaisingClient):
        def complete(self, *a, **k):
            return '{"call": "bullish"}'

    votes, unparsed2, _ = _analysis.collect_votes(
        _WouldSucceed(), cache, refresh=False, on_error="tolerate"
    )
    assert unparsed2 == n_calls  # read from the cached record, not re-derived
    some = _personas.SCENARIOS[0]["id"]
    assert votes[some]["optimist"] is None


def test_parse_call_json():
    assert analysis.parse_call('{"call": "bullish"}') == "bullish"
    assert analysis.parse_call('  {"call":"BEARISH"}  ') == "bearish"


def test_parse_call_keyword_fallback():
    assert analysis.parse_call("I am bullish on this") == "bullish"
    assert analysis.parse_call("clearly bearish here") == "bearish"


def test_parse_call_ambiguous_is_none():
    assert analysis.parse_call("bullish and bearish both") is None
    assert analysis.parse_call("no opinion") is None


def test_parse_call_record_reports_method():
    parsed = analysis.parse_call_record('{"call": "bullish"}')
    assert parsed["call"] == "bullish"
    assert parsed["parse_method"] == "json"
    parsed = analysis.parse_call_record("clearly bearish")
    assert parsed["call"] == "bearish"
    assert parsed["parse_method"] == "keyword"
    parsed = analysis.parse_call_record("no directional view")
    assert parsed["call"] is None
    assert parsed["parse_method"] == "none"


def test_prompt_cache_key_contains_version_and_hash():
    client = _RaisingClient()
    key = analysis.cache_key(client, "optimist", "s01")
    assert f"p{_personas.PROMPT_VERSION}" in key
    assert f"h{_personas.prompt_hash()}" in key


def test_prompt_text_is_balanced_schema_contract():
    systems = " ".join(p["system"].lower() for p in _personas.PERSONAS)
    assert "exactly one json object" in _personas.USER_TEMPLATE.lower()
    assert "no neutral" in _personas.USER_TEMPLATE.lower()
    assert "no mixed" in _personas.USER_TEMPLATE.lower()
    assert "no abstention" in _personas.USER_TEMPLATE.lower()
    assert _personas.CALL_SCHEMA["properties"]["call"]["enum"] == [
        "bullish",
        "bearish",
    ]
    assert "bullish" in _personas.USER_TEMPLATE
    assert "bearish" in _personas.USER_TEMPLATE
    assert "relentlessly" not in systems
    assert "deeply skeptical" not in systems
    assert "genuinely mixed cases" in systems
    assert "do not force optimism" in systems
    assert "do not force pessimism" in systems
    assert "choose bullish" in systems
    assert "choose bearish" in systems


def test_ollama_json_mode_uses_schema_object_when_supplied(monkeypatch):
    captured = {}

    def fake_post(url, payload, headers, timeout):
        captured["payload"] = payload
        return {"message": {"content": '{"call":"bullish"}'}}

    monkeypatch.setattr(_llm, "_post_json", fake_post)
    client = _llm.LLMClient(model="fake", temperature=0.1)
    out = client.complete(
        "system",
        "user",
        json_mode=True,
        schema=_personas.CALL_SCHEMA,
    )
    assert out == '{"call":"bullish"}'
    assert captured["payload"]["format"] == _personas.CALL_SCHEMA
    assert captured["payload"]["options"]["temperature"] == 0.1


def test_strict_parse_rejects_third_label_without_coercion():
    parsed = analysis.parse_call_record(
        '{"call":"neutral"}', schema_strict=True
    )
    assert parsed["call"] is None
    assert parsed["schema_valid"] is False
    assert "invalid call" in parsed["invalid_reason"]


def test_collect_votes_retries_invalid_schema_response():
    class _RetryClient(_RaisingClient):
        def __init__(self):
            self.calls = 0

        def complete(self, system, user, json_mode=False, schema=None):
            self.calls += 1
            assert schema == _personas.CALL_SCHEMA
            if self.calls == 1:
                return '{"call":"neutral"}'
            return '{"call":"bearish"}'

    client = _RetryClient()
    votes, unparsed, cache = _analysis.collect_votes(
        client, {}, refresh=True, on_error="tolerate"
    )
    first = _personas.SCENARIOS[0]["id"]
    assert votes[first]["optimist"] == "bearish"
    assert unparsed == 0
    record = cache[analysis.cache_key(client, "optimist", first)]
    assert record["call"] == "bearish"
    assert record["unparsed"] is False
    assert record["repair_used"] is True
    assert record["attempts"][0]["schema_valid"] is False
    assert record["attempts"][1]["schema_valid"] is True


def test_collect_votes_does_not_coerce_after_failed_retry():
    class _StillInvalidClient(_RaisingClient):
        def complete(self, system, user, json_mode=False, schema=None):
            return '{"call":"mixed"}'

    votes, unparsed, cache = _analysis.collect_votes(
        _StillInvalidClient(), {}, refresh=True, on_error="tolerate"
    )
    n_calls = len(_personas.SCENARIOS) * len(_personas.PERSONAS)
    assert unparsed == n_calls
    first = _personas.SCENARIOS[0]["id"]
    assert votes[first]["optimist"] is None
    record = next(iter(cache.values()))
    assert record["call"] is None
    assert record["unparsed"] is True
    assert len(record["attempts"]) == 2
    assert {a["call"] for a in record["attempts"]} == {None}


def test_normalize_cache_record_reads_legacy_shape():
    record = analysis.normalize_cache_record({"call": None, "unparsed": True})
    assert record["call"] is None
    assert record["unparsed"] is True
    assert record["parse_method"] == "none"


def test_safe_float_guards_complex():
    assert R.safe_float(0.5) == 0.5
    assert R.safe_float(complex(0, 1)) is None
    assert R.safe_float(None) is None


def test_safe_float_rejects_non_finite():
    assert R.safe_float(float("nan")) is None
    assert R.safe_float(float("inf")) is None
    assert R.safe_float(float("-inf")) is None


def test_constant_personas_flags_one_sided_votes():
    names = [p["name"] for p in _personas.PERSONAS]
    # Optimist always "a", pessimist always "b"; neutral varies.
    label_counts = {
        "a": {("a", "a", "b"): 5, ("a", "b", "b"): 3},
        "b": {("a", "b", "b"): 4},
    }
    flagged = analysis.constant_personas(label_counts)
    assert names[0] in flagged  # optimist: only "a"
    assert names[2] in flagged  # pessimist: only "b"
    assert names[1] not in flagged  # neutral: both seen


def test_constant_personas_none_when_all_vary():
    label_counts = {
        "a": {("a", "a", "a"): 2, ("b", "b", "b"): 1},
        "b": {("a", "b", "a"): 1},
    }
    assert analysis.constant_personas(label_counts) == []


def test_build_label_counts_unanimous_truth():
    # Every persona votes the authored truth -> two unanimous patterns.
    votes = {
        s["id"]: {p["key"]: s["truth"] for p in _personas.PERSONAS}
        for s in _personas.SCENARIOS
    }
    counts = analysis.build_label_counts(votes)
    n_a = sum(1 for s in _personas.SCENARIOS if s["truth"] == "bullish")
    n_b = sum(1 for s in _personas.SCENARIOS if s["truth"] == "bearish")
    assert counts["a"] == {("a", "a", "a"): n_a}
    assert counts["b"] == {("b", "b", "b"): n_b}


def _solution(acc, prev_a=0.5):
    return {
        "prevalence": {"a": prev_a, "b": 1 - prev_a},
        "accuracy": [{"a": x, "b": y} for x, y in acc],
    }


def test_select_best_prefers_better_than_random():
    good = _solution([(0.9, 0.9), (0.8, 0.8), (0.7, 0.7)])
    mirror = _solution([(0.1, 0.1), (0.2, 0.2), (0.3, 0.3)])
    assert R.select_best([good, mirror]) is good
    assert R.select_best([mirror, good]) is good


def test_build_comparison_picks_real_solution():
    sup = SimpleNamespace(
        evaluation_float={
            "prevalence": {"a": 0.5},
            "accuracy": [
                {"a": 1.0, "b": 0.9},
                {"a": 0.9, "b": 1.0},
                {"a": 0.2, "b": 1.0},
            ],
        }
    )
    good = _solution([(1.0, 0.85), (1.0, 1.0), (0.2, 1.0)], prev_a=0.46)
    mirror = _solution([(0.0, 0.15), (0.0, 0.0), (0.8, 0.0)], prev_a=0.54)
    ei = SimpleNamespace(evaluation_float=[mirror, good])
    comp = analysis.build_comparison(sup, ei)
    assert comp["recovered_prevalence_a"] == 0.46
    assert (
        comp["per_persona"][0]["recovered_accuracy_unsupervised"]["a"] == 1.0
    )


def test_recovery_stats_mae_and_prevalence():
    comparison = {
        "true_prevalence_a": 0.5,
        "recovered_prevalence_a": 0.45,
        "per_persona": [
            {
                "persona": "p1",
                "true_accuracy": {"a": 1.0, "b": 0.8},
                "recovered_accuracy_unsupervised": {"a": 0.9, "b": 0.8},
            },
            {
                "persona": "p2",
                "true_accuracy": {"a": 0.6, "b": 1.0},
                "recovered_accuracy_unsupervised": {"a": 0.5, "b": 1.0},
            },
        ],
    }
    stats = analysis.recovery_stats(comparison)
    # abs errors: 0.1, 0.0, 0.1, 0.0 -> mean 0.05
    assert abs(stats["recovery_mae"] - 0.05) < 1e-9
    assert abs(stats["prevalence_abs_error"] - 0.05) < 1e-9
    assert stats["n_accuracy_terms"] == 4


def test_recovery_stats_skips_missing_terms():
    comparison = {
        "true_prevalence_a": 0.5,
        "recovered_prevalence_a": None,
        "per_persona": [
            {
                "persona": "p1",
                "true_accuracy": {"a": 1.0, "b": 0.8},
                "recovered_accuracy_unsupervised": {"a": None, "b": 0.7},
            },
        ],
    }
    stats = analysis.recovery_stats(comparison)
    assert stats["n_accuracy_terms"] == 1  # only the b term counted
    assert stats["recovery_mae"] is None
    assert abs(stats["available_terms_mae"] - 0.1) < 1e-9
    assert stats["complete"] is False
    assert stats["prevalence_abs_error"] is None


def _votes_from(fn):
    """Build a votes dict {scenario_id: {persona_key: call}} from fn(key, i)."""
    votes = {}
    for i, s in enumerate(_personas.SCENARIOS):
        votes[s["id"]] = {
            p["key"]: fn(p["key"], i) for p in _personas.PERSONAS
        }
    return votes


def test_vote_variation_flags_constant():
    votes = _votes_from(
        lambda key, i: (
            "bullish"
            if key == "optimist"
            else (
                "bearish"
                if key == "pessimist"
                else ("bullish" if i % 2 else "bearish")
            )
        )
    )
    var = analysis.vote_variation(votes)
    names = [p["name"] for p in _personas.PERSONAS]
    assert var[names[0]]["is_constant"] is True  # optimist: only bullish
    assert var[names[0]]["modal_fraction"] == 1.0
    assert var[names[2]]["is_constant"] is True  # pessimist: only bearish
    assert var[names[1]]["is_constant"] is False  # neutral varies
    assert var[names[1]]["modal_fraction"] < 1.0


def test_pairwise_disagreement_bounds():
    # optimist & neutral always bullish; pessimist always bearish.
    votes = _votes_from(
        lambda key, i: "bearish" if key == "pessimist" else "bullish"
    )
    dis = analysis.pairwise_disagreement(votes)
    names = [p["name"] for p in _personas.PERSONAS]
    assert dis["pairs"][f"{names[0]} vs {names[1]}"] == 0.0  # identical
    assert dis["pairs"][f"{names[0]} vs {names[2]}"] == 1.0  # always differ
    assert 0.0 <= dis["mean"] <= 1.0


def test_evaluability_diagnostic_predicts_degenerate():
    votes = _votes_from(lambda key, i: "bullish")  # everyone constant
    diag = analysis.evaluability_diagnostic(votes)
    assert diag["predicted_evaluable"] is False
    assert len(diag["constant_classifiers"]) == 3


def test_evaluability_diagnostic_predicts_evaluable():
    offset = {"optimist": 0, "neutral": 1, "pessimist": 0}
    votes = _votes_from(
        lambda key, i: "bullish" if (i + offset[key]) % 2 else "bearish"
    )
    diag = analysis.evaluability_diagnostic(votes)
    assert diag["predicted_evaluable"] is True
    assert diag["constant_classifiers"] == []


def test_prompt_health_reports_entropy_and_parse_methods():
    votes = _votes_from(lambda key, i: "bullish" if i % 2 else "bearish")
    records = {
        s["id"]: {
            p["key"]: {
                "parse_method": "schema_json",
                "attempts": [
                    {
                        "schema_valid": True,
                        "parse_method": "schema_json",
                    }
                ],
                "repair_used": False,
                "unparsed": False,
            }
            for p in _personas.PERSONAS
        }
        for s in _personas.SCENARIOS
    }
    health = analysis.prompt_health(votes, records)
    for row in health.values():
        assert row["n_unparsed"] == 0
        assert row["parse_rate"] == 1.0
        assert row["primary_parse_rate"] == 1.0
        assert row["final_parse_rate"] == 1.0
        assert row["n_repaired"] == 0
        assert row["parse_methods"]["schema_json"] == len(_personas.SCENARIOS)
        assert row["entropy_bits"] > 0.9


def test_recovery_stats_per_label():
    comparison = {
        "true_prevalence_a": 0.5,
        "recovered_prevalence_a": 0.5,
        "per_persona": [
            {
                "persona": "p1",
                "true_accuracy": {"a": 1.0, "b": 0.8},
                "recovered_accuracy_unsupervised": {"a": 0.9, "b": 0.8},
            },
            {
                "persona": "p2",
                "true_accuracy": {"a": 0.6, "b": 1.0},
                "recovered_accuracy_unsupervised": {"a": 0.4, "b": 1.0},
            },
        ],
    }
    stats = analysis.recovery_stats(comparison)
    # label a errors: 0.1, 0.2 -> 0.15 ; label b errors: 0.0, 0.0 -> 0.0
    assert abs(stats["recovery_mae_by_label"]["a"] - 0.15) < 1e-9
    assert abs(stats["recovery_mae_by_label"]["b"] - 0.0) < 1e-9


def test_recovery_stats_worst_term_and_max():
    comparison = {
        "true_prevalence_a": 0.5,
        "recovered_prevalence_a": 0.5,
        "per_persona": [
            {
                "persona": "p1",
                "true_accuracy": {"a": 1.0, "b": 0.8},
                "recovered_accuracy_unsupervised": {"a": 0.9, "b": 0.8},
            },
            {
                "persona": "p2",
                "true_accuracy": {"a": 0.6, "b": 1.0},
                "recovered_accuracy_unsupervised": {"a": 0.3, "b": 1.0},
            },
        ],
    }
    stats = analysis.recovery_stats(comparison)
    assert abs(stats["max_abs_error"] - 0.3) < 1e-9
    assert stats["worst_term"]["persona"] == "p2"
    assert stats["worst_term"]["label"] == "a"
    assert abs(stats["worst_term"]["abs_error"] - 0.3) < 1e-9


def test_recovery_stats_worst_term_none_when_empty():
    comparison = {
        "true_prevalence_a": None,
        "recovered_prevalence_a": None,
        "per_persona": [
            {
                "persona": "p1",
                "true_accuracy": {"a": None, "b": None},
                "recovered_accuracy_unsupervised": {"a": None, "b": None},
            },
        ],
    }
    stats = analysis.recovery_stats(comparison)
    assert stats["max_abs_error"] is None
    assert stats["worst_term"] is None
    assert stats["recovery_mae"] is None


def test_descriptive_stats_basic():
    stats = analysis.descriptive_stats([0.1, 0.3, 0.2])
    assert stats["n"] == 3
    assert stats["min"] == 0.1
    assert stats["max"] == 0.3
    assert abs(stats["mean"] - 0.2) < 1e-9
    assert abs(stats["median"] - 0.2) < 1e-9
    assert abs(stats["spread"] - 0.2) < 1e-9
    assert stats["std"] > 0


def test_descriptive_stats_drops_none_and_handles_empty():
    assert analysis.descriptive_stats([])["n"] == 0
    assert analysis.descriptive_stats([None, None])["mean"] is None
    s = analysis.descriptive_stats([0.5, None, 0.5])
    assert s["n"] == 2
    assert s["std"] == 0.0
    assert s["median"] == 0.5


def test_sampling_noise_floor_matches_binomial_se():
    # Q=24 over 2 labels -> 12 questions/label -> sqrt(0.25/12).
    import math

    floor = analysis.sampling_noise_floor(24)
    assert abs(floor - math.sqrt(0.25 / 12)) < 1e-12
    assert analysis.sampling_noise_floor(0) is None


def test_sampling_noise_floor_minority_label():
    # An imbalanced deck: the minority label backs the weakest estimate, so the
    # conservative floor uses its count, not the balanced n/2.
    import math

    # 32 questions, 12 on the minority label -> sqrt(0.25/12), larger than the
    # balanced sqrt(0.25/16).
    floor = analysis.sampling_noise_floor(32, min_label_count=12)
    assert abs(floor - math.sqrt(0.25 / 12)) < 1e-12
    assert floor > analysis.sampling_noise_floor(32)
    assert analysis.sampling_noise_floor(32, min_label_count=0) is None


def test_min_truth_label_count_matches_deck():
    # Derived straight from the authored answer key; the minority is bearish.
    from collections import Counter

    counts = Counter(s["truth"] for s in _personas.SCENARIOS)
    assert analysis.min_truth_label_count() == min(counts.values())


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"  [PASS] {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"  [FAIL] {fn.__name__}: {exc}")
    print(f"\n{len(tests) - failures} passed, {failures} failed")
    return 1 if failures else 0


def test_build_label_counts_skips_abstained_scenarios():
    votes = {
        s["id"]: {p["key"]: s["truth"] for p in _personas.PERSONAS}
        for s in _personas.SCENARIOS
    }
    # Knock out one persona's vote on the first scenario -> that scenario drops.
    first = _personas.SCENARIOS[0]["id"]
    votes[first][_personas.PERSONAS[0]["key"]] = None
    counts = analysis.build_label_counts(votes)
    total = sum(sum(d.values()) for d in counts.values())
    assert total == len(_personas.SCENARIOS) - 1


def test_vote_variation_counts_abstentions():
    votes = {
        s["id"]: {p["key"]: "bullish" for p in _personas.PERSONAS}
        for s in _personas.SCENARIOS
    }
    first = _personas.SCENARIOS[0]["id"]
    votes[first][_personas.PERSONAS[1]["key"]] = None
    vv = analysis.vote_variation(votes)
    neutral = _personas.PERSONAS[1]["name"]
    assert vv[neutral]["n_unparsed"] == 1
    assert vv[neutral]["n"] == len(_personas.SCENARIOS) - 1


def test_evaluability_diagnostic_unparsed_blocks_evaluable():
    # All judges vary, but one abstention exists -> not evaluable.
    votes = {}
    for i, s in enumerate(_personas.SCENARIOS):
        votes[s["id"]] = {
            p["key"]: ("bullish" if i % 2 else "bearish")
            for p in _personas.PERSONAS
        }
    first = _personas.SCENARIOS[0]["id"]
    votes[first][_personas.PERSONAS[0]["key"]] = None
    diag = analysis.evaluability_diagnostic(votes)
    assert diag["n_unparsed"] == 1
    assert diag["predicted_evaluable"] is False


def test_pairwise_disagreement_ignores_abstentions():
    # Judges always agree where both vote; abstentions are not disagreements.
    votes = {
        s["id"]: {p["key"]: "bullish" for p in _personas.PERSONAS}
        for s in _personas.SCENARIOS
    }
    first = _personas.SCENARIOS[0]["id"]
    votes[first][_personas.PERSONAS[0]["key"]] = None
    pw = analysis.pairwise_disagreement(votes)
    assert pw["mean"] == 0.0  # no disagreements among compared cells


if __name__ == "__main__":
    raise SystemExit(_main())
