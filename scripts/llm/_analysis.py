"""Thin LLM-voting glue: the only persona-pipeline piece that talks to a model.

The computation (response parsing, cache-record diagnostics, vote analysis,
recovery scoring) lives in ``python/src/ntqr/llm/`` (``ntqr.llm.analysis``,
``ntqr.llm.recovery_scoring``). This module keeps just the collect-votes loop:
prompt formatting, client calls with schema + repair retry, and cache writes.
The orchestrators (``trader_personas_evaluation.py``,
``multi_llm_personas_evaluation.py``) call :func:`collect_votes` and delegate
everything else to ntqr.
"""

from __future__ import annotations

import _llm
from ntqr.llm import personas as _personas
from ntqr.llm.analysis import (
    cache_key,
    normalize_cache_record,
    parse_call_record,
)


def _cache_response_record(attempts):
    accepted = next((a for a in attempts if a.get("call") is not None), None)
    last = attempts[-1] if attempts else {}
    call = accepted.get("call") if accepted else None
    return {
        "call": call,
        "unparsed": call is None,
        "raw": (accepted or last).get("raw"),
        "parse_method": (accepted or last).get("parse_method", "none"),
        "schema_valid": bool((accepted or last).get("schema_valid", False)),
        "invalid_reason": (
            None if accepted else last.get("invalid_reason", "no attempts")
        ),
        "attempts": attempts,
        "repair_used": len(attempts) > 1 and attempts[0].get("call") is None,
        "n_attempts": len(attempts),
        "primary_parse_valid": bool(
            attempts and attempts[0].get("schema_valid", False)
        ),
        "final_parse_valid": call is not None,
        "prompt_version": _personas.PROMPT_VERSION,
        "prompt_hash": _personas.prompt_hash(),
    }


def _attempt_record(attempt_no, raw, parsed):
    return {
        "attempt": attempt_no,
        "raw": raw,
        "call": parsed["call"],
        "parse_method": parsed["parse_method"],
        "schema_valid": parsed["schema_valid"],
        "invalid_reason": parsed["invalid_reason"],
    }


def collect_votes(
    client, cache, refresh, on_error="raise", on_cache_update=None
):
    """Return (votes_by_scenario, unparsed_count, cache) querying/caching LLM.

    ``votes_by_scenario[scenario_id][persona_key]`` is ``"bullish"`` /
    ``"bearish"``, or ``None`` when the model's reply could not be parsed into a
    vote. Votes are NEVER fabricated: an unparseable reply (and, when tolerated,
    an unreachable model) is recorded as ``None`` -- an honest abstention -- and
    counted in ``unparsed_count``. The cache key includes backend, model, and
    temperature, so one cache file holds many runs without collision. Cache
    entries are backward-compatible records. New records additionally retain
    ``raw``, ``parse_method``, ``prompt_version``, and ``prompt_hash`` so prompt
    failures can be audited after the sweep.

    ``on_error`` governs an unreachable model (timeout / connection failure):

    * ``"raise"`` (default) -- propagate ``LLMUnavailable`` so the single-model
      demo fails loudly.
    * ``"tolerate"`` -- record the cell as ``None`` and continue, so one
      unreachable model never aborts the multi-model sweep.

    A reply that arrives but cannot be parsed is always recorded as ``None``,
    independent of ``on_error`` -- the model genuinely did not produce a vote.
    """
    votes: dict[str, dict[str, str | None]] = {}
    unparsed = 0
    for scenario in _personas.SCENARIOS:
        votes[scenario["id"]] = {}
        user = _personas.USER_TEMPLATE.format(scenario=scenario["text"])
        for persona in _personas.PERSONAS:
            key = cache_key(client, persona["key"], scenario["id"])
            if not refresh and key in cache:
                record = normalize_cache_record(cache[key])
                call = record["call"]
                cell_unparsed = bool(record["unparsed"])
            else:
                attempts = []
                try:
                    raw = client.complete(
                        persona["system"],
                        user,
                        json_mode=True,
                        schema=_personas.CALL_SCHEMA,
                    )
                    parsed = parse_call_record(raw, schema_strict=True)
                    attempts.append(_attempt_record(1, raw, parsed))
                    if parsed["call"] is None:
                        repair_user = _personas.REPAIR_TEMPLATE.format(
                            user_prompt=user,
                            reason=parsed["invalid_reason"],
                            raw=raw,
                        )
                        raw = client.complete(
                            persona["system"],
                            repair_user,
                            json_mode=True,
                            schema=_personas.CALL_SCHEMA,
                        )
                        parsed = parse_call_record(raw, schema_strict=True)
                        attempts.append(_attempt_record(2, raw, parsed))
                except _llm.LLMUnavailable:
                    if on_error != "tolerate":
                        raise
                    parsed = parse_call_record(None, schema_strict=True)
                    attempts.append(
                        _attempt_record(len(attempts) + 1, None, parsed)
                    )
                cache[key] = _cache_response_record(attempts)
                record = normalize_cache_record(cache[key])
                call = record["call"]
                cell_unparsed = bool(record["unparsed"])
                if on_cache_update is not None:
                    on_cache_update(cache)
            if cell_unparsed:
                unparsed += 1
            votes[scenario["id"]][persona["key"]] = call
    return votes, unparsed, cache


