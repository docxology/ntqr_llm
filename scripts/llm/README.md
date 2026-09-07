# llm/

LLM persona evaluation pipeline. Entry points: `trader_personas_evaluation.py`
and `multi_llm_personas_evaluation.py` (query Ollama models, then evaluate
votes with ntqr). Write to `outputs/llm/`.

Scripts-side helpers: `_llm.py` (Ollama/Anthropic client) and `_analysis.py`
(the thin collect-votes loop: prompt formatting, schema + repair retry,
response cache). All computation — response parsing, cache-record diagnostics,
vote analysis, recovery scoring, persona/scenario tables, synthetic and
bootstrap harnesses — lives in `python/src/ntqr/llm/` (`ntqr.llm.analysis`,
`ntqr.llm.recovery_scoring`, `ntqr.llm.simulation`, `ntqr.llm.bootstrap`,
`ntqr.llm.personas`).
