# AGENTS.md - scripts/llm

Leaf: multi_llm_personas_evaluation.py (headline entry, needs Ollama) +
trader_personas_evaluation.py. Scripts-side helpers: `_llm.py` (LLM client)
and `_analysis.py` (collect-votes glue). All computation is imported from
`python/src/ntqr/llm/` (`ntqr.llm.analysis`, `ntqr.llm.recovery_scoring`,
`ntqr.llm.personas`). Writes to ../outputs/llm/...

Refresh: .venv/bin/python scripts/llm/multi_llm_personas_evaluation.py --set temperature=0.1 --set timeout=45

Verify: uv run pytest tests/test_llm_analysis.py tests/test_recovery.py tests/test_simulation.py tests/test_bootstrap.py -q

Upstream policy: see `../AGENTS.md` and /Volumes/external_drive/Git/template/projects/ongoing/AGENTS.md for repo-wide conventions; do not duplicate them here.
