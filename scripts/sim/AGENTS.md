# AGENTS.md - scripts/sim

Leaves: bootstrap_recovery_ci.py, synthetic_recovery_study.py - deterministic
offline validation (no Ollama) writing to ../outputs/sim/<stem>/. Both import
their computation from `python/src/ntqr/llm/` (`ntqr.llm.simulation`,
`ntqr.llm.bootstrap`, `ntqr.llm.personas`).

Verify: uv run pytest tests/test_simulation.py tests/test_bootstrap.py -q

Upstream policy: see `../AGENTS.md` for this repository's conventions. The shared policy that governs template projects is maintained in the template repository, not restated here.
