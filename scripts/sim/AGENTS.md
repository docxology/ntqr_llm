# AGENTS.md - scripts/sim

Leaves: bootstrap_recovery_ci.py, synthetic_recovery_study.py - deterministic
offline validation (no Ollama) writing to ../outputs/sim/<stem>/. Both import
their computation from `python/src/ntqr/llm/` (`ntqr.llm.simulation`,
`ntqr.llm.bootstrap`, `ntqr.llm.personas`).

Verify: uv run pytest tests/test_simulation.py tests/test_bootstrap.py -q

Upstream policy: see `../AGENTS.md` and /Volumes/external_drive/Git/template/projects/ongoing/AGENTS.md for repo-wide conventions; do not duplicate them here.
