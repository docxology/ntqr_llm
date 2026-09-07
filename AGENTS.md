# AGENTS.md - ntqr_llm root

Child dirs: docs/ (technical docs + essays; has its own AGENTS.md), figures/ (static repo figures), docs/manuscript/ (authored manuscript source + config.yaml), output/ (final rendered deliverables), outputs/ (analysis results), python/ (ntqr library: src + tests), scripts/ (thin-orchestrator analyses + rendering), tests/ (scripts-layer tests). Files: pyproject.toml, uv.lock, README.md, CITATION.cff, codemeta.json, .zenodo.json, LICENSE, NOTICE.

Invariants: thin-orchestrator layout - business logic lives in python/src/ntqr/, scripts only orchestrate. outputs/ is fully regenerable via the documented refresh chain (see outputs/AGENTS.md); output/ is rendered, not hand-edited. The manuscript's numbers are injected via {{TOKEN}} placeholders; never hard-code a generated number into manuscript prose. Verify: uv run pytest tests/ -v && uv run pytest python/tests/ -v

Upstream policy: see `../AGENTS.md` and /Volumes/external_drive/Git/template/projects/ongoing/AGENTS.md for repo-wide conventions; do not duplicate them here.
