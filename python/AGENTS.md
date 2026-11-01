# AGENTS.md - python

Child dirs: src/ (ntqr library), tests/ (package tests: r2/, r3/). The ntqr library is the business-logic layer - all real computation lives here, per the thin-orchestrator layout.

Verify: uv run pytest python/tests/ -v

Upstream policy: see `../AGENTS.md` for this repository's conventions. The shared policy that governs template projects is maintained in the template repository, not restated here.
