# AGENTS.md - tests

Leaf: scripts-layer tests - test_llm_analysis.py, test_recovery.py, test_bootstrap.py, test_simulation.py, test_manuscript_render.py, test_validation_report.py. Package tests live in python/tests/. Zero-mock philosophy per docs/testing_philosophy.md.

Verify: uv run pytest tests/ -v

Upstream policy: see `../AGENTS.md` for this repository's conventions. The shared policy that governs template projects is maintained in the template repository, not restated here.

Research-tool tests cover real profiling, atomic raster/vector export, citation integrity and prepared computations; no timing thresholds or mocked empirical observations.
