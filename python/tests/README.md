# python/tests/

pytest suite for `ntqr`. Run with `make test` (from [`../`](../)) or `pytest`
here.

Layout mirrors the source: shared-module tests at the top level, with `r2/`
and `r3/` subfolders for the per-response-count implementations.

| Test | Covers |
|------|--------|
| `test_alarms.py` | `ntqr.alarms` |
| `test_algebra.py` | `ntqr.algebra` |
| `test_evaluations.py` | `ntqr.evaluations` |
| `test_raxioms.py` | `ntqr.raxioms` |
| `test_statistics.py` | `ntqr.statistics` |
| `test_testsketches.py` | `ntqr.testsketches` |
| [`r2/`](r2/) | R=2 tree |
| [`r3/`](r3/) | R=3 tree |
