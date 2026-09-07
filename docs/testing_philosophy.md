# Testing philosophy — ntqr_llm_personas

Two non-negotiable principles govern this project: **zero mocks** and **no
fabricated data**. Both exist because the manuscript publishes numbers, and a
test that passes against a fiction is worse than no test.

## Zero-mock

No `unittest.mock`, `MagicMock`, `@patch`, or stubs anywhere. Tests exercise real
code paths against real data.

```python
# ✅ real algebra on real vote counts
def test_error_independent_recovers_prevalence():
    lc = build_label_counts(votes)               # real tuples
    tvc = TrioLabelVoteCounts(lc).to_TrioVoteCounts()
    ei = ErrorIndependentEvaluation(tvc)
    assert 0.0 <= float(ei.alpha_prevalence_estimates()[1]) <= 1.0

# ❌ a mock tests the mock, not the algebra
def test_mocked():
    with patch("ntqr...alpha_prevalence_estimates", return_value=[0.4, 0.6]):
        ...   # passes regardless of whether the algebra is correct
```

The numeric core `ntqr.llm.analysis` (with `ntqr.llm.recovery_scoring`) is
**pure** (no I/O, no plotting, no
network), so `tests/test_llm_analysis.py` runs it directly with real vote
fixtures — no Ollama, no mock needed. `tests/test_manuscript_render.py` renders
the real templates against real values and asserts the determinism contract
(unknown / leftover tokens raise).

## No fabricated data

This is the project's defining rule and it is enforced in the collection code,
not just in tests:

- An unparseable model reply is an **abstention** (`None`), never an invented
  vote. (`_analysis.parse_call`).
- A model with *any* unparseable response is reported **non-compliant** and
  excluded from the unsupervised solve — it still contributes its parsed votes to
  the disagreement comparison. (`evaluate_model`).
- A degenerate solve (a constant classifier) yields a SymPy `nan`; because
  `float(nan)` succeeds, `safe_float()` rejects non-finite values explicitly so a
  `NaN` is never averaged into a statistic — the model is named as a failure
  instead.

The consequence: every solved row in `summary.json` rests on real, parsed model
responses, and every number in the manuscript traces to one of them.

## Why this matters here

The manuscript is generated from the data (see [rendering_pipeline.md](rendering_pipeline.md)).
A mocked or fabricated test would let a false number reach the published prose
silently. A real test fails the moment the data changes — which is exactly the
signal to re-render. The `render.py --check` gate closes the loop: prose that no
longer matches the artifacts fails the build.

## Test inventory

| Suite | Covers |
|-------|--------|
| `tests/test_llm_analysis.py` | the pure numeric core: label counts, vote variation, pairwise disagreement, the evaluability diagnostic, recovery statistics, the sampling-noise floor |
| `tests/test_recovery.py` | the shared scoring core (`ntqr.llm.recovery_scoring`): the finite-float guard, mean accuracy, the better-than-random tie-break, and per-term absolute errors |
| `tests/test_simulation.py` | the synthetic harness against the *real* ntqr solver: vote-count conservation, seeded reproducibility, exact sample truth, recovery on error-independent data, the `~1/√Q` convergence fit and its robustness across ensembles, the imaginary-root alarm (anti-correlation fires, common-mode blind spot, no false alarm), the tie-break inversion toward random, and the EI-vs-majority-voting trade-off (EI wins for strong judges, MV more robust in the danger zone) |
| `tests/test_bootstrap.py` | the scenario bootstrap: count conservation, point estimate equals the full-sample solve, well-formed CIs and degenerate counts, and seed reproducibility |
| `tests/test_validation_report.py` | the offline validation matrix: every load-bearing statistical check passes, and the report writes a parseable summary, sidecar, and figure |
| `tests/test_manuscript_render.py` | `_values.compute_values` shape, the renderer's fail-closed token contract, no render drift, template cross-reference syntax, hydrated `output/manuscript/`, 16 embedded figures, public NTQR citation, and parseable sidecars |
| `python/` (vendored `ntqr`) | the algebra itself: evaluators, data sketches, axioms |

## Adding tests

1. Put project tests in `tests/`; they self-manage `sys.path` (insert `scripts/`,
   `scripts/llm`, `scripts/manuscript`) — no package install needed.
2. Use real data: real vote fixtures, real JSON artifacts, the real renderer.
3. If you add a manuscript token, assert it exists in `compute_values()` output.
4. Re-run the [verification routine](VERIFICATION.md) and update counts if needed.
