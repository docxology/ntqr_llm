# NTQR 0.9 migration

The authority for the installed version is the [published 0.9 release](https://pypi.org/project/ntqr/0.9/).
The [documentation landing page](https://ntqr.readthedocs.io/en/latest/) still
labels itself 0.8 at migration time (2026-10-17); it is useful conceptual background.

`uv sync --locked --extra dev` now installs `python/` editable as a declared
`ntqr==0.9` dependency, including SciPy, Numba and typing_extensions. Root pytest
collects both study and library suites. `.python-version` selects Python 3.12;
CI also checks 3.13. Scripts consistently use the source in this checkout and
reject an already-imported different NTQR package.

`python/UPSTREAM.json` records the published wheel SHA-256 and every imported
source/notebook hash. The original MIT license and attribution remain in
`python/LICENSE.txt`. `ntqr.llm` is the local study extension. The trio evaluator,
data sketches, and binary/ternary axioms are byte-identical to the 0.9 wheel.
The supplied notebooks are upstream reference material; notebook execution is
separate from the offline script/test verification below.

## Changed upstream interfaces

- `ResponseVariables.label_response_to_observable(counts)` now returns counts.
- `ResponseVariables.label_response_to_q(ql)` now returns label totals.
- Exact consistent-set generators yield sparse `HashablePoint` rows, converted
  to JSON through `row.toarray().ravel().tolist()`.
- `random_points` and `random_correct_cuboid_points` on `ConsistentSet` have
  been replaced by stream APIs. Their Numba random stream is not controlled by
  Python's local `random.Random` seed.

The possible/consistent-set example delegates to `ntqr.llm.consistency` and
uses bounded exact enumeration with seeded reservoir sampling without replacement.
Its default is R=3, N=3, Q=6. Synthetic true-label counts now match `ql` exactly.
`max_allocations` bounds the number of candidate per-event label allocations
before enumeration. Requested samples above the set size return the whole set.

## Local corrections

The released 0.9 exact generator initialized the slack label from its per-label
capacity rather than the observed counts. If an event count exceeds that
capacity, generated points can lose observations. The local patch initializes
from actual observations and constrains residual slack counts. Regression tests
compare all results against exhaustive allocations, including boundary label
counts. The exact correct-cuboid projection now flattens sparse rows explicitly.
These fixes are recorded separately from original wheel hashes.

The legacy alarm demo now validates response totals and safety factors and caps
Q by default (binary 30, ternary 6). Its ternary default is Q=3; explicitly
raising max_q acknowledges the expensive enumeration. It continues using the
original algebra, without claiming an LP speedup.

The persona-disagreement function no longer shadows its imported persona module.
Manuscript hydration preserves README/AGENTS guides in generated directories.

## Verify

```bash
uv sync --locked --extra dev
uv run --no-sync pytest
uv run --no-sync python scripts/run_all.py --only ntqr --only baselines
uv run --no-sync python scripts/manuscript/render.py
uv run --no-sync python scripts/z_generate_manuscript_variables.py
uv run --no-sync python scripts/manuscript/render.py --check
```

Recorded model votes and published PDFs retain their historical provenance.
The migration does not label old model calls as newly collected observations.
The trio evaluator is unchanged; offline recovery, bootstrap and claim checks
exercise it with the study fixtures and recorded artifacts.
