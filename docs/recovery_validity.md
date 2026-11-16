# Recovery validity and reliable artifacts

The variation diagnostic is a preliminary screen. All three judges can vary
while one or more pair covariances are zero. Conversely, a constant judge makes
the implemented prevalence quadratic unidentifiable; that does not prove that
all interior latent models are impossible. The maintained manuscript now makes
these distinctions explicit, including the difference between an exactly
symmetric vote sketch and a merely balanced authored answer key.

## Physical solutions

`ntqr.llm.recovery_scoring.solution_validity(solution)` checks both prevalence
probabilities, their sum, the three binary classifier rows, finiteness, and the
[0, 1] bounds. It reports issues without clipping or fabricating values.
`select_physical(solutions)` returns the higher-mean-accuracy admissible branch,
or `None` when no complete physical branch is available.

`select_best` remains available only for callers explicitly reproducing historical
raw algebraic experiments. The maintained `solve_recovery` path uses
`select_physical`: nonphysical roots make recovery unavailable, and convergence
and bootstrap summaries include only complete physical recoveries on the same
resample support. A finite root from the raw API is not automatically a
probability. Partial solutions no longer outrank complete recoveries, and
`recovery_stats` reserves
`recovery_mae` for complete finite comparisons. `available_terms_mae` explicitly
labels a partial diagnostic rather than presenting it as a complete recovery.

Live evaluations require a physical solution via
`build_comparison(..., physical_only=True)`. Failure of the majority-voting
baseline does not erase a successful error-independent recovery. Historical
imaginary-alarm and tie-break sweeps pass `physical_only=False` to preserve their
raw-finite-root measurements.

## Synthetic dependence

The study's configured accuracies are pre-corruption parameters. A common-mode
event forces all three judges to err, multiplying each post-common accuracy by
`1 - common_mode`. For an anti pair `(i, j)`, judge `j` mirrors the opposite of
judge `i` with probability `anti_strength`, so its effective accuracy is
`(1-s) * acc_j + s * (1-acc_i)` after applying the common-mode adjustment to
both terms. `correlation_sweep` serializes these `effective_accuracies` for every
row rather than presenting dependence and marginal-skill changes as one effect.
Independent generation retains its historical seeded RNG stream. The alarm
sweep explicitly uses raw mode so its rate counts imaginary roots, not physical
solution unavailability.

## Audit recorded votes without an answer key

```bash
uv run python scripts/validation/recovery_validity_report.py
```

The report under `outputs/validation/recovery_validity_report/` reads only vote
records, not scenario truths, and binds each input to its SHA-256. It separates
missing votes, constant classifiers, singular pair moments, nonfinite/complex
roots, nonphysical real roots, and physical solutions. It does not prove error
independence or remove the label mirror ambiguity. The better-than-random
assumption is still required for choosing a branch.

The pure API is `ntqr.llm.validity_audit.audit_votes(votes_by_judge)`, with three
aligned rows of `a`, `b`, or `None` (an explicit abstention).

## Artifact and cache guarantees

- Shared script outputs are strict UTF-8 JSON. Nonfinite numeric values become
  `null`; symbolic values retain an explicit exact representation. Sets are
  serialized deterministically, and colliding encoded keys are rejected.
- Output records include their NTQR version, schema version and a timezone-aware
  UTC timestamp. `SOURCE_DATE_EPOCH` makes that timestamp reproducible.
- JSON, text and LLM caches are replaced atomically using complete temporary
  files in the destination directory. Existing file modes are preserved.
- The manuscript bridge validates all source figures and sidecars and composes
  every output before publishing any replacement. It preserves directory guides
  and unrelated files. `output/data/render_manifest.json` binds generated files
  to their SHA-256 hashes. Individual file replacement is atomic; the manifest
  can detect an interrupted multi-file update.

```bash
uv run python scripts/manuscript/render.py
uv run python scripts/z_generate_manuscript_variables.py
uv run python scripts/z_generate_manuscript_variables.py --check
```

The final command is read-only and checks every declared figure, sidecar,
section, auxiliary source and manifest against its current source. It now runs
in CI along with the manuscript text drift check. These atomic writes protect
single-writer checkpoints; they are not a multi-writer cache merge protocol.
