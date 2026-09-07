# Rendering pipeline — ntqr_llm_personas

## Why the manuscript is *generated*

The manuscript contains dozens of numeric claims (recovery MAE, prevalence,
disagreement, pair moments, the two prevalence roots…). If those were typed by
hand they would silently drift from the data the moment a model or scenario
changed. Instead the prose lives in templates with `{{TOKEN}}` placeholders, and
every token is filled from the run's JSON artifacts: the LLM sweep, the
deterministic simulations, the scenario bootstrap, and the validation matrix.
**The prose cannot drift from the data, because the numbers are never typed.**

## The path

```
outputs/llm/.../summary.json       outputs/sim/.../summary.json       outputs/validation/.../summary.json
        │                                      │                                      │
        └──────────────────────────────────────┴──────────────────────────────────────┘
                                               ▼
scripts/manuscript/_values.py :: compute_values(out)
        │   reads only generated JSON artifacts; returns dict[str, str]
        │   of ready-to-insert strings (scalars + whole markdown tables)
        ▼
scripts/manuscript/render.py :: render_all()
        │   for each templates/<stem>.md.tmpl:
        │     substitute every {{TOKEN}};  raise KeyError on unknown token;
        │     raise KeyError on any leftover {{…}} after substitution
        ▼
docs/manuscript/{manuscript,supplemental,README}.md          (GENERATED)
        │
        ▼
scripts/z_generate_manuscript_variables.py
        │   writes output/data/manuscript_variables.json,
        │   splits the main manuscript into numbered template sections,
        │   copies PNGs + sidecars into output/figures/
        ▼
output/manuscript/*.md + output/figures/*.png           (template renderer input)
```

`render.py` is ~110 lines and does exactly three things: load values, substitute
tokens with a strict regex (`\{\{([A-Z0-9_]+)\}\}`), and write (or, with
`--check`, diff) the targets. Token resolution is **fail-closed**: an unknown
token (not in `_values`) or a leftover token (typo, undefined) aborts the render
with a named error. This is the determinism guarantee.

## Two modes

```bash
.venv/bin/python scripts/manuscript/render.py          # write the three files
.venv/bin/python scripts/manuscript/render.py --check  # diff vs disk; non-zero on drift
.venv/bin/python scripts/z_generate_manuscript_variables.py
```

`--check` renders into memory and compares against the on-disk files. It is the
CI / pre-commit gate: if `manuscript.md` was hand-edited, or the data changed
without a re-render, `--check` exits non-zero and names the drifted files.

`z_generate_manuscript_variables.py` is the bridge to the sibling
`docxology/template` renderer. It uses the same value map as `render.py`, writes
`output/data/manuscript_variables.json`, emits section files such as
`00_abstract.md` and `03_results.md`, and copies all 16 embedded figure PNGs and
their data sidecars into `output/figures/`. The template renderer then supplies
the cover page and resolves Pandoc-crossref labels for figures, tables,
equations, and sections.

PDF style is project-local: `docs/manuscript/config.yaml` sets the compact
`margin=0.58in` geometry used by Pandoc, and `docs/manuscript/preamble.md` pins
cross-reference, citation, and URL hyperlinks to red. These are copied into
`output/manuscript/` by `z_generate_manuscript_variables.py` before the template
renderer runs.

## Token kinds

| Kind | Example | Source |
|------|---------|--------|
| Scalar | `{{REC_MAE}}`, `{{TRUE_PREV_A}}`, `{{TEMP}}` | a formatted number from generated JSON artifacts |
| Spelled-out | `{{N_MODELS_W}}` ("four") | `_word()` over a count |
| Data-driven prose | `{{ABSTRACT_RECOVERY}}`, `{{SAMPLING_LIMITATION}}`, `{{DISAGREEMENT_NARRATIVE}}` | a sentence whose wording **branches on the data** (exact vs approximate; deterministic vs stochastic; which model is lowest) so adjectives can never contradict the numbers |
| Whole table | `{{TABLE_S3_RECOVERY}}`, `{{TABLE_CONSTANTS}}`, `{{S2_ALL}}` | a complete markdown table built from the artifacts, robust to model count |
| Validation table | `{{VALIDATION_TABLE}}` | the offline claim-check matrix from `outputs/validation/statistical_validation/summary.json` |
| Worked example | `{{WE_ROOT_LO}}`, `{{WE_DISCRIMINANT}}`, `{{WE_PAIR_01}}` | the prevalence algebra recomputed from the best model's real votes via `ntqr.r2` |

The data-driven-prose tokens are the key to honesty: e.g. `RECOVERY_VERB` says
"reproduced the authored truth exactly" only when MAE is literally 0, otherwise
"tracked the authored truth to a mean absolute error of X" — so the manuscript
adapts truthfully to any run.

## Adding a number to the manuscript

1. Compute it in `scripts/manuscript/_values.py` and assign `v["NEW_TOKEN"] = …`
   (format with `f2`/`f3`/`f4`/`frac`; read only from the JSON artifacts).
2. Use `{{NEW_TOKEN}}` in `templates/*.md.tmpl`.
3. `render.py` — if the token is undefined, it fails loudly; if defined but
   unused, that is harmless.
4. **Never type a number directly into a `.tmpl`.** If you cannot name the token
   that backs a number, it does not belong in the manuscript.

## Editing prose

Edit `templates/*.md.tmpl`, never `docs/manuscript/*.md` (the latter is overwritten on
every render and is guarded by the `<!-- GENERATED FILE -->` banner). Structural
math (equations, the evaluability proposition) is static and lives in the
template directly; only *data-derived numbers* must be tokens.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `FileNotFoundError: missing summary.json under outputs/llm` | the sweep has not run | run `scripts/llm/multi_llm_personas_evaluation.py` |
| `FileNotFoundError: missing simulation/bootstrap/validation summary.json` | a deterministic companion artifact has not been regenerated | run the named `scripts/sim/...` or `scripts/validation/...` command from the error |
| `KeyError: template references unknown tokens: X` | `{{X}}` used but not defined in `_values.py` | define `v["X"]` or fix the typo |
| `KeyError: unresolved tokens after render` | a literal `{{…}}` survived | a token name has a lowercase/format the regex rejects |
| `--check` reports DRIFT | `manuscript.md` edited by hand, or data changed | re-run `render.py` (edit the `.tmpl`, not the output) |
