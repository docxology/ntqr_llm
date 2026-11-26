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
        │   for each docs/manuscript/templates/<stem>.md.tmpl:
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

Recorded supporting artwork is exported separately by
`scripts/validation/render_supporting_figures.py` into
`outputs/validation/render_supporting_figures/`. It reads the preserved
simulation/bootstrap/validation summaries, copies scientific sidecars
byte-for-byte, and writes a new input-hash manifest. This step renders existing
data; it does not resample scenarios, rerun simulations or call models.
The manuscript and bridge consume these derived figures, not overwritten
producer artwork.

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
`docxology/template` renderer. Within each bridge invocation it computes the value
map once and passes that identical mapping to `render_all`, so the variables JSON,
numeric evidence, and prose come from the same artifact snapshot. It writes
`output/data/manuscript_variables.json`, writes the numeric components of those tokens to
`output/data/manuscript_numeric_evidence.json`, emits section files such as
`00_abstract.md` and `03_results.md`, and copies all 17 embedded figure PNGs and
their data sidecars into `output/figures/`. The template renderer then supplies
the cover page and resolves Pandoc-crossref labels for figures, tables,
equations, and sections.

PDF style is project-local: `docs/manuscript/config.yaml` sets the compact
`margin=0.58in` geometry used by Pandoc, and `docs/manuscript/preamble.md` pins
cross-reference, citation, and URL hyperlinks to red. These are copied into
`output/manuscript/` by `z_generate_manuscript_variables.py` before the template
renderer runs. The generated configuration declares that the bridge owns the
render roster, so Stage 04 binds the exact numbered inputs instead of treating
the reading copies as additional sections.

The preamble uses the existing `float` package's `H` placement for figures.
Figures and captions stay with their source text instead of entering a pending
float queue around long tables. PDF qualification must find all 17 numbered
captions and the complete results prose, with no `Overfull \vbox` diagnostics;
an API success or an asset-loading count alone does not prove complete pages.

## Pinned PDF and HTML build

Install Git, uv, XeLaTeX with BibTeX, and a compatible Pandoc and
`pandoc-crossref` pair available on `PATH`. Pandoc 3.1.13 and
`pandoc-crossref` 0.3.17.1 were used for the pinned build. Use a fresh working
checkout inside Template. Starting from this study's repository root,
run:

```bash
study_root=$(pwd)
git clone https://github.com/docxology/template.git ../template-render
cd ../template-render
git checkout --detach 87704feae0a4490ef7dad9c931d268e8c30273c2
uv sync --locked --no-dev --python 3.14

git clone "$study_root" projects/working/ntqr_llm
(
  cd projects/working/ntqr_llm
  uv sync --locked --extra dev --python 3.14
  uv run --no-sync python scripts/manuscript/render.py --check
  uv run --no-sync python scripts/z_generate_manuscript_variables.py --check
)

uv run --no-sync python scripts/pipeline/stage_03_render.py --project working/ntqr_llm
uv run --no-sync python scripts/pipeline/stage_04_validate.py --project working/ntqr_llm
```

Stage 03 uses the study's `.venv` to hydrate the numbered manuscript stages
from checked-in evidence. The outputs are
`projects/working/ntqr_llm/output/pdf/ntqr_llm_combined.pdf` and
`projects/working/ntqr_llm/output/web/index.html`. Stage 04 writes a
validation report and provenance receipt under the same project's
`output/reports/`. The tracked version PDF retains its original publication
filename. Check that the combined outputs include the abstract, methods,
results, discussion, figure specifications, supplement, and bibliography,
with all 17 figure captions and resolved citations.

### Released PDF producer

The published `docxology/template` v5.0.0 release provides
`infrastructure.rendering.RenderManager.render_combined_pdf`.
Its annotated tag resolves to commit
`e932a5d2a4f5c75f354352b43e54039eac9f63cf`. Bind that released source,
its installed environment and native Pandoc/TeX toolchain separately from this
project's source gates; a working checkout advertising a newer version is not
release evidence. No unrelated template release blocks the offline gates.

From that admitted release environment, with explicit `SOURCE_ROOT` and a new
`DESTINATION_ROOT`, the direct API needs no project registration or analysis run:

```python
import os
from pathlib import Path
import shutil
from infrastructure.rendering import (
    RenderingConfig, RenderManager, discover_manuscript_files,
)

root = Path(os.environ["SOURCE_ROOT"]).resolve()
target = Path(os.environ["DESTINATION_ROOT"]).resolve()
manuscript = root / "output/manuscript"
figures = target / "figures"
shutil.copytree(root / "output/figures", figures)
config = RenderingConfig(
    manuscript_dir=str(manuscript), figures_dir=str(figures),
    output_dir=str(target), pdf_dir=str(target / "pdf"),
)
pdf = RenderManager(config).render_combined_pdf(
    discover_manuscript_files(manuscript), manuscript, "ntqr_llm_personas",
)
print(pdf)
```

The adjacent `figures/` staging copy is required by combined PDF relative image
paths. Keep original PDFs and evidence untouched. A successful API return alone
does not qualify scientific claims: inspect the actual PDF for all figures,
resolved tokens/references, red links and the project-local margin geometry.

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
2. Use `{{NEW_TOKEN}}` in `docs/manuscript/templates/*.md.tmpl`.
3. `render.py` — if the token is undefined, it fails loudly; if defined but
   unused, that is harmless.
4. **Never type a number directly into a `.tmpl`.** If you cannot name the token
   that backs a number, it does not belong in the manuscript.

## Editing prose

Edit `docs/manuscript/templates/*.md.tmpl`. The files marked with the
`<!-- GENERATED FILE -->` banner are rewritten by the local renderer. Structural
math (equations, the evaluability proposition) is static and lives in the
template directly; only *data-derived numbers* must be tokens.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `FileNotFoundError: missing summary.json under outputs/llm` | recorded study input is missing | restore the tracked study artifacts from the intended revision |
| `FileNotFoundError: missing simulation/bootstrap/validation summary.json` | a deterministic companion artifact has not been regenerated | run the named `scripts/sim/...` or `scripts/validation/...` command from the error |
| `KeyError: template references unknown tokens: X` | `{{X}}` used but not defined in `_values.py` | define `v["X"]` or fix the typo |
| `KeyError: unresolved tokens after render` | a literal `{{…}}` survived | a token name has a lowercase/format the regex rejects |
| `--check` reports DRIFT | `manuscript.md` edited by hand, or data changed | re-run `render.py` (edit the `.tmpl`, not the output) |
