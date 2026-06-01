# Manuscript guide — ntqr_llm_personas

## What the manuscript is

A single generated study, `manuscript/manuscript.md`, plus a
`manuscript/supplemental.md` and a short `manuscript/README.md`. All three are
**rendered** from templates by [the rendering pipeline](rendering_pipeline.md);
do not edit the `.md` outputs directly.

## Source of truth

| You want to change… | Edit… |
|---------------------|-------|
| prose, structure, equations, figure captions | `scripts/manuscript/templates/*.md.tmpl` |
| a number, a table, a data-driven sentence | `scripts/manuscript/_values.py` (then reference the new `{{TOKEN}}`) |
| metadata (title, authors, keywords, DOI) | `manuscript/config.yaml` |
| PDF style (margins, link colors, LaTeX packages) | `manuscript/config.yaml` and `manuscript/preamble.md` |
| the experiment (models, scenarios, personas) | `scripts/llm/_personas.py`, `multi_llm_personas_evaluation.py` — then re-run the sweep |

## Structure of the main manuscript

| Section | Content |
|---------|---------|
| Abstract | data-driven recovery + failure summary (`{{ABSTRACT_RECOVERY}}`, `{{ABSTRACT_FAILURE}}`, `{{ABSTRACT_DISAGREEMENT}}`) |
| §1 Introduction | the NTQR stance; lineage across crowdsourcing, weak supervision, LLM-as-judge, and unlabeled accuracy estimation; the three questions; what "prediction" means without a task model |
| §1 Related evaluation paradigms | compact comparison of supervised evaluation, Dawid-Skene, agreement-rate estimation, weak supervision, LLM-as-judge, majority voting/self-consistency, and NTQR |
| §2.1 | evaluation object and data flow, including the validation-only truth boundary |
| §2.2–2.5 | personas, the 64-scenario test, model-family provenance citations, structured-output collection, and the three evaluators |
| **§2.6 The algebra** | universal evaluation polynomial; data sketch + moments; the prevalence quadratic and closed form; the **Proposition (per-judge evaluability)**; the singularity at ½; the imaginary-root alarm |
| §2.7–2.9 | statistics, the evaluability diagnostic, reproducibility |
| §3.1–3.6 | recovery ranking across evaluable models; failure modes; *disagreement ≠ evaluability*; the label-free diagnostic; MV vs error-independent; reading the figures |
| §4 Discussion | what it shows, the warning, why strong models fail, limitations, future work |
| Figures | formal per-figure specification (claim → encoding → thresholds → sidecar) |
| References | generated bibliography; cite NTQR via public NTQR 0.8 documentation (`ntqr_0_8_docs`) and add new `[@key]` citations only after adding matching BibTeX |

## Structure of the supplemental

`S1` tie-break · **`S1.5` worked example** (the algebra on real votes) · `S2`
per-model vote counts (`{{S2_ALL}}`, robust to any model count) · `S3` full
per-persona recovery · `S4` personas, scenarios, authored truth · `S5`
reproduction · `S6` the diagnostic defined · `S7` output inventory.

## Conventions

- **Numbers are tokens.** Never type a digit-as-claim into a `.tmpl`. If a number
  appears, a `{{TOKEN}}` backs it and `_values.py` computes it from the data.
- **Equations are static.** Symbolic math (LaTeX `$…$` / `$$…$$`) is structural,
  not data, so it lives in the template directly.
- **Figures are embedded** with `![caption](../outputs/llm/multi_llm_personas_evaluation/<stem>.png)`
  at their first reference; captions may interpolate tokens (e.g. `{{REC_MAE}}`).
- **Data-driven prose** (exact vs approximate, singular vs plural, which model is
  lowest) is computed in `_values.py`, never asserted in the template — this is
  what keeps adjectives honest at any temperature or model count.
- **Citations are checked.** Every `[@key]` in the templates must have a matching
  entry in `manuscript/references.bib`; `tests/test_manuscript_render.py`
  enforces this before the PDF render.

## Worked example wiring (the deepest token path)

`_values.py::_worked_example()` reconstructs the best model's `TrioVoteCounts`
from `label_vote_counts.json`, runs `ErrorIndependentEvaluation`, and exposes the
real data-sketch moments, the discriminant `1−4c/a`, and the two prevalence roots
as `{{WE_*}}` tokens. It is guarded: if `ntqr` is unavailable the tokens become
`n/a` and the render still succeeds. This is the template's proof that the
formalism operates on the actual votes, not a toy.

## After any edit

```bash
.venv/bin/python scripts/manuscript/render.py        # regenerate
.venv/bin/python scripts/manuscript/render.py --check # confirm no drift
.venv/bin/python -m pytest tests/ -q                  # render + analysis tests
```
