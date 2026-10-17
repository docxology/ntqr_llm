# Documentation — ntqr_llm_personas

Technical documentation for the algebraic-evaluation study that recovers
LLM-persona accuracies from unlabeled votes. For the science, read the generated
[`docs/manuscript/manuscript.md`](../manuscript/manuscript.md); for the software, read
here.

## Index

| Document | Purpose |
|----------|---------|
| [architecture.md](architecture.md) | system layers and the full data flow (personas → Ollama → votes → `ntqr` → simulation/bootstrap/validation artifacts → render) |
| [rendering_pipeline.md](rendering_pipeline.md) | the `{{TOKEN}}`-injection pipeline; how validated numbers reach the prose; the `--check` drift gate |
| [manuscript_guide.md](manuscript_guide.md) | how to author/extend the manuscript (edit `.tmpl`, add tokens to `_values.py`) |
| [testing_philosophy.md](testing_philosophy.md) | zero-mock + no-fabricated-data, and why they matter for a generated manuscript |
| [VERIFICATION.md](VERIFICATION.md) | the post-change verification routine and checklist |
| [essays/](essays/) | conceptual background on algebraic evaluation (Dawid–Skene, Regnault, the algebraic-geometry primer) |

## Getting started

```bash
# 1. regenerate the headline LLM data (needs Ollama + requested models)
.venv/bin/python scripts/llm/multi_llm_personas_evaluation.py --set temperature=0.1 --set timeout=45

# 2. regenerate deterministic offline validation artifacts
.venv/bin/python scripts/sim/synthetic_recovery_study.py
.venv/bin/python scripts/sim/bootstrap_recovery_ci.py
.venv/bin/python scripts/validation/statistical_validation_report.py

# 3. render the manuscript from templates + data (numbers injected, never typed)
.venv/bin/python scripts/manuscript/render.py

# 4. verify: no drift, tests green
.venv/bin/python scripts/manuscript/render.py --check
.venv/bin/python -m pytest tests/ -q
```

## Key concepts

- **Generated manuscript** — every data-derived claim is a `{{TOKEN}}` filled
  from generated JSON artifacts; the prose cannot drift from the data.
- **Algebraic (NTQR) evaluation** — recover per-judge accuracies from votes alone,
  no answer key; two logically consistent solutions, selected by a label-free
  tie-break. Cite the public NTQR 0.9 documentation as `ntqr_0_9_docs`.
- **Research context** — the manuscript now situates NTQR beside Dawid-Skene
  truth inference, weak supervision, agreement-rate estimators, LLM-as-judge
  evaluation, structured-output collection, and majority-vote/self-consistency
  baselines; citation keys are checked against `docs/manuscript/references.bib`.
- **Evaluability is per-judge** — a single constant classifier (a judge that never
  varies) collapses the algebra regardless of how much the ensemble disagrees
  (proved in manuscript §2.5, shown on real votes in Supplemental §S1.5).
- **No fabrication** — unparseable replies are abstentions; non-compliant models
  are named, never solved on invented votes.
- **Sixteen embedded figures**, each with a parseable data sidecar so every
  plotted value is machine-readable.

- [NTQR 0.9 migration](ntqr09.md): release provenance, sparse API changes, local fixes, and offline verification.

- [Recovery validity and reliable artifacts](recovery_validity.md): physical solutions, a truth-free audit, atomic checkpoints, and render manifests.
