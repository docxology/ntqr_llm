# Manuscript: algebraic evaluation of LLM-persona ensembles

<!-- GENERATED FILE — do not edit by hand.
     Source: docs/manuscript/templates/README.md.tmpl
     Rebuild: python scripts/manuscript/render.py -->

A short, reproducible IMRAD study built on this repository's NTQR machinery and
the multi-LLM persona pipeline in `scripts/llm/`.

- **[manuscript.md](manuscript.md)** — the paper (Abstract, Introduction,
  contextualized related work, Methods, Results, Discussion, References).
- **[supplemental.md](supplemental.md)** — tie-break derivation, full per-model
  vote tables, per-persona recovery, the scenario list, reproduction commands,
  and the evaluability-diagnostic definition.

> **These files are generated.** Do not hand-edit `manuscript.md`,
> `supplemental.md`, or this README. Every number is injected from generated
> JSON artifacts by `scripts/manuscript/render.py`; edit the templates in
> `docs/manuscript/templates/` and rerun the renderer.
> For the sibling `template/` PDF renderer, run
> `scripts/z_generate_manuscript_variables.py` to write
> `output/manuscript/`, `output/data/manuscript_variables.json`, and
> `output/figures/`.

## The one-paragraph version

Three versioned system-prompted trader personas
(`v3_binary_schema`, `4d157f84885a`) judge
64 binary market scenarios; the same trio runs through six local
LLMs. On the five models whose three judges all varied (`mistral:latest`, `gemma4:latest`, `gemma3:4b`, `gemma2:2b`, `granite4.1:3b`), purely algebraic,
answer-key-free evaluation recovered persona accuracies to a mean absolute error of **0.012**, within the 0.102 sampling-noise floor across all six per-label accuracy terms, with no labels -- including a persona's genuinely poor bullish accuracy of **0.57**, recovered as **0.59**. The other model collapsed at least one persona into a **constant classifier** (a judge that voted one way on all 64 scenarios), which makes the error-independent algebra unsolvable. The non-obvious
result: Aggregate disagreement separated this run only because the unevaluable model(s) collapsed to 0.00; the five evaluable models spanned 0.03–0.23 — so evaluability is gated by a
**per-judge** condition (every judge must vary), not by ensemble disagreement.
A label-free **evaluability diagnostic** predicted exactly which models the
algebra could solve, with no answer key.

The Introduction and Methods situate the result against Dawid-Skene truth
inference, weak supervision, agreement-rate estimators, LLM-as-judge work,
structured-output collection, and majority-vote/self-consistency baselines.

## Data, validation, and figures

The [rendering guide](../rendering_pipeline.md) shows how to verify the
recorded study and produce the PDF and HTML offline. The [study overview](../study.md)
summarizes the empirical result and its limits.

The checked-in records include the sweep summary and per-model votes under
`outputs/llm/multi_llm_personas_evaluation/`, the deterministic study and
bootstrap under `outputs/sim/`, and the reports under `outputs/validation/`.
The manuscript embeds 17 figures with parseable data
sidecars. See Supplement §§S5 and S7 for the reproduction and artifact inventory.
