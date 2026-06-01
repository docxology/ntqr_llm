# Abstract {#sec:abstract}

Algebraic (NTQR) evaluation infers how accurate a group of noisy classifiers
was on a finite test **using only their responses — no answer key**. We test
this end to end on real large language models. Three trader "personas"
(optimistic, neutral, pessimistic), instantiated as system prompts, each make a
binary `bullish`/`bearish` call on the same 64 market scenarios; we run the
identical trio through six locally-hosted models via Ollama. For each
model we recover per-persona, per-label accuracy with
`ErrorIndependentEvaluation` (unsupervised) and score it against the authored
ground truth (supervised), which is used only as a check.

On the five models whose three judges all varied (`mistral:latest`, `gemma4:latest`, `gemma3:4b`, `gemma2:2b`, `granite4.1:3b`), the unsupervised algebra
recovered persona accuracies to a mean absolute error of **0.012**, within the 0.102 sampling-noise floor across all six per-label accuracy terms, with no labels -- including a persona's genuinely poor bullish accuracy of **0.57**, recovered as **0.59**. The other model collapsed at least one persona into a **constant classifier** (a judge that voted one way on all 64 scenarios), which makes the error-independent algebra unsolvable.

The central, non-obvious result: **inter-judge disagreement does not imply
evaluability.** Aggregate disagreement separated this run only because the unevaluable model(s) collapsed to 0.00; the five evaluable models spanned 0.03–0.23. What gates
evaluation is a *per-judge* condition — every judge must vary (and answer) — not
an *ensemble* one. We formalize this as a label-free **evaluability diagnostic**
(a judge whose modal-vote fraction reaches 1.0 is a constant classifier; an
unparseable vote is an abstention) that predicted exactly which models would be
evaluable, before any solve and without ground truth. This is a concrete
instance of the safety property the NTQR logic promises: it warns you when an
ensemble is not good enough to be evaluated. A scenario bootstrap puts a 95% CI
of [0.000, 0.038] on the recovery MAE (well inside the
0.102 noise floor), and a deterministic synthetic study generalizes the
recovery beyond the finite set of real evaluable models — error falls like `1/√Q` (slope
-0.58, stable across ensembles) — while mapping two honest limits: the
built-in failure alarm catches anti-correlated judges with no false positives yet
can miss positively-correlated (shared-training) errors, and the two-solution
tie-break inverts once judges are no longer clearly better than random — exactly
where simple majority-voting evaluation, though biased, is the more robust
fallback.
