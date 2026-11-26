# Study overview

## The study in one paragraph

Three trader "personas" (optimist / neutral / pessimist), instantiated as
versioned system prompts, each make a binary `bullish`/`bearish` call on the
same 64 market scenarios. We run the identical trio through six locally-hosted
models via Ollama
and, for each model, recover per-persona, per-label accuracy with
`ntqr.r2.ErrorIndependentEvaluation` (**unsupervised** — no answer key), scoring
it against an authored truth used only as a check. The headline, non-obvious
result: **inter-judge disagreement does not imply evaluability.** What gates a
solution is a *per-judge* condition — every judge must actually vary — not an
*ensemble* one. A label-free preliminary screen matches the evaluated-model set in this study.
Passing that screen alone does not guarantee a complete physical solution; the
[recovery-validity audit](recovery_validity.md) checks the actual roots.

## The algebra (why it works, and how it fails)

For a single judge the universal evaluation polynomial is
`f_a = P_a·P_{i,a} + P_b·(1 − P_{i,b})` — exact for any noisy judge on any finite
test. A trio of error-independent judges closes the system: the prevalence of
label `a` is a root of a quadratic in the trio's vote-pattern moments, giving
**two** logically consistent solutions (the truth and its mirror image). When the
judges are in fact error-*correlated*, the prevalence root can go imaginary — an
label-free indication that the observed vote moments are incompatible with the
error-independent model. See
[`docs/essays/SimplestExampleOfEvaluationWithAlgebraicGeometry.md`](essays/SimplestExampleOfEvaluationWithAlgebraicGeometry.md)
and the [manuscript](manuscript/manuscript.md).

The real recovery ranking is hardened and generalised by deterministic, offline
checks:

- **A confidence interval on the real number** (manuscript §3.1): bootstrapping
  the authored scenarios puts a 95% CI on the recovery MAE that sits well inside
  the sampling-noise floor — the headline is not a lucky single draw.
- **Convergence across many ensembles** (§3.7): over hundreds of synthetic
  error-independent ensembles the recovery error falls like sampling noise
  (`~1/√Q`), with the slope stable across diverse prevalences and accuracies.
- **Two honest failure boundaries**: the imaginary-root alarm is *sufficient but
  not necessary* — it catches an anti-correlated judge pair with zero false
  positives, yet forced **common-mode** errors (the shared-pre-training analogue)
  silently degrade recovery without tripping it; this stress design also lowers
  effective judge accuracy, which is reported per configuration (§3.7); and the
  two-solution tie-break inverts once judges are no longer clearly better than
  random (§3.8).
- **A method trade-off, not a winner** (§3.9): the exact error-independent algebra
  is more *accurate* than simple majority-voting where its assumption holds, but
  majority-voting is more *robust* in that tie-break danger zone — so the harness
  reports both.
- **A validation matrix** (§3.10): an offline report reads the LLM, simulation,
  bootstrap, and figure-sidecar artifacts and checks every load-bearing
  statistical claim before the manuscript is rendered.
