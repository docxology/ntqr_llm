# Discussion {#sec:discussion}

**What this shows.** On real, contemporary LLM-persona ensembles that satisfy
the evaluability condition, purely algebraic, answer-key-free evaluation recovered per-judge accuracies closely (recovery MAE 0.012, within the 0.102 sampling-noise floor) -- including correctly identifying and quantifying a *bad* judge. That is the positive
existence result: the NTQR logic is not only theoretically universal but
operationally effective on real model vote matrices, while still refusing to
solve a matrix whose judges carry no variation signal.

**How it fits the literature.** The result is complementary to crowdsourcing,
weak-supervision, and LLM-as-judge measurement-error methods rather than a
replacement for them [@zheng_2017_vldb; @ratner_2017_snorkel;
@zheng_2023_llm_judge; @chen_2026_efficient_judge]. Those approaches ask how to
infer or aggregate task truth under a model of workers, labeling functions, or
judges; majority-vote and self-consistency baselines ask which answer should win
under better-than-random aggregation assumptions [@condorcet_jury_sep;
@wang_2022_self_consistency]. This experiment asks a prior algebraic question:
before using a judge ensemble as an evaluator, does the unlabeled response
matrix contain the variation and agreement structure required for evaluation at
all? That is the practical role of the NTQR alarm lineage
[@corrada_emmanuel_2024_alarm; @corrada_emmanuel_2025_nka].

**The sharper lesson is the warning.** The repository frames the safety value of
algebraic evaluation as its ability to "warn you if the ensemble is not good
enough." We make that warning concrete and show it fires correctly:
one of six models was not evaluable, and a label-free diagnostic predicted it in advance. Crucially, the warning is
*not* "did the ensemble disagree enough?"; the failing model sits at low disagreement because its personas collapsed into constant classifiers. The
correct warning is per-judge: *at least one of your judges is a constant classifier.* For
practitioners assembling LLM juries, this reframes ensemble design: maximizing
diversity/disagreement is neither sufficient nor the right target; ensuring every
judge actually exercises both responses across the test is the necessary
condition.

**Implications for evaluation engineering.** The operational lesson is to treat
LLM evaluation as measurement engineering, not just answer aggregation. First,
make the interface boring: schema-constrained, low-temperature collection and
explicit repair diagnostics should make parseability a controlled boundary, not
the headline result. Second, check the unlabeled matrix before trusting any
recovered score: a constant judge is not a weak signal, it is no signal for the
trio algebra. Third, distinguish *good judges* from *recoverable judges*:
[@fig:validation-accuracy-heatmap] shows that the scientific value is not only
finding high accuracies, but recovering poor or label-specific accuracies without
looking at the answer key. That is what makes the method relevant to LLM-as-judge
systems where the evaluator itself is part of the risk surface.

**Why strong models fail this way.** A strongly dispositioned persona prompt is
a recipe for a constant classifier: a model follows its instruction ("lean
bullish") so rigidly that it casts the same call on every scenario. The very
capability that makes a model a good instruction-follower can make it a useless
*evaluable* judge — in this sweep a capable mid-sized model collapsed personas
into constant classifiers ([@sec:failure-modes]). Crucially this is *not*
cured by sampling: it appears here even at temperature 0.1
(stochastic decoding), so the failure is prompt-induced degeneracy, not a
decoding artifact. This is a caution about over-prompting LLM-as-judge
ensembles.

**Limitations (and what would falsify the claims).**

- **Single run, single test.** `T = 1`, `Q = 64` — a single stochastic draw at temperature 0.1. The recovery result is one sample, not a population estimate; because decoding is stochastic, repeating the sweep would now yield a distribution of recovery MAE (a fixed temperature-0 run could not). A stronger design would repeat the draw many times and report that distribution, and also vary the scenario set.
  Decoding is fixed, but the scenario set is itself a finite sample, so
  [@sec:recovery-result] bootstraps over it: the recovery MAE carries a 95% CI of
  [0.000, 0.038] and 2% of resamples are
  degenerate — the test-draw uncertainty this limitation flags, now quantified.
- **Recovery needs clearly-better-than-random judges.** The two-solution
  tie-break ([@sec:tie-break]) assumes the ensemble beats random;
  [@sec:tiebreak-boundary] shows it stays exact down to accuracy
  0.82 but inverts (up to 85%
  catastrophic) as judges approach chance. The evaluated `mistral:latest` judges are well clear of
  that boundary, but a borderline jury is not safe.
- **Five evaluable models.** `mistral:latest`, `gemma4:latest`, `gemma3:4b`, `gemma2:2b`, `granite4.1:3b` cleared the evaluability bar, so the head-to-head recovery ranking compares five real models. The recovery-quality claim rests on those finite real vote matrices; the *failure-mode* and *diagnostic* claims rest on all six. The deterministic synthetic study
  ([@sec:synthetic-validation]) extends the recovery-quality claim across
  200 ensembles per test size where the ground truth is known by
  construction, showing the real result is on-curve, not a fluke.
- **Diagnostic is necessary, not proven sufficient.** "No constant classifier"
  is a *necessary* condition for the error-independent solve; this study shows
  it was also sufficient on these six models, but six models is
  not a powered test of sufficiency. A model could in principle vary on every
  judge yet still violate error-independence and recover poorly. The honest claim
  is: the diagnostic is a cheap, label-free *necessary-condition* screen that
  caught every failure here.
- **Authored truth.** "Truth" is the authors' defensible call, not a market
  outcome; it calibrates the check, not the unsupervised recovery.

**Future work.** Repeat across temperatures and scenario draws to get a recovery
distribution; extend beyond the trio (larger `N`, `R>2`); test whether a
softened persona prompt (or light temperature) restores variation in the
collapsed judges and thereby restores evaluability. The synthetic study
([@sec:synthetic-validation])
begins the last of these — it shows positive common-mode correlation silently
degrades recovery without tripping the alarm — and a natural next step is to
*measure* the realized error-correlation of real LLM juries (e.g. from shared
base models) and predict the resulting recovery bias.

---

## Identifiability and reproducible computation {#sec:identifiability-and-computation}

Latent-class theory distinguishes generic identifiability from identifiability
at every parameter value [@allman2009]. Here, variation is only a necessary
screen; physical-root checks enforce the probability constraints of the fitted
model, and a better-than-random assumption orients its label mirror. A physical
solution cannot verify the independence assumption that generated it.

The supervised scores are exact for the authored finite deck. Scenario
bootstrap intervals describe the stated resampling procedure conditional on
that deck and the recorded responses; they do not include prompt-selection,
model-selection, model-update, or market-population uncertainty.

Runtime profiles in `benchmarks/runtime/` use Python 3.15, deterministic inputs,
unprofiled timing samples, and separate tracing and memory passes. Native
sampling availability is recorded explicitly. Computational improvements and
figure regeneration do not add model observations to the empirical study.
`docs/research_map.json` links scoped claims to their sources and artifacts;
the offline checker guards reference integrity rather than certifying claims.
