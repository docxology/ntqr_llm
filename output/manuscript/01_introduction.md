# Introduction {#sec:introduction}

Evaluation is the "forgotten twin" of learning: most AI work optimizes
training, while the question *how good were these judges, really?* is usually
answered only when labels exist. The unlabeled version of that question is old:
Dawid-Skene-style observer-error models, annotator-expertise models, truth
inference in crowdsourcing, and weak-supervision systems all use repeated
judgments to infer reliability without a conventional answer key
[@dawid_skene_1979; @yan_2010_annotator; @zheng_2017_vldb; @ratner_2017_snorkel].
Modern LLM-as-judge work makes the same problem newly practical and newly
fragile: LLM judges can scale evaluation, but their errors, biases, and
measurement uncertainty have to be modeled rather than wished away
[@zheng_2023_llm_judge; @chen_2026_efficient_judge].

That is also why majority vote and self-consistency are not enough as
evaluation *instrumentation*. They can select an answer, but they do not by
themselves tell us whether the voters were reliable, whether one judge was
systematically wrong on one label, or whether the response matrix has enough
variation to estimate reliability at all. A high-integrity evaluator needs to
separate three questions that are often collapsed in practice: did the interface
return valid measurements, is the vote matrix algebraically evaluable, and how
does the selected estimate compare with external truth when such truth is
available?

Algebraic evaluation (the **NTQR** logic [@ntqr_0_8_docs;
@corrada_emmanuel_2023_ntqr_logic; @corrada_emmanuel_2023_streaming]; see the
repository root [README](../README.md)) takes a deliberately narrower stance.
It constructs postulates — true for *any* finite test, containing no probability
theory and no model of the task domain — that link the observed agreements and
disagreements of an ensemble to the unknown statistics of their correctness.
Agreement-rate estimators showed how much unlabeled signal is present in
classifier agreements [@platanios_2014_uai; @platanios_2016_icml]; NTQR uses a
complete algebraic postulate system to ask when that signal is sufficient to
recover judge accuracies, and when the response matrix itself warns that the
evaluation is not justified.

A crucial caveat in that same framing is that the logic "is not magical … GIGO
also applies to evaluation on unlabeled data," and its distinctive safety value
is that it "can warn you if the ensemble is not good enough to perform a
reliable evaluation." This paper turns that promise into a measured,
reproducible demonstration on contemporary LLMs, and sharpens the warning into
a quantitative diagnostic. The claim is not that NTQR solves crowdsourcing or
judge aggregation generally. The narrow claim is that, under its stated
algebraic assumptions, NTQR can recover per-judge accuracies from a finite
unlabeled vote matrix — or emit an evaluability warning when the matrix lacks
the required structure [@corrada_emmanuel_2024_alarm;
@corrada_emmanuel_2025_nka].

We ask three questions:

1. **Does unsupervised algebraic recovery actually work on real LLMs?** When an
   LLM ensemble *is* well-behaved, how close does the answer-key-free recovery
   come to the truth?
2. **What breaks it?** Across a spread of models — strong and deliberately weak
   — what failure mode dominates, and what is its mechanism?
3. **Can we predict failure without labels?** Is there a statistic, computable
   from the votes alone, that tells us in advance whether an ensemble is
   evaluable at all?

Our contributions: (i) an end-to-end, cached, reproducible pipeline that runs
the same persona trio across many local models and scores unsupervised recovery
against authored truth; (ii) the empirical finding that **disagreement is not
evaluability** — the binding constraint is per-judge variation, not ensemble
disagreement; and (iii) a label-free **evaluability diagnostic** that, in this
study, perfectly predicted which models the algebra could solve.

The experiment is therefore not asking an LLM to predict markets better than
another LLM. It is asking whether a finite matrix of binary judgments contains
enough internal structure to evaluate the judges themselves. That distinction is
what makes the method useful when a task model is unavailable: the algebra
evaluates the response process, while authored truth is reserved for the
controlled validation step.

## Reader map: what is observed, recovered, and checked {#sec:reader-map}

Because the idea is unconventional, the central bookkeeping matters. The
unsupervised evaluator never sees the authored truth; it sees only the vote
matrix. The truth appears later, as a validation target for this controlled
study. The simulations then ask whether the same algebra behaves the same way
when the true data-generating process is known by construction.

| Object | Observed without labels? | Validation-only? | Meaning in this study |
|---|---:|---:|---|
| Scenario text | yes | no | A finite authored question containing only the market facts the model may use. |
| Persona vote | yes | no | One parsed `bullish` or `bearish` response, or an abstention if parsing fails. |
| Bullish / bearish labels | yes | no | Operational response labels: expected next-quarter direction up or down under the stated facts. |
| Authored truth | no | yes | The defensible answer used after the unsupervised solve to check recovery accuracy. |
| Prevalence | recovered | checked | Fraction of scenarios whose authored truth is `bullish` (`a`). |
| Persona accuracy | recovered | checked | Per-persona, per-label correctness rates inferred from vote-pattern structure. |
| Best predictor | recovered convention | checked | The higher-mean-accuracy algebraic branch selected without seeing truth. |
| Task/world model | no | no | Not estimated here; the algebra evaluates judges on a finite test rather than learning market semantics. |

: What is observed, recovered, and checked. The answer key is not an input to the unsupervised solve; it is the controlled-study check. {#tbl:evaluation-objects}

The important negative statement is as load-bearing as the positive one: without
a task/world model, the algebra does **not** learn what financial markets will
do. It evaluates a finite response matrix. In this study, `bullish` and
`bearish` are operational labels defined by the authored scenario facts:
`bullish` means the defensible next-quarter directional call is up under the
stated facts; `bearish` means down. The estimator then asks whether the pattern
of judge agreements is sufficient to recover each judge's per-label accuracy on
that finite test.

| ID | Operational truth | Why this is the authored answer | Scenario |
|---|---|---|---|
| s01 | bullish | Growth, estimate beat, and raised guidance all point to upside. | Revenue grew 42% YoY, beat estimates, and management raised full-year guidance. |
| s02 | bearish | Repeated misses plus a dividend cut indicate deteriorating fundamentals. | The company missed earnings for the third straight quarter and cut its dividend. |
| s19 | bullish | Mixed but positive: the earnings beat is the operational tie-break despite full valuation. | Earnings beat but guidance was merely in line; shares are near all-time highs. |
| s20 | bearish | Mixed but negative: margin compression and weak macro outweigh sales growth. | Sales grew but margins thinned and the macro outlook for the sector is murky. |
| s23 | bullish | Mixed but positive: cost cuts plus reinvestment create a turnaround path. | New management is cutting costs aggressively and reinvesting in a growing niche. |
| s24 | bearish | Mixed but negative: low valuation is outweighed by structural decline. | The stock is cheap on paper but sits in a structurally declining industry. |

: Representative operational questions and answers. The full 64-scenario answer key is generated in the supplemental; these rows show how `bullish` and `bearish` are defined without a learned task model. {#tbl:scenario-examples}

## Related evaluation paradigms {#sec:related-evaluation-paradigms}

The experiment is easiest to read if NTQR is placed beside the neighboring
evaluation paradigms it resembles but does not replace. The table below keeps
the comparison concrete: what is observed, what assumption does the work, what
the method returns, and what kind of warning it can or cannot provide.

| Paradigm | Observed input | Main assumption | Output | Failure warning |
|---|---|---|---|---|
| Supervised evaluation | Votes plus labels | Trusted labels | Empirical accuracies | None about unlabeled evaluability |
| Dawid-Skene / expertise | Repeated labels | Latent truth model | Worker error estimates | Model-fit warning |
| Agreement-rate estimation | Classifier agreements | Independence / priors | Accuracy estimates | Assumption sensitivity |
| Weak supervision | Labeling functions | Generative dependency model | Probabilistic labels | Dependency diagnostics |
| LLM-as-judge | Model judgments | Judge reliability model | Scalable evaluation | Bias / measurement-error correction |
| Majority vote / self-consistency | Multiple votes | Better-than-random voters | Aggregate answer | No per-judge recovery guarantee |
| NTQR in this paper | Unlabeled vote sketch | Algebraic postulates | Accuracies or alarm | Constant-classifier / imaginary-root failure |

: Neighboring evaluation paradigms. Citations in the surrounding text identify the canonical references for each row; the key distinction is that NTQR returns either a label-free algebraic recovery or a label-free warning that recovery is not justified. {#tbl:related-evaluation-paradigms}
