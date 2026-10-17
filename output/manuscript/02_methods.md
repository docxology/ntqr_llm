# Methods {#sec:methods}

## Evaluation object and data flow {#sec:method-data-flow}

The unit of analysis is a **model-specific vote matrix**: three personas by
64 scenarios, with each cell containing exactly one parsed `bullish` or
`bearish` call. The unsupervised NTQR solve never consumes scenario text, market
semantics, model logits, chain-of-thought, or the authored answer key. It
receives only unlabeled vote-pattern counts derived from that matrix.

The controlled experiment adds two layers around that solve. First, a
schema-constrained collection interface makes parsing a measured engineering
boundary rather than a hidden source of invented labels. Ollama's structured
output interface accepts a JSON Schema in the `format` field and its public
guidance recommends lower-temperature decoding for more reliable schema
adherence [@ollama_structured_outputs_2024; @ollama_structured_blog_2024].
Structured-output reliability is itself an active evaluation problem
[@structured_output_reliability_2025]; here we make it a measured interface
check and then move the scientific question to the completed binary vote
matrix.
Second, the authored scenario truth is kept out of the solve and used only
afterward to score recovery error. [@fig:evaluation-pipeline] shows this
separation explicitly.

![Pipeline boundary for the experiment. Scenario facts and system prompts create parsed persona votes; the unsupervised NTQR solve sees only the resulting unlabeled agreement sketch. Authored truth is held out until the validation step, where recovered accuracies are compared with supervised truth.](../figures/evaluation_pipeline.png){#fig:evaluation-pipeline width=95%}

## The classifiers: three LLM personas {#sec:classifiers}

We use three system-prompted "trader" personas as noisy binary judges, chosen
so their *mistakes are not identical* — exactly the error structure
error-independent evaluation needs:

- **Optimist** — leans `bullish` on ambiguous setups.
- **Neutral** — weighs the stated facts with no directional bias.
- **Pessimist** — leans `bearish` on ambiguous setups.

Three personas form a trio, matching the three-classifier `R=2` evaluators in
the public NTQR package documentation [@ntqr_0_9_docs]. Each persona reads the
same scenario and returns a single JSON call. The prompt contract is versioned
as `v3_binary_schema`
(`4d157f84885a`) and appears verbatim in [@sec:personas-scenarios].
We treat the personas as instruction-following interfaces, not as human trader
models; instruction-following behavior itself is a product of modern LLM
alignment and fine-tuning practice [@ouyang_2022_instructgpt].

## The test: 64 binary scenarios {#sec:test-deck}

The test is **T = 1** test of **Q = 64** one-line market scenarios with
**R = 2** possible responses (`bullish`→label `a`, `bearish`→label `b`). Each
scenario carries an authored `truth` — the defensible call given only the stated
facts. Most scenarios are clear-cut; six are deliberately ambiguous (mixed
signals) so the personas' dispositions pull their calls apart.

**The deck is deliberately *not* balanced.** The authored prevalence is
**40 bullish, 24 bearish** — true prevalence of label `a`
= **0.625**. The imbalance is a design choice motivated by the
implemented evaluator's singularity when its recovered prevalence is exactly
1/2. At that point its two algebraic roots coalesce and some accuracy ratios
become 0/0. An exactly balanced answer key does not force that equality in a
finite sampled sketch: empirical error correlations can shift the roots.
Tilting the generating prevalence to 0.625 avoids centering the
study on that symmetric case, while leaving 24 bearish scenarios
for assessing label variation. Neither imbalance nor variation guarantees a
complete physical solution
([@sec:failure-modes]). The full
scenario list and truths are in
[@sec:personas-scenarios].

The authored truth is **never** an input to the unsupervised evaluator. It is
used only to build a `SupervisedEvaluation` as a check against which recovery is
scored.

This distinction also defines the meaning of "best predictor" in the paper. The
best model is not the model with the most plausible financial prose. It is the
evaluable vote matrix whose selected unsupervised solution has the smallest
validation-only recovery MAE against the authored labels.

## The models {#sec:models}

We run the identical trio through six locally-hosted chat models via
Ollama at temperature 0.1 (stochastic decoding): `mistral:latest`, `gemma4:latest`, `gemma3:4b`, `gemma2:2b`, `granite4.1:3b`,
`smollm2:135m-instruct-q4_K_S`. The models span a range of sizes and capabilities, from small
sub-instruction-following models to multi-billion-parameter chat models. Votes
are cached per `(backend, model, temperature, prompt version, prompt hash,
persona, scenario)`, so the sweep is reproducible and resumable without mixing
prompt designs.

The local Ollama tags are the experimental identifiers. Public model-family
references provide provenance only: Mistral for `mistral:latest`
[@jiang_2023_mistral], the Gemma-family reports for the local Gemma tags
[@gemma_team_2024; @gemma2_team_2024; @gemma3_team_2025], IBM Granite 4.1 for
`granite4.1:3b` [@ibm_granite_4_1_2026], and SmolLM2 as family context for the
quantized local `smollm2:135m-instruct-q4_K_S` tag
[@ben_allal_2025_smollm2]. None of those citations enters the solve; the solve
uses only the cached votes and the model names recorded in `summary.json`.

| Local Ollama tag | Public family citation | Run status | What the status means here |
|---|---|---|---|
| `granite4.1:3b` | IBM Granite 4.1 [@ibm_granite_4_1_2026] | evaluable; recovery MAE 0.054 | All three personas varied, every cell parsed, and the error-independent solve returned a selected branch. |
| `mistral:latest` | Mistral family [@jiang_2023_mistral] | best recovered model; recovery MAE 0.012 | All three personas varied, every cell parsed, and the error-independent solve returned a selected branch. |
| `gemma2:2b` | Gemma and Gemma 2 family [@gemma_team_2024; @gemma2_team_2024] | evaluable; recovery MAE 0.026 | All three personas varied, every cell parsed, and the error-independent solve returned a selected branch. |
| `gemma3:4b` | Gemma family reports [@gemma_team_2024; @gemma3_team_2025] | evaluable; recovery MAE 0.024 | All three personas varied, every cell parsed, and the error-independent solve returned a selected branch. |
| `gemma4:latest` | Gemma family reports [@gemma_team_2024; @gemma3_team_2025] | evaluable; recovery MAE 0.024 | All three personas varied, every cell parsed, and the error-independent solve returned a selected branch. |
| `smollm2:135m-instruct-q4_K_S` | SmolLM2 family [@ben_allal_2025_smollm2] | not solved; constant persona(s): Optimist, Neutral, Pessimist | At least one judge never varied, so the trio carries no recoverable error signal for that judge. |

: Local model lineup, public family provenance, and data-driven run status. The citations identify model families, while the solve uses only cached votes and the recorded local tags. {#tbl:model-lineup}

Each requested model is asked the same binary task under the same schema.
If a primary response violates the schema, the collector allows one
model-driven repair attempt using the same scenario facts and the same binary
contract. A cell is accepted only if the model returns one of the two allowed
labels; the collector never maps `neutral`, `mixed`, malformed JSON, or an
unavailable model into a vote.

## The evaluators {#sec:evaluators}

For each model we encode the trio's votes as `TrioLabelVoteCounts` and run three
`ntqr.r2` evaluators [@ntqr_0_9_docs; @corrada_emmanuel_2023_streaming]:

- **`ErrorIndependentEvaluation`** (unsupervised) — recovers prevalence and each
  persona's per-label accuracy from agreement structure alone, with no answer
  key. It returns **two** algebraically consistent solutions (the labelling we
  want and its near-zero-accuracy mirror image).
- **`MajorityVotingEvaluation`** (unsupervised) — a simpler unsupervised
  baseline, also returning two solutions.
- **`SupervisedEvaluation`** (the check) — the answer-key truth.

**Two-solution tie-break.** From the two unsupervised solutions we keep the one
with higher mean accuracy — the standard, answer-key-free "experts are better
than random" assumption. We deliberately do *not* break the tie by closeness to
the true prevalence: that would peek at the ground truth and would fail whenever
the two prevalences are symmetric about 0.5. The derivation is in
[@sec:tie-break].

## The algebra: from votes to accuracies {#sec:formalism}

This section makes the "exact algebra" concrete, because the whole study turns
on its structure — including *why* it fails. The derivation follows the NTQR
complete-postulate and streaming-algorithm line of work
[@corrada_emmanuel_2023_ntqr_logic; @corrada_emmanuel_2023_streaming], with the
agreement-rate literature as its nearest statistical antecedent
[@platanios_2014_uai; @platanios_2016_icml]. We follow the construction in
[`SimplestExampleOfEvaluationWithAlgebraicGeometry.md`](../docs/essays/SimplestExampleOfEvaluationWithAlgebraicGeometry.md)
and the public NTQR implementation documented for
`ntqr.r2.evaluators.ErrorIndependentEvaluation` [@ntqr_0_9_docs].

**The universal evaluation polynomial.** Write the two labels as $\alpha$
(`bullish`, `a`) and $\beta$ (`bearish`, `b`), the unknown test prevalence of
$\alpha$ as $P_\alpha$ (with $P_\beta = 1 - P_\alpha$), and a single judge's
per-label accuracies as $P_{i,\alpha}$ and $P_{i,\beta}$. The observed frequency
with which that judge votes $\alpha$ decomposes — by definition, with no
probability model — into "correct on $\alpha$ items" plus "wrong on $\beta$
items":

$$
f_\alpha \;=\; P_\alpha\,P_{i,\alpha} \;+\; P_\beta\,\bigl(1 - P_{i,\beta}\bigr).
$$ {#eq:universal-evaluation-polynomial}

This polynomial is *exact and universal*: any noisy judge on any finite test
satisfies it identically. It contains no domain knowledge — that emptiness is
precisely what makes it immune to the out-of-distribution failures of trained
estimators. It is an evaluation *thermometer*, not a model.

**The trio raises the rank.** One judge gives two such equations in three
unknowns — underdetermined. Three error-independent judges sharing one test
supply enough agreement structure to close the system. The `ntqr` data sketch is
the eight trio vote-pattern frequencies $f_{v_0 v_1 v_2}$ (each $v_i \in
\{\alpha,\beta\}$, $\sum f = 1$). From them the evaluator forms three orders of
statistic (shown here for label $\beta$; `datasketches.py`):

$$
f_{\beta_i} = \!\!\sum_{v_j,v_k}\!\! f_{\dots v_i=\beta\dots}, \qquad
   \delta_{(i,j)} = f_{\beta_i \beta_j} - f_{\beta_i} f_{\beta_j}, \qquad
   m_{012} = \!\prod_i f_{\beta_i} + \!\!\sum_{(i,j,k)}\!\! f_{\beta_i}\,\delta_{(j,k)}.
$$ {#eq:trio-moments}

Here $f_{\beta_i}$ is judge $i$'s marginal $\beta$-vote frequency, the **pair
moment** $\delta_{(i,j)}$ is the (sample) covariance of judges $i,j$ voting
$\beta$ — a *measured* agreement, not a modelled one — and $m_{012}$ is the trio
frequency moment. The prevalence of $\alpha$ is then the root of a **quadratic**

$$
a\,P_\alpha^{2} + b\,P_\alpha + c \;=\; 0, \qquad
   a = \Delta^{2} + 4\!\!\prod_{(i,j)}\!\delta_{(i,j)}, \quad
   b = -a, \quad
   c = \prod_{(i,j)}\!\delta_{(i,j)}.
$$ {#eq:prevalence-quadratic}

where $\Delta = f_{\beta\beta\beta} - m_{012}$ is the gap between the all-$\beta$
vote frequency and the trio frequency moment $m_{012}$ (see
`alpha_prevalence_quadratic_terms`). Because $b = -a$, the closed form collapses
to a single $\pm$ pair:

$$
P_\alpha \;=\; \tfrac12 \;\mp\; \tfrac12\sqrt{\,1 - 4c/a\,}.
$$ {#eq:prevalence-roots}

**Two solutions, by construction.** The $\mp$ is not numerical noise: absent
labels, a nondegenerate physical solve admits two mirror evaluation
points — the labelling we want and its near-mirror image with the labels
swapped. Each judge's per-label accuracy is then recovered as an exact *linear*
function of $P_\alpha$ (`classifier_a_label_accuracy` /
`classifier_b_label_accuracy`), so fixing the prevalence root fixes all six
accuracy terms. The tie-break ([@sec:evaluators]) selects between the two points without
peeking at the truth.

**The singularity at $P_\alpha = \tfrac12$.** When the prevalence quadratic
has discriminant zero, its two roots coalesce at $\tfrac12$, and this evaluator's
downstream accuracy ratios can become $0/0$. Exact symmetry can therefore make
the implemented estimator singular. A 50/50 authored answer key alone does not
force finite sampled votes onto that singularity: sample error correlations
can shift the algebraic roots. The deliberately unbalanced deck in
[@sec:test-deck] avoids centering the generating prevalence on the symmetric
case; it does not guarantee identifiability or physical roots.

**The built-in alarm.** When the judges are in fact error-*correlated* on the
test, the discriminant $1 - 4c/a$ goes negative and the prevalence root becomes
imaginary — the current evaluator returns complex symbolic values, which the
physical-solution guard rejects. An
imaginary prevalence is an *iron-clad, label-free* detection that the
error-independence assumption is violated: the algebra refuses to return a
number it cannot justify. This is the "warning light" property at the center of
the safety framing and the later no-knowledge alarm formulation
[@corrada_emmanuel_2024_alarm; @corrada_emmanuel_2025_nka] — and the per-judge
constant-classifier failure of [@sec:failure-modes] is
its discrete cousin, which the following makes precise.

> **Proposition (constant-judge non-identification).** *If any judge $i$ casts
> one label on all $Q$ items, the error-independent trio's prevalence quadratic
> does not identify isolated prevalence roots, regardless of the other judges.*
>
> *Proof.* A constant judge has $f_{\beta_i}\in\{0,1\}$, and every pair moment
> touching it vanishes. Thus $c=\prod\delta_{(i,j)}=0$. If $f_{\beta_i}=0$,
> both $f_{\beta\beta\beta}$ and $m_{012}$ vanish. If $f_{\beta_i}=1$, both
> equal the remaining pair's joint frequency. In either case $\Delta=0$, so
> $a=b=c=0$: the polynomial is identically zero, not a quadratic with two
> isolated roots. The implemented estimator is singular and returns undefined
> terms. This proves non-identification, not the absence of all possible
> interior physical evaluations. ∎

**This is why disagreement is the wrong axis.** The degeneracy is triggered by a
*per-judge* quantity ($\delta_{(i,\cdot)} = 0$ for a single $i$) and is blind to
the *ensemble* quantity of how much the judges collectively disagree. Two judges
can disagree violently while the third sits constant: the product still vanishes
and the system is still unsolvable. Conversely the worked example in
[@sec:worked-example]
shows all three pair moments strictly positive on real votes, a positive
discriminant, and the two mirror-image roots resolving to the truth.
[@sec:disagreement-not-evaluability] is the empirical face of this proposition;
[@sec:diagnostic-result] turns it into a label-free screen.

## Statistics {#sec:statistics}

- **Recovery MAE** (headline): mean absolute error between recovered and true
  per-persona, per-label accuracies. We also report it split by label.
- **Prevalence absolute error**: |recovered − true| prevalence of label `a`.
- **Sampling-noise floor**: a reference scale for the recovery MAE. A per-label
  accuracy is estimated from only the questions carrying that label; the
  worst-case binomial standard error of such a proportion is `sqrt(0.25 / k)`,
  where `k` is the minority label's question count (here `k = 24`,
  giving a floor of **0.102**). A recovery MAE at or below this floor is
  smaller than the worst-case uncertainty of the supervised estimate it is
  compared against. We use it as a scale reference, not as a formal hypothesis
  test.
- **Degeneracy guard**: a constant classifier makes the algebra unsolvable and
  yields a SymPy `nan`; because `float(nan)` succeeds, we reject non-finite
  values explicitly (`math.isfinite`) and report the model as a *named failure*
  rather than averaging `NaN` into a meaningless statistic.
- **Scenario bootstrap**: uncertainty for the finite scenario deck is estimated
  by nonparametric resampling of scenarios with replacement, following the
  bootstrap logic of Efron and Tibshirani [@efron_tibshirani_1993].
- **Evaluability diagnostic** (label-free; see [@sec:evaluability-diagnostic]).

## The evaluability diagnostic {#sec:evaluability-diagnostic}

Computed from the votes alone — no answer key, no solve:

- **Vote variation** — for each judge, the fraction of scenarios on its single
  most common call (`modal_fraction`). A value of 1.0 means the judge never
  varied: a **constant classifier**, which carries no error signal.
- **Pairwise disagreement** — for each pair of judges, the fraction of scenarios
  on which they differ; and its mean over pairs.
- **Preliminary screen** — `predicted_evaluable` requires no constant judge and
  no abstentions. This is a necessary-condition screen, not a sufficient test.
  Varying judges can still have zero pair covariance, complex roots, or
  out-of-range accuracy estimates. The offline `recovery_validity_report.py`
  separately checks exact pair moments and all recovered probabilities without
  an answer key. Current live scoring requires a complete physical solution.

## Reproducibility {#sec:reproducibility}

Everything is exact and offline-reproducible from the cached votes. The numeric
core (`python/src/ntqr/llm/`) is pure — no I/O, no plotting — and covered by
124 unit tests that run with no network and no Ollama. The
manuscript itself is *generated*: every number below is injected from
`summary.json` by `scripts/manuscript/render.py`, so the prose cannot drift from
the data. Exact commands are in [@sec:reproduction].
