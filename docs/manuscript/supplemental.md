# Supplemental material {#sec:supplemental}

<!-- GENERATED FILE — do not edit by hand.
     Source: scripts/manuscript/templates/supplemental.md.tmpl
     Numbers: scripts/manuscript/_values.py  (read from generated JSON artifacts)
     Rebuild: python scripts/manuscript/render.py -->

Companion to the main manuscript. All numbers here are reproduced
from generated artifacts: the cached LLM sweep in
`outputs/llm/multi_llm_personas_evaluation/` (Ollama backend, temperature
0.1, N=3 personas, T=1 test, Q=64 scenarios, R=2 responses),
the deterministic simulation/bootstrap outputs, and the offline validation
matrix.

- [S1 — The two-solution tie-break](#sec:tie-break)
- [S1.5 — Worked example: the prevalence algebra on real votes](#sec:worked-example)
- [S1.6 — Exactly when the imaginary-root alarm fires (and its blind spot)](#sec:alarm-derivation)
- [S2 — Full per-model label vote counts](#sec:vote-counts)
- [S3 — Full per-persona recovery (mistral:latest)](#sec:full-recovery)
- [S4 — Personas, scenarios, and authored truth](#sec:personas-scenarios)
- [S5 — Reproduction](#sec:reproduction)
- [S6 — The evaluability diagnostic, defined](#sec:diagnostic-definition)
- [S7 — Output file inventory](#sec:output-inventory)

## S1 — The two-solution tie-break {#sec:tie-break}

`ErrorIndependentEvaluation` (and `MajorityVotingEvaluation`) each return **two**
algebraically consistent solutions for the trio's prevalence and per-persona
accuracies. The two are a symmetry pair: the labelling we intend, and its
*mirror image* in which every judge is treated as near-random/anti-correct and
the prevalence is reflected about 0.5. Both satisfy the postulates exactly — the
algebra alone cannot distinguish them.

We select with the standard answer-key-free assumption that **the judges are
better than random**: keep the solution with the higher mean per-judge,
per-label accuracy (`ntqr.llm.recovery_scoring::select_best`).

Two tempting alternatives are rejected on principle:

1. **Pick the solution closest to the true prevalence.** This peeks at the
   ground truth, so it is not unsupervised. It also fails outright when the two
   candidate prevalences are symmetric about 0.5 (e.g. 0.46 vs 0.54), where
   "closest to 0.5" is a coin flip.
2. **Pick by internal consistency.** Both solutions are *exactly* consistent —
   that is the whole point — so consistency cannot break the tie.

For mistral:latest, the selected solution had mean accuracy ≈ 0.89
against its mirror's ≈ 0.11; the tie-break is unambiguous. See the
two solutions verbatim in
`outputs/llm/multi_llm_personas_evaluation/models/mistral_latest/evaluation_error_independent.json`.

## S1.5 — Worked example: the prevalence algebra on real votes {#sec:worked-example}

This section runs the [@sec:formalism] formalism on **`mistral:latest`'s actual votes** — no
toy numbers. Every value below is recomputed by `_values.py` from the on-disk
`label_vote_counts.json` through `ntqr.r2`, so it cannot drift from the data.

**Step 1 — the data sketch.** From the trio's votes we read the all-bearish
vote-pattern frequency and the frequency moments (label `b`):

| Quantity | Symbol | Value |
|---|---|---|
| All-`b` vote frequency | $f_{\beta\beta\beta}$ | 0.2969 |
| Trio frequency moment | $m_{012}$ | 0.2735 |
| Gap | $\Delta = f_{\beta\beta\beta} - m_{012}$ | 0.0233 |
| Pair moment (Opt,Neu) | $\delta_{(0,1)}$ | 0.1809 |
| Pair moment (Opt,Pes) | $\delta_{(0,2)}$ | 0.1067 |
| Pair moment (Neu,Pes) | $\delta_{(1,2)}$ | 0.1404 |
| Product of pair moments | $\prod_{(i,j)}\delta_{(i,j)}$ | 2.71e-03 |

: Real-vote frequency moments for `mistral:latest`, recomputed from `label_vote_counts.json`. {#tbl:worked-frequency-moments}

All three pair moments are **strictly positive** — every judge varies and the
pairs co-vary — so the product does not vanish and the system is non-degenerate
(contrast [@sec:failure-modes], where a constant classifier drives a pair moment to zero).

**Step 2 — the prevalence quadratic.** With $a = \Delta^2 + 4\prod\delta$ and
$c = \prod\delta$, the discriminant of $P_\alpha = \tfrac12 \mp
\tfrac12\sqrt{1 - 4c/a}$ evaluates to

$$
1 - 4c/a = 0.0479 \;>\; 0 \quad\Longrightarrow\quad
   \tfrac12\sqrt{1 - 4c/a} = 0.1094.
$$ {#eq:worked-discriminant}

The discriminant is **positive**, so the prevalence is real — the algebra finds
no error-correlation violation (had it been negative, `ntqr` would raise
`PrevalenceImaginaryException`; see [@sec:formalism]).

**Step 3 — the two solutions.** The $\mp$ yields the symmetry pair

$$
P_\alpha \in \{\,0.3906,\; 0.6094\,\}.
$$ {#eq:worked-roots}

mirror images about $\tfrac12$. The tie-break ([@sec:tie-break]) selects
**0.6094** (higher mean accuracy), against the authored true
prevalence of **0.6250** — an absolute error of **0.0156**,
recovered with no answer key. Fixing this root then fixes all six per-persona,
per-label accuracies via the linear maps of [@sec:formalism] (tabulated in
[@sec:full-recovery]).

## S1.6 — Exactly when the imaginary-root alarm fires (and its blind spot) {#sec:alarm-derivation}

The [@sec:synthetic-validation] simulation finds, empirically, that the imaginary-prevalence alarm trips
on an anti-correlated judge pair but **never** on positive common-mode
correlation. That asymmetry is not a quirk of the draws — it falls straight out
of the quadratic. Substituting $a = \Delta^2 + 4\prod\delta$ and $c = \prod\delta$
into the discriminant and simplifying,

$$
D \;=\; 1 - \frac{4c}{a}
   \;=\; \frac{\Delta^2 + 4\prod\delta - 4\prod\delta}{\Delta^2 + 4\prod\delta}
   \;=\; \frac{\Delta^2}{\Delta^2 + 4\prod\delta},
$$ {#eq:alarm-discriminant}

where $\Delta = f_{\beta\beta\beta} - m_{012}$ and $\prod\delta =
\delta_{(0,1)}\,\delta_{(0,2)}\,\delta_{(1,2)}$ is the product of the three
pairwise vote-moments. The numerator $\Delta^2 \ge 0$ is a square, so (for
$\Delta \neq 0$) the alarm condition $D < 0$ is **exactly**

$$
\Delta^2 + 4\prod\delta < 0
   \quad\Longleftrightarrow\quad
   \prod\delta \;<\; -\,\tfrac14\,\Delta^2 \;\le\; 0.
$$ {#eq:alarm-condition}

Two consequences, both visible in [@sec:synthetic-validation]:

1. **The alarm is sufficient, never a false positive.** $D < 0$ forces
   $\prod\delta < 0$ — an *odd* number of the three pairs are anti-correlated in
   their votes. No genuinely error-independent test produces that in expectation,
   which is why the independent control fired in 0% of trials.
2. **It is blind to positive common-mode correlation.** If every pair co-varies
   positively ($\prod\delta > 0$), then $D \in (0, 1]$ for *any* strength of that
   correlation: the root stays real and the alarm is structurally silent. Shared
   errors only push every $\delta_{(i,j)}$ **more** positive, enlarging the
   denominator and driving $D \to 0^{+}$ without ever crossing it — precisely the
   [@sec:synthetic-validation] curve where recovery degraded from 0.016 to
   0.042 while the alarm never sounded. The real worked
   example above sits in this safe regime: all three $\delta_{(i,j)}$ are
   positive, so $D = 0.0479 > 0$.

The practical reading: the imaginary-root alarm is a genuine, answer-key-free
*sufficient* check, but a silent $D$ is **not** a certificate of
error-independence — only an absence of the specific anti-correlated signature it
can see.

## S2 — Full per-model label vote counts {#sec:vote-counts}

Vote patterns are ordered `(Optimist, Neutral, Pessimist)`; `a` = bullish,
`b` = bearish. The outer key is the **authored truth** of the scenario. Counts
sum to 64 (40 bullish `a`, 24 bearish `b`).

**`granite4.1:3b`** — evaluable (every judge varies):

| Truth | Vote pattern | Count |
|---|---|---|
| a (bullish) | a,a,a | 33 |
| a | a,b,b | 3 |
| a | a,a,b | 3 |
| a | b,b,b | 1 |
| b (bearish) | b,b,b | 24 |

Per-judge: Optimist a×39/b×25, Neutral a×36/b×28, Pessimist a×33/b×31 → **no constant classifier**.

**`mistral:latest`** — evaluable (every judge varies):

| Truth | Vote pattern | Count |
|---|---|---|
| a (bullish) | a,a,a | 23 |
| a | a,a,b | 16 |
| a | a,b,b | 1 |
| b (bearish) | b,b,b | 19 |
| b | a,b,b | 5 |

Per-judge: Optimist a×45/b×19, Neutral a×39/b×25, Pessimist b×41/a×23 → **no constant classifier**.

**`gemma2:2b`** — evaluable (every judge varies):

| Truth | Vote pattern | Count |
|---|---|---|
| a (bullish) | a,a,a | 33 |
| a | a,a,b | 5 |
| a | a,b,b | 1 |
| a | b,b,b | 1 |
| b (bearish) | b,b,b | 24 |

Per-judge: Optimist a×39/b×25, Neutral a×38/b×26, Pessimist a×33/b×31 → **no constant classifier**.

**`gemma3:4b`** — evaluable (every judge varies):

| Truth | Vote pattern | Count |
|---|---|---|
| a (bullish) | a,a,a | 34 |
| a | a,a,b | 4 |
| a | a,b,b | 1 |
| a | b,b,b | 1 |
| b (bearish) | b,b,b | 19 |
| b | a,b,b | 5 |

Per-judge: Optimist a×44/b×20, Neutral a×38/b×26, Pessimist a×34/b×30 → **no constant classifier**.

**`gemma4:latest`** — evaluable (every judge varies):

| Truth | Vote pattern | Count |
|---|---|---|
| a (bullish) | a,a,a | 37 |
| a | b,b,b | 1 |
| a | a,a,b | 1 |
| a | a,b,b | 1 |
| b (bearish) | b,b,b | 23 |
| b | a,a,b | 1 |

Per-judge: Optimist a×40/b×24, Neutral a×39/b×25, Pessimist a×37/b×27 → **no constant classifier**.

**`smollm2:135m-instruct-q4_K_S`** — Optimist *and* Neutral *and* Pessimist constant:

| Truth | Vote pattern | Count |
|---|---|---|
| a (bullish) | a,a,a | 40 |
| b (bearish) | a,a,a | 24 |

Optimist, Neutral, Pessimist never varied → **two constant classifiers (Optimist, Neutral, Pessimist)**.

## S3 — Full per-persona recovery (mistral:latest) {#sec:full-recovery}

Unsupervised recovery (`ErrorIndependentEvaluation`, better-than-random branch)
vs. supervised truth. Source:
`models/mistral_latest/comparison.json` and `recovery_stats.json`.

| Persona | Label | True | Recovered | Abs. error |
|---|---|---|---|---|
| Optimist | a | 1.0000 | 1.0000 | 0.0000 |
| Optimist | b | 0.7917 | 0.7600 | 0.0317 |
| Neutral | a | 0.9750 | 1.0000 | 0.0250 |
| Neutral | b | 1.0000 | 1.0000 | 0.0000 |
| Pessimist | a | 0.5750 | 0.5897 | 0.0147 |
| Pessimist | b | 1.0000 | 1.0000 | 0.0000 |

: Full per-persona recovery table for `mistral:latest`, with the selected unsupervised branch compared to the supervised check. {#tbl:full-recovery}

- **Recovery MAE** = 0.0119 over six accuracy terms
  (close recovery, within the 0.102 sampling-noise floor).
- **Recovery MAE by label**: a = 0.0132, b = 0.0106.
- **Prevalence (label a)**: true 0.6250, recovered 0.6094, abs.
  error 0.0156.
- **Majority-voting recovery MAE** = 0.0119 (identical to
  error-independent here).

Exact fractions (from `evaluation_*.json`): recovered prevalence =
39/64; Optimist(b) = 19/25; Neutral = 1; Pessimist(a) =
23/39. The recovered values are close to the supervised truth, not exact; the residual error is the finite-test recovery error reported above and remains within the 0.102 sampling-noise floor.

## S4 — Personas, scenarios, and authored truth {#sec:personas-scenarios}

### Persona system prompts (verbatim)

Prompt contract: `v3_binary_schema` (`4d157f84885a`).

| Persona | System prompt |
|---|---|
| Optimist | You classify next-quarter equity direction from the stated facts only. Use the Optimist decision policy: in genuinely mixed cases, give slightly more weight to upside catalysts and recovery paths. If the evidence is exactly balanced, choose bullish. Do not force optimism when explicit negative evidence dominates; such cases should be bearish. Return valid JSON only. |
| Neutral | You classify next-quarter equity direction from the stated facts only. Use the Neutral decision policy: weigh positive and negative evidence symmetrically, with no directional prior. Make the call the evidence best supports; if it is close, still choose the stronger stated side. Return valid JSON only. |
| Pessimist | You classify next-quarter equity direction from the stated facts only. Use the Pessimist decision policy: in genuinely mixed cases, give slightly more weight to downside risks and execution failure. If the evidence is exactly balanced, choose bearish. Do not force pessimism when explicit positive evidence dominates; such cases should be bullish. Return valid JSON only. |

: Versioned persona system prompts. The optimist and pessimist policies bias only genuinely mixed cases and explicitly allow evidence to override the persona. {#tbl:persona-prompts}

Each persona receives the scenario and this user prompt:

> Market scenario:<br>[scenario text]<br><br>Task: classify the asset's expected direction over the next quarter.<br>Use only the facts stated above; do not assume outside market data.<br>Choose exactly one allowed value for call: bullish or bearish.<br>No neutral, no mixed, no abstention, and no explanation are allowed. If evidence is mixed, choose the allowed side implied by your persona policy.<br>Return exactly one JSON object and no other text, for example {"call":"bullish"} or {"call":"bearish"}.

### Scenarios and authored truth (Q = 64)

| ID | Truth | Scenario |
|---|---|---|
| s01 | bullish | Revenue grew 42% YoY, beat estimates, and management raised full-year guidance. |
| s02 | bearish | The company missed earnings for the third straight quarter and cut its dividend. |
| s03 | bullish | A blockbuster drug just received FDA approval ahead of schedule with a broad label. |
| s04 | bearish | Regulators opened a fraud probe and the CFO abruptly resigned this morning. |
| s05 | bullish | Free cash flow doubled, the firm initiated a large buyback, and debt fell sharply. |
| s06 | bearish | A key patent was invalidated and two generic competitors announced launches. |
| s07 | bullish | The company signed a multi-year supply contract with the largest buyer in its sector. |
| s08 | bearish | Inventory is piling up, margins compressed 600bps, and demand guidance was withdrawn. |
| s09 | bullish | Same-store sales accelerated, traffic is up, and the loyalty program is growing fast. |
| s10 | bearish | A major data breach exposed millions of customers and class-action suits are filed. |
| s11 | bullish | The central bank signaled rate cuts and the company is highly rate-sensitive on the upside. |
| s12 | bearish | The main product was recalled for a safety defect and shipments are halted. |
| s13 | bullish | Order backlog hit a record and lead times extended, signaling robust demand. |
| s14 | bearish | A larger rival slashed prices 30% to take share in the company's core market. |
| s15 | bullish | Gross margins expanded for the fifth straight quarter as input costs fell. |
| s16 | bearish | The firm issued a going-concern warning and is in breach of its loan covenants. |
| s17 | bullish | A respected activist investor took a 9% stake and won three board seats. |
| s18 | bearish | Key engineering talent is leaving and the flagship project slipped a full year. |
| s19 | bullish | Earnings beat but guidance was merely in line; shares are near all-time highs. *(mixed)* |
| s20 | bearish | Sales grew but margins thinned and the macro outlook for the sector is murky. *(mixed)* |
| s21 | bullish | A turnaround plan is showing early traction though leverage remains elevated. *(mixed)* |
| s22 | bearish | A one-time gain flattered profit while underlying organic growth stalled. *(mixed)* |
| s23 | bullish | New management is cutting costs aggressively and reinvesting in a growing niche. *(mixed)* |
| s24 | bearish | The stock is cheap on paper but sits in a structurally declining industry. *(mixed)* |
| s25 | bullish | A new product line sold out within hours and the company doubled its production targets. |
| s26 | bullish | The company won a multi-year government contract that triples its addressable market. |
| s27 | bullish | Subscriber growth reaccelerated and monthly churn fell to a multi-year low. |
| s28 | bullish | A long-running patent suit was settled in the company's favor with a large damages award. |
| s29 | bullish | Operating leverage kicked in: revenue rose 20% while operating expenses stayed flat. |
| s30 | bullish | The firm raised full-year guidance twice this quarter as bookings outpaced every forecast. |
| s31 | bullish | A strategic partner took a minority stake at a premium and committed to joint development. |
| s32 | bullish | Margins hit a record after the firm exited its lowest-margin segment and raised prices. |
| s33 | bullish | Posted record annual revenue and free cash flow and initiated its first-ever dividend. |
| s34 | bullish | A flagship product launch drew record preorders and sold out within a day. |
| s35 | bullish | Regulators granted a breakthrough designation that fast-tracks the lead program. |
| s36 | bullish | Closed an all-cash, immediately accretive acquisition that added no new debt. |
| s37 | bullish | Its credit rating was upgraded to investment grade and borrowing costs fell sharply. |
| s38 | bullish | Signed its largest-ever enterprise contract, a multi-year committed deal. |
| s39 | bullish | Margins expanded as a new automated plant came online ahead of schedule. |
| s40 | bullish | A major competitor exited the segment, leaving the firm the dominant supplier. |
| s41 | bullish | Recurring revenue crossed 80% of sales as customer churn hit a record low. |
| s42 | bullish | Won a landmark licensing deal that monetizes its previously idle patent portfolio. |
| s43 | bullish | Order backlog doubled year over year on accelerating enterprise demand. |
| s44 | bullish | Raised full-year guidance after a blowout quarter across every segment. |
| s45 | bullish | A cornerstone investor took a large stake at a premium to the market price. |
| s46 | bullish | Grew its net cash position while retiring high-coupon debt early. |
| s47 | bullish | International expansion beat plan and the new region turned profitable quickly. |
| s48 | bullish | A long-delayed product cleared its final certification with no conditions attached. |
| s49 | bearish | Slashed full-year guidance twice within a single month as demand evaporated. |
| s50 | bearish | A surprise goodwill writedown wiped out the entire quarter's profit. |
| s51 | bearish | Its largest customer terminated a contract and switched to a direct rival. |
| s52 | bearish | An accounting restatement triggered a delayed filing and an SEC inquiry. |
| s53 | bearish | A safety recall halted production of its top-selling product line. |
| s54 | bearish | Debt covenants were breached and lenders demanded immediate renegotiation. |
| s55 | bearish | The CEO and CFO resigned in the same week amid a board dispute. |
| s56 | bearish | A core patent expired and generic rivals immediately flooded the market. |
| s57 | bullish | Revenue beat expectations but the firm guided conservatively citing macro caution. |
| s58 | bearish | Margins held up but unit volumes quietly declined for a third straight quarter. |
| s59 | bullish | A costly restructuring is underway but early customer cohorts show real traction. |
| s60 | bearish | A buyback was announced even as several insiders sold into the news. |
| s61 | bullish | Cash burn continues but a recent raise extends the runway by two years. |
| s62 | bearish | Headline growth impressed but was driven entirely by one large one-off deal. |
| s63 | bullish | The dividend was trimmed to fund a high-return capacity expansion. |
| s64 | bearish | The stock screens cheap but sits in a slowly shrinking end market. |

: Full generated scenario list and authored operational truth. These truths are validation-only and are not visible to the unsupervised evaluator. {#tbl:scenario-truth-full}

Authored prevalence: **40 bullish (`a`) / 24 bearish
(`b`) → true prevalence of `a` = 0.625**. The deck is intentionally
tilted off an even 50/50: an exactly balanced answer key would put the true
prevalence at 1/2, a removable singularity of the error-independent evaluator
where its accuracy equations degenerate to `NaN` (see
[@sec:test-deck]). The
24 clear bearish scenarios keep even the bullish-leaning personas
casting both calls, so no judge collapses to a constant classifier.

## S5 — Reproduction {#sec:reproduction}

All commands run from the repository root. The sweep is reproducible offline
from the cached votes (`outputs/llm/.../cache.json`); only a fresh `--set
refresh=true` run needs a live Ollama.

```bash
# 1. Project test suite (no network, no Ollama, no TeX):
.venv/bin/python -m pytest tests/ -q                      # 108 tests
#   or: cd python && ../.venv/bin/python -m pytest -q     # 22 library tests

# 2. Regenerate the multi-LLM sweep from the cache (JSON + figures):
.venv/bin/python scripts/llm/multi_llm_personas_evaluation.py

# 2b. Regenerate the deterministic synthetic-validation study (no network):
.venv/bin/python scripts/sim/synthetic_recovery_study.py

# 2c. Bootstrap a CI on the real recovery MAE (deterministic, offline):
.venv/bin/python scripts/sim/bootstrap_recovery_ci.py

# 2d. Regenerate the offline statistical validation matrix:
.venv/bin/python scripts/validation/statistical_validation_report.py

# 3. Re-inject the numbers into the manuscript (generated from templates):
.venv/bin/python scripts/manuscript/render.py
.venv/bin/python scripts/manuscript/render.py --check     # verify no drift
.venv/bin/python scripts/z_generate_manuscript_variables.py

# 4. Run the whole orchestrator harness including the LLM category:
.venv/bin/python scripts/run_all.py --include llm

# 5. Re-query live models (requires `ollama serve` and the pulled models):
.venv/bin/python scripts/llm/multi_llm_personas_evaluation.py \
    --set refresh=true --set timeout=45

# 6. Run a custom model list:
.venv/bin/python scripts/llm/multi_llm_personas_evaluation.py \
    --set models='["granite4.1:3b","mistral:latest"]'
```

Lint/format used in this study:
`.venv/bin/ruff check scripts tests` and
`.venv/bin/black --check -l 79 scripts tests`.

## S6 — The evaluability diagnostic, defined {#sec:diagnostic-definition}

Computed from the vote matrix alone — no answer key, no solve
(`ntqr.llm.analysis`):

- **`vote_variation(votes)`** → per judge: `distinct_calls`, `call_counts`,
  `modal_fraction` = (count of the most common call) / Q, and
  `is_constant` = (only one distinct call seen). `modal_fraction == 1.0` ⇔
  `is_constant`.
- **`pairwise_disagreement(votes)`** → for each judge pair, the fraction of the
  Q scenarios on which they differ; plus the mean over the three pairs.
- **`evaluability_diagnostic(votes)`** → `predicted_evaluable` =
  (no judge `is_constant`), plus the named `constant_classifiers` and the mean
  pairwise disagreement.

**Predictive check (this study).** The orchestrator records, in
`summary.json → evaluability_diagnostic`, whether the label-free
`predicted_evaluable` set equals the set of models the ntqr solve actually
evaluated. Result: `diagnostic_matched_outcome = true` — predicted
evaluable = `gemma2:2b`, `gemma3:4b`, `gemma4:latest`, `granite4.1:3b`, `mistral:latest` = actually evaluated.

**Disagreement vs. evaluability (the key contrast).**

| Model | Mean pairwise disagreement | Constant classifier? | Evaluable |
|---|---|---|---|
| smollm2:135m-instruct-q4_K_S | 0.000 | yes (Optimist, Neutral, Pessimist) | no |
| gemma4:latest | 0.031 | no | **yes** |
| granite4.1:3b | 0.062 | no | **yes** |
| gemma2:2b | 0.062 | no | **yes** |
| gemma3:4b | 0.104 | no | **yes** |
| mistral:latest | 0.229 | no | **yes** |

In this low-temperature schema run, disagreement happens to sort the models because the only unevaluable model is a zero-disagreement constant-classifier collapse. That is not the NTQR solvability rule; it is a visible symptom of the per-judge rule. The diagnostic therefore keys on per-judge variation, not on ensemble disagreement.

## S7 — Output file inventory {#sec:output-inventory}

Per sweep, under `outputs/llm/multi_llm_personas_evaluation/`:

- `summary.json` — cross-model ranking, evaluability-diagnostic block, the
  requested/evaluated/failed model sets.
- `cache.json` — shared response cache, keyed by
  `backend / model / temperature / prompt-version / prompt-hash / persona /
  scenario`. New records include every raw attempt, parse method, schema-valid
  flag, invalid reason, parsed call, repair status, and prompt version/hash.
- `disagreement_vs_evaluability.png`, `recovery_mae_by_model.png`,
  `recovered_vs_true_scatter.png`, `evaluability_diagnostic.png`,
  `prompt_compliance.png`, `temperature_schema_comparison.png`,
  `scenario_hardness.png`, `best_model_recovery.png` — the figures, each with a
  `<stem>.data.json` sidecar.
- `models/<sanitized-model>/` — per model:
  `votes.json`, `label_vote_counts.json`, `evaluability_diagnostic.json`,
  `evaluation_error_independent.json`, `evaluation_majority_voting.json`,
  `evaluation_supervised.json`, `comparison.json`, `recovery_stats.json`,
  `comparison_majority_voting.json`, `recovery_stats_majority_voting.json`.

Degenerate models still emit `votes.json`, `label_vote_counts.json`, and
`evaluability_diagnostic.json` (the inputs that explain the failure); they do not
emit the recovery files, because there is no finite solution to record.

Additional generated artifacts:

- `outputs/sim/synthetic_recovery_study/` — deterministic convergence,
  dependence/alarm, tie-break, robustness, and EI-vs-MV sweeps; each embedded
  figure has a parseable `.data.json` sidecar.
- `outputs/sim/bootstrap_recovery_ci/` — scenario-bootstrap summary,
  `bootstrap_recovery_mae.png`, and `bootstrap_recovery_mae.data.json`.
- `outputs/validation/statistical_validation/` — offline validation report
  (`summary.json`), validation-matrix figure, and `validation_matrix.data.json`.
