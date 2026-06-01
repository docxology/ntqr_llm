# outputs/llm/

Generated results from [`../../scripts/llm/`](../../scripts/llm/). One subfolder
per orchestrator. Safe to delete and regenerate.

## trader_personas_evaluation/

Three LLM trader personas judge 32 market scenarios; their calls become noisy
classifier votes that `ntqr` evaluates with and without labels.

| File | What it holds |
|------|---------------|
| `comparison.json` | **the punchline** — per-persona true vs. unsupervised-recovered accuracy, and true vs. recovered prevalence |
| `votes.json` | each persona's call per scenario, the backend/model, and `fallback_parses` |
| `label_vote_counts.json` | the votes in ntqr's dataset format — replay any `scripts/ntqr/` evaluator on it via `--dataset-file` |
| `evaluation_error_independent.json` | unsupervised `ErrorIndependentEvaluation` (two solutions) |
| `evaluation_majority_voting.json` | unsupervised `MajorityVotingEvaluation` baseline |
| `evaluation_supervised.json` | `SupervisedEvaluation` using the authored truth (the check) |
| `cache.json` | cached model responses keyed by `backend\|model\|persona\|scenario` |

## multi_llm_personas_evaluation/

The same three personas run through several models, scored on how well the
*unsupervised* algebra recovered each persona's true accuracy.

| File | What it holds |
|------|---------------|
| `summary.json` | **the punchline** — models ranked by recovery MAE, plus prevalence error, pairwise disagreement, the `evaluability_diagnostic` block, skipped/failed models |
| `recovery_mae_by_model.png` | recovery MAE per model (lower = label-free recovery closer to truth) |
| `recovered_vs_true_scatter.png` | recovered vs true accuracy; points hug `y = x` when the math tracks truth |
| `evaluability_diagnostic.png` | per-persona modal-call fraction per model; red bars at 1.0 are constant classifiers that make the algebra unsolvable |
| `best_model_recovery.png` | true vs recovered per-persona, per-label accuracy for the evaluable model (MAE in title) |
| `models/<model>/` | per-model `votes`, `evaluation_*`, `comparison`, `recovery_stats`, `comparison_majority_voting`, `recovery_stats_majority_voting`, `evaluability_diagnostic`, `label_vote_counts` |
| `cache.json` | model responses, shared across models, keyed by `backend\|model\|persona\|scenario` |

A model with degenerate votes (unsolvable algebra) appears under
`failed_models` in `summary.json` and is excluded from the ranking and plots.
The non-obvious finding: those failed models tend to disagree **more** than the
one evaluable model — evaluability is gated by a per-judge condition (no
constant classifier), not by ensemble disagreement. The label-free
`evaluability_diagnostic` predicts which models the algebra can solve with no
answer key (`diagnostic_matched_outcome`). Full write-up in
[`../../manuscript/`](../../manuscript/).

## Reading comparison.json

`recovered_*` comes from `ErrorIndependentEvaluation` using only the votes — no
answer key. Of its two algebraic solutions, the higher-mean-accuracy one is kept
(the "experts are better than random" assumption). `true_*` uses the authored
ground truth and is shown only so you can see how close the label-free recovery
gets.

## Regenerate

```bash
python scripts/llm/trader_personas_evaluation.py        # needs ollama (or --set backend=anthropic)
python scripts/llm/multi_llm_personas_evaluation.py     # multi-model sweep + plots
```
