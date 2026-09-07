# scripts/

Analysis and orchestration scripts. Every script is a **thin orchestrator**:
argparse + path bootstrap (shared helpers `_common.py`/`_config.py`), logging,
and a single delegated call into `python/src/ntqr/`. `run_all.py` is the
full-refresh entry point; `configs/` holds input scenario configs. Grouped by
category — each script writes to `outputs/<category>/<script_stem>/`.

## Inventory

| Script | Purpose | Delegates to | Command |
|--------|---------|--------------|---------|
| `run_all.py` | Auto-discover and run every orchestrator, pass/fail summary | each category's `main()` | `.venv/bin/python scripts/run_all.py` |
| `z_generate_manuscript_variables.py` | Hydrate the manuscript token map for the docxology/template renderer; write section files, supplemental, figures + sidecars | `manuscript/_values.py`, `manuscript/render.py` | `.venv/bin/python scripts/z_generate_manuscript_variables.py` |
| `baselines/naive_agreement_accuracy.py` | Pairwise-agreement baseline (no labels, non-ntqr) | `_config` labeled-vote loaders | `.venv/bin/python scripts/baselines/naive_agreement_accuracy.py` |
| `baselines/raw_majority_vote.py` | Majority-vote consensus baseline (non-ntqr) | `_config` labeled-vote loaders | `.venv/bin/python scripts/baselines/raw_majority_vote.py` |
| `llm/trader_personas_evaluation.py` | One model, one trader-persona trio: collect votes, evaluate with ntqr | `llm/_llm.py`, `llm/_analysis.py`, `ntqr.llm.analysis`, `ntqr.llm.personas`, `ntqr.r2` | `.venv/bin/python scripts/llm/trader_personas_evaluation.py` |
| `llm/multi_llm_personas_evaluation.py` | Multi-model persona sweep, recovery ranking, figures (needs Ollama) | `llm/_llm.py`, `llm/_analysis.py`, `ntqr.llm.analysis`, `ntqr.llm.personas`, `ntqr.r2` | `.venv/bin/python scripts/llm/multi_llm_personas_evaluation.py` |
| `ntqr/error_independent_evaluation.py` | Error-independent evaluation of a labeled vote trio | `ntqr.r2.evaluators.ErrorIndependentEvaluation` | `.venv/bin/python scripts/ntqr/error_independent_evaluation.py` |
| `ntqr/majority_voting_evaluation.py` | Majority-voting evaluation, both a-priori branches | `ntqr.r2.evaluators.MajorityVotingEvaluation` | `.venv/bin/python scripts/ntqr/majority_voting_evaluation.py` |
| `llm/multi_llm_personas_evaluation.py` | Multi-model persona sweep, recovery ranking, figures (needs Ollama) | `llm/_llm.py`, `llm/_analysis.py`, `ntqr.llm.analysis`, `ntqr.llm.personas`, `ntqr.r2` | `.venv/bin/python scripts/llm/multi_llm_personas_evaluation.py` |
| `ntqr/pair_logical_alarm_plot.py` | Pair logical-alarm figure (Agg, no LaTeX) | `ntqr.plots` | `.venv/bin/python scripts/ntqr/pair_logical_alarm_plot.py` |
| `ntqr/possible_and_consistent_sets.py` | Possible vs. consistent evaluation-set sizes | `ntqr.evaluations` | `.venv/bin/python scripts/ntqr/possible_and_consistent_sets.py` |
| `ntqr/single_classifier_alarm.py` | Single-classifier logical alarm (R=2 and R=3) | `ntqr.alarms`, `ntqr.r2`, `ntqr.r3` | `.venv/bin/python scripts/ntqr/single_classifier_alarm.py` |
| `sim/synthetic_recovery_study.py` | Deterministic synthetic-ensemble recovery sweeps + figures | `ntqr.llm.simulation` | `.venv/bin/python scripts/sim/synthetic_recovery_study.py` |
| `sim/bootstrap_recovery_ci.py` | Scenario-bootstrap CI on the real model's recovery MAE | `ntqr.llm.bootstrap`, `ntqr.llm.personas` | `.venv/bin/python scripts/sim/bootstrap_recovery_ci.py` |
| `validation/statistical_validation_report.py` | Offline validation matrix over the paper's load-bearing claims | `ntqr` (reads `outputs/` artifacts) | `.venv/bin/python scripts/validation/statistical_validation_report.py` |
| `manuscript/render.py` | Hydrate `manuscript/templates/*.tmpl` with `_values` tokens; `--check` drift gate | `manuscript/_values.py` | `.venv/bin/python scripts/manuscript/render.py` |

`baselines/` and `configs/` also carry their own `README.md`/`AGENTS.md`;
`scripts/_common.py` (paths, JSON + provenance sidecars) and
`scripts/_config.py` (scenario/config loading) are shared helpers, not
orchestrators.
