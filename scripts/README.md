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
| `profile_runtime.py` | Bounded real-workload runtime profiling with source-bound receipts | `ntqr.llm.runtime_profile`, `ntqr.llm.profile_workloads` | `.venv/bin/python scripts/profile_runtime.py` |
| `check_research.py` | Offline reference/evidence integrity checks | `ntqr.llm.scholarship` | `.venv/bin/python scripts/check_research.py` |
| `baselines/naive_agreement_accuracy.py` | Pairwise-agreement baseline (no labels, non-ntqr) | `_config` labeled-vote loaders | `.venv/bin/python scripts/baselines/naive_agreement_accuracy.py` |
| `baselines/raw_majority_vote.py` | Majority-vote consensus baseline (non-ntqr) | `_config` labeled-vote loaders | `.venv/bin/python scripts/baselines/raw_majority_vote.py` |
| `llm/trader_personas_evaluation.py` | One model, one trader-persona trio: collect votes, evaluate with ntqr | `llm/_llm.py`, `llm/_analysis.py`, `ntqr.llm.analysis`, `ntqr.llm.personas`, `ntqr.r2` | `.venv/bin/python scripts/llm/trader_personas_evaluation.py` |
| `llm/multi_llm_personas_evaluation.py` | Multi-model persona sweep, recovery ranking, figures (needs Ollama) | `llm/_llm.py`, `llm/_analysis.py`, `ntqr.llm.analysis`, `ntqr.llm.personas`, `ntqr.r2` | `.venv/bin/python scripts/llm/multi_llm_personas_evaluation.py` |
| `ntqr/error_independent_evaluation.py` | Error-independent evaluation of a labeled vote trio | `ntqr.r2.evaluators.ErrorIndependentEvaluation` | `.venv/bin/python scripts/ntqr/error_independent_evaluation.py` |
| `ntqr/majority_voting_evaluation.py` | Majority-voting evaluation, both a-priori branches | `ntqr.r2.evaluators.MajorityVotingEvaluation` | `.venv/bin/python scripts/ntqr/majority_voting_evaluation.py` |
| `ntqr/supervised_evaluation.py` | Supervised (labeled) reference evaluation | `ntqr.r2.evaluators.SupervisedEvaluation` | `.venv/bin/python scripts/ntqr/supervised_evaluation.py` |
| `ntqr/pair_logical_alarm_plot.py` | Pair logical-alarm figure (Agg, no LaTeX) | `ntqr.plots` | `.venv/bin/python scripts/ntqr/pair_logical_alarm_plot.py` |
| `ntqr/possible_and_consistent_sets.py` | Possible vs. consistent evaluation-set sizes | `ntqr.evaluations` | `.venv/bin/python scripts/ntqr/possible_and_consistent_sets.py` |
| `ntqr/single_classifier_alarm.py` | Single-classifier logical alarm (R=2 and R=3) | `ntqr.alarms`, `ntqr.r2`, `ntqr.r3` | `.venv/bin/python scripts/ntqr/single_classifier_alarm.py` |
| `sim/synthetic_recovery_study.py` | Deterministic synthetic-ensemble recovery sweeps + figures | `ntqr.llm.simulation` | `.venv/bin/python scripts/sim/synthetic_recovery_study.py` |
| `sim/bootstrap_recovery_ci.py` | Scenario-bootstrap CI on the real model's recovery MAE | `ntqr.llm.bootstrap`, `ntqr.llm.personas` | `.venv/bin/python scripts/sim/bootstrap_recovery_ci.py` |
| `validation/statistical_validation_report.py` | Offline validation matrix over the paper's load-bearing claims | `ntqr` (reads `outputs/` artifacts) | `.venv/bin/python scripts/validation/statistical_validation_report.py` |
| `validation/recovery_validity_report.py` | Physical-recovery audit of recorded votes, no answer key or model calls | `ntqr.llm.validity_audit` | `.venv/bin/python scripts/validation/recovery_validity_report.py` |
| `validation/render_recorded_figures.py` | Model figures from persisted responses/comparisons only | `ntqr.llm.study_records`, `ntqr.llm.visualization` | `.venv/bin/python scripts/validation/render_recorded_figures.py` |
| `validation/render_supporting_figures.py` | Supporting figures from recorded summaries, without rerunning experiments | `ntqr.llm.supporting_figures` | `.venv/bin/python scripts/validation/render_supporting_figures.py` |
| `manuscript/render.py` | Hydrate `manuscript/templates/*.tmpl` with `_values` tokens; `--check` drift gate | `manuscript/_values.py` | `.venv/bin/python scripts/manuscript/render.py` |

`baselines/` and `configs/` also carry their own `README.md`/`AGENTS.md`;
`scripts/_common.py` (paths, JSON + provenance sidecars) and
`scripts/_config.py` (scenario/config loading) are shared helpers, not
orchestrators.
