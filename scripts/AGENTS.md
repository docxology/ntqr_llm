# AGENTS.md - scripts

Child dirs: baselines/, configs/, llm/, manuscript/ (with templates/), ntqr/,
sim/, validation/. Thin orchestrators only — import computation from
`python/src/ntqr/`. The LLM-persona study computation lives in
`python/src/ntqr/llm/` (`analysis`, `recovery_scoring`, `simulation`,
`bootstrap`, `personas`); scripts keep only the LLM client (`llm/_llm.py`),
the collect-votes glue (`llm/_analysis.py::collect_votes`), and the manuscript
token map (`manuscript/_values.py`).

## Design contract

- Scripts do orchestration only: path bootstrap (`_common.ensure_ntqr_importable()`), argparse, logging, and delegated calls into library entrypoints.
- Business/data/plot/analysis logic lives in `python/src/ntqr/` (algebra: `ntqr.r2`/`ntqr.r3`; LLM study: `ntqr.llm`) and is tested there and in `tests/`.
- Each analysis script writes to `outputs/<category>/<script_stem>/`; `run_all.py` is the full-refresh entry point; `configs/` holds input scenario configs.

## Script inventory

| Script | Purpose | Delegates to | Command |
|--------|---------|--------------|---------|
| `run_all.py` | Discover + run orchestrators, pass/fail summary | each category's `main()` | `.venv/bin/python scripts/run_all.py` |
| `z_generate_manuscript_variables.py` | Hydrate manuscript variables for the docxology/template renderer | `manuscript/_values.py`, `manuscript/render.py` | `.venv/bin/python scripts/z_generate_manuscript_variables.py` |
| `baselines/naive_agreement_accuracy.py` | Pairwise-agreement baseline (no labels, non-ntqr) | `_config` labeled-vote loaders | `.venv/bin/python scripts/baselines/naive_agreement_accuracy.py` |
| `baselines/raw_majority_vote.py` | Majority-vote consensus baseline (non-ntqr) | `_config` labeled-vote loaders | `.venv/bin/python scripts/baselines/raw_majority_vote.py` |
| `llm/trader_personas_evaluation.py` | One model, one persona trio: votes -> ntqr recovery | `llm/_llm.py`, `llm/_analysis.py`, `ntqr.llm.analysis`, `ntqr.llm.personas`, `ntqr.r2` | `.venv/bin/python scripts/llm/trader_personas_evaluation.py` |
| `llm/multi_llm_personas_evaluation.py` | Multi-model sweep, ranking, figures (needs Ollama) | same as trader + `ntqr.llm.simulation` figures path | `.venv/bin/python scripts/llm/multi_llm_personas_evaluation.py --set temperature=0.1 --set timeout=45` |
| `ntqr/error_independent_evaluation.py` | Error-independent evaluation of a labeled trio | `ntqr.r2.evaluators.ErrorIndependentEvaluation` | `.venv/bin/python scripts/ntqr/error_independent_evaluation.py` |
| `ntqr/majority_voting_evaluation.py` | Majority-voting evaluation, both a-priori branches | `ntqr.r2.evaluators.MajorityVotingEvaluation` | `.venv/bin/python scripts/ntqr/majority_voting_evaluation.py` |
| `ntqr/supervised_evaluation.py` | Supervised (labeled) reference evaluation | `ntqr.r2.evaluators.SupervisedEvaluation` | `.venv/bin/python scripts/ntqr/supervised_evaluation.py` |
| `ntqr/pair_logical_alarm_plot.py` | Pair logical-alarm figure (Agg, no LaTeX) | `ntqr.plots` | `.venv/bin/python scripts/ntqr/pair_logical_alarm_plot.py` |
| `ntqr/possible_and_consistent_sets.py` | Possible vs. consistent evaluation-set sizes | `ntqr.evaluations` | `.venv/bin/python scripts/ntqr/possible_and_consistent_sets.py` |
| `llm/multi_llm_personas_evaluation.py` | Multi-model sweep, ranking, figures (needs Ollama) | `llm/_llm.py`, `llm/_analysis.py`, `ntqr.llm.analysis`, `ntqr.llm.personas`, `ntqr.r2` | `.venv/bin/python scripts/llm/multi_llm_personas_evaluation.py --set temperature=0.1 --set timeout=45` |
| `sim/synthetic_recovery_study.py` | Deterministic synthetic-ensemble recovery sweeps | `ntqr.llm.simulation` | `.venv/bin/python scripts/sim/synthetic_recovery_study.py` |
| `sim/bootstrap_recovery_ci.py` | Scenario-bootstrap CI on the real recovery MAE | `ntqr.llm.bootstrap`, `ntqr.llm.personas` | `.venv/bin/python scripts/sim/bootstrap_recovery_ci.py` |
| `validation/statistical_validation_report.py` | Offline claim-checking matrix over generated artifacts | `ntqr` (reads `outputs/` artifacts) | `.venv/bin/python scripts/validation/statistical_validation_report.py` |
| `manuscript/render.py` | Hydrate templates with `_values` tokens; `--check` drift gate | `manuscript/_values.py` | `.venv/bin/python scripts/manuscript/render.py` |

Do not put business logic here. Verify: uv run pytest tests/ -v

Upstream policy: see `../AGENTS.md` and /Volumes/external_drive/Git/template/projects/ongoing/AGENTS.md for repo-wide conventions; do not duplicate them here.
