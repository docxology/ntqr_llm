# outputs/

Generated results from the orchestrators in [`../scripts/`](../scripts/). One
subfolder per orchestrator, grouped by category.

| Path | Produced by |
|------|-------------|
| [`ntqr/`](ntqr/) | `scripts/ntqr/*.py` — results from the real `ntqr` methods |
| [`baselines/`](baselines/) | `scripts/baselines/*.py` — non-ntqr comparison results |
| [`llm/`](llm/) | `scripts/llm/*.py` — LLM persona votes evaluated by `ntqr` (true vs. unsupervised-recovered accuracy) |
| [`sim/`](sim/) | `scripts/sim/*.py` — deterministic synthetic validation and bootstrap summaries |
| [`validation/`](validation/) | `scripts/validation/*.py` — offline manuscript claim-check matrix |

## This directory is generated

Everything here except this README is produced by code and safe to regenerate.
The full publication artifact refresh is:

```bash
.venv/bin/python scripts/llm/multi_llm_personas_evaluation.py --set temperature=0.1 --set timeout=45
.venv/bin/python scripts/sim/synthetic_recovery_study.py
.venv/bin/python scripts/sim/bootstrap_recovery_ci.py
.venv/bin/python scripts/validation/statistical_validation_report.py
.venv/bin/python scripts/manuscript/render.py
.venv/bin/python scripts/z_generate_manuscript_variables.py
```

Each orchestrator writes to `outputs/<category>/<script_stem>/`, creating the
folder on first run. JSON files carry a `generated_at` timestamp and a
`result` payload; algebraic (SymPy) values appear as `{"exact","float"}`.
