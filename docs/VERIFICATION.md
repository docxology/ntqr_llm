# Verification — ntqr_llm_personas

Routine to run after any change. All commands use `.venv/bin/python` directly
(the project sets an empty package set in `pyproject.toml`, so `uv run` would try
to build nothing; the vendored `ntqr` is the only editable install).

## 1. Tests (no network, no Ollama)

```bash
.venv/bin/python -m pytest tests/ -q          # project: render + analysis
.venv/bin/python -m pytest python/ -q         # vendored ntqr library
```

Expected: `tests/` all pass (`test_llm_analysis.py`, `test_recovery.py`,
`test_simulation.py`, `test_bootstrap.py`, `test_validation_report.py`,
`test_recovery_validity.py`, `test_research_tools.py`, `test_artifact_io.py`,
`test_modular_figures.py`, `test_ntqr09.py`, `test_manuscript_render.py`);
`python/` all pass. The shared scoring core
(`ntqr.llm.recovery_scoring`), the LLM numeric core (`ntqr.llm.analysis`), the
synthetic harness (`ntqr.llm.simulation`), the scenario bootstrap
(`ntqr.llm.bootstrap`), the offline validation report, and the renderer are all
tested with no I/O to models.

## 2. Lint / format

```bash
.venv/bin/ruff check scripts tests            # All checks passed!
.venv/bin/black --check -l 79 scripts tests   # all files already formatted
```

## 3. Render gate (the determinism check)

```bash
.venv/bin/python scripts/manuscript/render.py --check
```

Expected: `OK — 3 manuscript files match the data.` A non-zero exit means the
on-disk manuscript drifted from a fresh render — re-run `render.py` (and edit the
`.tmpl`, never the `.md`). This is the single most important gate: it proves
every number in the prose still equals the data.

## 4. Data regeneration (needs a running Ollama)

```bash
# resumes from prompt-versioned cache.json; writes summary.json + 10 method/LLM figures
.venv/bin/python scripts/llm/multi_llm_personas_evaluation.py --set temperature=0.1 --set timeout=45
```

The synthetic-validation study (§3.7–3.9) and the scenario bootstrap (§3.1) are
deterministic and need **no network**; regenerate them (and the figures they
back) with:

```bash
.venv/bin/python scripts/sim/synthetic_recovery_study.py   # convergence/alarm/tie-break/robustness
.venv/bin/python scripts/sim/bootstrap_recovery_ci.py      # CI on the real recovery MAE
.venv/bin/python scripts/validation/statistical_validation_report.py  # offline claim checks
.venv/bin/python scripts/manuscript/render.py
.venv/bin/python scripts/z_generate_manuscript_variables.py  # template output tree
(cd "${TEMPLATE_DIR:-../template}" && \
  uv run python scripts/03_render_pdf.py --project ntqr_llm_personas)
```

To restrict to specific models (e.g. only those with cached votes):

```bash
.venv/bin/python scripts/llm/multi_llm_personas_evaluation.py \
  --set models='["granite4.1:3b","mistral:latest","gemma2:2b","gemma3:4b"]'
```

Confirm the models are present and used:

```bash
curl -s http://localhost:11434/api/tags | python3 -c "import json,sys; \
print([m['name'] for m in json.load(sys.stdin)['models']])"
python3 -c "import json; r=json.load(open( \
'outputs/llm/multi_llm_personas_evaluation/summary.json')).get('result',{}); \
print('requested:', r.get('requested_models'))"
```

## 5. Manuscript sanity checks

```bash
python3 - <<'PY'
import re
from pathlib import Path
for path in [Path("docs/manuscript/manuscript.md"), Path("docs/manuscript/supplemental.md"), Path("docs/manuscript/README.md")]:
    assert not re.findall(r"\{\{[A-Z0-9_]+\}\}", path.read_text()), path
print("OK no unresolved manuscript tokens")
PY
grep -c '!\[' docs/manuscript/manuscript.md                             # expect 17 (figures embedded)
for f in evaluation_pipeline best_model_recovery recovered_vs_true_scatter \
         evaluability_diagnostic disagreement_vs_evaluability \
         recovery_mae_by_model prompt_compliance \
         temperature_schema_comparison scenario_hardness \
         validation_accuracy_heatmap; do
  test -f output/figures/$f.png && echo "OK $f" || echo "MISSING $f"
done
for f in synthetic_convergence synthetic_alarm synthetic_tiebreak synthetic_mv_vs_ei; do
  test -f output/figures/$f.png && echo "OK $f" || echo "MISSING $f"
done
test -f output/figures/bootstrap_recovery_mae.png \
  && echo "OK bootstrap" || echo "MISSING bootstrap"
test -f output/figures/bootstrap_recovery_mae.data.json \
  && echo "OK bootstrap sidecar" || echo "MISSING bootstrap sidecar"
test -f output/figures/validation_matrix.png \
  && echo "OK validation matrix" || echo "MISSING validation matrix"
test -f output/figures/validation_matrix.data.json \
  && echo "OK validation sidecar" || echo "MISSING validation sidecar"
test -f output/figures/physical_recovery_audit.png \
  && echo "OK physical audit" || echo "MISSING physical audit"
test -f output/figures/physical_recovery_audit.data.json \
  && echo "OK physical audit sidecar" || echo "MISSING physical audit sidecar"
```

## Checklist

| Check | Pass criteria |
|-------|---------------|
| Tests | `tests/` and `python/` all pass |
| Lint | ruff clean; black clean (`-l 79`) |
| Render gate | `render.py --check` → "OK — 3 … match the data" |
| No leftover tokens | `grep -c '{{'` is 0 in all three rendered files |
| Figures embedded | 17 `![]()` in `manuscript.md`; 10 method/LLM + 4 synthetic + 1 bootstrap + 1 validation + 1 physical-recovery audit PNG in `output/figures/`, with parseable data sidecars |
| Validation matrix | `scripts/validation/statistical_validation_report.py` reports 9/9 checks passed |
| Models | `summary.json` `requested_models` are pulled in Ollama |
| Template PDF | `output/pdf/ntqr_llm_personas_combined.pdf` exists after the targeted template render, has no unresolved `{{TOKEN}}` text, uses red hyperlinks, and uses the compact `margin=0.58in` geometry from `docs/manuscript/config.yaml` |

## Common issues

| Symptom | Fix |
|---------|-----|
| `missing summary.json under …` | run the sweep (§4); a fresh checkout has no data |
| `none of the requested models are available` | `ollama pull <model>`; check `curl …/api/tags` |
| `--check` reports DRIFT | re-run `render.py`; do not hand-edit `docs/manuscript/*.md` |
| a model runs for minutes per call | large model (e.g. `gemma4:latest`); restrict `--set models=[…]` to cached/fast ones |
