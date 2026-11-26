# Verify the study offline

From the repository root, install the development environment. The default
Python is pinned in [.python-version](../.python-version). The CI matrix
checks Python 3.12, 3.13, and 3.15.

```bash
uv sync --locked --extra dev
uv run --no-sync pytest
uv run --no-sync python scripts/check_research.py
uv run --no-sync python scripts/manuscript/render.py --check
uv run --no-sync python scripts/z_generate_manuscript_variables.py --check
```

The manuscript checks recompute the prose and bridge payload in memory and
compare them to the tracked artifacts. The research check validates citations
and the evidence map. See [the rendering pipeline](rendering_pipeline.md) to
qualify the PDF and HTML with the pinned Template revision.

## Recompute the physical-solution audit from recorded votes

```bash
audit_dir=$(mktemp -d)
uv run --no-sync python scripts/validation/recovery_validity_report.py \
  --output-dir "$audit_dir"
```

The command reads checked-in `outputs/llm/multi_llm_personas_evaluation/models/*/votes.json`
and writes a new summary and plot into the temporary directory. Each result
binds its input by SHA-256. The audit takes no answer key. Passing establishes
the presence of a physical solution under the error-independence model.
See [the recovery guide](recovery_validity.md) for the model assumptions.

The recorded observations, synthetic studies, bootstrap results, and
publication files remain distinct from this replay. See [research integrity](research_integrity.md)
to redraw figures from the recorded summaries and [configuration](configuration.md)
for the separate live collection operation.
