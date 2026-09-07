# AGENTS.md - scripts/manuscript

Child dir: templates/ (.tmpl sources). render.py hydrates the templates with
_values.py tokens and writes the rendered manuscript files; drift gate:
render.py --check. See docs/rendering_pipeline.md.

`scripts/z_generate_manuscript_variables.py` (scripts/ root) reuses
`_values.py` + `render.py` for the docxology/template renderer's output tree.
`_values.py` imports persona/scenario tables from `python/src/ntqr/llm`
(`ntqr.llm.personas`).

Verify: uv run pytest tests/test_manuscript_render.py -q

Upstream policy: see `../AGENTS.md` and /Volumes/external_drive/Git/template/projects/ongoing/AGENTS.md for repo-wide conventions; do not duplicate them here.
