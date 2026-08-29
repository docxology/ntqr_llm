# docs/ — Agent Notes (ntqr_llm)

## Layout

- `README.md` — documentation index and getting-started commands (real,
  executed paths from `scripts/`).
- `architecture.md` — system layers and data flow.
- `rendering_pipeline.md` — the `{{TOKEN}}` injection pipeline and `--check` drift gate.
- `manuscript_guide.md` — authoring rules for the generated manuscript.
- `testing_philosophy.md` — zero-mock / no-fabricated-data policy.
- `VERIFICATION.md` — post-change verification checklist.
- `essays/` — conceptual background on algebraic evaluation.

## Key facts for agents

- The manuscript under `manuscript/` is a **generated artifact**: numbers are
  injected from `outputs/llm/.../summary.json` by `scripts/manuscript/render.py`
  (see `manuscript/config.yaml` header and `docs/manuscript_guide.md`). Never
  hand-edit numeric claims into `manuscript/manuscript.md` or `.tmpl` templates.
- Tests are zero-mock by policy (`docs/testing_philosophy.md`).
- After doc edits, follow `docs/VERIFICATION.md`.

## Maintenance

Plain Markdown; keep this directory's index rows in `README.md` accurate when
adding files. Audit provenance: `docs/AGENTS.md` and the essays link fixes were
added 2026-08-29 by the repository-wide documentation audit. No docs CI gate
beyond the repo's own verification routine was found.
