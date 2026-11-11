# TODO

## Workflowz review — 2026-11-10

- [ ] **NTQRLLM-002 — P3: Make the notebook-copy CLI report failures through its process contract.** `python/src/ntqr/__init__.py:44-50` catches every notebook-copy exception and only prints `Error finding notebooks`, while `python/pyproject.toml:48-49` exposes this operation as the `ntqr-docs` console script. Let failures produce a nonzero exit (and retain the underlying exception) rather than looking successful to automation. Acceptance probe: from the repository root, invoke `uv run ntqr-docs` in an isolated temporary environment where the source notebook directory is unreadable or unavailable, assert nonzero exit and a diagnostic naming the failure, then invoke it in a normal temporary destination and assert copied notebooks plus zero exit. Dependencies/constraints: preserve the existing already-exists behavior unless separately specified; use temporary directories and do not modify checked-in notebooks.
