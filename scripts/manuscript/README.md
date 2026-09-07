# manuscript/

Manuscript rendering: `render.py` hydrates `templates/*.tmpl` (manuscript,
README, supplemental) with values from `_values.py` (persona/scenario tables
from `ntqr.llm.personas`; numbers pulled from `outputs/` summaries).
`scripts/z_generate_manuscript_variables.py` reuses `_values.py` and
`render.py` to produce the docxology/template renderer's variables and output
tree.
