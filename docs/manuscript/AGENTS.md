# AGENTS.md - manuscript

Leaf: authored manuscript source - manuscript.md, preamble.md, supplemental.md, references.bib, config.yaml (paper metadata). README.md explains authoring.

Source layer: numbers must appear as {{TOKEN}} placeholders resolved at render time (see docs/rendering_pipeline.md); never paste generated values directly into prose. output/ holds the rendered result. Drift gate: scripts/manuscript/render.py --check.

Upstream policy: see `../AGENTS.md` for this repository's conventions. The shared policy that governs template projects is maintained in the template repository, not restated here.
