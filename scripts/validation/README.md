# validation/

`statistical_validation_report.py` — offline claim-check: recomputes/reloads result JSONs in `outputs/` and emits the validation matrix backing manuscript claims.

`recovery_validity_report.py` audits recorded votes for singular moments and complete physical NTQR roots, with source hashes and no model calls or answer key.

Offline publication figures: `render_recorded_figures.py` renders the model panels from persisted votes/comparisons; `render_supporting_figures.py` renders stored simulation/bootstrap/check summaries. Both export PNG and SVG without model calls.
