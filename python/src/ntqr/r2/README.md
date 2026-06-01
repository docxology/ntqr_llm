# ntqr.r2

R=2 (binary classification) logic and algebra. A single classifier's
evaluation is the integer triple `(Q_a, R_a_a, R_b_b)`.

| Module | Purpose |
|--------|---------|
| `evaluators.py` | `ErrorIndependentEvaluation`, `MajorityVotingEvaluation`, `SupervisedEvaluation` (re-exported at package root) |
| `evaluations.py` | Finite-set evaluation computation, `SingleClassifierEvaluations` |
| `datasketches.py` | Vote-count containers with exact + float views |
| `raxioms.py` | Axioms in integer space |
| `paxioms.py` | Axioms in percentage (rational) space |
| `examples.py` | `uciadult_label_counts` bundled fixture data |
| `uci_adult_20240415.py` | Source UCI Adult evaluation data |
| `plots.py` | R=2 evaluation plots |

Try it: `python -m ntqr.r2.evaluators` runs the module's demo. Worked
orchestrators are at [`../../../../../scripts/ntqr/`](../../../../../scripts/ntqr/).
