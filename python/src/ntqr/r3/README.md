# ntqr.r3

R=3 (three-class) logic and algebra. With N responders there are 3^N decision
patterns; the axioms are constructed from single-responder (N=1) statistics
upward.

| Module | Purpose |
|--------|---------|
| `raxioms.py` | Axioms of evaluation for R=3 tests |
| `evaluations.py` | Finite-set evaluations for three responses |

This tree mirrors the more complete [`../r2/`](../r2/) implementation. The
single-classifier alarm orchestrator exercises R=3 via `alarm_r3.json` — see
[`../../../../../scripts/ntqr/`](../../../../../scripts/ntqr/).
