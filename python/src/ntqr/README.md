# ntqr

Logical and algebraic evaluation of noisy decision makers (classifiers) on
**unlabeled** test data. The package treats evaluation as exact symbolic
algebra over integer decision-event counts (SymPy) — it needs no ground-truth
labels and assumes no probability model.

## Quick import

```python
import ntqr
ntqr.__version__                      # "0.9"
from ntqr import ErrorIndependentEvaluation, MajorityVotingEvaluation, SupervisedEvaluation
from ntqr import uciadult_label_counts, TrioLabelVoteCounts, TrioVoteCounts
```

## Modules

| Module | Purpose |
|--------|---------|
| `statistics.py` | Construct the Q (answer-key) and R (response) count variables |
| `raxioms.py` | The invariant axioms classification tests must obey, in label-response integer spaces |
| `algebra.py` | Helpers for manipulating linear axiom expressions |
| `labels.py` | `Label` / `Labels` types for variable subscripts |
| `testsketches.py` | Question-aligned test sketches (observed agreements/disagreements) |
| `evaluations.py` | Axiom-based evaluation superclasses; possible/consistent sets |
| `alarms.py` | Logical alarms detecting misaligned classifiers |
| `plots.py` | Plotting utilities (uses LaTeX by default) |
| `r2/` | R=2 binary-classification implementation |
| `r3/` | R=3 three-class implementation |
| `llm/` | LLM-persona study: response parsing + vote analysis (`analysis`), recovery-scoring core (`recovery_scoring`), synthetic + bootstrap validation harnesses (`simulation`, `bootstrap`), persona/scenario tables (`personas`) |
| `notebooks/` | Tutorials and worked examples |

## Worked examples

Runnable, self-contained orchestrators that call these methods on the bundled
UCI Adult data live at the repo root under
[`../../../../scripts/ntqr/`](../../../../scripts/ntqr/), writing results to
[`../../../../outputs/ntqr/`](../../../../outputs/ntqr/).
