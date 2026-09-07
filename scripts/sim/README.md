# sim/

Deterministic synthetic studies delegating all computation to
`python/src/ntqr/llm/`: `synthetic_recovery_study.py` (recovery across
synthetic ensemble configurations via `ntqr.llm.simulation`) and
`bootstrap_recovery_ci.py` (bootstrap CIs on recovery MAE via
`ntqr.llm.bootstrap`). Write to `outputs/sim/`.
