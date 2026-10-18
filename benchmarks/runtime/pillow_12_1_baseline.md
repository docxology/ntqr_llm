# Runtime measurement receipt

Python 3.15.0; CPython; arm64.

Source binding: `13aa53359c407c5871413fdc2414a0f947752356935685fcdde5b2639780cc5c`.

All timing values below come from unprofiled samples after the declared warmup. Allocation peaks come from a separate instrumented pass.

| Workload | Median CPU ms | Median wall ms | Wall range ms | Python peak KiB |
|---|---:|---:|---:|---:|
| recovery | 52.394 | 694.296 | 388.883–1133.881 | 190.9 |
| enumeration | 0.881 | 0.881 | 0.832–1.224 | 26.3 |
| validity | 1.845 | 1.843 | 1.772–1.976 | 11.8 |

Ranges are descriptive, not confidence intervals. Parent CPU excludes external solver CPU; Python allocations exclude native heaps and process RSS.

## Workloads and traces

### recovery

Real error-independent recovery from 96 seeded synthetic observations

Input SHA-256: `5f0f7fa3349c6480c08b80a4fce0d408e00bbe9c113a9f5dc8362d501d2ab309`. Tracing backend: `profiling.tracing`; 336413 instrumented calls.

Top cumulative tracing frames (instrumented time; nested frames overlap):

- `python/src/ntqr/llm/profile_workloads.py:31 <lambda>`: 1 calls, 0.825981 seconds cumulative.
- `python/src/ntqr/llm/simulation.py:201 solve_recovery`: 1 calls, 0.825971 seconds cumulative.
- `python/src/ntqr/r2/evaluators.py:376 __init__`: 1 calls, 0.825281 seconds cumulative.
- `site-packages/sympy/simplify/simplify.py:435 simplify`: 14 calls, 0.801079 seconds cumulative.
- `python/src/ntqr/r2/evaluators.py:459 classifier_a_label_accuracy`: 6 calls, 0.398886 seconds cumulative.

### enumeration

Exact consistent-set enumeration, counting without retaining sample points

Input SHA-256: `8cca71212ebb7eb460dfd15a8cad5ba36659441e61929909d2a48a8c4028465d`. Tracing backend: `profiling.tracing`; 6251 instrumented calls.

Top cumulative tracing frames (instrumented time; nested frames overlap):

- `python/src/ntqr/llm/profile_workloads.py:37 <lambda>`: 1 calls, 0.002522 seconds cumulative.
- `python/src/ntqr/llm/consistency.py:39 summarize_sets`: 1 calls, 0.002514 seconds cumulative.
- `python/src/ntqr/evaluations.py:524 set_generator`: 4 calls, 0.001320 seconds cumulative.
- `python/src/ntqr/evaluations.py:606 <genexpr>`: 9 calls, 0.000957 seconds cumulative.
- `python/src/ntqr/evaluations.py:1005 is_valid_point`: 3 calls, 0.000918 seconds cumulative.

### validity

Answer-key-free physical audit of a seeded three-judge vote matrix

Input SHA-256: `b3dc49dc543aaebc8dfa1e63981f9f972fa50e5905541a9b6f9e532e1c68da8a`. Tracing backend: `profiling.tracing`; 22763 instrumented calls.

Top cumulative tracing frames (instrumented time; nested frames overlap):

- `python/src/ntqr/llm/profile_workloads.py:43 <lambda>`: 1 calls, 0.040258 seconds cumulative.
- `python/src/ntqr/llm/validity_audit.py:17 audit_votes`: 1 calls, 0.040246 seconds cumulative.
- `python/src/ntqr/r2/evaluators.py:376 __init__`: 1 calls, 0.040024 seconds cumulative.
- `python/src/ntqr/r2/evaluators.py:459 classifier_a_label_accuracy`: 6 calls, 0.036433 seconds cumulative.
- `python/src/ntqr/r2/datasketches.py:328 classifier_label_frequency`: 231 calls, 0.035509 seconds cumulative.

## Availability and limits

Native sampling: not requested. Cold import profile: available.

The JSON receipt retains individual samples, bounded caller–callee edges, allocation sites, dependency/build context, input hashes and source bindings. These software measurements are separate from scientific model observations.
