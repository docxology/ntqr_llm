# Runtime measurement receipt

Python 3.15.0; CPython; arm64.

Source binding: `8ac3b9171ce0c3f1348aa346fa73fa3b3dabdce95772b4de2b3e2efb85cc5736`.

All timing values below come from unprofiled samples after the declared warmup. Allocation peaks come from a separate instrumented pass.

| Workload | Median CPU ms | Median wall ms | Wall range ms | Python peak KiB |
|---|---:|---:|---:|---:|
| recovery | 38.929 | 43.733 | 37.213–50.879 | 190.8 |
| enumeration | 0.869 | 0.869 | 0.765–1.072 | 25.8 |
| validity | 1.861 | 1.860 | 1.834–1.874 | 11.8 |

Ranges are descriptive, not confidence intervals. Parent CPU excludes external solver CPU; Python allocations exclude native heaps and process RSS.

## Workloads and traces

### recovery

Real error-independent recovery from 96 seeded synthetic observations

Input SHA-256: `5f0f7fa3349c6480c08b80a4fce0d408e00bbe9c113a9f5dc8362d501d2ab309`. Tracing backend: `profiling.tracing`; 336413 instrumented calls.

Top cumulative tracing frames (instrumented time; nested frames overlap):

- `python/src/ntqr/llm/profile_workloads.py:31 <lambda>`: 1 calls, 0.116428 seconds cumulative.
- `python/src/ntqr/llm/simulation.py:201 solve_recovery`: 1 calls, 0.116420 seconds cumulative.
- `python/src/ntqr/r2/evaluators.py:376 __init__`: 1 calls, 0.115837 seconds cumulative.
- `site-packages/sympy/simplify/simplify.py:435 simplify`: 14 calls, 0.108257 seconds cumulative.
- `python/src/ntqr/r2/evaluators.py:505 classifier_b_label_accuracy`: 6 calls, 0.046630 seconds cumulative.

### enumeration

Exact consistent-set enumeration, counting without retaining sample points

Input SHA-256: `8cca71212ebb7eb460dfd15a8cad5ba36659441e61929909d2a48a8c4028465d`. Tracing backend: `profiling.tracing`; 6251 instrumented calls.

Top cumulative tracing frames (instrumented time; nested frames overlap):

- `python/src/ntqr/llm/profile_workloads.py:37 <lambda>`: 1 calls, 0.002675 seconds cumulative.
- `python/src/ntqr/llm/consistency.py:39 summarize_sets`: 1 calls, 0.002668 seconds cumulative.
- `python/src/ntqr/evaluations.py:524 set_generator`: 4 calls, 0.001486 seconds cumulative.
- `python/src/ntqr/evaluations.py:606 <genexpr>`: 9 calls, 0.001134 seconds cumulative.
- `python/src/ntqr/evaluations.py:1005 is_valid_point`: 3 calls, 0.000905 seconds cumulative.

### validity

Answer-key-free physical audit of a seeded three-judge vote matrix

Input SHA-256: `b3dc49dc543aaebc8dfa1e63981f9f972fa50e5905541a9b6f9e532e1c68da8a`. Tracing backend: `profiling.tracing`; 22763 instrumented calls.

Top cumulative tracing frames (instrumented time; nested frames overlap):

- `python/src/ntqr/llm/profile_workloads.py:43 <lambda>`: 1 calls, 0.005493 seconds cumulative.
- `python/src/ntqr/llm/validity_audit.py:17 audit_votes`: 1 calls, 0.005483 seconds cumulative.
- `python/src/ntqr/r2/evaluators.py:376 __init__`: 1 calls, 0.005242 seconds cumulative.
- `python/src/ntqr/r2/datasketches.py:475 label_pairs_frequency_moments`: 26 calls, 0.003113 seconds cumulative.
- `python/src/ntqr/r2/datasketches.py:442 pair_frequency_moment`: 78 calls, 0.003092 seconds cumulative.

## Availability and limits

Native sampling: not requested. Cold import profile: available.

The JSON receipt retains individual samples, bounded caller–callee edges, allocation sites, dependency/build context, input hashes and source bindings. These software measurements are separate from scientific model observations.
