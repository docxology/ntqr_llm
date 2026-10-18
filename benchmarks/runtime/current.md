# Runtime measurement receipt

Python 3.15.0; CPython; arm64.

Source binding: `a9c69de98cd1a3f853dfd6066974fa3b063d34ac7ab96f5af3a32850cdb11bd8`.

All timing values below come from unprofiled samples after the declared warmup. Allocation peaks come from a separate instrumented pass.

| Workload | Median CPU ms | Median wall ms | Wall range ms | Python peak KiB |
|---|---:|---:|---:|---:|
| recovery | 34.711 | 34.986 | 31.669–36.482 | 191.2 |
| enumeration | 0.867 | 0.866 | 0.845–1.237 | 25.5 |
| validity | 1.855 | 1.869 | 1.623–2.356 | 11.8 |

Ranges are descriptive, not confidence intervals. Parent CPU excludes external solver CPU; Python allocations exclude native heaps and process RSS.

## Same-context descriptive comparison

Ratios are current/baseline; values below one indicate lower observed cost. No significance or causal speedup claim is implied.

| Workload | Comparable | CPU ratio | Wall ratio | Reasons |
|---|---|---:|---:|---|
| recovery | True | 0.892 | 0.800 | |
| enumeration | True | 0.998 | 0.997 | |
| validity | True | 0.997 | 1.005 | |

## Workloads and traces

### recovery

Real error-independent recovery from 96 seeded synthetic observations

Input SHA-256: `5f0f7fa3349c6480c08b80a4fce0d408e00bbe9c113a9f5dc8362d501d2ab309`. Tracing backend: `profiling.tracing`; 336413 instrumented calls.

Top cumulative tracing frames (instrumented time; nested frames overlap):

- `python/src/ntqr/llm/profile_workloads.py:31 <lambda>`: 1 calls, 0.118318 seconds cumulative.
- `python/src/ntqr/llm/simulation.py:201 solve_recovery`: 1 calls, 0.118312 seconds cumulative.
- `python/src/ntqr/r2/evaluators.py:376 __init__`: 1 calls, 0.117724 seconds cumulative.
- `.venv315/lib/python3.15/site-packages/sympy/simplify/simplify.py:435 simplify`: 14 calls, 0.109651 seconds cumulative.
- `python/src/ntqr/r2/evaluators.py:505 classifier_b_label_accuracy`: 6 calls, 0.052048 seconds cumulative.

### enumeration

Exact consistent-set enumeration, counting without retaining sample points

Input SHA-256: `8cca71212ebb7eb460dfd15a8cad5ba36659441e61929909d2a48a8c4028465d`. Tracing backend: `profiling.tracing`; 6145 instrumented calls.

Top cumulative tracing frames (instrumented time; nested frames overlap):

- `python/src/ntqr/llm/profile_workloads.py:37 <lambda>`: 1 calls, 0.002390 seconds cumulative.
- `python/src/ntqr/llm/consistency.py:39 summarize_sets`: 1 calls, 0.002383 seconds cumulative.
- `python/src/ntqr/evaluations.py:524 set_generator`: 4 calls, 0.001315 seconds cumulative.
- `python/src/ntqr/evaluations.py:606 <genexpr>`: 9 calls, 0.000941 seconds cumulative.
- `python/src/ntqr/evaluations.py:1005 is_valid_point`: 3 calls, 0.000852 seconds cumulative.

### validity

Answer-key-free physical audit of a seeded three-judge vote matrix

Input SHA-256: `b3dc49dc543aaebc8dfa1e63981f9f972fa50e5905541a9b6f9e532e1c68da8a`. Tracing backend: `profiling.tracing`; 22763 instrumented calls.

Top cumulative tracing frames (instrumented time; nested frames overlap):

- `python/src/ntqr/llm/profile_workloads.py:43 <lambda>`: 1 calls, 0.005958 seconds cumulative.
- `python/src/ntqr/llm/validity_audit.py:17 audit_votes`: 1 calls, 0.005946 seconds cumulative.
- `python/src/ntqr/r2/evaluators.py:376 __init__`: 1 calls, 0.005706 seconds cumulative.
- `python/src/ntqr/r2/datasketches.py:475 label_pairs_frequency_moments`: 26 calls, 0.003507 seconds cumulative.
- `python/src/ntqr/r2/datasketches.py:442 pair_frequency_moment`: 78 calls, 0.003484 seconds cumulative.

## Availability and limits

Native sampling: unavailable. Cold import profile: available.

The JSON receipt retains individual samples, bounded caller–callee edges, allocation sites, dependency/build context, input hashes and source bindings. These software measurements are separate from scientific model observations.
