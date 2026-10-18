"""Bounded, source-bound runtime measurements; tracing and memory are separate passes.

Supports Python 3.12+; Python 3.15 uses its native profiling.tracing module.
No profiler artifacts are loaded from pickle or accepted as executable inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import os
import sys
import sysconfig
import pstats
import statistics
import time
import tracemalloc
from typing import Callable


@dataclass(frozen=True)
class Workload:
    name: str
    run: Callable[[], object]
    description: str
    input_sha256: str


def source_binding(root: Path, paths) -> dict:
    files = {
        str(path.resolve().relative_to(root.resolve())): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(paths)
        if path.is_file()
    }
    encoded = json.dumps(files, sort_keys=True).encode()
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "files": files}


def runtime_context() -> dict:
    distributions = sorted(
        (dist.metadata["Name"], dist.version)
        for dist in importlib.metadata.distributions()
        if dist.metadata["Name"]
    )
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "hash_seed": os.environ.get("PYTHONHASHSEED", "random"),
        "build": sys.version,
        "cache_tag": sys.implementation.cache_tag,
        "free_threaded_build": bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
        "gil_enabled": getattr(sys, "_is_gil_enabled", lambda: True)(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": dict(distributions),
    }


def _location(key, root):
    filename, line, function = key
    path = Path(filename)
    if path.is_absolute():
        try:
            filename = str(path.relative_to(root))
        except ValueError:
            # Preserve enough package context without publishing user paths.
            parts = path.parts
            marker = "site-packages" if "site-packages" in parts else "lib"
            filename = (
                "/".join(parts[parts.index(marker) :]) if marker in parts else path.name
            )
    return {"file": filename, "line": line, "function": function}


def measure(
    workload: Workload, *, root: Path, repeats: int = 5, warmups: int = 1, top: int = 30
) -> dict:
    if type(repeats) is not int or repeats < 2 or repeats > 100:
        raise ValueError("repeats must be an integer in [2, 100]")
    if type(warmups) is not int or not 0 <= warmups <= 20:
        raise ValueError("warmups must be an integer in [0, 20]")
    if type(top) is not int or not 1 <= top <= 200:
        raise ValueError("top must be an integer in [1, 200]")
    for _ in range(warmups):
        workload.run()
    samples = []
    for _ in range(repeats):
        wall, cpu = time.perf_counter_ns(), time.process_time_ns()
        workload.run()
        samples.append(
            {
                "wall_ns": time.perf_counter_ns() - wall,
                "cpu_ns": time.process_time_ns() - cpu,
            }
        )
    try:
        from profiling.tracing import Profile

        backend = "profiling.tracing"
    except ImportError:
        from cProfile import Profile

        backend = "cProfile"
    profiler = Profile()
    profiler.runcall(workload.run)
    stats = pstats.Stats(profiler)
    functions = []
    edges = []
    for key, (primitive, calls, own, cumulative, callers) in stats.stats.items():
        functions.append(
            {
                **_location(key, root),
                "primitive_calls": primitive,
                "calls": calls,
                "self_seconds": own,
                "cumulative_seconds": cumulative,
            }
        )
        for caller, value in callers.items():
            if isinstance(value, tuple):
                edges.append(
                    {
                        "caller": _location(caller, root),
                        "callee": _location(key, root),
                        "calls": value[1],
                        "cumulative_seconds": value[3],
                    }
                )
    # Memory is measured separately so its instrumentation cannot contaminate timing.
    if tracemalloc.is_tracing():
        raise RuntimeError(
            "memory profiling cannot replace an existing tracemalloc session"
        )
    tracemalloc.start(10)
    try:
        workload.run()
        current, peak = tracemalloc.get_traced_memory()
        allocation_sites = [
            {
                **_location(
                    (
                        entry.traceback[0].filename,
                        entry.traceback[0].lineno,
                        "allocation",
                    ),
                    root,
                ),
                "bytes": entry.size,
                "count": entry.count,
            }
            for entry in tracemalloc.take_snapshot().statistics("lineno")[:top]
        ]
    finally:
        tracemalloc.stop()
    return {
        "name": workload.name,
        "description": workload.description,
        "input_sha256": workload.input_sha256,
        "warmups": warmups,
        "unprofiled_samples": samples,
        "wall_median_ns": statistics.median(x["wall_ns"] for x in samples),
        "cpu_median_ns": statistics.median(x["cpu_ns"] for x in samples),
        "wall_min_ns": min(x["wall_ns"] for x in samples),
        "wall_max_ns": max(x["wall_ns"] for x in samples),
        "tracing": {
            "backend": backend,
            "total_calls": stats.total_calls,
            "total_seconds": stats.total_tt,
            "functions": sorted(functions, key=lambda x: -x["cumulative_seconds"])[
                :top
            ],
            "call_edges": sorted(edges, key=lambda x: -x["cumulative_seconds"])[:top],
        },
        "memory": {
            "method": "tracemalloc; Python allocations, not process RSS or native heaps",
            "current_bytes": current,
            "peak_bytes": peak,
            "allocation_sites": allocation_sites,
        },
    }


def compare_reports(before: dict, after: dict) -> dict:
    reasons = []
    if before.get("runtime") != after.get("runtime"):
        reasons.append("runtime_or_dependencies_differ")
    previous = {row["name"]: row for row in before["workloads"]}
    comparisons = []
    for row in after["workloads"]:
        old = previous.get(row["name"])
        issues = list(reasons)
        if old is None or old.get("input_sha256") != row.get("input_sha256"):
            issues.append("workload_inputs_differ_or_missing")
        if old is not None and (
            old["warmups"] != row["warmups"]
            or len(old["unprofiled_samples"]) != len(row["unprofiled_samples"])
        ):
            issues.append("measurement_design_differs")
        comparisons.append(
            {
                "name": row["name"],
                "comparable": not issues,
                "reasons": issues,
                "cpu_ratio_after_over_before": (
                    row["cpu_median_ns"] / old["cpu_median_ns"]
                    if not issues and old["cpu_median_ns"] > 0
                    else None
                ),
                "wall_ratio_after_over_before": (
                    row["wall_median_ns"] / old["wall_median_ns"]
                    if not issues and old["wall_median_ns"] > 0
                    else None
                ),
            }
        )
    return {
        "before_source": before["source"]["sha256"],
        "after_source": after["source"]["sha256"],
        "interpretation": "Descriptive ratios on one host; no significance, cross-runtime speedup, or generalization claim.",
        "workloads": comparisons,
    }


def render_profile_markdown(report: dict) -> str:
    """Human-readable view of the receipt, generated from the same observations."""
    lines = [
        "# Runtime measurement receipt",
        "",
        f"Python {report['runtime']['python']}; {report['runtime']['implementation']}; {report['runtime']['machine']}.",
        "",
        f"Source binding: `{report['source']['sha256']}`.",
        "",
        "All timing values below come from unprofiled samples after the declared warmup. Allocation peaks come from a separate instrumented pass.",
        "",
        "| Workload | Median CPU ms | Median wall ms | Wall range ms | Python peak KiB |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["workloads"]:
        lines.append(
            f"| {row['name']} | {row['cpu_median_ns']/1e6:.3f} | {row['wall_median_ns']/1e6:.3f} | {row['wall_min_ns']/1e6:.3f}–{row['wall_max_ns']/1e6:.3f} | {row['memory']['peak_bytes']/1024:.1f} |"
        )
    lines += [
        "",
        "Ranges are descriptive, not confidence intervals. Parent CPU excludes external solver CPU; Python allocations exclude native heaps and process RSS.",
    ]
    if "comparison" in report:
        lines += [
            "",
            "## Same-context descriptive comparison",
            "",
            "Ratios are current/baseline; values below one indicate lower observed cost. No significance or causal speedup claim is implied.",
            "",
            "| Workload | Comparable | CPU ratio | Wall ratio | Reasons |",
            "|---|---|---:|---:|---|",
        ]
        for row in report["comparison"]["workloads"]:
            cpu = row["cpu_ratio_after_over_before"]
            wall = row["wall_ratio_after_over_before"]
            lines.append(
                f"| {row['name']} | {row['comparable']} | {cpu:.3f} | {wall:.3f} | |"
                if row["comparable"]
                else f"| {row['name']} | False | — | — | {', '.join(row['reasons'])} |"
            )
    lines += ["", "## Workloads and traces", ""]
    for row in report["workloads"]:
        lines += [
            f"### {row['name']}",
            "",
            row["description"],
            "",
            f"Input SHA-256: `{row['input_sha256']}`. Tracing backend: `{row['tracing']['backend']}`; {row['tracing']['total_calls']} instrumented calls.",
            "",
            "Top cumulative tracing frames (instrumented time; nested frames overlap):",
            "",
        ]
        for frame in row["tracing"]["functions"][:5]:
            lines.append(
                f"- `{frame['file']}:{frame['line']} {frame['function']}`: {frame['calls']} calls, {frame['cumulative_seconds']:.6f} seconds cumulative."
            )
        lines.append("")
    lines += [
        "## Availability and limits",
        "",
        f"Native sampling: {report.get('sampling',{}).get('status','not requested')}. Cold import profile: {report.get('cold_imports',{}).get('status','not requested')}.",
        "",
        "The JSON receipt retains individual samples, bounded caller–callee edges, allocation sites, dependency/build context, input hashes and source bindings. These software measurements are separate from scientific model observations.",
        "",
    ]
    return "\n".join(lines)
