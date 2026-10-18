"""Run bounded runtime profiles; outputs are separate from scientific results."""

from __future__ import annotations
import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python/src"))
from ntqr.llm.runtime_profile import (
    measure,
    runtime_context,
    source_binding,
    compare_reports,
)
from ntqr.llm.profile_workloads import workloads


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "benchmarks/runtime/current.json"
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workload", action="append")
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--workload-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    selected = [
        work for work in workloads() if not args.workload or work.name in args.workload
    ]
    if not selected:
        parser.error("no matching workload")
    if args.workload_only:
        for work in selected:
            for _ in range(10):
                work.run()
        return 0
    source_paths = (
        list(ROOT.glob("src/**/*.py"))
        + list(ROOT.glob("python/src/**/*.py"))
        + [Path(__file__), ROOT / "uv.lock", ROOT / "PROFILE_REFERENCE.json"]
    )
    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": runtime_context(),
        "source": source_binding(ROOT, source_paths),
        "measurement_design": "Warm up once; unprofiled timing, deterministic tracing, and memory in separate passes. Imports excluded.",
        "workloads": [
            measure(work, root=ROOT, repeats=args.repeats) for work in selected
        ],
    }
    reference = ROOT / "PROFILE_REFERENCE.json"
    if reference.is_file():
        report["baseline_reference"] = json.loads(reference.read_text(encoding="utf-8"))
    if args.sample:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        sample = args.output.with_suffix(".flamegraph.html")
        command = [
            sys.executable,
            "-m",
            "profiling.sampling",
            "run",
            "--flamegraph",
            "-d",
            "15",
            "-o",
            str(sample),
            str(Path(__file__)),
            "--workload-only",
        ]
        for work in selected:
            command.extend(["--workload", work.name])
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=45, cwd=ROOT
            )
            report["sampling"] = {
                "status": (
                    "available"
                    if result.returncode == 0 and sample.exists()
                    else "unavailable"
                ),
                "returncode": result.returncode,
                "artifact": sample.name if sample.exists() else None,
                "detail": (result.stderr + result.stdout)[-2000:]
                .replace(str(ROOT), "<repo>")
                .replace(str(Path.home()), "<home>"),
            }
        except subprocess.TimeoutExpired:
            report["sampling"] = {
                "status": "unavailable",
                "detail": "bounded sampling timeout",
            }
    # Import timing is a fresh subprocess, separated from steady-state samples.
    try:
        imported = subprocess.run(
            [
                sys.executable,
                "-X",
                "importtime",
                "-c",
                "import ntqr.llm.profile_workloads; ntqr.llm.profile_workloads.workloads()",
            ],
            capture_output=True,
            text=True,
            timeout=45,
            cwd=ROOT,
        )
        imports = []
        for line in imported.stderr.splitlines():
            if line.startswith("import time:"):
                parts = line.removeprefix("import time:").split("|")
                try:
                    imports.append(
                        {
                            "self_us": int(parts[0]),
                            "cumulative_us": int(parts[1]),
                            "module": parts[2].strip(),
                        }
                    )
                except (ValueError, IndexError):
                    pass
        report["cold_imports"] = {
            "status": "available" if imported.returncode == 0 else "unavailable",
            "returncode": imported.returncode,
            "largest_cumulative": sorted(
                imports, key=lambda row: -row["cumulative_us"]
            )[:30],
        }
    except subprocess.TimeoutExpired:
        report["cold_imports"] = {
            "status": "unavailable",
            "detail": "bounded fresh-process import timeout",
        }
    if args.compare:
        report["comparison"] = compare_reports(
            json.loads(args.compare.read_text()), report
        )
    from ntqr.llm.artifacts import save_json

    save_json(args.output, report)
    from ntqr.llm.runtime_figure import plot_runtime_report

    plot_runtime_report(report, args.output.with_suffix(".png"))
    from ntqr.llm.runtime_profile import render_profile_markdown
    from ntqr.llm.artifacts import atomic_write_bytes

    atomic_write_bytes(
        args.output.with_suffix(".md"), render_profile_markdown(report).encode("utf-8")
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    if "PYTHONHASHSEED" not in os.environ:
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(
            sys.executable,
            [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        )
    raise SystemExit(main())
