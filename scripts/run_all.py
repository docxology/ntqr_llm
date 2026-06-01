"""Thin driver that runs every orchestrator and reports a pass/fail summary.

Each orchestrator is a standalone, runnable module under a category folder
(``scripts/ntqr/``, ``scripts/baselines/``, ``scripts/llm/``, ...). Categories
are **auto-discovered** -- any subfolder of ``scripts/`` holding ``*.py`` files
is a category, so adding a new method module needs no edit here. Orchestrators
are loaded by file path (so the ``scripts/ntqr/`` folder name does not clash
with the installed ``ntqr`` package) and each ``main()`` is called in turn.

Some categories need external services (e.g. ``llm`` needs a running Ollama).
A module declares that with a top-level ``REQUIRES = ["ollama"]``; if the
requirement is unmet, the orchestrator is SKIPPED, not failed. Optional
categories listed in ``OPTIONAL_CATEGORIES`` are not run by default -- pass
``--all`` or ``--include llm`` to include them.

Usage:
    python scripts/run_all.py                 # ntqr + baselines (offline)
    python scripts/run_all.py --all           # everything, incl. llm
    python scripts/run_all.py --include llm   # add one optional category
    python scripts/run_all.py --only ntqr     # restrict to one category
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import traceback
import urllib.request
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
import _common  # noqa: E402

# Folders under scripts/ that are not method categories.
# "manuscript" holds the prose renderer (render.py), not ntqr orchestrators;
# it is a post-processing step run explicitly after the sweep, not a method
# category, so it is excluded from auto-discovery.
_NOT_CATEGORIES = {"configs", "tests", "manuscript", "__pycache__"}
# Categories not run unless explicitly requested (network/cost side effects).
OPTIONAL_CATEGORIES = {"llm"}


def discover_categories() -> list[str]:
    return sorted(
        p.name
        for p in SCRIPTS_DIR.iterdir()
        if p.is_dir()
        and p.name not in _NOT_CATEGORIES
        and not p.name.startswith(".")
        and any(p.glob("*.py"))
    )


def discover(categories: list[str]) -> list[Path]:
    files: list[Path] = []
    for category in categories:
        cat_dir = SCRIPTS_DIR / category
        files.extend(
            sorted(
                p for p in cat_dir.glob("*.py") if not p.name.startswith("_")
            )
        )
    return files


def _ollama_available() -> bool:
    try:
        with urllib.request.urlopen(
            "http://localhost:11434/api/tags", timeout=2
        ) as resp:
            return resp.status == 200
    except Exception:
        return False


REQUIREMENT_CHECKS = {"ollama": _ollama_available}


def load_module(path: Path):
    name = f"orchestrator_{path.parent.name}_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def unmet_requirements(module) -> list[str]:
    return [
        req
        for req in getattr(module, "REQUIRES", [])
        if not REQUIREMENT_CHECKS.get(req, lambda: True)()
    ]


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run orchestrators.")
    parser.add_argument("--all", action="store_true", help="include optional")
    parser.add_argument(
        "--include", action="append", default=[], metavar="CAT"
    )
    parser.add_argument("--only", action="append", default=[], metavar="CAT")
    return parser.parse_args(argv)


def select_categories(args: argparse.Namespace) -> list[str]:
    available = discover_categories()
    if args.only:
        return [c for c in available if c in set(args.only)]
    if args.all:
        return available
    chosen = set(args.include)
    return [
        c for c in available if c not in OPTIONAL_CATEGORIES or c in chosen
    ]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _common.ensure_ntqr_importable()

    categories = select_categories(args)
    results: list[tuple[str, str, str]] = []  # (label, status, detail)
    for path in discover(categories):
        label = f"{path.parent.name}/{path.name}"
        try:
            module = load_module(path)
            missing = unmet_requirements(module)
            if missing:
                results.append((label, "SKIP", f"needs {', '.join(missing)}"))
                continue
            module.main([])  # empty argv: don't leak run_all's flags
            results.append((label, "PASS", ""))
        except Exception as exc:  # surface, never mask
            traceback.print_exc()
            results.append((label, "FAIL", repr(exc)))

    _common.banner("run_all summary")
    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    for label, status, detail in results:
        counts[status] += 1
        print(f"  [{status}] {label}{('  ' + detail) if detail else ''}")

    print(
        f"\n{counts['PASS']} passed, {counts['FAIL']} failed, "
        f"{counts['SKIP']} skipped "
        f"(categories: {', '.join(categories) or 'none'})"
    )
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
