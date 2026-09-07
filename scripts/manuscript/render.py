"""Render the manuscript from templates + the single-source-of-truth values.

The prose lives in ``templates/*.md.tmpl`` with ``{{TOKEN}}`` placeholders;
every token is filled from ``_values.compute_values`` (which reads only the
JSON artifacts). This makes the manuscript a *generated* artifact: it cannot
drift from the data, because the numbers are never typed by hand.

Two modes:
    render            write manuscript.md / supplemental.md / README.md
    --check           render to memory and diff against the files on disk;
                      exit non-zero if they differ (a CI / pre-release gate)

Run:
    python scripts/manuscript/render.py
    python scripts/manuscript/render.py --check
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _values  # noqa: E402

REPO_ROOT = HERE.parents[1]
OUT_DIR = REPO_ROOT / "outputs" / "llm" / "multi_llm_personas_evaluation"
TEMPLATE_DIR = HERE / "templates"
MANUSCRIPT_DIR = REPO_ROOT / "docs" / "manuscript"

# template stem -> output file
TARGETS = {
    "manuscript": MANUSCRIPT_DIR / "manuscript.md",
    "supplemental": MANUSCRIPT_DIR / "supplemental.md",
    "README": MANUSCRIPT_DIR / "README.md",
}

_TOKEN = re.compile(r"\{\{([A-Z0-9_]+)\}\}")


def render_one(template_text: str, values: dict[str, str], stem: str) -> str:
    """Substitute every ``{{TOKEN}}``; raise on unknown or leftover tokens."""
    missing: set[str] = set()

    def _sub(match: re.Match) -> str:
        key = match.group(1)
        if key not in values:
            missing.add(key)
            return match.group(0)
        return values[key]

    out = _TOKEN.sub(_sub, template_text)
    if missing:
        raise KeyError(
            f"{stem}: template references unknown tokens: "
            f"{', '.join(sorted(missing))}"
        )
    leftover = _TOKEN.findall(out)
    if leftover:
        raise KeyError(
            f"{stem}: unresolved tokens after render: {', '.join(leftover)}"
        )
    return out


def render_all() -> dict[Path, str]:
    """Return {output_path: rendered_text} for every template."""
    values = _values.compute_values(OUT_DIR)
    rendered: dict[Path, str] = {}
    for stem, dest in TARGETS.items():
        tmpl = TEMPLATE_DIR / f"{stem}.md.tmpl"
        if not tmpl.exists():
            raise FileNotFoundError(f"missing template: {tmpl}")
        rendered[dest] = render_one(tmpl.read_text(), values, stem)
    return rendered


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="verify on-disk files match a fresh render; non-zero on drift",
    )
    args = ap.parse_args(argv)

    rendered = render_all()

    if args.check:
        drifted = []
        for dest, text in rendered.items():
            on_disk = dest.read_text() if dest.exists() else None
            if on_disk != text:
                drifted.append(dest)
        if drifted:
            print("DRIFT — these files differ from a fresh render:")
            for d in drifted:
                print(f"  {d.relative_to(REPO_ROOT)}")
            print("Run: python scripts/manuscript/render.py")
            return 1
        print(f"OK — {len(rendered)} manuscript files match the data.")
        return 0

    for dest, text in rendered.items():
        dest.write_text(text if text.endswith("\n") else text + "\n")
        print(f"-> wrote {dest.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
