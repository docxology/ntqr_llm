#!/usr/bin/env python3
"""Hydrate manuscript variables for the docxology/template renderer.

The standalone project renderer writes ``docs/manuscript/{manuscript,supplemental}.md``.
The sibling template renderer prefers ``output/manuscript/*.md`` and
``output/figures/*.png``. This script bridges those interfaces without changing
the experiment math: it computes the same `_values.py` token map, writes it to
``output/data/manuscript_variables.json``, splits the rendered main manuscript
into template-style section files, writes the supplemental as an ``S01`` file,
and copies every embedded figure plus its machine-readable sidecar.
"""

from __future__ import annotations

import json
import argparse
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANUSCRIPT_DIR = ROOT / "docs" / "manuscript"
OUT_DIR = ROOT / "outputs" / "llm" / "multi_llm_personas_evaluation"
OUTPUT_MANUSCRIPT = ROOT / "output" / "manuscript"
OUTPUT_DATA = ROOT / "output" / "data"
OUTPUT_FIGURES = ROOT / "output" / "figures"

sys.path.insert(0, str(ROOT / "scripts" / "manuscript"))
import _values  # noqa: E402
import render  # noqa: E402

FIGURES: tuple[tuple[Path, Path | None], ...] = (
    (
        ROOT
        / "outputs"
        / "llm"
        / "multi_llm_personas_evaluation"
        / "evaluation_pipeline.png",
        ROOT
        / "outputs"
        / "llm"
        / "multi_llm_personas_evaluation"
        / "evaluation_pipeline.data.json",
    ),
    (
        ROOT
        / "outputs"
        / "llm"
        / "multi_llm_personas_evaluation"
        / "disagreement_vs_evaluability.png",
        ROOT
        / "outputs"
        / "llm"
        / "multi_llm_personas_evaluation"
        / "disagreement_vs_evaluability.data.json",
    ),
    (
        ROOT
        / "outputs"
        / "llm"
        / "multi_llm_personas_evaluation"
        / "best_model_recovery.png",
        ROOT
        / "outputs"
        / "llm"
        / "multi_llm_personas_evaluation"
        / "best_model_recovery.data.json",
    ),
    (
        ROOT
        / "outputs"
        / "llm"
        / "multi_llm_personas_evaluation"
        / "recovered_vs_true_scatter.png",
        ROOT
        / "outputs"
        / "llm"
        / "multi_llm_personas_evaluation"
        / "recovered_vs_true_scatter.data.json",
    ),
    (
        ROOT
        / "outputs"
        / "llm"
        / "multi_llm_personas_evaluation"
        / "validation_accuracy_heatmap.png",
        ROOT
        / "outputs"
        / "llm"
        / "multi_llm_personas_evaluation"
        / "validation_accuracy_heatmap.data.json",
    ),
    (
        ROOT
        / "outputs"
        / "llm"
        / "multi_llm_personas_evaluation"
        / "evaluability_diagnostic.png",
        ROOT
        / "outputs"
        / "llm"
        / "multi_llm_personas_evaluation"
        / "evaluability_diagnostic.data.json",
    ),
    (
        ROOT
        / "outputs"
        / "llm"
        / "multi_llm_personas_evaluation"
        / "prompt_compliance.png",
        ROOT
        / "outputs"
        / "llm"
        / "multi_llm_personas_evaluation"
        / "prompt_compliance.data.json",
    ),
    (
        ROOT
        / "outputs"
        / "llm"
        / "multi_llm_personas_evaluation"
        / "temperature_schema_comparison.png",
        ROOT
        / "outputs"
        / "llm"
        / "multi_llm_personas_evaluation"
        / "temperature_schema_comparison.data.json",
    ),
    (
        ROOT
        / "outputs"
        / "llm"
        / "multi_llm_personas_evaluation"
        / "scenario_hardness.png",
        ROOT
        / "outputs"
        / "llm"
        / "multi_llm_personas_evaluation"
        / "scenario_hardness.data.json",
    ),
    (
        ROOT
        / "outputs"
        / "llm"
        / "multi_llm_personas_evaluation"
        / "recovery_mae_by_model.png",
        ROOT
        / "outputs"
        / "llm"
        / "multi_llm_personas_evaluation"
        / "recovery_mae_by_model.data.json",
    ),
    (
        ROOT
        / "outputs"
        / "sim"
        / "synthetic_recovery_study"
        / "synthetic_convergence.png",
        ROOT / "outputs" / "sim" / "synthetic_recovery_study" / "convergence.data.json",
    ),
    (
        ROOT / "outputs" / "sim" / "synthetic_recovery_study" / "synthetic_alarm.png",
        ROOT / "outputs" / "sim" / "synthetic_recovery_study" / "correlation.data.json",
    ),
    (
        ROOT
        / "outputs"
        / "sim"
        / "bootstrap_recovery_ci"
        / "bootstrap_recovery_mae.png",
        ROOT
        / "outputs"
        / "sim"
        / "bootstrap_recovery_ci"
        / "bootstrap_recovery_mae.data.json",
    ),
    (
        ROOT
        / "outputs"
        / "sim"
        / "synthetic_recovery_study"
        / "synthetic_tiebreak.png",
        ROOT / "outputs" / "sim" / "synthetic_recovery_study" / "tiebreak.data.json",
    ),
    (
        ROOT
        / "outputs"
        / "sim"
        / "synthetic_recovery_study"
        / "synthetic_mv_vs_ei.png",
        ROOT / "outputs" / "sim" / "synthetic_recovery_study" / "mv_vs_ei.data.json",
    ),
    (
        ROOT
        / "outputs"
        / "validation"
        / "statistical_validation"
        / "validation_matrix.png",
        ROOT
        / "outputs"
        / "validation"
        / "statistical_validation"
        / "validation_matrix.data.json",
    ),
)

SECTION_FILES = {
    "abstract": "00_abstract.md",
    "introduction": "01_introduction.md",
    "methods": "02_methods.md",
    "results": "03_results.md",
    "discussion": "04_discussion.md",
    "figure-specification": "05_figure_specification.md",
    "references": "99_references.md",
}

HEADING_RE = re.compile(r"(?m)^# .+? \{#sec:([^}]+)\}\s*$")
TOKEN_RE = re.compile(r"\{\{[A-Z0-9_]+\}\}")


def _template_path_text(text: str) -> str:
    return text.replace("../output/figures/", "../figures/")


def _split_main(text: str) -> dict[str, str]:
    matches = list(HEADING_RE.finditer(text))
    if not matches:
        raise ValueError("rendered manuscript has no template section labels")
    sections: dict[str, str] = {}
    for i, match in enumerate(matches):
        label = match.group(1)
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        if label in SECTION_FILES:
            sections[SECTION_FILES[label]] = text[start:end].strip() + "\n"
    missing = sorted(set(SECTION_FILES.values()) - set(sections))
    if missing:
        raise ValueError(f"missing required manuscript sections: {missing}")
    return sections


def figure_payloads(figures=FIGURES) -> dict[Path, bytes]:
    """Read and validate every declared input before replacing any generated file."""
    payloads = {}
    for image, sidecar in figures:
        image_target = OUTPUT_FIGURES / image.name
        if image_target in payloads:
            raise ValueError(f"duplicate generated figure filename: {image.name}")
        payloads[image_target] = image.read_bytes()
        if sidecar is not None:
            source = sidecar.read_bytes()
            json.loads(source)
            sidecar_target = OUTPUT_FIGURES / f"{image.stem}.data.json"
            if sidecar_target in payloads:
                raise ValueError(
                    f"duplicate generated sidecar filename: {sidecar_target.name}"
                )
            payloads[sidecar_target] = source
    return payloads


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check all declared outputs without writing",
    )
    args = parser.parse_args(argv)
    from ntqr.llm.artifacts import atomic_write_bytes

    values = _values.compute_values(OUT_DIR)
    rendered = render.render_all()
    main_text = rendered[MANUSCRIPT_DIR / "manuscript.md"]
    supplemental = rendered[MANUSCRIPT_DIR / "supplemental.md"]
    payloads = figure_payloads()
    payloads[OUTPUT_DATA / "manuscript_variables.json"] = (
        json.dumps(values, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    for filename, section in _split_main(main_text).items():
        body = _template_path_text(section)
        if TOKEN_RE.search(body):
            raise KeyError(f"unresolved token in generated section {filename}")
        payloads[OUTPUT_MANUSCRIPT / filename] = body.encode("utf-8")
    supp_body = _template_path_text(supplemental.strip() + "\n")
    if TOKEN_RE.search(supp_body):
        raise KeyError("unresolved token in generated supplemental")
    payloads[OUTPUT_MANUSCRIPT / "S01_supplemental.md"] = supp_body.encode("utf-8")
    for name in ("config.yaml", "preamble.md", "references.bib"):
        payloads[OUTPUT_MANUSCRIPT / name] = (MANUSCRIPT_DIR / name).read_bytes()
    manifest = {
        "schema_version": 1,
        "files": {
            str(path.relative_to(ROOT)): hashlib.sha256(source).hexdigest()
            for path, source in sorted(payloads.items())
        },
    }
    payloads[OUTPUT_DATA / "render_manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if args.check:
        stale = [
            str(path.relative_to(ROOT))
            for path, source in payloads.items()
            if not path.is_file() or path.read_bytes() != source
        ]
        if stale:
            print("Stale generated artifacts: " + ", ".join(stale), file=sys.stderr)
            return 1
        print(f"OK — {len(payloads)} generated artifacts match their sources.")
        return 0
    # All validation has completed; replace only files this bridge owns.
    for path, source in payloads.items():
        atomic_write_bytes(path, source)

    print(OUTPUT_DATA / "manuscript_variables.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
