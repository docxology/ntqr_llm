"""Tests for the manuscript injection pipeline (scripts/manuscript/).

These guard the determinism guarantee: the renderer fills every ``{{TOKEN}}``
from the JSON-derived values and fails loudly on unknown or leftover tokens, and
the on-disk manuscript matches a fresh render of the current data. Runs under
pytest *and* standalone:

    pytest scripts/tests/test_manuscript_render.py
    python scripts/tests/test_manuscript_render.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS / "manuscript"))

import _values  # noqa: E402
import render  # noqa: E402

OUT = SCRIPTS.parent / "outputs" / "llm" / "multi_llm_personas_evaluation"
REFERENCES = SCRIPTS.parent / "manuscript" / "references.bib"
TEMPLATES = SCRIPTS / "manuscript" / "templates"


def _values_or_skip():
    """Compute values, or signal skip if the sweep has not been run yet."""
    if not (OUT / "summary.json").exists():
        return None
    return _values.compute_values(OUT)


def _bib_keys() -> set[str]:
    return set(re.findall(r"@\w+\{([^,\s]+)", REFERENCES.read_text()))


def _template_citation_keys() -> set[str]:
    text = "\n".join(
        path.read_text() for path in sorted(TEMPLATES.glob("*.md.tmpl"))
    )
    keys = set(re.findall(r"(?<!\w)@([A-Za-z0-9_:-]+)", text))
    return {key for key in keys if ":" not in key}


def _section(text: str, start: str, end: str | None = None) -> str:
    start_i = text.index(start)
    end_i = text.index(end, start_i) if end else len(text)
    return text[start_i:end_i]


def test_render_substitutes_known_tokens():
    v = {"A": "1", "B": "two"}
    out = render.render_one("x={{A}}, y={{B}}", v, "t")
    assert out == "x=1, y=two"


def test_render_rejects_unknown_token():
    try:
        render.render_one("{{NOPE}}", {"A": "1"}, "t")
        raise AssertionError("expected KeyError for unknown token")
    except KeyError as exc:
        assert "NOPE" in str(exc)


def test_render_rejects_leftover_after_partial():
    # A token the value map does not define must raise, never pass silently.
    try:
        render.render_one("{{A}} {{B}}", {"A": "ok"}, "t")
        raise AssertionError("expected KeyError for missing B")
    except KeyError as exc:
        assert "B" in str(exc)


def test_values_are_all_strings():
    v = _values_or_skip()
    if v is None:
        print("[SKIP] no summary.json — run the sweep first")
        return
    assert v, "expected a non-empty value map"
    for key, val in v.items():
        assert isinstance(val, str), f"{key} is {type(val).__name__}, not str"


def test_headline_numbers_match_artifacts():
    v = _values_or_skip()
    if v is None:
        print("[SKIP] no summary.json — run the sweep first")
        return
    summary = _values._load(OUT / "summary.json")
    # Q, model counts, and the diagnostic verdict are the load-bearing claims.
    assert v["Q"] == str(summary["n_scenarios"])
    assert v["N_EVALUABLE"] == str(len(summary["evaluated_models"]))
    assert v["N_UNEVALUABLE"] == str(len(summary["failed_models"]))
    assert (
        v["DIAG_MATCHED"]
        == str(
            summary["evaluability_diagnostic"]["diagnostic_matched_outcome"]
        ).lower()
    )
    assert (
        v["DIS_GATES"]
        == str(
            summary["cross_model_statistics"][
                "disagreement_gates_evaluability"
            ]
        ).lower()
    )


def test_headline_run_is_low_temperature_schema_contract():
    if not (OUT / "summary.json").exists():
        print("[SKIP] no summary.json — run the sweep first")
        return
    summary = _values._load(OUT / "summary.json")
    assert summary["temperature"] == 0.1
    assert summary["prompt_version"] == "v3_binary_schema"
    assert summary["prompt_hash"]


def test_on_disk_manuscript_matches_render():
    """The committed manuscript must equal a fresh render (no hand-drift)."""
    if not (OUT / "summary.json").exists():
        print("[SKIP] no summary.json — run the sweep first")
        return
    rendered = render.render_all()
    drifted = [
        dest.name
        for dest, text in rendered.items()
        if (dest.read_text() if dest.exists() else None) != text
    ]
    assert not drifted, (
        f"manuscript files drifted from the data: {drifted}. "
        "Run: python scripts/manuscript/render.py"
    )


def test_rendered_files_have_no_leftover_tokens():
    """Freshly rendered text must contain no unresolved manuscript tokens."""
    if not (OUT / "summary.json").exists():
        print("[SKIP] no summary.json — run the sweep first")
        return
    for dest, text in render.render_all().items():
        assert not render._TOKEN.findall(
            text
        ), f"{dest.name} still contains unresolved template tokens"


def test_rendered_manuscript_figure_inventory_has_sidecars():
    """Every embedded manuscript PNG should exist and have a data sidecar."""
    if not (OUT / "summary.json").exists():
        print("[SKIP] no summary.json — run the sweep first")
        return
    repo = SCRIPTS.parent
    manuscript = render.render_all()[repo / "manuscript" / "manuscript.md"]
    links = re.findall(r"!\[.*?\]\(([^)]+\.png)\)", manuscript, re.S)
    expected_sidecars = {
        "../output/figures/evaluation_pipeline.png": "../output/figures/evaluation_pipeline.data.json",
        "../output/figures/disagreement_vs_evaluability.png": "../output/figures/disagreement_vs_evaluability.data.json",
        "../output/figures/best_model_recovery.png": "../output/figures/best_model_recovery.data.json",
        "../output/figures/recovered_vs_true_scatter.png": "../output/figures/recovered_vs_true_scatter.data.json",
        "../output/figures/validation_accuracy_heatmap.png": "../output/figures/validation_accuracy_heatmap.data.json",
        "../output/figures/evaluability_diagnostic.png": "../output/figures/evaluability_diagnostic.data.json",
        "../output/figures/prompt_compliance.png": "../output/figures/prompt_compliance.data.json",
        "../output/figures/temperature_schema_comparison.png": "../output/figures/temperature_schema_comparison.data.json",
        "../output/figures/scenario_hardness.png": "../output/figures/scenario_hardness.data.json",
        "../output/figures/recovery_mae_by_model.png": "../output/figures/recovery_mae_by_model.data.json",
        "../output/figures/synthetic_convergence.png": "../output/figures/synthetic_convergence.data.json",
        "../output/figures/synthetic_alarm.png": "../output/figures/synthetic_alarm.data.json",
        "../output/figures/bootstrap_recovery_mae.png": "../output/figures/bootstrap_recovery_mae.data.json",
        "../output/figures/synthetic_tiebreak.png": "../output/figures/synthetic_tiebreak.data.json",
        "../output/figures/synthetic_mv_vs_ei.png": "../output/figures/synthetic_mv_vs_ei.data.json",
        "../output/figures/validation_matrix.png": "../output/figures/validation_matrix.data.json",
    }
    assert len(links) == 16
    assert set(links) == set(expected_sidecars)
    if not (repo / "output" / "figures").exists():
        print("[SKIP] template figure hydration has not been run")
        return
    for image, sidecar in expected_sidecars.items():
        image_path = (repo / "manuscript" / image).resolve()
        sidecar_path = (repo / "manuscript" / sidecar).resolve()
        assert image_path.exists(), image
        assert sidecar_path.exists(), sidecar
        doc = json.loads(sidecar_path.read_text())
        assert isinstance(doc.get("result", doc), dict), sidecar


def test_nonzero_recovery_does_not_claim_exact_recovery():
    """When recovery MAE is nonzero, generated prose must say close, not exact."""
    v = _values_or_skip()
    if v is None:
        print("[SKIP] no summary.json — run the sweep first")
        return
    if v["REC_MAE"] == "0.000":
        return
    rendered = {
        dest.name: " ".join(text.split())
        for dest, text in render.render_all().items()
    }
    banned = {
        "README.md": [
            "recovered every per-persona accuracy **exactly**",
            "bullish accuracy exactly",
        ],
        "manuscript.md": [
            "recovered per-judge accuracies exactly",
            "Reading the four figures together",
        ],
        "supplemental.md": [
            "accuracy terms (exact recovery)",
            "recovered values match the supervised truth exactly because",
        ],
    }
    for name, snippets in banned.items():
        for snippet in snippets:
            assert snippet not in rendered[name]


def test_manuscript_has_reader_map_and_validation_matrix():
    if not (OUT / "summary.json").exists():
        print("[SKIP] no summary.json — run the sweep first")
        return
    repo = SCRIPTS.parent
    manuscript = render.render_all()[repo / "manuscript" / "manuscript.md"]
    assert (
        "## Reader map: what is observed, recovered, and checked {#sec:reader-map}"
        in manuscript
    )
    assert (
        "| Object | Observed without labels? | Validation-only? | Meaning in this study |"
        in manuscript
    )
    assert (
        "## Statistical validation matrix {#sec:validation-matrix}"
        in manuscript
    )
    assert (
        "## Evaluation object and data flow {#sec:method-data-flow}"
        in manuscript
    )
    assert "evaluation_pipeline.png" in manuscript
    assert "validation_matrix.png" in manuscript


def test_public_ntqr_citation_is_present():
    if not (OUT / "summary.json").exists():
        print("[SKIP] no summary.json — run the sweep first")
        return
    repo = SCRIPTS.parent
    manuscript = render.render_all()[repo / "manuscript" / "manuscript.md"]
    references = REFERENCES.read_text()
    assert "ntqr_0_8_docs" in references
    assert "NTQR 0.8 Documentation" in references
    assert "Corrada-Emmanuel" in references
    assert "Lee, Walker" in references
    assert "Sloat, Adam" in references
    assert "[@ntqr_0_8_docs]" in manuscript
    assert "2. `ntqr` Python package" not in manuscript
    assert "@misc{ntqr_library_2026" not in references


def test_pdf_style_uses_red_links_tight_margins_and_no_placeholder_doi():
    repo = SCRIPTS.parent
    preamble = (repo / "manuscript" / "preamble.md").read_text()
    config = (repo / "manuscript" / "config.yaml").read_text()
    for key in (
        "linkcolor=red",
        "urlcolor=red",
        "citecolor=red",
        "anchorcolor=red",
        "filecolor=red",
    ):
        assert key in preamble
    assert "allcolors=blue" not in preamble
    assert 'geometry: "margin=0.58in"' in config
    assert 'doi: ""' in config
    assert 'version_doi: ""' in config
    assert 'github_repository: "docxology/ntqr_llm"' in config
    assert 'repository_url: "https://github.com/docxology/ntqr_llm"' in config
    assert "TBD" not in config


def test_public_release_metadata_files_match_config():
    repo = SCRIPTS.parent
    citation = repo / "CITATION.cff"
    zenodo = repo / ".zenodo.json"
    codemeta = repo / "codemeta.json"
    for path in (
        citation,
        zenodo,
        codemeta,
        repo / "LICENSE",
        repo / "NOTICE",
    ):
        assert path.exists(), path.name
    citation_text = citation.read_text()
    zenodo_json = json.loads(zenodo.read_text())
    codemeta_json = json.loads(codemeta.read_text())
    assert "Recovering LLM-Persona Accuracies" in citation_text
    assert "version: 1.0.0" in citation_text
    assert "https://github.com/docxology/ntqr_llm" in citation_text
    assert "docxology/projects" not in citation_text
    assert zenodo_json["version"] == "1.0.0"
    assert zenodo_json["license"] == "Apache-2.0"
    assert codemeta_json["version"] == "1.0.0"
    assert codemeta_json["codeRepository"] == (
        "https://github.com/docxology/ntqr_llm"
    )


def test_public_docs_do_not_leak_private_paths_or_repo_slug():
    repo = SCRIPTS.parent
    public_docs = [
        repo / "README.md",
        repo / "docs" / "README.md",
        repo / "docs" / "VERIFICATION.md",
        repo / "manuscript" / "config.yaml",
        repo / "CITATION.cff",
        repo / ".zenodo.json",
        repo / "codemeta.json",
    ]
    if (repo / "manuscript" / "manuscript.md").exists():
        public_docs.append(repo / "manuscript" / "manuscript.md")
    banned = ["/Users/4d", "docxology/projects"]
    for path in public_docs:
        text = path.read_text()
        for phrase in banned:
            assert phrase not in text, f"{path.name}: {phrase}"


def test_figure_sidecar_captions_have_no_stale_story():
    repo = SCRIPTS.parent
    fig_dir = repo / "output" / "figures"
    if not fig_dir.exists():
        print("[SKIP] template figure hydration has not been run")
        return
    banned = [
        "lone evaluable model",
        "single evaluable model",
        "reproduces all six per-persona accuracies",
        "statistically indistinguishable",
    ]
    for sidecar in fig_dir.glob("*.data.json"):
        doc = json.loads(sidecar.read_text())
        body = doc.get("result", doc)
        caption = str(doc.get("caption", body.get("caption", "")))
        metadata = doc.get("metadata", body.get("metadata"))
        provenance = doc.get("provenance")
        assert caption.strip(), f"{sidecar.name} has no caption"
        assert (
            isinstance(metadata, dict) and metadata
        ), f"{sidecar.name} has no metadata/provenance block"
        assert (
            isinstance(provenance, dict) and provenance
        ), f"{sidecar.name} has no top-level provenance alias"
        lowered = caption.lower()
        for phrase in banned:
            assert phrase not in lowered, f"{sidecar.name}: {phrase}"


def test_template_citations_are_defined_in_bibliography():
    missing = _template_citation_keys() - _bib_keys()
    assert not missing


def test_rendered_citations_are_defined_in_bibliography():
    if not (OUT / "summary.json").exists():
        print("[SKIP] no summary.json — run the sweep first")
        return
    rendered = render.render_all()
    text = "\n".join(rendered.values())
    keys = set(re.findall(r"(?<!\w)@([A-Za-z0-9_:-]+)", text))
    citation_keys = {key for key in keys if ":" not in key}
    missing = citation_keys - _bib_keys()
    assert not missing


def test_template_pdf_log_has_no_undefined_crossrefs_if_rendered():
    log = SCRIPTS.parent / "output" / "pdf" / "_combined_manuscript.log"
    if not log.exists():
        print("[SKIP] template PDF has not been rendered")
        return
    text = log.read_text(errors="ignore")
    assert not re.search(r"LaTeX Warning: Reference `[^`]+'.*undefined", text)
    assert "There were undefined references" not in text


def test_rendered_validation_sidecar_count_is_current():
    if not (OUT / "summary.json").exists():
        print("[SKIP] no summary.json — run the sweep first")
        return
    repo = SCRIPTS.parent
    manuscript = render.render_all()[repo / "manuscript" / "manuscript.md"]
    assert "14/14 sidecars" not in manuscript
    assert "16/16 sidecars" in manuscript


def test_no_stale_single_evaluable_claim_when_multiple_models_are_evaluable():
    if not (OUT / "summary.json").exists():
        print("[SKIP] no summary.json — run the sweep first")
        return
    summary = _values._load(OUT / "summary.json")
    if len(summary["evaluated_models"]) <= 1:
        return
    rendered = " ".join("\n".join(render.render_all().values()).split())
    banned = [
        "single evaluable real model",
        "single evaluable model",
        "the one evaluable model",
    ]
    for phrase in banned:
        assert phrase not in rendered


def test_required_context_citations_are_in_rendered_sections():
    if not (OUT / "summary.json").exists():
        print("[SKIP] no summary.json — run the sweep first")
        return
    repo = SCRIPTS.parent
    manuscript = render.render_all()[repo / "manuscript" / "manuscript.md"]
    introduction = _section(manuscript, "# Introduction", "# Methods")
    methods = _section(manuscript, "# Methods", "# Results")
    discussion = _section(manuscript, "# Discussion", "---")
    for key in {
        "dawid_skene_1979",
        "yan_2010_annotator",
        "zheng_2017_vldb",
        "ratner_2017_snorkel",
        "zheng_2023_llm_judge",
        "chen_2026_efficient_judge",
        "platanios_2014_uai",
        "platanios_2016_icml",
        "corrada_emmanuel_2023_ntqr_logic",
        "corrada_emmanuel_2023_streaming",
        "corrada_emmanuel_2024_alarm",
        "corrada_emmanuel_2025_nka",
    }:
        assert f"@{key}" in introduction
    for key in {
        "ntqr_0_8_docs",
        "ollama_structured_outputs_2024",
        "ollama_structured_blog_2024",
        "structured_output_reliability_2025",
        "ouyang_2022_instructgpt",
        "jiang_2023_mistral",
        "gemma_team_2024",
        "gemma2_team_2024",
        "gemma3_team_2025",
        "ibm_granite_4_1_2026",
        "ben_allal_2025_smollm2",
        "efron_tibshirani_1993",
    }:
        assert f"@{key}" in methods
    for key in {
        "condorcet_jury_sep",
        "wang_2022_self_consistency",
        "zheng_2017_vldb",
        "ratner_2017_snorkel",
        "zheng_2023_llm_judge",
        "chen_2026_efficient_judge",
    }:
        assert f"@{key}" in discussion


def test_template_crossref_syntax_is_present():
    if not (OUT / "summary.json").exists():
        print("[SKIP] no summary.json — run the sweep first")
        return
    repo = SCRIPTS.parent
    manuscript = render.render_all()[repo / "manuscript" / "manuscript.md"]
    figure_labels = re.findall(
        r"!\[.*?\]\([^)]+\.png\)\{#fig:[^}]+}", manuscript, re.S
    )
    assert len(figure_labels) == 16
    assert "{#tbl:evaluation-objects}" in manuscript
    assert "{#tbl:scenario-examples}" in manuscript
    assert "{#tbl:model-lineup}" in manuscript
    assert "{#tbl:related-evaluation-paradigms}" in manuscript
    assert "{#eq:universal-evaluation-polynomial}" in manuscript
    assert "{#eq:prevalence-quadratic}" in manuscript
    assert "[@fig:evaluation-pipeline]" in manuscript
    assert "[@fig:disagreement-vs-evaluability]" in manuscript
    assert "[@fig:validation-accuracy-heatmap]" in manuscript
    assert "[@tbl:synthetic-convergence]" in manuscript
    assert "**Figure 1.**" not in manuscript
    assert "**Table 1.**" not in manuscript


def test_template_hydration_outputs_are_valid():
    out_dir = SCRIPTS.parent / "output" / "manuscript"
    data_file = (
        SCRIPTS.parent / "output" / "data" / "manuscript_variables.json"
    )
    if not out_dir.exists() or not data_file.exists():
        print("[SKIP] template hydration has not been run")
        return
    expected = {
        "00_abstract.md",
        "01_introduction.md",
        "02_methods.md",
        "03_results.md",
        "04_discussion.md",
        "05_figure_specification.md",
        "99_references.md",
        "S01_supplemental.md",
    }
    files = {p.name for p in out_dir.glob("*.md")}
    assert expected <= files
    values = json.loads(data_file.read_text())
    assert values["Q"]
    for md in out_dir.glob("*.md"):
        text = md.read_text()
        assert not render._TOKEN.findall(text), md.name
    assert (out_dir / "preamble.md").exists()
    assert (out_dir / "references.bib").exists()


def _run_standalone() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  [PASS] {t.__name__}")
            passed += 1
        except AssertionError as exc:
            print(f"  [FAIL] {t.__name__}: {exc}")
    print(f"\n{passed} passed, {len(tests) - passed} failed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
