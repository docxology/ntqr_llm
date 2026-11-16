"""Use recorded model evidence to verify independently selectable figure builders."""

from pathlib import Path
import json
import hashlib
import subprocess
import sys
import pytest
from ntqr.llm.study_records import load_recorded_study, model_dirname
from ntqr.llm.visualization import plot_models
from ntqr.llm.consistency import summarize_sets

ROOT = Path(__file__).resolve().parents[1]


def test_recorded_study_loader_keeps_failed_models_out_of_scored_rows(
    tmp_path,
):
    source = ROOT / "outputs/llm/multi_llm_personas_evaluation"
    rows, context, binding = load_recorded_study(source)
    failed = [row for row in rows if "error" in row]
    assert failed
    assert all("comparison" not in row for row in failed)
    assert context["figure_inputs_sha256"] == binding["sha256"]
    assert all("/" in key or key.endswith(".json") for key in binding["files"])
    paths = plot_models(
        rows,
        tmp_path,
        context,
        figures=["recovery", "best_model"],
        source_dir=source,
    )
    assert len(paths) == 2
    for path in paths:
        assert path.with_suffix(".svg").exists()
        data = json.loads(path.with_suffix(".data.json").read_text())
        assert (
            data["result"]["metadata"]["experiment"]["figure_inputs_sha256"]
            == binding["sha256"]
        )
    with pytest.raises(ValueError, match="unknown figures"):
        plot_models(rows, tmp_path, figures=["not_a_figure"])


def test_model_dirname_preserves_recorded_artifact_layout():
    assert model_dirname("mistral:latest") == "mistral_latest"
    assert model_dirname("granite4.1:3b") == "granite4.1_3b"
    assert model_dirname("vendor/model:tag") == "vendor_model_tag"


def test_zero_reservoir_preserves_exact_counts_and_seeded_sampling():
    full = summarize_sets(("a", "b"), ("0", "1"), (2, 2), seed=7, n_samples=3)
    count_only = summarize_sets(
        ("a", "b"), ("0", "1"), (2, 2), seed=7, n_samples=0
    )
    assert full["consistent_set_count"] == count_only["consistent_set_count"]
    assert count_only["consistent_random_points"] == []
    assert full == summarize_sets(
        ("a", "b"), ("0", "1"), (2, 2), seed=7, n_samples=3
    )


def test_recorded_supporting_cli_preserves_producers_and_sidecars(tmp_path):
    simulation = ROOT / "outputs/sim/synthetic_recovery_study"
    originals = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in simulation.iterdir()
        if path.is_file()
    }
    destination = tmp_path / "rendered"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/validation/render_supporting_figures.py"),
            "--output-dir",
            str(destination),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in originals
    } == originals
    for filename in (
        "convergence.data.json",
        "correlation.data.json",
        "tiebreak.data.json",
        "mv_vs_ei.data.json",
    ):
        assert (destination / filename).read_bytes() == (
            simulation / filename
        ).read_bytes()
    manifest = json.loads((destination / "render_sources.json").read_text())
    sources = manifest["result"]["inputs"]
    for filename in ("summary.json", "mv_vs_ei.data.json"):
        path = simulation / filename
        assert sources[str(path.relative_to(ROOT))] == originals[path]
