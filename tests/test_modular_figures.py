"""Use recorded model evidence to verify independently selectable figure builders."""

from pathlib import Path
import json
import pytest
from ntqr.llm.study_records import load_recorded_study
from ntqr.llm.visualization import plot_models
from ntqr.llm.consistency import summarize_sets

ROOT = Path(__file__).resolve().parents[1]


def test_recorded_study_loader_keeps_failed_models_out_of_scored_rows(tmp_path):
    source = ROOT / "outputs/llm/multi_llm_personas_evaluation"
    rows, context, binding = load_recorded_study(source)
    failed = [row for row in rows if "error" in row]
    assert failed
    assert all("comparison" not in row for row in failed)
    assert context["figure_inputs_sha256"] == binding["sha256"]
    assert all("/" in key or key.endswith(".json") for key in binding["files"])
    paths = plot_models(
        rows, tmp_path, context, figures=["recovery", "best_model"], source_dir=source
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


def test_zero_reservoir_preserves_exact_counts_and_seeded_sampling():
    full = summarize_sets(("a", "b"), ("0", "1"), (2, 2), seed=7, n_samples=3)
    count_only = summarize_sets(("a", "b"), ("0", "1"), (2, 2), seed=7, n_samples=0)
    assert full["consistent_set_count"] == count_only["consistent_set_count"]
    assert count_only["consistent_random_points"] == []
    assert full == summarize_sets(("a", "b"), ("0", "1"), (2, 2), seed=7, n_samples=3)
