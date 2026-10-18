"""Real computation checks for profiling, publication and scholarship boundaries."""

import pytest
from ntqr.llm.runtime_profile import Workload, measure, source_binding, compare_reports
from ntqr.llm.scholarship import bibliography_entries, audit_research
from ntqr.llm.figure_export import export_figure


def test_profile_keeps_unprofiled_samples_separate_and_binds_sources(tmp_path):
    path = tmp_path / "source.py"
    path.write_text("value = 1\n")
    binding = source_binding(tmp_path, [path])
    workload = Workload(
        "arithmetic",
        lambda: sum(i * i for i in range(100)),
        "real arithmetic",
        "fixed-input",
    )
    result = measure(workload, root=tmp_path, repeats=2, warmups=0)
    assert len(result["unprofiled_samples"]) == 2
    assert result["cpu_median_ns"] > 0
    assert result["tracing"]["total_calls"] > 0
    assert result["memory"]["peak_bytes"] >= result["memory"]["current_bytes"]
    assert result["tracing"]["backend"] in {"profiling.tracing", "cProfile"}
    path.write_text("value = 2\n")
    assert source_binding(tmp_path, [path])["sha256"] != binding["sha256"]
    report = {"runtime": {"python": "3.15.0"}, "source": binding, "workloads": [result]}
    changed = {**report, "runtime": {"python": "3.12.13"}}
    comparison = compare_reports(report, changed)["workloads"][0]
    assert not comparison["comparable"]
    assert comparison["cpu_ratio_after_over_before"] is None
    with pytest.raises(ValueError, match="repeats"):
        measure(workload, root=tmp_path, repeats=0)


def test_citation_integrity_rejects_duplicate_entries_and_missing_evidence():
    bib = "@article{known,\n author={Person},\n title={A {nested} title},\n year={2009},\n doi={10.1234/example}\n}"
    assert bibliography_entries(bib)["known"]["title"] == "A {nested} title"
    with pytest.raises(ValueError, match="duplicate"):
        bibliography_entries(bib + "\n" + bib)
    with pytest.raises(ValueError, match="unclosed"):
        bibliography_entries(bib[:-1])
    registry = {
        "claims": [
            {
                "id": "claim",
                "reference_keys": ["known"],
                "evidence_paths": ["../outside"],
                "limitations": ["bounded"],
            }
        ]
    }
    result = audit_research(
        bib,
        {"paper": "See [@missing] and [@fig:diagram]."},
        registry,
        existing_paths=set(),
    )
    assert "missing_citation:missing" in result["errors"]
    assert any("unsafe_evidence" in error for error in result["errors"])
    assert not any("fig:diagram" in error for error in result["errors"])


def test_figure_export_includes_selectable_vector_text_and_closes_figures(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    before = dict(matplotlib.rcParams)
    fig, ax = plt.subplots()
    ax.plot([1, 2], [2, 3], marker="o")
    ax.set_title("Measured observations")
    path = export_figure(
        fig,
        tmp_path / "measurements.png",
        description="A measured sequence with two observations",
    )
    assert path.read_bytes().startswith(b"\x89PNG")
    svg = path.with_suffix(".svg").read_text()
    assert "<text" in svg and "A measured sequence" in svg
    assert not plt.fignum_exists(fig.number)
    assert dict(matplotlib.rcParams) == before
