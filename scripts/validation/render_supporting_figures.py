"""Render supporting figures from recorded summaries without rerunning experiments."""

from pathlib import Path
import sys
import json
import argparse
import hashlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common

_common.ensure_ntqr_importable()


def main(argv=None):
    from ntqr.llm.artifacts import atomic_write_bytes
    from ntqr.llm import figure_export, supporting_figures
    from ntqr.llm.supporting_figures import (
        plot_convergence,
        plot_alarm,
        plot_tiebreak,
        plot_mv_vs_ei,
        plot_bootstrap,
        plot_validation_matrix,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    root = _common.REPO_ROOT / "outputs"
    simulation = root / "sim/synthetic_recovery_study"
    bootstrap = root / "sim/bootstrap_recovery_ci"
    validation = root / "validation/statistical_validation"
    out = args.output_dir or root / "validation/render_supporting_figures"
    if out.resolve() in {
        simulation.resolve(),
        bootstrap.resolve(),
        validation.resolve(),
    }:
        raise ValueError(
            "recorded rendering must not replace producer artifacts"
        )

    sidecars = [
        simulation / name
        for name in (
            "convergence.data.json",
            "correlation.data.json",
            "tiebreak.data.json",
            "mv_vs_ei.data.json",
        )
    ] + [
        bootstrap / "bootstrap_recovery_mae.data.json",
        validation / "validation_matrix.data.json",
    ]
    inputs = sidecars + [
        simulation / "summary.json",
        bootstrap / "summary.json",
        validation / "summary.json",
    ]
    source_bytes = {path: path.read_bytes() for path in inputs}

    def load(path):
        record = json.loads(source_bytes[path])
        return record.get("result", record)

    study = load(simulation / "summary.json")
    out.mkdir(parents=True, exist_ok=True)
    paths = [
        builder(
            study,
            out,
            "Recorded synthetic study; no new trials or model calls",
        )
        for builder in (
            plot_convergence,
            plot_alarm,
            plot_tiebreak,
            plot_mv_vs_ei,
        )
    ]
    bootstrap_summary = load(bootstrap / "summary.json")
    bootstrap_summary["_mae_samples"] = load(
        bootstrap / "bootstrap_recovery_mae.data.json"
    )["samples"]
    paths.append(
        plot_bootstrap(
            bootstrap_summary,
            out,
            "Recorded scenario bootstrap; no new resamples or model calls",
        )
    )
    paths.append(
        plot_validation_matrix(
            load(validation / "summary.json"),
            out,
        )
    )
    for source in sidecars:
        atomic_write_bytes(out / source.name, source_bytes[source])
    rendering_sources = [
        Path(__file__),
        Path(_common.__file__),
        Path(supporting_figures.__file__),
        Path(figure_export.__file__),
    ]
    _common.save_json(
        out,
        "render_sources.json",
        {
            "inputs": {
                str(path.relative_to(_common.REPO_ROOT)): hashlib.sha256(
                    data
                ).hexdigest()
                for path, data in source_bytes.items()
            },
            "rendering_sources": {
                str(path.relative_to(_common.REPO_ROOT)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in rendering_sources
            },
            "figures": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for image in paths
                for path in (image, image.with_suffix(".svg"))
            },
            "new_trials": False,
            "model_calls": False,
        },
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
