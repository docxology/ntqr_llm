"""Render supporting figures from recorded summaries without rerunning experiments."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common

_common.ensure_ntqr_importable()
from ntqr.llm.supporting_figures import (
    plot_convergence,
    plot_alarm,
    plot_tiebreak,
    plot_mv_vs_ei,
    plot_bootstrap,
    plot_validation_matrix,
)
import json


def main():
    root = _common.REPO_ROOT / "outputs"

    def load(path):
        record = json.loads(path.read_text(encoding="utf-8"))
        return record.get("result", record)

    simulation = root / "sim/synthetic_recovery_study"
    study = load(simulation / "summary.json")
    footer = "Recorded synthetic study; no new experiments or model calls"
    for builder in (plot_convergence, plot_alarm, plot_tiebreak, plot_mv_vs_ei):
        print(builder(study, simulation, footer))
    bootstrap = root / "sim/bootstrap_recovery_ci"
    bootstrap_summary = load(bootstrap / "summary.json")
    bootstrap_summary["_mae_samples"] = load(
        bootstrap / "bootstrap_recovery_mae.data.json"
    )["samples"]
    print(plot_bootstrap(bootstrap_summary, bootstrap, footer))
    validation = root / "validation/statistical_validation"
    print(plot_validation_matrix(load(validation / "summary.json"), validation))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
