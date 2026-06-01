"""Orchestrate ntqr's pair logical-alarm plot and save it as a PNG.

The ntqr plotting helper sets ``text.usetex=True``, which needs a LaTeX
install. We render with the non-interactive Agg backend and disable usetex
before saving, so the axis labels fall back to matplotlib's built-in mathtext
(``$...$``) and the figure is produced without a LaTeX toolchain.

Plot parameters (labels, classifiers, qs answer-key point, observed responses,
dpi) are configurable through ``params``. Defaults reproduce the canonical
N=2, Q=10 example.

Real ntqr methods exercised:
    ntqr.plots.plot_pair_logical_alarm_at_qs
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _common  # noqa: E402
import _config  # noqa: E402

_common.ensure_ntqr_importable()

import ntqr.plots  # noqa: E402

DEFAULTS = {
    "labels": ["a", "b"],
    "classifiers": ["i", "j"],
    "qs": [5, 5],
    "responses": [[3, 2], [4, 1]],
    "dpi": 150,
}


def main(argv: list[str] | None = None) -> None:
    config = _config.build_config(
        "ntqr",
        "pair_logical_alarm_plot",
        default_params=DEFAULTS,
        argv=argv,
    )
    out = _common.output_dir(config.category, config.name)

    labels = tuple(config.get("labels", DEFAULTS["labels"]))
    classifiers = tuple(config.get("classifiers", DEFAULTS["classifiers"]))
    qs = tuple(config.get("qs", DEFAULTS["qs"]))
    responses = [
        list(r) for r in config.get("responses", DEFAULTS["responses"])
    ]
    dpi = config.get("dpi", DEFAULTS["dpi"])

    ntqr.plots.plot_pair_logical_alarm_at_qs(
        labels, classifiers, qs, responses
    )
    # Helper forced LaTeX rendering; fall back to mathtext for headless save.
    plt.rcParams["text.usetex"] = False
    png = out / "pair_logical_alarm.png"
    plt.gcf().savefig(png, dpi=dpi, bbox_inches="tight")
    plt.close("all")

    _common.save_json(
        out,
        "parameters.json",
        {
            "labels": list(labels),
            "classifiers": list(classifiers),
            "qs": list(qs),
            "responses": responses,
            "figure": png.name,
        },
    )

    _common.banner("plot_pair_logical_alarm_at_qs")
    print(f"-> wrote figure {png} ({png.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
