"""Orchestrate ntqr's logical alarm for misaligned classifiers, for both R=2
and R=3 tests.

The alarm uses only the classifiers' label-response counts and a safety
specification (each classifier must be better than 1/factor at every label) to
decide, fully unsupervised, whether at least one classifier is misaligned.

The R=2 and R=3 scenarios (Q, safety factors, observed responses) are
configurable through ``params`` -- pass ``--config FILE.json`` or repeated
``--set r2.Q=12`` style overrides. Defaults reproduce the canonical example.

Real ntqr methods exercised:
    ntqr.r2.raxioms.SingleClassifierAxioms / ntqr.r3.raxioms.SingleClassifierAxioms
    ntqr.r2.evaluations.SingleClassifierEvaluations / ntqr.r3.evaluations....
    ntqr.alarms.SingleClassifierAxiomsAlarm
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _common  # noqa: E402
import _config  # noqa: E402

_common.ensure_ntqr_importable()

from ntqr.llm.alarm_demo import run_alarm  # noqa: E402
import ntqr.r2.evaluations  # noqa: E402
import ntqr.r2.raxioms  # noqa: E402
import ntqr.r3.evaluations  # noqa: E402
import ntqr.r3.raxioms  # noqa: E402

DEFAULTS = {
    "r2": {
        "labels": ["a", "b"],
        "classifiers": ["i", "j", "k"],
        "Q": 10,
        "factors": [2, 2],
        "responses": [[7, 3], [6, 4], [8, 2]],
    },
    "r3": {
        "labels": ["a", "b", "c"],
        "classifiers": ["i", "j", "k"],
        "Q": 3,
        "factors": [2, 2, 2],
        "responses": [[1, 1, 1], [2, 1, 0], [1, 2, 0]],
    },
}


def main(argv: list[str] | None = None) -> None:
    config = _config.build_config(
        "ntqr",
        "single_classifier_alarm",
        default_params=DEFAULTS,
        argv=argv,
    )
    out = _common.output_dir(config.category, config.name)

    r2 = run_alarm(
        config.get("r2", DEFAULTS["r2"]),
        ntqr.r2.raxioms.SingleClassifierAxioms,
        ntqr.r2.evaluations.SingleClassifierEvaluations,
    )
    r3 = run_alarm(
        config.get("r3", DEFAULTS["r3"]),
        ntqr.r3.raxioms.SingleClassifierAxioms,
        ntqr.r3.evaluations.SingleClassifierEvaluations,
    )

    _common.save_json(out, "alarm_r2.json", r2)
    _common.save_json(out, "alarm_r3.json", r3)

    _common.banner("SingleClassifierAxiomsAlarm")
    print(f"R=2 are_misaligned: {r2['are_misaligned']}")
    print(f"R=3 are_misaligned: {r3['are_misaligned']}")
    print(f"-> wrote results to {out}")


if __name__ == "__main__":
    main()
