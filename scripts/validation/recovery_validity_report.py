"""Audit recorded votes for physical NTQR recovery, without an answer key or model calls."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common

_common.ensure_ntqr_importable()
from ntqr.llm.validity_audit import audit_persona_records


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=_common.REPO_ROOT / "outputs/llm/multi_llm_personas_evaluation/models",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    inputs = sorted(args.models_dir.glob("*/votes.json"))
    if not inputs:
        raise FileNotFoundError(f"no recorded votes under {args.models_dir}")
    audits = []
    for path in inputs:
        source = path.read_bytes()
        record = json.loads(source)
        row = audit_persona_records(record.get("result", record))
        audits.append(
            {
                **row,
                "source_file": f"{path.parent.name}/votes.json",
                "source_sha256": hashlib.sha256(source).hexdigest(),
            }
        )
    out = args.output_dir or _common.output_dir(
        "validation", "recovery_validity_report"
    )
    _common.save_json(
        out,
        "summary.json",
        {
            "uses_answer_key": False,
            "n_models": len(audits),
            "status_counts": dict(Counter(row["status"] for row in audits)),
            "models": audits,
            "interpretation": "A physical solution is a consistency check under error independence; it does not prove that assumption or resolve the label mirror without a better-than-random assumption.",
        },
    )
    print(f"Audited {len(audits)} recorded models: {out / 'summary.json'}")


if __name__ == "__main__":
    main()
