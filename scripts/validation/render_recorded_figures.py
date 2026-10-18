"""Regenerate model figures exclusively from persisted responses and comparisons."""

from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common

_common.ensure_ntqr_importable()
from ntqr.llm.study_records import load_recorded_study
from ntqr.llm.visualization import plot_models


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=_common.REPO_ROOT / "outputs/llm/multi_llm_personas_evaluation",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--figure", action="append")
    args = parser.parse_args(argv)
    rows, ctx, manifest = load_recorded_study(args.source_dir)
    out = args.output_dir or args.source_dir
    paths = plot_models(rows, out, ctx, figures=args.figure, source_dir=args.source_dir)
    _common.save_json(out, "figure_inputs.json", manifest)
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
