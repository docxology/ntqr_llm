"""Configurable, composable settings for the thin orchestrators.

The orchestrators stay thin: they import ``ntqr``, call its public methods, and
save results. This module is the one place that turns *configuration* into the
inputs those calls need, so a single script can be pointed at different data or
parameters without editing its body.

Two ideas live here:

1. **A dataset registry** -- the composition seam. A "dataset" is a labeled
   vote-count mapping (``{label: {votes: count}}``). Built-ins are registered
   by name (``uci_adult``); other method modules (e.g. ``scripts/llm/``)
   register their generated data or persist it to a JSON file that any
   evaluator orchestrator can then consume via ``--dataset-file``.

2. **``OrchestratorConfig``** -- a small, uniform config object every
   orchestrator builds from (defaults -> optional JSON file -> CLI overrides).
   Method-specific knobs live in ``params`` so the object stays general.

Resolution order (lowest to highest precedence):
    code defaults  ->  --config FILE.json  ->  individual CLI flags
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

# scripts/_config.py -> scripts/ -> repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = Path(__file__).resolve().parent / "configs"

# A labeled vote-count mapping: {label: {(vote, vote, ...): count}}.
LabelVoteCounts = Mapping[str, Mapping[tuple, int]]
DatasetLoader = Callable[[], LabelVoteCounts]

# Name -> zero-arg loader returning labeled vote counts. Other modules extend
# this with register_dataset(); keep loaders lazy so importing this module
# never forces an ntqr import.
DATASETS: dict[str, DatasetLoader] = {}


def register_dataset(name: str, loader: DatasetLoader) -> None:
    """Register a labeled-vote-count loader under ``name`` (idempotent)."""
    DATASETS[name] = loader


def _uci_adult() -> LabelVoteCounts:
    from ntqr.r2.examples import uciadult_label_counts

    return uciadult_label_counts


register_dataset("uci_adult", _uci_adult)


# --- label-vote-counts JSON (de)serialization -----------------------------
# Tuple keys can't be JSON keys, so vote patterns are stored as "a,a,a".


def label_vote_counts_to_json(counts: LabelVoteCounts) -> dict:
    return {
        label: {",".join(votes): count for votes, count in by_votes.items()}
        for label, by_votes in counts.items()
    }


def label_vote_counts_from_json(data: Mapping[str, Mapping[str, int]]) -> dict:
    return {
        label: {
            tuple(key.split(",")): int(count)
            for key, count in by_votes.items()
        }
        for label, by_votes in data.items()
    }


def save_label_vote_counts(path: Path, counts: LabelVoteCounts) -> Path:
    path.write_text(
        json.dumps(label_vote_counts_to_json(counts), indent=2) + "\n"
    )
    return path


def load_label_vote_counts_file(path: Path) -> dict:
    return label_vote_counts_from_json(json.loads(Path(path).read_text()))


# --- the config object -----------------------------------------------------


@dataclass
class OrchestratorConfig:
    """Uniform configuration shared by every orchestrator.

    ``dataset`` / ``dataset_file`` only matter for the evaluator orchestrators
    that consume labeled vote counts. Alarm/plot/set orchestrators read their
    synthetic knobs straight from ``params``.
    """

    category: str
    name: str
    dataset: str = "uci_adult"
    dataset_file: str | None = None
    labels: tuple[str, ...] = ("a", "b")
    params: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    @property
    def data_source(self) -> str:
        """Human label for the active data source (for banners/summaries)."""
        if self.dataset_file:
            return Path(self.dataset_file).name
        return self.dataset

    def load_label_vote_counts(self) -> dict:
        """Resolve this config's labeled vote counts.

        ``--dataset-file`` (an explicit JSON path) wins over the named
        ``--dataset`` from the registry.
        """
        if self.dataset_file:
            return load_label_vote_counts_file(Path(self.dataset_file))
        if self.dataset not in DATASETS:
            raise KeyError(
                f"unknown dataset {self.dataset!r}; "
                f"known: {sorted(DATASETS)} (or pass --dataset-file)"
            )
        return dict(DATASETS[self.dataset]())


def build_config(
    category: str,
    name: str,
    *,
    default_params: Mapping[str, Any] | None = None,
    default_dataset: str = "uci_adult",
    default_labels: tuple[str, ...] = ("a", "b"),
    argv: list[str] | None = None,
) -> OrchestratorConfig:
    """Construct an ``OrchestratorConfig`` from defaults, a JSON file, and CLI.

    Recognised flags:
        --config FILE.json   merge a JSON config (keys: dataset, dataset_file,
                             labels, params)
        --dataset NAME       pick a registered dataset
        --dataset-file FILE  load labeled vote counts from JSON (overrides
                             --dataset)
        --output-name NAME   override the output subfolder name
        --set key=value      set one params entry; value parsed as JSON, then
                             falls back to a bare string
    """
    parser = argparse.ArgumentParser(prog=f"{category}/{name}", add_help=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset-file", type=str, default=None)
    parser.add_argument("--output-name", type=str, default=None)
    parser.add_argument("--set", action="append", default=[], metavar="K=V")
    args = parser.parse_args(argv)

    config = OrchestratorConfig(
        category=category,
        name=name,
        dataset=default_dataset,
        labels=tuple(default_labels),
        params=dict(default_params or {}),
    )

    if args.config:
        blob = json.loads(_resolve_config_path(args.config).read_text())
        if "dataset" in blob:
            config.dataset = blob["dataset"]
        if "dataset_file" in blob:
            config.dataset_file = blob["dataset_file"]
        if "labels" in blob:
            config.labels = tuple(blob["labels"])
        config.params.update(blob.get("params", {}))

    if args.dataset:
        config.dataset = args.dataset
    if args.dataset_file:
        config.dataset_file = args.dataset_file
    if args.output_name:
        config.name = args.output_name
    for item in args.set:
        key, _, raw = item.partition("=")
        config.params[key.strip()] = _parse_scalar(raw)

    return config


def _resolve_config_path(value: str) -> Path:
    """Allow ``--config name`` (looked up in ``scripts/configs/``) or a path."""
    candidate = Path(value)
    if candidate.exists():
        return candidate
    named = CONFIGS_DIR / value
    if named.exists():
        return named
    named_json = CONFIGS_DIR / f"{value}.json"
    if named_json.exists():
        return named_json
    raise FileNotFoundError(f"config not found: {value}")


def _parse_scalar(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw
