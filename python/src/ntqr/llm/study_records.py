"""Load recorded study results for offline visualization, without model clients.

The summary determines the current evaluation status. Historical comparison
files left by an earlier run cannot turn a failed model into a solved model.
"""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
from . import personas


def load_recorded_study(directory):
    directory = Path(directory)
    inputs = {}

    def read(path):
        source = path.read_bytes()
        inputs[str(path.relative_to(directory))] = hashlib.sha256(source).hexdigest()
        record = json.loads(source)
        return record.get("result", record)

    record = json.loads((directory / "summary.json").read_bytes())
    summary = read(directory / "summary.json")
    ranking = summary["ranking_by_recovery_mae"]
    by_model = {
        row["model"]: row
        for row in ranking
        if row.get("recovery_mae") is not None and not row.get("error")
    }
    rows = []
    for model in summary["requested_models"]:
        model_dir = directory / "models" / model.replace(":", "_").replace("/", "_")
        if not (model_dir / "votes.json").is_file():
            continue
        diagnostic = read(model_dir / "evaluability_diagnostic.json")
        health = read(model_dir / "prompt_health.json")
        row = {
            "model": model,
            "prompt_health": health,
            "predicted_evaluable": diagnostic["predicted_evaluable"],
            "constant_classifiers": diagnostic["constant_classifiers"],
            "mean_pairwise_disagreement": diagnostic["mean_pairwise_disagreement"],
        }
        # Bind every file consumed by the figure builders, including raw votes.
        read(model_dir / "votes.json")
        if model in by_model:
            row.update(by_model[model])
            row["comparison"] = read(model_dir / "comparison.json")
        else:
            row["error"] = "not evaluated in the recorded study summary"
        rows.append(row)
    baseline = directory / "baseline_t0.7_summary.json"
    if baseline.exists():
        read(baseline)
    identity = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    ctx = {
        "backend": summary["backend"],
        "temperature": summary["temperature"],
        "n_personas": len(personas.PERSONAS),
        "n_scenarios": summary["n_scenarios"],
        "n_labels": len(personas.LABELS),
        "n_models_requested": len(summary["requested_models"]),
        "n_models_evaluated": len(by_model),
        "prompt_version": summary.get("prompt_version"),
        "prompt_hash": summary.get("prompt_hash"),
        "date": record.get("generated_at", "")[:10],
        "figure_inputs_sha256": identity,
        "collection": "recorded responses; no model calls",
    }
    return rows, ctx, {"schema_version": 1, "sha256": identity, "files": inputs}
