"""Physical-recovery status visualization; no answer key is required."""

from pathlib import Path
from .artifacts import save_record
from .figure_export import export_figure, publication_figure


@publication_figure
def plot_validity_report(report, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["models"]
    if not rows:
        raise ValueError("validity report must contain model records")
    fig, ax = plt.subplots(
        figsize=(10, max(3.5, 0.62 * len(rows) + 1.5)), layout="constrained"
    )
    colors = [
        "#0072B2" if row["physical_solution_available"] else "#A74600" for row in rows
    ]
    positions = list(range(len(rows)))
    ax.barh(positions, [1] * len(rows), color=colors, height=0.62)
    ax.set_yticks(positions, [row["model"] for row in rows])
    for i, row in enumerate(rows):
        ax.text(
            0.03,
            i,
            row["status"].replace("_", " "),
            va="center",
            color="white",
            weight="bold",
            fontsize=11,
        )
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_title("Physical recovery audit of recorded votes", loc="left", weight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.supxlabel(
        "Model-consistency check only: passing does not prove error independence or resolve label orientation.",
        fontsize=10,
    )
    caption = "Recorded votes only; no answer key or new model calls. Each status is labeled directly; color is redundant."
    output_dir = Path(output_dir)
    path = output_dir / "physical_recovery_audit.png"
    export_figure(fig, path, description=caption)
    save_record(
        output_dir,
        "physical_recovery_audit.data.json",
        {
            "figure": path.name,
            "caption": caption,
            "models": [
                {
                    "model": row["model"],
                    "status": row["status"],
                    "source_sha256": row["source_sha256"],
                }
                for row in rows
            ],
            "uses_answer_key": False,
            "metadata": {
                "study": "recorded-physical-recovery-audit",
                "n_models": len(rows),
                "uses_answer_key": False,
            },
            "exports": {"raster": path.name, "vector": path.with_suffix(".svg").name},
        },
    )
    return path
