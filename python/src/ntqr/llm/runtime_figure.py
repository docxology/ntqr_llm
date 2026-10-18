"""Descriptive runtime distributions and Python allocation peaks from real receipts."""

from pathlib import Path
from .figure_export import export_figure, publication_figure


@publication_figure
def plot_runtime_report(report, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["workloads"]
    fig, axes = plt.subplots(
        1, 2, figsize=(11, max(4, 0.8 * len(rows) + 2)), layout="constrained"
    )
    labels = [row["name"] for row in rows]
    y = list(range(len(rows)))
    median = [row["cpu_median_ns"] / 1e6 for row in rows]
    low = [
        median[i] - min(s["cpu_ns"] for s in row["unprofiled_samples"]) / 1e6
        for i, row in enumerate(rows)
    ]
    high = [
        max(s["cpu_ns"] for s in row["unprofiled_samples"]) / 1e6 - median[i]
        for i, row in enumerate(rows)
    ]
    axes[0].errorbar(median, y, xerr=[low, high], fmt="o", color="#0072B2", capsize=4)
    for i, row in enumerate(rows):
        samples = [sample["cpu_ns"] / 1e6 for sample in row["unprofiled_samples"]]
        axes[0].scatter(
            samples, [i] * len(samples), marker="|", color="#555555", s=85, zorder=3
        )
    axes[0].set_yticks(y, labels)
    axes[0].set_xlabel("Parent-process CPU milliseconds (unprofiled)")
    axes[0].set_title(
        "Median, range, and individual timing samples", loc="left", fontsize=11
    )
    axes[1].barh(
        y, [row["memory"]["peak_bytes"] / 1024 for row in rows], color="#009E73"
    )
    axes[1].set_yticks(y, labels)
    axes[1].set_xlabel("Python traced allocation peak (KiB)")
    axes[1].set_title(
        "Separate instrumented pass; excludes native heaps", loc="left", fontsize=11
    )
    for ax in axes:
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle(
        f"Bounded NTQR runtime evidence — Python {report['runtime']['python']}",
        weight="bold",
    )
    caption = "Fixed seeded inputs; timing excludes imports and profiling. Ranges describe host variability, not confidence intervals."
    fig.supxlabel(caption, fontsize=9)
    return export_figure(fig, Path(path), description=caption)
