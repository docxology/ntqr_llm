"""Independent supporting figure builders over recorded simulation summaries."""

from __future__ import annotations
from pathlib import Path
from .figure_export import export_figure, publication_figure
from .artifacts import generated_at


@publication_figure
def plot_convergence(summary: dict, out_dir: Path, footer: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["text.usetex"] = False
    rows = [r for r in summary["convergence"]["rows"] if r["accuracy_mae_mean"]]
    qs = [r["q"] for r in rows]
    acc = [r["accuracy_mae_mean"] for r in rows]
    acc_sd = [r["accuracy_mae_sd"] for r in rows]
    prev = [r["prevalence_abs_error_mean"] for r in rows]
    fit = summary["convergence"]["sqrt_q_fit_accuracy"]

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.errorbar(
        qs,
        acc,
        yerr=acc_sd,
        fmt="o-",
        capsize=3,
        color="#1f77b4",
        label="accuracy MAE (mean ± sd)",
    )
    ax.plot(qs, prev, "s--", color="#2ca02c", label="prevalence abs. error")
    # reference 1/sqrt(Q) anchored at the first point
    ref = [acc[0] * (qs[0] / q) ** 0.5 for q in qs]
    ax.plot(qs, ref, ":", color="#888888", label=r"$\propto 1/\sqrt{Q}$ reference")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Test size Q (log scale)")
    ax.set_ylabel("Unsupervised recovery error (log scale)")
    ax.set_title(
        "NTQR recovery error falls like sampling noise\n"
        f"(fitted slope {fit['slope']:.2f}, "
        f"$R^2$={fit['r_squared']:.3f}; ideal $-0.5$)"
    )
    ax.legend(frameon=False)
    ax.grid(True, which="both", alpha=0.25)
    fig.text(0.5, 0.005, footer, ha="center", fontsize=6, color="#666666")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    path = out_dir / "synthetic_convergence.png"
    export_figure(
        fig,
        path,
        description="Recorded study summary; descriptive visualization, not a new experiment",
    )
    plt.close(fig)
    return path


@publication_figure
def plot_alarm(summary: dict, out_dir: Path, footer: str) -> Path:
    """Two panels: alarm rate per config, and the common-mode blind-spot mechanism.

    Left: how often the imaginary-prevalence alarm fires under each dependence
    structure (independent control, positive common-mode, anti-correlated pair).
    Right: the mechanism behind the blind spot -- as common-mode correlation
    strengthens, the discriminant ``D`` falls toward 0 but never below (so the
    alarm stays silent) while the recovery error climbs (so accuracy silently
    degrades). The anti-correlation discriminant is heavy-tailed, so its firing
    is read from the *rate* (left), not a mean; we keep the right panel to the
    monotone common-mode story to avoid implying a clean mean there.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["text.usetex"] = False
    rows = summary["correlation"]["rows"]
    labels = [r["label"] for r in rows]
    rates = [100 * r["alarm_rate"] for r in rows]

    def _color(label: str) -> str:
        if label.startswith("independent"):
            return "#2ca02c"
        if label.startswith("common"):
            return "#ff7f0e"
        return "#d62728"

    colors = [_color(label) for label in labels]
    x = range(len(labels))

    # Common-mode blind-spot series: independent (strength 0) + common-mode rows.
    blind = [r for r in rows if r["label"].startswith(("independent", "common"))]
    strengths = [r["common_mode"] for r in blind]
    discs = [r["discriminant_mean"] for r in blind]
    silent_err = [r["silent_prevalence_abs_error_mean"] for r in blind]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 5.2))
    ax1.bar(x, rates, color=colors)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax1.set_ylabel("Imaginary-prevalence alarm fires (% of trials)")
    ax1.set_title(
        "The alarm trips on anti-correlation,\nnot on common-mode correlation"
    )
    ax1.set_ylim(0, 105)

    ax2.axhspan(-1, 0, color="#d62728", alpha=0.07)
    ax2.plot(
        strengths,
        discs,
        "o-",
        color="#ff7f0e",
        label="discriminant $D$ (mean)",
    )
    ax2.axhline(0, color="#d62728", lw=1.2, ls="--", label="alarm threshold $D=0$")
    ax2.set_xlabel("Common-mode error-correlation strength")
    ax2.set_ylabel("Mean prevalence discriminant  $D = 1 - 4c/a$")
    ax2.set_ylim(min(0, min(discs)) - 0.004, max(discs) + 0.006)
    ax2b = ax2.twinx()
    ax2b.plot(
        strengths,
        silent_err,
        "s--",
        color="#1f77b4",
        label="recovery error (silent)",
    )
    ax2b.set_ylabel("Prevalence abs. error (alarm silent)", color="#1f77b4")
    ax2b.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.set_title(
        "Blind spot: D approaches 0 but never crosses,\n"
        "while recovery silently degrades"
    )
    lines1, lab1 = ax2.get_legend_handles_labels()
    lines2, lab2 = ax2b.get_legend_handles_labels()
    ax2.legend(
        lines1 + lines2,
        lab1 + lab2,
        frameon=False,
        fontsize=8,
        loc="center right",
    )

    fig.text(0.5, 0.005, footer, ha="center", fontsize=6, color="#666666")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path = out_dir / "synthetic_alarm.png"
    export_figure(
        fig,
        path,
        description="Recorded study summary; descriptive visualization, not a new experiment",
    )
    plt.close(fig)
    return path


@publication_figure
def plot_tiebreak(summary: dict, out_dir: Path, footer: str) -> Path:
    """The better-than-random tie-break's safe region vs judge accuracy."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["text.usetex"] = False
    rows = summary["tie_break"]["rows"]
    acc = [r["accuracy"] for r in rows]
    cat = [100 * r["catastrophic_rate"] for r in rows]
    perr = [r["prevalence_abs_error_mean"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.plot(
        acc,
        cat,
        "o-",
        color="#d62728",
        label="catastrophic mirror-selection (%)",
    )
    ax.axvline(0.5, color="#7f7f7f", ls=":", lw=1, label="random ($acc=0.5$)")
    safe = summary["tie_break"]["safe_accuracy"]
    if safe is not None:
        ax.axvline(
            safe,
            color="#2ca02c",
            ls="--",
            lw=1.2,
            label=f"clean down to acc={safe:.2f}",
        )
    ax.set_xlabel("Per-label judge accuracy (symmetric trio)")
    ax.set_ylabel("Catastrophic mirror-selection (% of trials)")
    ax.set_ylim(-3, 103)
    ax.invert_xaxis()  # strong judges on the left, approaching random rightward
    ax2 = ax.twinx()
    ax2.plot(
        acc,
        perr,
        "s--",
        color="#1f77b4",
        label="mean prevalence error",
    )
    ax2.set_ylabel("Mean prevalence abs. error", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")
    ax.set_title(
        "The better-than-random tie-break is safe for strong judges\n"
        "and inverts as the ensemble approaches random"
    )
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, frameon=False, fontsize=8)
    fig.text(0.5, 0.005, footer, ha="center", fontsize=6, color="#666666")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    path = out_dir / "synthetic_tiebreak.png"
    export_figure(
        fig,
        path,
        description="Recorded study summary; descriptive visualization, not a new experiment",
    )
    plt.close(fig)
    return path


@publication_figure
def plot_mv_vs_ei(summary: dict, out_dir: Path, footer: str) -> Path:
    """EI vs MV recovery MAE across accuracy, one panel per prevalence."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["text.usetex"] = False
    block = summary["mv_vs_ei"]
    prevalences = block["prevalences"]
    rows = block["rows"]

    fig, axes = plt.subplots(
        1, len(prevalences), figsize=(4.2 * len(prevalences), 4.8), sharey=True
    )
    if len(prevalences) == 1:
        axes = [axes]
    for ax, prev in zip(axes, prevalences):
        cells = [r for r in rows if r["prevalence_a"] == prev]
        accs = [r["accuracy"] for r in cells]
        ei = [r["ei_mae_mean"] for r in cells]
        mv = [r["mv_mae_mean"] for r in cells]
        ax.plot(accs, ei, "o-", color="#1f77b4", label="error-independent")
        ax.plot(accs, mv, "s--", color="#ff7f0e", label="majority-voting")
        ax.invert_xaxis()  # strong judges left, approaching random rightward
        ax.set_yscale("log")
        ax.set_title(f"prevalence {prev:.2f}")
        ax.set_xlabel("Per-label judge accuracy")
        ax.grid(True, which="both", alpha=0.2)
    axes[0].set_ylabel("Recovery MAE (log)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Error-independent algebra is more accurate when its assumption holds;\n"
        "majority voting is more robust in the tie-break danger zone "
        "(weak judges, extreme prevalence)",
        fontsize=11,
    )
    fig.text(0.5, 0.005, footer, ha="center", fontsize=6, color="#666666")
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    path = out_dir / "synthetic_mv_vs_ei.png"
    export_figure(
        fig,
        path,
        description="Recorded study summary; descriptive visualization, not a new experiment",
    )
    plt.close(fig)
    return path


@publication_figure
def plot_bootstrap(summary: dict, out_dir: Path, footer: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["text.usetex"] = False
    mae = summary["recovery_mae"]
    samples = summary["_mae_samples"]
    point = summary["point_recovery_mae"]
    floor = summary["sampling_noise_floor"]

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.hist(samples, bins=40, color="#1f77b4", alpha=0.75, edgecolor="white")
    ax.axvline(
        point,
        color="#d62728",
        lw=2,
        label=f"point estimate {point:.3f}",
    )
    ax.axvspan(
        mae["ci_lo"],
        mae["ci_hi"],
        color="#2ca02c",
        alpha=0.12,
        label=f"95% CI [{mae['ci_lo']:.3f}, {mae['ci_hi']:.3f}]",
    )
    if floor is not None:
        ax.axvline(
            floor,
            color="#7f7f7f",
            ls="--",
            lw=1.2,
            label=f"binomial SE reference {floor:.3f}",
        )
    ax.set_xlabel("Unsupervised recovery MAE (per scenario resample)")
    ax.set_ylabel("Bootstrap resamples")
    ax.set_title(
        f"Recovery MAE is stable under scenario resampling\n"
        f"({summary['n_boot']} resamples of {summary['n_scenarios']} scenarios; "
        f"{summary['degenerate_rate'] * 100:.1f}% degenerate)"
    )
    ax.legend(frameon=False)
    fig.text(0.5, 0.005, footer, ha="center", fontsize=6, color="#666666")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    path = out_dir / "bootstrap_recovery_mae.png"
    export_figure(
        fig,
        path,
        description="Recorded study summary; descriptive visualization, not a new experiment",
    )
    plt.close(fig)
    return path


@publication_figure
def plot_validation_matrix(summary: dict, out_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    checks = summary["checks"]
    labels = [row["label"] for row in checks]
    y = list(range(len(checks)))[::-1]
    colors = ["#16A34A" if row["passed"] else "#DC2626" for row in checks]

    fig, ax = plt.subplots(figsize=(13.4, 7.0))
    ax.barh(y, [1] * len(checks), color=colors, alpha=0.88)
    for yi, row in zip(y, checks):
        status = "PASS" if row["passed"] else "FAIL"
        ax.text(
            0.02,
            yi,
            status,
            va="center",
            ha="left",
            color="white",
            weight="bold",
            fontsize=10,
        )
        ax.text(
            1.03,
            yi,
            row["evidence"],
            va="center",
            ha="left",
            fontsize=8.6,
            color="#222222",
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(0, 1.95)
    ax.set_xticks([])
    ax.set_title(
        "Statistical validation matrix",
        fontsize=15,
        weight="bold",
    )
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.text(
        0.5,
        0.012,
        f"{summary['n_passed']}/{summary['n_checks']} checks passed | "
        f"generated {generated_at()[:10]}",
        ha="center",
        fontsize=8,
        color="#666666",
    )
    fig.text(
        0.5,
        0.94,
        "Every load-bearing statistical claim is checked against generated artifacts",
        ha="center",
        fontsize=10,
        color="#333333",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    path = out_dir / "validation_matrix.png"
    export_figure(
        fig,
        path,
        description="Statistical validation checks over recorded study artifacts",
    )
    plt.close(fig)
    return path
