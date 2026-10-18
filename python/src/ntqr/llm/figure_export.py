"""Atomic PNG and accessible, selectable-text SVG exports for publication figures."""

from __future__ import annotations
from io import BytesIO
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from .artifacts import atomic_write_bytes


def export_figure(fig, path, *, description="", dpi=200):
    path = Path(path)
    if path.suffix.lower() not in {".png", ".svg", ".pdf"}:
        raise ValueError("figure path must end in .png, .svg, or .pdf")
    targets = [path]
    if path.suffix.lower() == ".png":
        targets.append(path.with_suffix(".svg"))
    try:
        with matplotlib.rc_context(
            {"svg.fonttype": "none", "svg.hashsalt": "ntqr-0.9", "text.usetex": False}
        ):
            for target in targets:
                kind = target.suffix[1:].lower()
                metadata = (
                    {
                        "Title": (
                            fig._suptitle.get_text()
                            if fig._suptitle
                            else path.stem.replace("_", " ")
                        ),
                        "Description": description,
                        "Creator": "NTQR research figure pipeline",
                    }
                    if kind == "svg"
                    else {"Description": description}
                )
                if kind == "svg":
                    metadata["Date"] = None
                if kind == "pdf":
                    metadata = {
                        "Title": path.stem,
                        "Subject": description,
                        "Creator": "NTQR research figure pipeline",
                        "CreationDate": None,
                        "ModDate": None,
                    }
                buffer = BytesIO()
                fig.savefig(
                    buffer, format=kind, dpi=dpi, bbox_inches="tight", metadata=metadata
                )
                atomic_write_bytes(target, buffer.getvalue())
    finally:
        plt.close(fig)
    return path


def publication_figure(builder):
    """Contain global style changes and close newly-created figures on failure."""
    from functools import wraps

    @wraps(builder)
    def render(*args, **kwargs):
        existing = set(plt.get_fignums())
        try:
            with matplotlib.rc_context(
                {
                    "text.usetex": False,
                    "svg.fonttype": "none",
                    "svg.hashsalt": "ntqr-0.9",
                }
            ):
                return builder(*args, **kwargs)
        finally:
            for number in set(plt.get_fignums()) - existing:
                plt.close(number)

    return render
