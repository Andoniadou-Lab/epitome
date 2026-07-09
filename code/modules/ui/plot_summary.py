"""Small summary lines shown beneath plots (sample counts, matrix shape, etc.)."""

from __future__ import annotations

import streamlit as st


def plot_summary_caption(*segments: str | int | float | None) -> None:
    """Render a muted one-line summary below a plot, segments joined by middle dots."""
    parts: list[str] = []
    for segment in segments:
        if segment is None or segment == "":
            continue
        if isinstance(segment, float):
            parts.append(f"{segment:g}")
        elif isinstance(segment, int):
            parts.append(f"{segment:,}")
        else:
            parts.append(str(segment))
    if parts:
        st.caption(" · ".join(parts))


def heatmap_shape_caption(
    n_genes: int,
    n_columns: int,
    *,
    per_group: bool = False,
) -> None:
    """genes × samples/groups line used under expression heatmaps."""
    plot_summary_caption(
        f"{n_genes:,} genes × {n_columns:,} {'groups' if per_group else 'samples'}"
    )


def boxplot_sample_caption(
    gene: str,
    n_samples: int,
    *,
    sample_label: str = "samples",
    n_studies: int | None = None,
) -> None:
    segments: list[str | int] = [gene, f"across {n_samples} {sample_label}"]
    if n_studies is not None:
        segments.append(f"{n_studies} studies")
    plot_summary_caption(*segments)
