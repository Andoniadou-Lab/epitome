"""Small summary lines shown beneath plots (sample counts, matrix shape, etc.)."""

from __future__ import annotations

from collections.abc import Sequence

import streamlit as st

from modules.versioning import format_version_label, get_resolved_version


def plot_summary_caption(
    *segments: str | int | float | None,
    version: str | None = None,
    loader_key: str | None = None,
    loader_keys: Sequence[str] | None = None,
    resolved_version: str | None = None,
) -> None:
    """Render a muted one-line summary below a plot, segments joined by middle dots.

    When ``version`` is set, appends the data version (or ``Fall back to v_x.yy``
    if a lower version was used). Pass every loader the plot reads via
    ``loader_key``/``loader_keys`` so the label reflects this plot's data rather
    than unrelated loaders warmed elsewhere in the session.
    """
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
    if version is not None:
        actual = resolved_version or get_resolved_version(
            version, loader_key, loader_keys
        )
        parts.append(format_version_label(version, actual))
    if parts:
        st.caption(" · ".join(parts))


def version_note(
    version: str | None,
    loader_key: str | None = None,
    loader_keys: Sequence[str] | None = None,
) -> str:
    """Suffix for table summary lines: ``' · v_0.03'`` / ``' · Fall back to v_0.02'``.

    Returns an empty string when no version is given, so callers can append it
    unconditionally.
    """
    if version is None:
        return ""
    resolved = get_resolved_version(version, loader_key, loader_keys)
    return f" · {format_version_label(version, resolved)}"


def heatmap_shape_caption(
    n_genes: int,
    n_columns: int,
    *,
    per_group: bool = False,
    version: str | None = None,
    loader_key: str | None = None,
    loader_keys: Sequence[str] | None = None,
) -> None:
    """genes × samples/groups line used under expression heatmaps."""
    plot_summary_caption(
        f"{n_genes:,} genes × {n_columns:,} {'groups' if per_group else 'samples'}",
        version=version,
        loader_key=loader_key,
        loader_keys=loader_keys,
    )


def boxplot_sample_caption(
    gene: str,
    n_samples: int,
    *,
    sample_label: str = "samples",
    n_studies: int | None = None,
    version: str | None = None,
    loader_key: str | None = None,
    loader_keys: Sequence[str] | None = None,
) -> None:
    segments: list[str | int] = [gene, f"across {n_samples} {sample_label}"]
    if n_studies is not None:
        segments.append(f"{n_studies} studies")
    plot_summary_caption(
        *segments,
        version=version,
        loader_key=loader_key,
        loader_keys=loader_keys,
    )
