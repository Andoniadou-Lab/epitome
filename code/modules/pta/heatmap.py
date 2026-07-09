"""Heatmap construction and plotting for PTA bulk RNA-seq."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules.pta.cell_type_labels import (
    ordered_group_keys,
    ordered_sample_index,
    sort_pta_categories,
)

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#dbdb8d", "#9edae5", "#393b79",
]


def select_genes(expr, gene_list=None, top_variable=None):
    if gene_list:
        lower_to_actual = {g.lower(): g for g in expr.index}
        out = []
        for g in gene_list:
            key = g.strip().lower()
            if key in lower_to_actual and lower_to_actual[key] not in out:
                out.append(lower_to_actual[key])
        return out

    n = top_variable or 30
    variances = expr.var(axis=1).sort_values(ascending=False)
    return variances.head(n).index.tolist()


def build_matrix(
    expr,
    meta,
    genes,
    group_cols,
    per_group=False,
    zscore=False,
    merge_mixed: bool = True,
):
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    sub = expr.loc[genes]

    if per_group:
        keys = [meta[c] for c in group_cols]
        grouped = sub.T.groupby(keys).mean().T
        sorted_keys = ordered_group_keys(grouped.columns, group_cols, merge_mixed=merge_mixed)
        grouped = grouped[sorted_keys]
        if len(group_cols) == 1:
            ann_values = [list(grouped.columns)]
        else:
            ann_values = [[tup[i] for tup in grouped.columns] for i in range(len(group_cols))]
        annotations = [
            pd.Series([ann_values[k][i] for i in range(len(grouped.columns))], index=grouped.columns)
            for k in range(len(group_cols))
        ]
        sub = grouped
    else:
        ordered_index = ordered_sample_index(meta, group_cols, merge_mixed=merge_mixed)
        ordered_index = [s for s in ordered_index if s in sub.columns]
        sub = sub[ordered_index]
        annotations = [meta.loc[ordered_index, c] for c in group_cols]

    if zscore:
        means = sub.mean(axis=1)
        stds = sub.std(axis=1).replace(0, np.nan)
        sub = sub.sub(means, axis=0).div(stds, axis=0).fillna(0.0)

    return sub, annotations


def _category_colours(categories, override: dict[str, str] | None = None):
    if override:
        return {str(c): override.get(str(c), _FALLBACK_COLOUR(str(c), override)) for c in categories}
    return {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(categories)}


def _FALLBACK_COLOUR(label: str, existing: dict[str, str]) -> str:
    used = set(existing.values())
    for colour in PALETTE:
        if colour not in used:
            return colour
    return PALETTE[len(label) % len(PALETTE)]


def _annotation_trace(labels, samples, col_name, cat_colours):
    categories = list(cat_colours.keys())
    cat_to_code = {c: i for i, c in enumerate(categories)}
    n = len(categories)

    codes = [[cat_to_code[str(labels[s])] for s in samples]]
    customdata = [[str(labels[s]) for s in samples]]

    if n == 1:
        colorscale = [[0, cat_colours[categories[0]]], [1, cat_colours[categories[0]]]]
    else:
        colorscale = []
        for i, c in enumerate(categories):
            colorscale.append([i / n, cat_colours[c]])
            colorscale.append([(i + 1) / n, cat_colours[c]])

    return go.Heatmap(
        z=codes,
        x=samples,
        y=[col_name],
        colorscale=colorscale,
        showscale=False,
        zmin=0,
        zmax=max(n - 1, 1),
        customdata=customdata,
        hovertemplate="%{x}<br>" + col_name + ": %{customdata}<extra></extra>",
    )


def create_heatmap(
    matrix_df,
    annotations,
    group_cols,
    zscore=False,
    download_as="svg",
    annotation_color_maps: list[dict[str, str] | None] | None = None,
    merge_mixed: bool = True,
):
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    if matrix_df.shape[0] == 0:
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5, xref="paper", yref="paper",
            text="No genes matched your selection",
            showarrow=False, font=dict(size=15),
        )
        fig.update_layout(height=300)
        return fig, {"toImageButtonOptions": {"format": download_as, "filename": "rna_heatmap"}}

    samples = list(matrix_df.columns)
    genes = list(matrix_df.index)
    n_bars = len(group_cols)

    cat_colour_maps = []
    for i, ann in enumerate(annotations):
        cats = sort_pta_categories(
            ann.reindex(samples).astype(str).unique(),
            group_cols[i],
            merge_mixed=merge_mixed,
        )
        override = None
        if annotation_color_maps and i < len(annotation_color_maps):
            override = annotation_color_maps[i]
        cat_colour_maps.append(_category_colours(cats, override))

    value_label = "z-score" if zscore else "log1p(CPM)"
    expr_colorscale = "RdBu_r" if zscore else "Viridis"

    bar_h = 0.04
    rows = n_bars + 1
    row_heights = [bar_h] * n_bars + [1 - bar_h * n_bars]
    fig = make_subplots(
        rows=rows, cols=1,
        row_heights=row_heights,
        vertical_spacing=0.008,
        shared_xaxes=True,
    )

    for i, (ann, col_name) in enumerate(zip(annotations, group_cols)):
        labels = ann.reindex(samples).astype(str)
        fig.add_trace(
            _annotation_trace(labels, samples, col_name, cat_colour_maps[i]),
            row=i + 1, col=1,
        )

    fig.add_trace(
        go.Heatmap(
            z=matrix_df.values,
            x=samples,
            y=genes,
            colorscale=expr_colorscale,
            colorbar=dict(title=value_label, len=0.45, y=0.5, x=1.02),
            hovertemplate="Gene: %{y}<br>Col: %{x}<br>" + value_label + ": %{z:.2f}<extra></extra>",
        ),
        row=rows, col=1,
    )

    for col_name, cmap in zip(group_cols, cat_colour_maps):
        for cat, colour in cmap.items():
            fig.add_trace(
                go.Scatter(
                    x=[None], y=[None],
                    mode="markers",
                    marker=dict(size=12, color=colour, symbol="square"),
                    name=str(cat),
                    legendgroup=col_name,
                    legendgrouptitle_text=col_name,
                    showlegend=True,
                ),
                row=rows, col=1,
            )

    height = max(420, 30 * len(genes) + 80 + 70 * n_bars)
    bottom_margin = 120 if len(samples) <= 20 else 40
    width = int(height * 2.5)
    fig.update_layout(
        width=width,
        height=height,
        autosize=False,
        margin=dict(l=90, r=220, t=40, b=bottom_margin),
        plot_bgcolor="white",
        legend=dict(
            x=1.12, y=1, xanchor="left", yanchor="top",
            groupclick="toggleitem", itemsizing="constant",
            tracegroupgap=14,
        ),
    )
    for r in range(1, n_bars + 1):
        fig.update_xaxes(showticklabels=False, row=r, col=1)
    show_x_labels = len(samples) <= 20
    fig.update_xaxes(tickangle=90, showticklabels=show_x_labels, row=rows, col=1)

    config = {
        "toImageButtonOptions": {
            "format": download_as,
            "filename": "rna_heatmap",
            "width": width,
            "height": height,
            "scale": 2,
        }
    }
    return fig, config
