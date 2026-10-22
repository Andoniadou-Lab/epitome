"""Volcano plot for PTA differential-expression result tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from modules.pta.gene_annotation import format_clinical_approval_stage, format_clinical_target_drugs
from modules.pta.label_repel import (
    LABEL_FONT_FAMILY,
    PlotFrame,
    anchor_edge_x,
    repel_labels,
    text_box_size,
)


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.upper().isin({"TRUE", "1", "T", "YES"})


# Category column -> label, in the order labels are joined for combinations.
CATEGORY_COLUMNS = {
    "is_tf": "Transcription factor",
    "is_ligand": "Ligand",
    "is_receptor": "Receptor",
    "is_metabolism": "Metabolism",
    "is_clinical_target": "Clinical target",
}
COMBINATION_SEPARATOR = " + "


def gene_categories(df: pd.DataFrame, columns: list[str] | None = None) -> pd.Series:
    """``"Receptor + Clinical target"``-style label per gene ('' when none apply)."""
    columns = [c for c in (columns or list(CATEGORY_COLUMNS)) if c in df.columns]
    labels = pd.Series("", index=df.index, dtype=object)
    for col in columns:
        hit = _as_bool(df[col])
        name = CATEGORY_COLUMNS[col]
        labels[hit] = labels[hit].where(labels[hit] == "", labels[hit] + COMBINATION_SEPARATOR) + name
    return labels


def classify_points(
    df: pd.DataFrame,
    *,
    pval_threshold: float = 0.05,
    logfc_threshold: float = 1.0,
    highlight_tf: bool = False,
    highlight_ligand: bool = False,
    highlight_receptor: bool = False,
    highlight_metabolism: bool = False,
    highlight_clinical_target: bool = False,
    highlight_genes: list[str] | None = None,
) -> pd.Series:
    """Assign visual groups. All points are plotted; thresholds affect colour only.

    Category highlights apply to significant genes only. A gene in several
    checked categories gets a combined group such as ``"Receptor + Clinical target"``.
    """
    sig = (df["adj.P.Val"] < pval_threshold) & (df["logFC"].abs() >= logfc_threshold)
    groups = pd.Series("Not significant", index=df.index, dtype=object)
    groups[sig & (df["logFC"] > 0)] = "Upregulated"
    groups[sig & (df["logFC"] < 0)] = "Downregulated"

    checked = [
        col
        for col, on in (
            ("is_tf", highlight_tf),
            ("is_ligand", highlight_ligand),
            ("is_receptor", highlight_receptor),
            ("is_metabolism", highlight_metabolism),
            ("is_clinical_target", highlight_clinical_target),
        )
        if on
    ]
    if checked:
        categories = gene_categories(df, checked)
        highlighted = sig & (categories != "")
        groups[highlighted] = categories[highlighted]

    selected = set(highlight_genes or [])
    if selected and "gene" in df.columns:
        groups[df["gene"].isin(selected)] = "Selected genes"

    return groups


COLOR_MAP = {
    "Not significant": "#B0B0B0",
    "Upregulated": "#cc0000",
    "Downregulated": "#1f77b4",
    "Transcription factor": "#e377c2",
    "Ligand": "#2ca02c",
    "Receptor": "#ff7f0e",
    "Metabolism": "#17becf",
    "Clinical target": "#bcbd22",
    "Selected genes": "#000000",
    # Combinations seen in the v_0.04 annotations.
    "Ligand + Receptor": "#9467bd",
    "Ligand + Clinical target": "#006d2c",
    "Receptor + Clinical target": "#8c564b",
    "Transcription factor + Clinical target": "#ad1457",
    "Ligand + Receptor + Clinical target": "#393b79",
}
_COMBINATION_FALLBACK_COLORS = ["#7b4173", "#637939", "#8c6d31", "#843c39", "#5254a3", "#bd9e39"]

VOLCANO_EXPORT_PX = 800
MARGIN_LEFT_PX = 80
MARGIN_TOP_PX = 80
MARGIN_BOTTOM_PX = 70
LEGEND_CHAR_PX = 7.2
LABEL_FONT_SIZE = 16
LABEL_POINT_RADIUS_PX = 5.0
LABEL_OBSTACLE_REACH_PX = 200.0
NOT_SIGNIFICANT_OPACITY = 0.5
DIRECTION_OPACITY = 0.85
CATEGORY_OPACITY = 1.0
SELECTED_MARKER_OPACITY = 1.0

_HOVER_TEMPLATE = (
    "<b>%{text}</b><br>"
    "logFC: %{customdata[1]:.3f}<br>"
    "adj.P.Val: %{customdata[0]:.2e}<br>"
    "Categories: %{customdata[5]}<br>"
    "Approval stage: %{customdata[3]}<br>"
    "Drugs: %{customdata[4]}<extra></extra>"
)


def _group_colors(groups: list[str]) -> dict[str, str]:
    colors: dict[str, str] = {}
    fallback_i = 0
    for group in groups:
        if group in COLOR_MAP:
            colors[group] = COLOR_MAP[group]
        else:
            colors[group] = _COMBINATION_FALLBACK_COLORS[fallback_i % len(_COMBINATION_FALLBACK_COLORS)]
            fallback_i += 1
    return colors


def _legend_name(group: str, group_a: str | None, group_b: str | None) -> str:
    if group == "Upregulated" and group_a:
        return f"Upregulated in {group_a}"
    if group == "Downregulated" and group_b:
        return f"Upregulated in {group_b}"
    return group


def volcano_title(group_a: str, group_b: str) -> str:
    """Left/right title matching the x-axis: negative logFC = higher in B (left)."""
    return f"← {group_b} | {group_a} →"


def _hover_customdata(subset: pd.DataFrame) -> np.ndarray:
    if "clinical_approval_stage" in subset.columns:
        stages = subset["clinical_approval_stage"].map(format_clinical_approval_stage)
        stage_labels = stages.where(stages != "", "—").astype(str).values
    else:
        stage_labels = np.array(["—"] * len(subset))
    if "clinical_target_drugs" in subset.columns:
        drugs = subset["clinical_target_drugs"].map(format_clinical_target_drugs)
        drug_labels = drugs.where(drugs != "", "—").astype(str).values
    else:
        drug_labels = np.array(["—"] * len(subset))
    categories = gene_categories(subset)
    category_labels = categories.where(categories != "", "—").str.replace(
        COMBINATION_SEPARATOR, ", ", regex=False
    )
    return np.stack(
        [
            subset["adj.P.Val"],
            subset["logFC"],
            subset["gene"],
            stage_labels,
            drug_labels,
            category_labels.astype(str).values,
        ],
        axis=-1,
    )


def _volcano_frame(plot_df, fig, p_line, logfc_threshold):
    """Fixed axis ranges and margins, so data <-> pixel mapping is exact for label repel."""
    x = plot_df["logFC"].replace([np.inf, -np.inf], np.nan).dropna()
    y = plot_df["neg_log10_p"].replace([np.inf, -np.inf], np.nan).dropna()
    x_abs = max(float(x.abs().max()) if len(x) else 1.0, logfc_threshold, 0.5) * 1.08
    y_max = max(float(y.max()) if len(y) else 1.0, p_line, 1.0)
    legend_names = [t.name for t in fig.data if t.name] + ["Visual group"]
    legend_px = max(len(name) for name in legend_names) * LEGEND_CHAR_PX + 60
    margin = dict(
        l=MARGIN_LEFT_PX,
        r=int(min(max(legend_px, 120), 340)),
        t=MARGIN_TOP_PX,
        b=MARGIN_BOTTOM_PX,
        autoexpand=False,
    )
    frame = PlotFrame(
        x_range=(-x_abs, x_abs),
        y_range=(-0.02 * y_max, y_max * 1.1),
        width_px=VOLCANO_EXPORT_PX - margin["l"] - margin["r"],
        height_px=VOLCANO_EXPORT_PX - margin["t"] - margin["b"],
    )
    return frame, margin


def _add_repelled_labels(fig, labelled, plot_df, frame):
    """Gene labels pushed apart, joined to their points, never crossing logFC = 0."""
    anchors = frame.to_px(labelled["logFC"], labelled["neg_log10_p"])
    sizes = np.array([text_box_size(g, LABEL_FONT_SIZE) for g in labelled["gene"]])
    sides = np.sign(labelled["logFC"].to_numpy(dtype=float))
    others = plot_df.loc[~plot_df.index.isin(labelled.index)]
    obstacles = frame.to_px(others["logFC"], others["neg_log10_p"])
    reach = LABEL_OBSTACLE_REACH_PX
    near = np.zeros(len(obstacles), dtype=bool)
    for anchor in anchors:
        near |= (np.abs(obstacles - anchor) < reach).all(axis=1)
    centers, grows = repel_labels(
        anchors,
        sizes,
        sides,
        divider_px=float(frame.to_px([0.0], [0.0])[0, 0]),
        bounds_px=(frame.width_px, frame.height_px),
        obstacles=obstacles[near],
        point_radius=LABEL_POINT_RADIUS_PX,
        priority=np.argsort(labelled["adj.P.Val"].to_numpy(), kind="stable"),
    )

    # Text is anchored (in data units) at one edge and normally grows away from
    # logFC = 0, so it keeps its side when the on-screen chart is stretched. Plotly
    # draws the connector in screen pixels from the text box edge to the point, so the
    # two stay joined at any width.
    edge_px = np.column_stack([anchor_edge_x(centers, sizes, grows), centers[:, 1]])
    label_xy = frame.to_data(edge_px)
    # Plotly hides annotations whose tail falls outside the axis range.
    label_xy[:, 0] = np.clip(label_xy[:, 0], *frame.x_range)
    label_xy[:, 1] = np.clip(label_xy[:, 1], *frame.y_range)
    xanchors = np.where(grows > 0, "left", np.where(grows < 0, "right", "center"))
    for gene, x, y, (tx, ty), xanchor in zip(
        labelled["gene"], labelled["logFC"], labelled["neg_log10_p"], label_xy, xanchors
    ):
        fig.add_annotation(
            x=x,
            y=y,
            ax=tx,
            ay=ty,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            text=str(gene),
            xanchor=xanchor,
            yanchor="middle",
            showarrow=True,
            arrowhead=0,
            arrowwidth=0.8,
            arrowcolor="#555555",
            standoff=LABEL_POINT_RADIUS_PX,
            borderpad=0,
            font=dict(size=LABEL_FONT_SIZE, color="#333333", family=LABEL_FONT_FAMILY),
        )


def create_volcano_plot(
    df: pd.DataFrame,
    *,
    title: str,
    pval_threshold: float = 0.05,
    logfc_threshold: float = 1.0,
    highlight_tf: bool = False,
    highlight_ligand: bool = False,
    highlight_receptor: bool = False,
    highlight_metabolism: bool = False,
    highlight_clinical_target: bool = False,
    highlight_genes: list[str] | None = None,
    label_top_n: int = 0,
    download_as: str = "png",
    group_a: str | None = None,
    group_b: str | None = None,
) -> tuple[go.Figure, dict]:
    """``group_a`` is the positive-logFC (right) side, ``group_b`` the left side."""
    plot_df = df.copy()
    plot_df["neg_log10_p"] = -np.log10(plot_df["adj.P.Val"].clip(lower=1e-300))
    plot_df["group"] = classify_points(
        plot_df,
        pval_threshold=pval_threshold,
        logfc_threshold=logfc_threshold,
        highlight_tf=highlight_tf,
        highlight_ligand=highlight_ligand,
        highlight_receptor=highlight_receptor,
        highlight_metabolism=highlight_metabolism,
        highlight_clinical_target=highlight_clinical_target,
        highlight_genes=highlight_genes,
    )

    selected_genes = set(highlight_genes or [])
    fig = go.Figure()
    direction_groups = ["Not significant", "Downregulated", "Upregulated"]
    present = set(plot_df["group"]) - set(direction_groups) - {"Selected genes"}
    singles = [name for name in CATEGORY_COLUMNS.values() if name in present]
    combinations = sorted(
        present - set(singles),
        key=lambda g: (g.count(COMBINATION_SEPARATOR), g),
    )
    category_groups = singles + combinations
    colors = _group_colors(category_groups)
    for group in direction_groups + category_groups:
        subset = plot_df[plot_df["group"] == group]
        if selected_genes:
            subset = subset[~subset["gene"].isin(selected_genes)]
        if subset.empty:
            continue
        is_category = group in colors
        if group == "Not significant":
            opacity = NOT_SIGNIFICANT_OPACITY
        elif is_category:
            opacity = CATEGORY_OPACITY
        else:
            opacity = DIRECTION_OPACITY
        marker = dict(
            color=colors.get(group) or COLOR_MAP.get(group, "#888888"),
            size=7 if group == "Not significant" else 9,
            opacity=opacity,
        )
        if is_category:
            marker["line"] = dict(width=0.5, color="#ffffff")
        fig.add_trace(
            go.Scatter(
                x=subset["logFC"],
                y=subset["neg_log10_p"],
                mode="markers",
                name=_legend_name(group, group_a, group_b),
                marker=marker,
                text=subset["gene"],
                customdata=_hover_customdata(subset),
                hovertemplate=_HOVER_TEMPLATE,
            )
        )

    if selected_genes:
        highlighted = plot_df[plot_df["gene"].isin(selected_genes)]
        if not highlighted.empty:
            fig.add_trace(
                go.Scatter(
                    x=highlighted["logFC"],
                    y=highlighted["neg_log10_p"],
                    mode="markers",
                    name="Selected genes",
                    marker=dict(
                        color="#000000",
                        size=11,
                        opacity=SELECTED_MARKER_OPACITY,
                        line=dict(width=0.5, color="#ffffff"),
                    ),
                    text=highlighted["gene"],
                    customdata=_hover_customdata(highlighted),
                    hovertemplate=_HOVER_TEMPLATE,
                )
            )

    p_line = -np.log10(pval_threshold)
    frame, margin = _volcano_frame(plot_df, fig, p_line, logfc_threshold)

    label_genes = set(highlight_genes or [])
    if label_top_n > 0:
        label_genes.update(plot_df.nsmallest(label_top_n, "adj.P.Val")["gene"].tolist())
    if label_genes:
        labelled = plot_df[plot_df["gene"].isin(label_genes)].drop_duplicates("gene")
        _add_repelled_labels(fig, labelled, plot_df, frame)

    fig.add_hline(y=p_line, line_dash="dash", line_color="#888888", opacity=0.6)
    fig.add_vline(x=logfc_threshold, line_dash="dash", line_color="#888888", opacity=0.6)
    fig.add_vline(x=-logfc_threshold, line_dash="dash", line_color="#888888", opacity=0.6)

    fig.update_layout(
        title=title,
        xaxis_title="log2 fold change",
        yaxis_title="-log10(adj.P.Val)",
        width=VOLCANO_EXPORT_PX,
        height=VOLCANO_EXPORT_PX,
        margin=margin,
        xaxis=dict(range=list(frame.x_range), autorange=False),
        yaxis=dict(range=list(frame.y_range), autorange=False),
        legend=dict(title="Visual group", x=1.02, xanchor="left", y=1, yanchor="top"),
        hovermode="closest",
    )

    config = {
        "toImageButtonOptions": {
            "format": download_as,
            "filename": "pta_volcano",
            "width": VOLCANO_EXPORT_PX,
            "height": VOLCANO_EXPORT_PX,
            "scale": 2,
        }
    }
    return fig, config
