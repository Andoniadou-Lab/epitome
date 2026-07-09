"""Volcano plot for PTA differential-expression result tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from modules.pta.gene_annotation import format_clinical_approval_stage, format_clinical_target_drugs


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.upper().isin({"TRUE", "1", "T", "YES"})


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
    """Assign visual groups. All points are plotted; thresholds affect colour only."""
    is_tf = _as_bool(df["is_tf"]) if "is_tf" in df.columns else pd.Series(False, index=df.index)
    is_ligand = _as_bool(df["is_ligand"]) if "is_ligand" in df.columns else pd.Series(False, index=df.index)
    is_receptor = _as_bool(df["is_receptor"]) if "is_receptor" in df.columns else pd.Series(False, index=df.index)
    is_metabolism = (
        _as_bool(df["is_metabolism"]) if "is_metabolism" in df.columns else pd.Series(False, index=df.index)
    )
    is_clinical = (
        _as_bool(df["is_clinical_target"])
        if "is_clinical_target" in df.columns
        else pd.Series(False, index=df.index)
    )
    selected = set(highlight_genes or [])

    sig = (df["adj.P.Val"] < pval_threshold) & (df["logFC"].abs() >= logfc_threshold)
    groups = pd.Series("Not significant", index=df.index, dtype=object)
    groups[sig & (df["logFC"] > 0)] = "Upregulated"
    groups[sig & (df["logFC"] < 0)] = "Downregulated"

    if highlight_tf:
        groups[is_tf] = "Transcription factor"
    if highlight_ligand:
        groups[is_ligand] = "Ligand"
    if highlight_receptor:
        groups[is_receptor] = "Receptor"
    if highlight_metabolism:
        groups[is_metabolism] = "Metabolism"
    if highlight_clinical_target:
        groups[is_clinical] = "Clinical target"
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
}

VOLCANO_EXPORT_PX = 800
LABEL_FONT_SIZE = 16
MARKER_OPACITY = 0.5
SELECTED_MARKER_OPACITY = 1.0

_HOVER_TEMPLATE = (
    "<b>%{text}</b><br>"
    "logFC: %{customdata[1]:.3f}<br>"
    "adj.P.Val: %{customdata[0]:.2e}<br>"
    "Approval stage: %{customdata[3]}<br>"
    "Drugs: %{customdata[4]}<extra></extra>"
)


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
    return np.stack(
        [subset["adj.P.Val"], subset["logFC"], subset["gene"], stage_labels, drug_labels],
        axis=-1,
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
) -> tuple[go.Figure, dict]:
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
    order = [
        "Not significant",
        "Downregulated",
        "Upregulated",
        "Receptor",
        "Ligand",
        "Transcription factor",
        "Metabolism",
        "Clinical target",
    ]
    for group in order:
        subset = plot_df[plot_df["group"] == group]
        if selected_genes:
            subset = subset[~subset["gene"].isin(selected_genes)]
        if subset.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=subset["logFC"],
                y=subset["neg_log10_p"],
                mode="markers",
                name=group,
                marker=dict(
                    color=COLOR_MAP.get(group, "#888888"),
                    size=7 if group == "Not significant" else 9,
                    opacity=MARKER_OPACITY,
                ),
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

    label_genes = set(highlight_genes or [])
    if label_top_n > 0:
        label_genes.update(plot_df.nsmallest(label_top_n, "adj.P.Val")["gene"].tolist())
    if label_genes:
        labelled = plot_df[plot_df["gene"].isin(label_genes)]
        fig.add_trace(
            go.Scatter(
                x=labelled["logFC"],
                y=labelled["neg_log10_p"],
                mode="text",
                text=labelled["gene"],
                textposition="top center",
                textfont=dict(size=LABEL_FONT_SIZE, color="#333333"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    p_line = -np.log10(pval_threshold)
    fig.add_hline(y=p_line, line_dash="dash", line_color="#888888", opacity=0.6)
    fig.add_vline(x=logfc_threshold, line_dash="dash", line_color="#888888", opacity=0.6)
    fig.add_vline(x=-logfc_threshold, line_dash="dash", line_color="#888888", opacity=0.6)

    fig.update_layout(
        title=title,
        xaxis_title="log2 fold change",
        yaxis_title="-log10(adj.P.Val)",
        width=VOLCANO_EXPORT_PX,
        height=VOLCANO_EXPORT_PX,
        legend=dict(title="Visual group"),
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
