"""PTA bulk/pseudobulk box plots using epitome percentile boxplots."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from modules.boxplot import (
    apply_cell_type_tick_labels,
    create_box_strip_plot,
    prepare_grouped_x_positions,
)
from modules.expression_transform import log10_axis_label
from modules.gene_text import italic_gene
from modules.pta.cell_type_labels import sort_pta_categories
from modules.publication_export import (
    TEXT_COLOUR,
    base_publication_layout,
    journal_axes,
    text_width,
)

PTA_SEX_COLOR_MAP = {
    "Female": "#FFA500",
    "Male": "#63B3ED",
    "Unknown": "#B0B0B0",
}


BULK_HOVER_COLUMNS = (
    "Internal_ID",
    "Name",
    "Author",
    "Sex_pta",
    "Lineage_pta",
    "Cell_type_pta",
    "Subtype_pta",
    "Secretion_pta",
    "Disease_pta",
    "Sensitivity_pta",
)

PSEUDOBULK_HOVER_COLUMNS = (
    "Internal_ID",
    "SRA_ID",
    "GEO",
    "Name",
    "Author",
    "Modality",
    "broad_cluster_final",
    "Normal",
    "Tumor_pta",
    "Sex",
    "Final_sex",
    "Lineage",
    "Cell type",
    "Subtype",
    "Secretion",
    "Disease",
    "Recurrence",
)


def with_hover_columns(
    plot_df: pd.DataFrame, hover_columns=BULK_HOVER_COLUMNS
) -> tuple[pd.DataFrame, list[str]]:
    """Plot frame plus the metadata columns shown on dot hover.

    Hover columns stored as the index (bulk ``Internal_ID``) are moved into columns.
    """
    out = plot_df
    if out.index.name in hover_columns and out.index.name not in out.columns:
        out = out.reset_index()
    return out, [c for c in hover_columns if c in out.columns]


def _empty_figure(message: str, download_as: str = "png") -> tuple[go.Figure, dict]:
    fig = go.Figure()
    fig.add_annotation(x=0.5, y=0.5, text=message, font=dict(size=15), showarrow=False)
    fig.update_layout(
        title="No Data Available",
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
        height=600,
    )
    return fig, {"toImageButtonOptions": {"format": download_as}}


def add_comparison_column(
    df: pd.DataFrame,
    source_col: str,
    left_values,
    right_values,
    left_name: str,
    right_name: str,
    dest_col: str = "Comparison",
) -> pd.DataFrame:
    """Map current X-axis labels into two named comparison groups."""
    left_name = (left_name or "Left").strip() or "Left"
    right_name = (right_name or "Right").strip() or "Right"
    left_set = {str(v) for v in left_values}
    right_set = {str(v) for v in right_values} - left_set
    mapped = df[source_col].astype(str).map(
        lambda label: left_name if label in left_set else right_name if label in right_set else pd.NA
    )
    out = df.copy()
    out[dest_col] = mapped
    return out.dropna(subset=[dest_col])


def create_pta_boxplot(
    plot_df,
    gene: str,
    group_col: str,
    y_col: str = "Expression",
    secondary_group: str | None = None,
    hover_columns=None,
    y_label: str | None = None,
    color_map: dict[str, str] | None = None,
    merge_mixed: bool = True,
    download_as: str = "png",
    primary_order: list | None = None,
):
    """Box + strip plot for PTA bulk or pseudobulk expression."""
    if plot_df.empty:
        return _empty_figure("No data available", download_as)

    plot_df = plot_df.copy()
    plot_df["primary"] = plot_df[group_col].astype(str)
    hover_data = list(hover_columns) if hover_columns is not None else list(plot_df.columns)

    x_col = "primary"
    color_col = "primary"
    title = f"{italic_gene(gene)} expression by {group_col}"
    position_map = None
    primaries = None
    category_order = primary_order or sort_pta_categories(
        plot_df["primary"].unique(), group_col, merge_mixed=merge_mixed
    )
    secondary_order = None

    if secondary_group and secondary_group != group_col:
        plot_df[secondary_group] = plot_df[secondary_group].astype(str)
        secondary_order = sort_pta_categories(
            plot_df[secondary_group].unique(), secondary_group, merge_mixed=merge_mixed
        )
        plot_df, position_map, primaries = prepare_grouped_x_positions(
            plot_df,
            secondary_group,
            cell_type_col="primary",
            primary_order=category_order,
            secondary_order=secondary_order,
        )
        x_col = "x_position"
        color_col = secondary_group
        if color_map is None and secondary_group in ("Sex_pta", "Sex"):
            color_map = PTA_SEX_COLOR_MAP
        title = f"{italic_gene(gene)} expression by {group_col} and {secondary_group}"
    elif color_map is None and group_col in ("Sex_pta", "Sex"):
        color_map = PTA_SEX_COLOR_MAP

    fig = create_box_strip_plot(
        plot_df,
        x_col,
        y_col,
        color_col,
        color_map,
        title,
        hover_data=hover_data,
        category_order=category_order if x_col == "primary" else None,
        color_boxes=True,
        fill_alpha=0.45,
    )

    if position_map is not None and primaries is not None:
        apply_cell_type_tick_labels(fig, position_map, primaries)

    n_categories = len(plot_df["primary"].unique())
    if secondary_group and secondary_group != group_col:
        n_categories *= max(1, plot_df[secondary_group].nunique())
    dynamic_width = max(550, min(2400, n_categories * 120))

    fig.update_layout(
        xaxis_title=group_col,
        yaxis_title=y_label or log10_axis_label(gene),
        showlegend=True,
        xaxis={"tickangle": 45, "tickfont": {"size": 16}, "title_font": {"size": 18}},
        yaxis={"title_font": {"size": 18}, "tickfont": {"size": 14}},
        height=600,
        width=dynamic_width,
        plot_bgcolor="white",
    )

    config = {
        "toImageButtonOptions": {
            "format": download_as,
            "filename": f"{gene}_expression",
            "height": 700,
            "width": dynamic_width,
            "scale": 4,
        }
    }
    return fig, config


def _is_hover_target(trace) -> bool:
    return trace.type == "scatter" and trace.hoverinfo == "text" and trace.hovertext is not None


def boxplot_publication_figure(fig: go.Figure) -> tuple[go.Figure, int, int]:
    """Journal-style copy of a PTA box plot: compact boxes, large black labels, no title.

    Returns the figure with its export width and height in pixels.
    """
    out = go.Figure(fig)
    out.data = tuple(t for t in out.data if not _is_hover_target(t))
    grouped = out.layout.xaxis.tickvals is not None

    box_traces = [t for t in out.data if t.type == "box" and t.width not in (0, None)]
    if grouped:
        positions = [x for t in box_traces for x in (t.x or ()) if x is not None]
        n_slots = int(max(positions)) + 1 if positions else 1
        labels = list(out.layout.xaxis.ticktext or ())
    else:
        labels = list(dict.fromkeys(str(x) for t in box_traces for x in (t.x or ())))
        n_slots = max(1, len(labels))

    tick_px, title_px, point_px = 26, 28, 5
    slot_px = 36 if grouped else 50
    for trace in out.data:
        if trace.type != "box":
            continue
        if trace.width == 0:
            trace.marker.size = point_px
        else:
            trace.width = 0.8 if grouped else 0.65
            trace.line.width = 2

    axis_text = dict(tickfont=dict(size=tick_px, color=TEXT_COLOUR))
    title_font = dict(size=title_px, color=TEXT_COLOUR)
    base_publication_layout(out)
    out.update_layout(
        title=dict(text=""),
        showlegend=grouped,
        legend=dict(font=dict(size=tick_px - 4, color=TEXT_COLOUR), itemsizing="constant"),
        margin=dict(t=20, l=20, r=20, b=20),
        xaxis=dict(**axis_text, title_font=title_font, tickangle=45),
        yaxis=dict(**axis_text, title_font=title_font),
    )
    journal_axes(out)

    label_px = max((text_width(lab, tick_px) for lab in labels), default=0) * 0.71
    width = 110 + n_slots * slot_px + (190 if grouped else 0) + label_px * 0.6
    height = 540 + label_px * 0.7
    return out, int(width), int(height)


def boxplot_download_frame(
    plot_df: pd.DataFrame,
    columns: list[str],
    value_col: str = "Expression",
    value_label: str | None = None,
) -> pd.DataFrame:
    """Plotted values plus the grouping and hover metadata, one row per sample."""
    keep = [c for c in dict.fromkeys([*columns, value_col]) if c in plot_df.columns]
    out = plot_df[keep]
    if out.index.name and out.index.name not in keep:
        out = out.reset_index()
    else:
        out = out.reset_index(drop=True)
    if value_label:
        out = out.rename(columns={value_col: value_label})
    return out
