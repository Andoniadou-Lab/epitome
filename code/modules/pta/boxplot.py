"""PTA bulk/pseudobulk box plots using epitome percentile boxplots."""

from __future__ import annotations

import plotly.graph_objects as go

from modules.boxplot import (
    apply_cell_type_tick_labels,
    create_box_strip_plot,
    prepare_grouped_x_positions,
)
from modules.pta.cell_type_labels import sort_pta_categories

PTA_SEX_COLOR_MAP = {
    "Female": "#FFA500",
    "Male": "#63B3ED",
    "Unknown": "#B0B0B0",
}


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
):
    """Box + strip plot for PTA bulk or pseudobulk expression."""
    if plot_df.empty:
        return _empty_figure("No data available", download_as)

    plot_df = plot_df.copy()
    plot_df["primary"] = plot_df[group_col].astype(str)
    hover_data = list(hover_columns) if hover_columns is not None else list(plot_df.columns)

    x_col = "primary"
    color_col = "primary"
    title = f"{gene} expression by {group_col}"
    position_map = None
    primaries = None
    category_order = sort_pta_categories(
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
        title = f"{gene} expression by {group_col} and {secondary_group}"
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
    )

    if position_map is not None and primaries is not None:
        apply_cell_type_tick_labels(fig, position_map, primaries)

    n_categories = len(plot_df["primary"].unique())
    if secondary_group and secondary_group != group_col:
        n_categories *= max(1, plot_df[secondary_group].nunique())
    dynamic_width = max(550, min(2400, n_categories * 120))

    fig.update_layout(
        xaxis_title=group_col,
        yaxis_title=y_label or f"{gene} expression (log1p CPM)",
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
