"""Percentile box plots: Q1/median/Q3 box with 5th/95th percentile whiskers."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

_DEFAULT_COLOR = "#2E4057"
_HOVER_MARKERS = ("q05", "q50", "q95")

SEX_COLOR_MAP = {"Female": "#FFA500", "Male": "#63B3ED"}


def _sorted_unique(series: pd.Series, order: list | None = None) -> list:
    values = series.dropna().unique()
    if order:
        str_values = [str(v) for v in values]
        rank = {str(label): i for i, label in enumerate(order)}
        default_rank = len(rank)
        return sorted(str_values, key=lambda value: (rank.get(value, default_rank), value))
    if pd.api.types.is_numeric_dtype(series):
        return sorted(values)
    return sorted(values, key=str)


def _box_percentile_stats(y_values: pd.Series) -> dict[str, float]:
    y_values = y_values.dropna()
    return {f"q{p:02d}": float(y_values.quantile(p / 100)) for p in (5, 25, 50, 75, 95)}


def _parse_rgb(color: str) -> tuple[int, int, int]:
    color = str(color).strip()
    if color.startswith("#"):
        hex_color = color.lstrip("#")
        if len(hex_color) == 3:
            hex_color = "".join(ch * 2 for ch in hex_color)
        return (
            int(hex_color[0:2], 16),
            int(hex_color[2:4], 16),
            int(hex_color[4:6], 16),
        )
    if color.startswith("rgb("):
        return tuple(int(part.strip()) for part in color[4:-1].split(","))
    return 46, 64, 87


def _apply_box_style(trace, color: str, fill_alpha: float = 0.22, line_width: float = 2.5) -> None:
    r, g, b = _parse_rgb(color)
    trace.update(
        fillcolor=f"rgba({r},{g},{b},{fill_alpha})",
        line=dict(color=f"rgb({r},{g},{b})", width=line_width),
    )


def _box_hover_text(name, x_label, stats: dict[str, float]) -> str:
    title = name if str(name) == str(x_label) else f"{name} · {x_label}"
    return (
        f"<b>{title}</b><br>"
        f"Lower whisker (5th percentile): {stats['q05']:.4g}<br>"
        f"Q1 (25th percentile): {stats['q25']:.4g}<br>"
        f"Median: {stats['q50']:.4g}<br>"
        f"Q3 (75th percentile): {stats['q75']:.4g}<br>"
        f"Upper whisker (95th percentile): {stats['q95']:.4g}<br>"
    )


def _iter_box_groups(df, x_col, color_col=None, category_order=None):
    """Yield ``(sub_df, trace_name)`` groups matching plotly.express box layout."""
    if color_col is None:
        yield df, ""
    elif color_col == x_col:
        for x_cat in _sorted_unique(df[x_col], category_order):
            yield df[df[x_col] == x_cat], x_cat
    else:
        for color_val, color_df in df.groupby(color_col, sort=False):
            yield color_df, color_val


def _resolve_color(name, color_discrete_map, default=_DEFAULT_COLOR):
    if not color_discrete_map:
        return default
    return color_discrete_map.get(str(name), default)


def _collect_box_stats(sub_df, x_col, y_col, category_order=None):
    """Return parallel lists of x categories and percentile stats for one trace."""
    x_vals, lower, q1s, meds, q3s, upper = [], [], [], [], [], []
    for x_cat in _sorted_unique(sub_df[x_col], category_order):
        vals = sub_df.loc[sub_df[x_col] == x_cat, y_col].dropna()
        if vals.empty:
            continue
        stats = _box_percentile_stats(vals)
        x_vals.append(x_cat)
        lower.append(stats["q05"])
        q1s.append(stats["q25"])
        meds.append(stats["q50"])
        q3s.append(stats["q75"])
        upper.append(stats["q95"])
    return x_vals, lower, q1s, meds, q3s, upper


def _add_box_trace(
    fig,
    sub_df,
    x_col,
    y_col,
    trace_name,
    box_color,
    *,
    boxpoints=False,
    box_width=0.6,
    fill_alpha=0.22,
    line_width=2.5,
    showlegend=False,
    category_order=None,
) -> None:
    x_vals, lower, q1s, meds, q3s, upper = _collect_box_stats(
        sub_df, x_col, y_col, category_order
    )
    if not x_vals:
        return

    trace = go.Box(
        x=x_vals,
        lowerfence=lower,
        q1=q1s,
        median=meds,
        q3=q3s,
        upperfence=upper,
        name=str(trace_name),
        boxpoints="all" if boxpoints else False,
        width=box_width,
        whiskerwidth=0.5,
        showlegend=showlegend,
        hoverinfo="skip",
    )
    _apply_box_style(trace, box_color, fill_alpha, line_width)
    fig.add_trace(trace)


def create_custom_box(
    df,
    x_col,
    y_col,
    color_col=None,
    color_discrete_map=None,
    boxpoints=False,
    title=None,
    showlegend=None,
    box_width=0.6,
    fill_alpha=0.22,
    line_width=2.5,
    category_order=None,
):
    """Build percentile boxes; add strip points via ``overlay_strip``."""
    fig = go.Figure()
    if showlegend is None:
        showlegend = color_col is not None

    for sub_df, trace_name in _iter_box_groups(df, x_col, color_col, category_order):
        _add_box_trace(
            fig,
            sub_df,
            x_col,
            y_col,
            trace_name,
            _resolve_color(trace_name, color_discrete_map),
            boxpoints=boxpoints,
            box_width=box_width,
            fill_alpha=fill_alpha,
            line_width=line_width,
            showlegend=showlegend and trace_name != "",
            category_order=category_order,
        )

    layout = dict(boxmode="overlay", boxgap=0, boxgroupgap=0)
    if title:
        layout["title"] = title
    fig.update_layout(**layout)
    return fig


def overlay_strip(fig, strip_fig) -> go.Figure:
    """Add jittered strip points on top of box traces."""
    for trace in strip_fig.data:
        trace.update(marker=dict(opacity=0.4, size=7), showlegend=False)
        fig.add_trace(trace)
    fig.update_layout(boxmode="overlay", boxgap=0, boxgroupgap=0)
    return fig


def add_box_hover_targets(fig, df, x_col, y_col, color_col=None, category_order=None) -> go.Figure:
    """Near-invisible markers for percentile-labelled box hover (after strip layer)."""
    hover_x, hover_y, hover_text = [], [], []

    for sub_df, trace_name in _iter_box_groups(df, x_col, color_col, category_order):
        for x_cat in _sorted_unique(sub_df[x_col], category_order):
            vals = sub_df.loc[sub_df[x_col] == x_cat, y_col].dropna()
            if vals.empty:
                continue
            stats = _box_percentile_stats(vals)
            text = _box_hover_text(trace_name, x_cat, stats)
            for key in _HOVER_MARKERS:
                hover_x.append(x_cat)
                hover_y.append(stats[key])
                hover_text.append(text)

    if hover_x:
        fig.add_trace(
            go.Scatter(
                x=hover_x,
                y=hover_y,
                mode="markers",
                marker=dict(size=36, color="rgba(100,100,100,0.01)"),
                hoverinfo="text",
                hovertext=hover_text,
                showlegend=False,
                hoverlabel=dict(bgcolor="white", font_size=13, namelength=-1),
            )
        )
    return fig


def prepare_grouped_x_positions(
    plot_df,
    group_col,
    cell_type_col="cell_type",
    primary_order: list | None = None,
    secondary_order: list | None = None,
):
    """Add ``x_position`` column with spacing between cell-type clusters."""
    plot_df = plot_df.sort_values([cell_type_col, group_col])
    cell_types = primary_order or sorted(plot_df[cell_type_col].unique())
    secondary_groups = secondary_order or sorted(plot_df[group_col].unique())

    position_map = {}
    pos = 0
    for cell_type in cell_types:
        for group in secondary_groups:
            position_map[(cell_type, str(group))] = pos
            pos += 1
        pos += 1

    plot_df = plot_df.copy()
    plot_df["x_position"] = plot_df.apply(
        lambda row: position_map[(row[cell_type_col], str(row[group_col]))],
        axis=1,
    )
    return plot_df, position_map, cell_types


def apply_cell_type_tick_labels(fig, position_map, cell_types):
    tick_positions, tick_labels = [], []
    for cell_type in cell_types:
        positions = [p for (ct, _), p in position_map.items() if ct == cell_type]
        tick_positions.append(sum(positions) / len(positions))
        tick_labels.append(cell_type)
    fig.update_xaxes(ticktext=tick_labels, tickvals=tick_positions, tickangle=45)


def create_box_strip_plot(
    plot_df,
    x_col,
    y_col,
    color_col,
    color_discrete_map,
    title,
    hover_data=None,
    category_order=None,
):
    """Box layer + strip overlay + percentile hover targets."""
    fig = create_custom_box(
        plot_df,
        x_col=x_col,
        y_col=y_col,
        color_col=color_col,
        color_discrete_map=color_discrete_map,
        title=title,
        category_order=category_order,
    )
    strip_fig = px.strip(
        plot_df,
        x=x_col,
        y=y_col,
        color=color_col,
        color_discrete_map=color_discrete_map,
        hover_data=hover_data,
    )
    overlay_strip(fig, strip_fig)
    add_box_hover_targets(fig, plot_df, x_col, y_col, color_col, category_order)
    return fig


def add_connected_sample_lines(fig, plot_df, x_col, y_col, sra_col="SRA_ID", marker_size=6):
    for _, sra_df in plot_df.groupby(sra_col):
        fig.add_trace(
            go.Scatter(
                x=sra_df[x_col],
                y=sra_df[y_col],
                mode="lines+markers",
                line=dict(color="gray", width=1),
                marker=dict(size=marker_size, opacity=0.6),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    return fig
