import plotly.graph_objects as go

from .boxplot import (
    SEX_COLOR_MAP,
    add_connected_sample_lines,
    apply_cell_type_tick_labels,
    create_box_strip_plot,
    prepare_grouped_x_positions,
)
from .utils import create_color_mapping, to_array


def create_expression_plot(
    matrix,
    genes,
    meta_data,
    gene_name,
    additional_group=None,
    connect_dots=False,
    selected_cell_types=None,
    download_as="png",
):
    """Box + strip plot for gene expression with 5th/95th percentile whiskers."""
    gene_idx = genes[genes[0] == gene_name].index[0]
    plot_df = meta_data.copy()
    plot_df["Expression"] = to_array(matrix[gene_idx, :])
    plot_df["cell_type"] = plot_df["new_cell_type"]

    if selected_cell_types:
        plot_df = plot_df[plot_df["cell_type"].isin(selected_cell_types)]

    if plot_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            text="No data available for the selected cell types",
            font=dict(size=15),
            showarrow=False,
        )
        fig.update_layout(
            title="No Data Available",
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            height=600,
        )
        return fig, {"toImageButtonOptions": {"format": "svg"}}

    color_map = create_color_mapping(plot_df["cell_type"].unique())
    x_col, color_col, strip_colors = "cell_type", "cell_type", color_map
    title = f"{gene_name} Expression by Cell Type"
    position_map = None
    cell_types = None

    if additional_group:
        plot_df, position_map, cell_types = prepare_grouped_x_positions(
            plot_df, additional_group
        )
        x_col, color_col = "x_position", additional_group
        strip_colors = SEX_COLOR_MAP if additional_group == "Comp_sex" else None
        title = (
            f"{gene_name} Expression by Cell Type and Sex"
            if additional_group == "Comp_sex"
            else f"{gene_name} Expression by Cell Type and {additional_group}"
        )

    fig = create_box_strip_plot(
        plot_df,
        x_col,
        "Expression",
        color_col,
        strip_colors,
        title,
        hover_data=meta_data.columns,
    )

    if position_map is not None:
        apply_cell_type_tick_labels(fig, position_map, cell_types)

    if connect_dots:
        add_connected_sample_lines(
            fig, plot_df, x_col, "Expression", marker_size=7
        )

    dynamic_width = max(550, min(2400, plot_df["cell_type"].nunique() * 120))
    fig.update_layout(
        xaxis_title="Cell Type",
        yaxis_title=f"{gene_name} Expression (log10)",
        showlegend=True,
        xaxis={"tickangle": 45, "tickfont": {"size": 25}, "title_font": {"size": 25}},
        yaxis={"title_font": {"size": 25}, "tickfont": {"size": 22}},
        height=600,
        width=dynamic_width,
    )

    config = {
        "toImageButtonOptions": {
            "format": download_as,
            "filename": f"{gene_name}_expression",
            "height": 700,
            "width": dynamic_width,
            "scale": 4,
        }
    }
    return fig, config
