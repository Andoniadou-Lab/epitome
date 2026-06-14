import plotly.graph_objects as go

from .boxplot import (
    SEX_COLOR_MAP,
    add_connected_sample_lines,
    apply_cell_type_tick_labels,
    create_box_strip_plot,
    prepare_grouped_x_positions,
)
from .utils import create_color_mapping, to_array


def create_chromvar_plot(
    matrix, features, meta_data, motif_name, additional_group=None, connect_dots=False
):
    """Box + strip plot for ChromVAR enrichment with 5th/95th percentile whiskers."""
    motif_idx = features.index(motif_name)
    plot_df = meta_data.copy()
    plot_df["Enrichment"] = to_array(matrix[motif_idx, :])

    color_map = create_color_mapping(plot_df["cell_type"].unique())
    x_col, color_col, strip_colors = "cell_type", "cell_type", color_map
    title = f"{motif_name} Enrichment by Cell Type"
    position_map = None
    cell_types = None

    if additional_group:
        plot_df, position_map, cell_types = prepare_grouped_x_positions(
            plot_df, additional_group
        )
        x_col, color_col = "x_position", additional_group
        strip_colors = SEX_COLOR_MAP if additional_group == "Comp_sex" else None
        title = (
            f"{motif_name} Enrichment by Cell Type and Sex"
            if additional_group == "Comp_sex"
            else f"{motif_name} Enrichment by Cell Type and {additional_group}"
        )

    fig = create_box_strip_plot(
        plot_df,
        x_col,
        "Enrichment",
        color_col,
        strip_colors,
        title,
        hover_data=["GEO"],
    )

    if position_map is not None:
        apply_cell_type_tick_labels(fig, position_map, cell_types)

    if connect_dots:
        add_connected_sample_lines(fig, plot_df, x_col, "Enrichment")

    fig.update_layout(
        xaxis_title="Cell Type",
        yaxis_title=f"{motif_name} Enrichment Score",
        showlegend=True,
        xaxis={"tickangle": 45, "tickfont": {"size": 20}, "title_font": {"size": 20}},
        yaxis={"title_font": {"size": 20}, "tickfont": {"size": 20}},
        height=600,
        width=None,
    )

    config = {
        "toImageButtonOptions": {
            "format": "svg",
            "filename": f"{motif_name}_enrichment",
            "height": 600,
            "width": 1200,
            "scale": 2,
        }
    }
    return fig, config
