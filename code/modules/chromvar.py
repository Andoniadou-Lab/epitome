import plotly.express as px
import plotly.graph_objects as go
from .utils import create_color_mapping

def create_chromvar_plot(
    matrix, features, meta_data, motif_name, additional_group=None, connect_dots=False
):
    """
    Create a box plot for ChromVAR data with strip plot overlay, supporting additional grouping and connected dots
    """
    # Convert matrix to array if needed
    if hasattr(matrix, "toarray"):
        matrix = matrix.tocsr()

    # Get motif values
    motif_idx = features.index(motif_name)
    motif_values = matrix[motif_idx].toarray().flatten()

    # Create plot dataframe
    plot_df = meta_data.copy()
    plot_df["Enrichment"] = motif_values
    print(plot_df.head())

    # Create consistent color mapping
    color_map = create_color_mapping(plot_df["cell_type"].unique())

    if additional_group:
        # Sort data by cell type and additional group
        plot_df = plot_df.sort_values(["cell_type", additional_group])

        # Create a categorical x-axis that groups by cell type first
        cell_types = sorted(plot_df["cell_type"].unique())
        secondary_groups = sorted(plot_df[additional_group].unique())

        # Create a position mapping for proper spacing
        position_map = {}
        current_pos = 0
        for cell_type in cell_types:
            for group in secondary_groups:
                position_map[(cell_type, str(group))] = current_pos
                current_pos += 1
            current_pos += 1  # Add extra space between cell types

        # Create x positions for plotting
        plot_df["x_position"] = plot_df.apply(
            lambda row: position_map[(row["cell_type"], str(row[additional_group]))],
            axis=1,
        )

        if additional_group == "Comp_sex":
            # Define special colors for female (0) and male (1)
            sex_color_map = {
                "Female": "#FFA500",  # female
                "Male": "#63B3ED",     # male
            }

            # Create box plot with proper spacing
            fig = px.box(
                plot_df,
                x="x_position",
                y="Enrichment",
                color=additional_group,
                color_discrete_map=sex_color_map,
                points=False,
                hover_data=["GEO"],
                title=f"{motif_name} Enrichment by Cell Type and {additional_group}",
            )

            # Create matching strip plot
            strip_fig = px.strip(
                plot_df,
                x="x_position",
                y="Enrichment",
                color=additional_group,
                color_discrete_map=sex_color_map,
                hover_data=["GEO"],
            )
        else:
            # Create box plot with default colors for other groupings
            fig = px.box(
                plot_df,
                x="x_position",
                y="Enrichment",
                color=additional_group,
                points=False,
                hover_data=["GEO"],
                title=f"{motif_name} Enrichment by Cell Type and {additional_group}",
            )

            # Create matching strip plot
            strip_fig = px.strip(
                plot_df,
                x="x_position",
                y="Enrichment",
                color=additional_group,
                hover_data=["GEO"],
            )

        # Update x-axis to show cell type labels centered for each group
        tick_positions = []
        tick_labels = []
        for cell_type in cell_types:
            # Get positions for this cell type
            cell_type_positions = [
                pos for (ct, _), pos in position_map.items() if ct == cell_type
            ]
            tick_positions.append(sum(cell_type_positions) / len(cell_type_positions))
            tick_labels.append(cell_type)

        fig.update_xaxes(ticktext=tick_labels, tickvals=tick_positions, tickangle=45)
    else:
        fig = px.box(
            plot_df,
            x="cell_type",
            y="Enrichment",
            color="cell_type",
            points=False,
            color_discrete_map=color_map,
            hover_data=["GEO"],
            title=f"{motif_name} Enrichment by Cell Type",
        )

        # Add strip plot
        strip_fig = px.strip(
            plot_df,
            x="cell_type",
            y="Enrichment",
            color="cell_type",
            color_discrete_map=color_map,
            hover_data=["GEO"],
        )

    # Update strip plot traces to match the box plot
    for trace in strip_fig.data:
        trace.update(marker=dict(opacity=0.4, size=6), showlegend=False,hoverinfo="text" )
        fig.add_trace(trace)

    # Optionally connect dots with the same SRA_ID
    if connect_dots:
        for sra_id in plot_df["SRA_ID"].unique():
            sra_df = plot_df[plot_df["SRA_ID"] == sra_id]
            # Use x_position if it exists (for grouped data), otherwise use cell type
            x_values = (
                sra_df["x_position"]
                if "x_position" in sra_df.columns
                else sra_df["cell_type"]
            )
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=sra_df["Enrichment"],
                    mode="lines+markers",
                    line=dict(color="gray", width=1),
                    marker=dict(size=6, opacity=0.6),
                    name=sra_id,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

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
