import plotly.express as px
import plotly.graph_objects as go
from .utils import create_color_mapping

def create_expression_plot(
    matrix,
    genes,
    meta_data,
    gene_name,
    additional_group=None,
    connect_dots=False,
    selected_cell_types=None,
):
    """
    Create a box plot for gene expression data with consistent colors and overlay a strip plot

    Parameters:
    -----------
    matrix : array-like
        Expression matrix
    genes : pandas.DataFrame
        Gene information
    meta_data : pandas.DataFrame
        Metadata information
    gene_name : str
        Name of the gene to plot
    additional_group : str, optional
        Column name to use for additional grouping
    connect_dots : bool
        Whether to connect dots with the same SRA_ID
    selected_cell_types : list, optional
        List of specific cell types to display, if None all cell types are shown
    """
    import plotly.express as px
    import plotly.graph_objects as go
    from .utils import create_color_mapping

    # Meta data and matrix should already be filtered before being passed to this function
    gene_idx = genes[genes[0] == gene_name].index[0]
    expression_values = (
        matrix[gene_idx, :].A1
        if hasattr(matrix[gene_idx, :], "A1")
        else matrix[gene_idx, :]
    )

    plot_df = meta_data.copy()
    plot_df["Expression"] = expression_values

    # Filter by selected cell types if provided
    if selected_cell_types and len(selected_cell_types) > 0:
        plot_df = plot_df[plot_df["new_cell_type"].isin(selected_cell_types)]

    # If no data after filtering, return error
    if len(plot_df) == 0:
        fig = go.Figure()
        fig.add_annotation(
            x=0.5,
            y=0.5,
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
        config = {"toImageButtonOptions": {"format": "svg"}}
        return fig, config

    # Create consistent color mapping
    color_map = create_color_mapping(plot_df["new_cell_type"].unique())

    if additional_group:
        # Sort data by cell type and additional group
        plot_df = plot_df.sort_values(["new_cell_type", additional_group])

        # Create a categorical x-axis that groups by cell type first
        cell_types = sorted(plot_df["new_cell_type"].unique())
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
            lambda row: position_map[
                (row["new_cell_type"], str(row[additional_group]))
            ],
            axis=1,
        )

        # Create a custom color map for Comp_sex
        if additional_group == "Comp_sex":
            # Define special colors for female (0) and male (1)
            sex_color_map = {
                0: "#FFA500",  # female
                1: "#63B3ED",     # male
            }
            # Create box plot with custom colors for sex
            fig = px.box(
                plot_df,
                x="x_position",
                y="Expression",
                color=additional_group,
                color_discrete_map=sex_color_map,
                points=False,
                hover_data=meta_data.columns,
                title=f"{gene_name} Expression by Cell Type and Sex",
            )
            
            # Create matching strip plot with the same color scheme
            strip_fig = px.strip(
                plot_df,
                x="x_position",
                y="Expression",
                color=additional_group,
                color_discrete_map=sex_color_map,
                hover_data=meta_data.columns,
            )
        else:
            # Create box plot with default colors for other groupings
            fig = px.box(
                plot_df,
                x="x_position",
                y="Expression",
                color=additional_group,
                points=False,
                hover_data=meta_data.columns,
                title=f"{gene_name} Expression by Cell Type and {additional_group}",
            )
            
            # Create matching strip plot
            strip_fig = px.strip(
                plot_df,
                x="x_position",
                y="Expression",
                color=additional_group,
                hover_data=meta_data.columns,
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
        # Standard plot by cell type
        fig = px.box(
            plot_df,
            x="new_cell_type",
            y="Expression",
            color="new_cell_type",
            points=False,
            color_discrete_map=color_map,
            hover_data=meta_data.columns,
            title=f"{gene_name} Expression by Cell Type",
        )

        # Add strip plot
        strip_fig = px.strip(
            plot_df,
            x="new_cell_type",
            y="Expression",
            color="new_cell_type",
            color_discrete_map=color_map,
            hover_data=meta_data.columns,
        )

    # Update strip plot traces to match the box plot
    for trace in strip_fig.data:
        trace.update(marker=dict(opacity=0.4, size=6), showlegend=False)
        fig.add_trace(trace)

    # Optionally connect dots with the same SRA_ID
    if connect_dots:
        for sra_id in plot_df["SRA_ID"].unique():
            sra_df = plot_df[plot_df["SRA_ID"] == sra_id]
            # Use x_position if it exists (for grouped data), otherwise use cell type
            x_values = (
                sra_df["x_position"]
                if "x_position" in sra_df.columns
                else sra_df["new_cell_type"]
            )
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=sra_df["Expression"],
                    mode="lines+markers",
                    line=dict(color="gray", width=1),
                    marker=dict(size=7, opacity=0.6),
                    name=sra_id,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    # Calculate dynamic width based on number of cell types
    num_cell_types = len(plot_df["new_cell_type"].unique())
    base_width = 120  # Base width per cell type in pixels
    dynamic_width = max(
        900, min(2400, num_cell_types * base_width)
    )  # Min 900px, Max 2400px

    fig.update_layout(
        xaxis_title="Cell Type",
        yaxis_title=f"{gene_name} Expression (log10)",
        showlegend=True,
        xaxis={"tickangle": 45, "tickfont": {"size": 25}, "title_font": {"size": 30}},
        yaxis={"title_font": {"size": 30}, "tickfont": {"size": 30}},
        height=600,
        width=dynamic_width,
    )

    config = {
        "toImageButtonOptions": {
            "format": "svg",
            "filename": f"{gene_name}_expression",
            "height": 800,
            "width": dynamic_width,
            "scale": 2,
        }
    }

    return fig, config