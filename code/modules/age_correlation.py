import plotly.express as px
import numpy as np
import streamlit as st


def create_age_correlation_plot(
    matrix,
    genes,
    meta_data,
    gene_name,
    cell_type,
    use_log_age=False,
    remove_zeros=False,
    color_by=None,
    show_trendline=True,
    data_type_filter=None,
    version="v_0.01",
):
    """
    Create a scatter plot of gene expression vs age for a specific cell type

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
    cell_type : str
        Cell type to analyze
    use_log_age : bool
        Whether to use log10 scale for age
    color_by : str
        Column name to use for coloring points (e.g., 'Comp_sex' or 'Modality')
    show_trendline : bool
        Whether to display the linear trendline
    data_type_filter : str or None
        Filter to a specific data type (e.g., "sc", "sn", None for all)
    version : str
        Version of the dataset
    """
    from .data_loader import load_aging_genes  # Import here to avoid circular import
    import plotly.express as px
    import numpy as np

    # remove those where Age_numeric is < 0
    matrix = matrix[:, meta_data["Age_numeric"] >= 0]
    meta_data = meta_data[meta_data["Age_numeric"] >= 0]

    gene_idx = genes[genes[0] == gene_name].index[0]
    expression_values = (
        matrix[gene_idx, :].A1
        if hasattr(matrix[gene_idx, :], "A1")
        else matrix[gene_idx, :]
    )

    plot_df = meta_data.copy()
    plot_df["Expression"] = expression_values
    # if remove_zeros
    if remove_zeros:
        plot_df = plot_df[plot_df["Expression"] > 0]

    # Filter for cell type
    cell_type_df = plot_df[plot_df["new_cell_type"] == cell_type].copy()

    # Apply data type filter if specified
    if data_type_filter:
        cell_type_df = cell_type_df[cell_type_df["Modality"] == data_type_filter]

        # Check if we have data after filtering
        if len(cell_type_df) == 0:
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_annotation(
                x=0.5,
                y=0.5,
                text=f"No {data_type_filter} data available for {gene_name} in {cell_type}",
                showarrow=False,
                font=dict(size=30),
            )
            fig.update_layout(
                height=650,
                width=750,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
            )
            return fig, {}, 0, 1.0, load_aging_genes(version)

    # Add log10 transformation for age if selected
    if use_log_age:
        age_name = "log10_Age_days"
        # Add a small constant to handle zero/negative values
        min_age = cell_type_df["Age_numeric"].min()
        offset = abs(min_age) + 0.1 if min_age <= 0 else 0
        cell_type_df[age_name] = np.log10(cell_type_df["Age_numeric"] + offset)
        x_title = "log₁₀(Age)"
        #add hint that it is log10
        st.info(
            "Note: Age is shown on a log₁₀ scale. This transformation helps to visualize the data better. To see real age values, please uncheck the 'Use log10 scale for age' option or hover over the datapoints."
        )
    else:
        age_name = "Age_days"
        cell_type_df[age_name] = cell_type_df["Age_numeric"]
        x_title = "Age (days)"

    # Create scatter plot with optional coloring
    if color_by:
        # Convert to string type for discrete coloring
        cell_type_df[color_by] = cell_type_df[color_by].astype(str)

        # Create color mapping
        color_map = {}
        for i, val in enumerate(cell_type_df[color_by].unique()):
            color_map[val] = px.colors.qualitative.Plotly[i]

        # Create plot with or without trendline
        if show_trendline:
            fig = px.scatter(
                cell_type_df,
                x=age_name,
                y="Expression",
                color=color_by,
                color_discrete_map=color_map,
                title=f"{gene_name} Expression vs {x_title} in {cell_type}"
                + (f" ({data_type_filter} only)" if data_type_filter else ""),
                opacity=0.6,
                trendline="ols",
            )
        else:
            fig = px.scatter(
                cell_type_df,
                x=age_name,
                y="Expression",
                color=color_by,
                color_discrete_map=color_map,
                title=f"{gene_name} Expression vs {x_title} in {cell_type}"
                + (f" ({data_type_filter} only)" if data_type_filter else ""),
                opacity=0.6,
            )
    else:
        # Create plot with or without trendline
        if show_trendline:
            fig = px.scatter(
                cell_type_df,
                x=age_name,
                y="Expression",
                title=f"{gene_name} Expression vs {x_title} in {cell_type}"
                + (f" ({data_type_filter} only)" if data_type_filter else ""),
                opacity=0.6,
                trendline="ols",
            )
        else:
            fig = px.scatter(
                cell_type_df,
                x=age_name,
                y="Expression",
                title=f"{gene_name} Expression vs {x_title} in {cell_type}"
                + (f" ({data_type_filter} only)" if data_type_filter else ""),
                opacity=0.6,
            )

    # Update marker size and opacity
    fig.update_traces(
        marker=dict(size=15), selector=dict(mode="markers")  # Increased marker size
    )

    # Update trendline appearance if present
    # Replace the trendline update section in age_correlation.py
    if show_trendline:
        for trace in fig.data:
            if trace.mode == "lines":
                # Get min and max values for axes
                x_min = min(cell_type_df[age_name])
                x_max = max(cell_type_df[age_name])

                # Calculate line equation (y = mx + b) from two points
                x_vals = np.array([x_min, x_max])
                y_vals = np.array([trace.y[0], trace.y[-1]])
                m = (y_vals[1] - y_vals[0]) / (x_vals[1] - x_vals[0])
                b = y_vals[0] - m * x_vals[0]

                # Calculate x-intercept (where y=0)
                if m != 0:
                    x_intercept = -b / m
                else:
                    x_intercept = None

                # Calculate y-intercept (where x=0)
                y_intercept = b

                # Create new line segment that stays in positive quadrant
                new_x = []
                new_y = []

                # Add y-intercept if positive and within range
                if y_intercept >= 0 and x_min <= 0 <= x_max:
                    new_x.append(0)
                    new_y.append(y_intercept)

                # Add x-intercept if positive and within range
                if (
                    x_intercept is not None
                    and x_intercept >= 0
                    and x_min <= x_intercept <= x_max
                ):
                    new_x.append(x_intercept)
                    new_y.append(0)

                # Add original points if in positive quadrant
                for i in range(len(trace.x)):
                    if trace.x[i] >= 0 and trace.y[i] >= 0:
                        new_x.append(trace.x[i])
                        new_y.append(trace.y[i])

                # Sort points by x-value
                if len(new_x) >= 2:
                    sorted_pairs = sorted(zip(new_x, new_y))
                    trace.x = [pair[0] for pair in sorted_pairs]
                    trace.y = [pair[1] for pair in sorted_pairs]
                    trace.update(
                        line=dict(width=3, dash="solid", color="rgba(255, 0, 0, 0.8)")
                    )

    # Improve layout with better proportions and larger fonts
    fig.update_layout(
        xaxis_title=dict(text=x_title, font=dict(size=30)),
        yaxis_title=dict(text=f"{gene_name} (log₁₀ CPM)", font=dict(size=30)),
        title=dict(
            text=f"{gene_name} Expression vs {x_title} in {cell_type}"
            + (f" ({data_type_filter} only)" if data_type_filter else ""),
            font=dict(size=30),
            x=0.5,  # Centered title
            xanchor="center",
        ),
        legend=dict(font=dict(size=30), title=dict(font=dict(size=30))),
        height=650,  # Slightly taller
        width=750,  # Slightly less wide
        margin=dict(l=100, r=50, t=100, b=100),  # Increased margins
        xaxis=dict(
            tickfont=dict(size=30),
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=True,
            showgrid=True,
            gridcolor="lightgray",
        ),
        yaxis=dict(
            tickfont=dict(size=30),
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=True,
            showgrid=True,
            gridcolor="lightgray",
        ),
        plot_bgcolor="white",
        autosize=False,
    )

    # Configure download options
    config = {
        "toImageButtonOptions": {
            "format": "svg",
            "filename": f"{gene_name}_age_correlation"
            + (f"_{data_type_filter}" if data_type_filter else ""),
            "height": 800,
            "width": 950,
            "scale": 2,
        }
    }

    # Calculate correlation statistics
    # Only get correlation stats if trendline is shown or we need them for return values
    if (
        show_trendline and len(cell_type_df) >= 2
    ):  # Need at least two points for correlation
        try:
            correlation_stats = px.get_trendline_results(fig)
            r_squared = correlation_stats.px_fit_results.iloc[0].rsquared
            p_value = correlation_stats.px_fit_results.iloc[0].pvalues[1]
        except Exception:
            # Fallback to manual calculation if something goes wrong
            from scipy import stats

            slope, intercept, r_value, p_value, std_err = stats.linregress(
                cell_type_df[age_name], cell_type_df["Expression"]
            )
            r_squared = r_value**2
    else:
        # Calculate manually for return values even if trendline isn't shown
        if len(cell_type_df) >= 2:
            from scipy import stats

            slope, intercept, r_value, p_value, std_err = stats.linregress(
                cell_type_df[age_name], cell_type_df["Expression"]
            )
            r_squared = r_value**2
        else:
            # Not enough data for correlation
            r_squared = 0
            p_value = 1.0

    # Load aging genes for the table
    aging_genes_df = load_aging_genes(version)

    return fig, config, r_squared, p_value, aging_genes_df
