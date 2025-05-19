import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from .utils import create_color_mapping
import scipy


def filter_dotplot_data(
    proportion_matrix,
    expression_matrix,
    rows1,
    rows2,
    meta_data,
    selected_samples=None,
    selected_authors=None,
    age_range=None,
    only_normal=False,
):
    """
    Filter dotplot data based on selected criteria
    """
    import numpy as np
    import pandas as pd
    import scipy.sparse as sparse

    # Convert matrices to CSR format if they are sparse
    if sparse.issparse(proportion_matrix):
        proportion_matrix = proportion_matrix.tocsr()
    if sparse.issparse(expression_matrix):
        expression_matrix = expression_matrix.tocsr()

    # Parse row information
    def parse_row_info(rows):
        split_info = rows.iloc[:, 0].str.split("_", n=1)
        return pd.DataFrame(
            {
                "SRA_ID": [x[0] for x in split_info],
                "cell_type": [x[1] if len(x) > 1 else "" for x in split_info],
            }
        )

    # Get row information
    row_info1 = parse_row_info(rows1)
    row_info2 = parse_row_info(rows2)

    # Get valid SRA_IDs from metadata based on filters
    valid_sra_ids = set(meta_data["SRA_ID"].unique())

    # Apply filters to meta_data first
    if selected_samples:
        valid_sra_ids &= set(
            meta_data[meta_data["Name"].isin(selected_samples)]["SRA_ID"].unique()
        )

    if selected_authors:
        valid_sra_ids &= set(
            meta_data[meta_data["Author"].isin(selected_authors)]["SRA_ID"].unique()
        )

    if age_range and "Age_numeric" in meta_data.columns:
        # Convert age values to float, handling both string and numeric inputs
        try:
            age_values = pd.to_numeric(
                meta_data["Age_numeric"].replace(",", ".", regex=True), errors="coerce"
            )
            # Create age mask, handling NaN values
            age_mask = (
                age_values.notna()
                & (age_values >= float(age_range[0]))
                & (age_values <= float(age_range[1]))
            )
            valid_sra_ids &= set(meta_data[age_mask]["SRA_ID"].unique())
        except Exception as e:
            print(f"Error in age filtering: {str(e)}")
            # If age filtering fails, continue without it
            pass

    if only_normal:
        valid_sra_ids &= set(meta_data[meta_data["Normal"] == 1]["SRA_ID"].unique())

    # Create masks for both matrices
    mask1 = row_info1["SRA_ID"].isin(valid_sra_ids).values  # Convert to numpy array
    mask2 = row_info2["SRA_ID"].isin(valid_sra_ids).values  # Convert to numpy array

    # Filter matrices
    filtered_prop_matrix = proportion_matrix[mask1]
    filtered_expr_matrix = expression_matrix[mask2]

    # Filter rows
    filtered_rows1 = rows1[mask1].copy()
    filtered_rows2 = rows2[mask2].copy()

    filtered_curation = meta_data[meta_data["SRA_ID"].isin(valid_sra_ids)]

    return (
        filtered_curation,
        filtered_prop_matrix,
        filtered_expr_matrix,
        filtered_rows1,
        filtered_rows2,
    )


def create_dotplot(
    proportion_matrix,
    expression_matrix,
    genes1,
    genes2,
    rows1,
    rows2,
    selected_genes,
    selected_cell_types=None,
    color_scheme="Red"
):
    """
    Create a dot plot with properly filtered data

    Parameters:
    -----------
    proportion_matrix : array-like
        Matrix of proportion values
    expression_matrix : array-like
        Matrix of expression values
    genes1, genes2 : pandas.DataFrame
        Gene information for proportion and expression matrices
    rows1, rows2 : pandas.DataFrame
        Row information for proportion and expression matrices
    selected_genes : list
        List of genes to display
    selected_cell_types : list, optional
        List of cell types to display (default: None, shows all cell types)
    """
    try:
        # Data processing
        proportion_matrix = (
            proportion_matrix.toarray()
            if hasattr(proportion_matrix, "toarray")
            else np.array(proportion_matrix)
        )
        expression_matrix = (
            expression_matrix.toarray()
            if hasattr(expression_matrix, "toarray")
            else np.array(expression_matrix)
        )

        genes_list1 = [str(gene) for gene in genes1[genes1.columns[0]].tolist()]
        genes_list2 = [str(gene) for gene in genes2[genes2.columns[0]].tolist()]
        row_data = [str(row) for row in rows1[rows1.columns[0]].tolist()]
        cell_types = [row.split("_")[1] if "_" in row else row for row in row_data]

        plot_data = []
        for gene in selected_genes:
            gene_str = str(gene)
            if gene_str not in genes_list1 or gene_str not in genes_list2:
                continue

            gene_idx1 = genes_list1.index(gene_str)
            gene_idx2 = genes_list2.index(gene_str)

            # Get values and ensure they're properly converted to floats
            proportions = proportion_matrix[:, gene_idx1].flatten()
            expressions = expression_matrix[:, gene_idx2].flatten()

            temp_df = pd.DataFrame(
                {
                    "cell_type": cell_types,
                    "proportion": proportions,
                    "expression": expressions,
                }
            )

            # Filter by selected cell types if provided
            if selected_cell_types:
                temp_df = temp_df[temp_df["cell_type"].isin(selected_cell_types)]

            # Group by cell type to get averages
            grouped = (
                temp_df.groupby("cell_type")
                .agg(
                    {
                        "proportion": "mean",
                        "expression": "mean",
                        "cell_type": "size",  # This gives us the count of datasets
                    }
                )
                .rename(columns={"cell_type": "Number_of_datasets"})
            )

            for cell_type, row in grouped.iterrows():
                plot_data.append(
                    {
                        "Gene": gene_str,
                        "Cell_Type": str(cell_type),
                        "Proportion": float(row["proportion"]),
                        "Mean_Expression": float(row["expression"]),
                        "Number_of_datasets": int(row["Number_of_datasets"]),
                    }
                )

        if not plot_data:
            raise ValueError("No valid data to plot")

        plot_df = pd.DataFrame(plot_data)

        # Define consistent dot size
        DOT_SIZE = 450
        PLOT_WIDTH = 1200

        if color_scheme == "Red":
            color= [[0, "lightgrey"], [1, "red"]]
        elif color_scheme == "Blue":
            color= [[0, "lightgrey"], [1, "#0000FF"]]
        elif color_scheme == "Viridis":
            color= px.colors.sequential.Viridis
        elif color_scheme == "Cividis":
            color= px.colors.sequential.Cividis

        # Create main plot
        fig1 = go.Figure()
        fig1.add_trace(
            go.Scatter(
                x=plot_df["Gene"],
                y=plot_df["Cell_Type"],
                mode="markers",
                marker=dict(
                    size=plot_df["Proportion"] * DOT_SIZE,
                    color=plot_df["Mean_Expression"],
                    colorscale=color,
                    showscale=True,
                    colorbar=dict(
                        title=dict(
                            text="Expression Level",
                            side="right",
                            font=dict(size=30),
                        ),
                        tickfont=dict(size=30),
                        x=0.79,
                    ),
                    sizemode="area",
                ),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    + "Gene: %{x}<br>"
                    + "Expression: %{customdata[1]:.2f}<br>"
                    + "Proportion: %{customdata[0]:.2f}<br>"
                    + "Datasets: %{customdata[2]}<br>"
                    + "<extra></extra>"
                ),
                customdata=plot_df[
                    ["Proportion", "Mean_Expression", "Number_of_datasets"]
                ].values,
                name="",
            )
        )
        # Create legend plot
        legend_sizes = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
        legend_df = pd.DataFrame(
            {
                "x": np.zeros(len(legend_sizes)),
                "y": [f"Prop.: {size:.2f}" for size in legend_sizes],
                "size": legend_sizes,
            }
        )

        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=legend_df["x"],
                y=legend_df["y"],
                mode="markers",
                marker=dict(
                    size=legend_df["size"] * DOT_SIZE, color="gray", sizemode="area"
                ),
                # larger font
                name="",
            )
        )

        fig2.update_layout(
            yaxis=dict(
                tickfont=dict(size=30),  # Increase font size for y-axis labels
            ),
            margin=dict(
                l=10, r=10, t=10, b=10
            ),  # Adjust margins to accommodate larger text
            height=400,  # Increase height to accommodate larger text
        )

        # Combine plots
        fig = go.Figure(
            data=[t for t in fig1.data]
            + [t.update(xaxis="x2", yaxis="y2") for t in fig2.data]
        )

        # Update layout
        fig.update_layout(
            height=800,
            width=PLOT_WIDTH,
            title="Gene Expression Dot Plot",
            showlegend=False,
            margin=dict(l=10, r=10, t=50, b=10),
            xaxis=dict(
                title="Genes",
                tickangle=45,
                domain=[0, 0.80],
                tickfont=dict(size=30 * min(1, 12 / len(selected_genes))),
                title_font=dict(size=30),
            ),
            yaxis=dict(
                title="Cell Types",
                gridcolor="lightgray",
                tickfont=dict(size=30),
                title_font=dict(size=30),
            ),
            xaxis2=dict(domain=[0.95, 1], visible=False),
            yaxis2=dict(
                anchor="x2",
                overlaying="y",
                side="right",
                showgrid=False,
                scaleanchor="x2",
                scaleratio=1,
                # tick size
                tickfont=dict(size=20),
            ),
            plot_bgcolor="white",
        )

        # Configure hover behavior
        fig.update_traces(hoverlabel=dict(bgcolor="white", font_size=12))

        # Download configuration
        config = {
            "toImageButtonOptions": {
                "format": "svg",
                "filename": "dotplot_with_legend",
                "height": 800,
                "width": PLOT_WIDTH,
                "scale": 2,
            }
        }

        return fig, config

    except Exception as e:
        print(f"Error in dot plot creation: {str(e)}")
        raise


def create_ligand_receptor_plot(
    liana_df,
    selected_source=None,
    selected_target=None,
    top_n=50,
    sort_by="magnitude",
    order_by="sender",
):
    """
    Create a dot plot for ligand-receptor interactions, showing unique interactions across cell types

    Parameters:
    -----------
    liana_df : pandas.DataFrame
        DataFrame containing ligand-receptor interactions
    selected_source : list, optional
        List of source cell types to include
    selected_target : list, optional
        List of target cell types to include
    top_n : int
        Number of top interactions to display
    sort_by : str
        Column to sort interactions by ('magnitude' or 'specificity')
    order_by : str
        How to order the x-axis ('sender' or 'target')
    """
    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np
    import math

    # Cap the maximum number of interactions at 40
    top_n = min(top_n, 40)

    # Filter by selected source and target if provided
    if selected_source:
        liana_df = liana_df[liana_df["source"].isin(selected_source)]
    if selected_target:
        liana_df = liana_df[liana_df["target"].isin(selected_target)]

    # Create unique interaction identifier
    liana_df["interaction"] = (
        liana_df["ligand_complex"] + " → " + liana_df["receptor_complex"]
    )

    # Find top interactions based on best rank for each unique interaction
    sort_column = "magnitude_rank" if sort_by == "magnitude" else "specificity_rank"
    best_ranks = liana_df.groupby("interaction")[sort_column].min().reset_index()
    top_interactions = best_ranks.nsmallest(top_n, sort_column)["interaction"].tolist()

    # Filter original dataframe to only include top interactions
    plot_df = liana_df[liana_df["interaction"].isin(top_interactions)].copy()

    # Create cell pair labels and order them based on user preference
    if order_by == "sender":
        # Order by source cell type first, then target
        plot_df["cell_pair"] = plot_df["source"] + " → " + plot_df["target"]
        unique_sources = sorted(plot_df["source"].unique())
        unique_targets = sorted(plot_df["target"].unique())
        ordered_pairs = []
        for source in unique_sources:
            source_targets = plot_df[plot_df["source"] == source]["target"].unique()
            for target in sorted(source_targets):
                ordered_pairs.append(f"{source} → {target}")
    else:  # order_by == 'target'
        # Order by target cell type first, then source
        plot_df["cell_pair"] = plot_df["target"] + " ← " + plot_df["source"]
        unique_targets = sorted(plot_df["target"].unique())
        unique_sources = sorted(plot_df["source"].unique())
        ordered_pairs = []
        for target in unique_targets:
            target_sources = plot_df[plot_df["target"] == target]["source"].unique()
            for source in sorted(target_sources):
                ordered_pairs.append(f"{target} ← {source}")

    # Calculate dynamic plot height
    height_per_interaction = 28
    plot_height = max(700, len(top_interactions) * height_per_interaction)

    # Convert ranks to -log10 scale for displayed data only

    # add 0.01
    #if 0 or less, set to 0.01
    plot_df["magnitude_rank"] = [ 0.1 if x <= 0 else x for x in plot_df["magnitude_rank"] ]
    plot_df["specificity_rank"] = [ 0.1 if x <= 0 else x for x in plot_df["specificity_rank"] ]
    
    plot_df["magnitude_rank"] = -np.log10(plot_df["magnitude_rank"])
    plot_df["specificity_rank"] = -np.log10(plot_df["specificity_rank"])

    
    

    

    # Calculate color range based on actually displayed data
    color_min = plot_df["specificity_rank"].min()
    color_max = plot_df["specificity_rank"].max()

    # Define consistent dot size for normalization
    DOT_SIZE = 300

    # Calculate the min and max magnitude ranks for consistent scaling
    magnitude_rank_min = plot_df["magnitude_rank"].min()
    magnitude_rank_max = plot_df["magnitude_rank"].max()

    #if these are equal, set to 0 and 1
    if magnitude_rank_min == magnitude_rank_max:
        magnitude_rank_min = 0
        magnitude_rank_max = 1


    # Create main figure
    fig = go.Figure()

    # Ensure all unique cell pairs are represented on x-axis
    cell_pairs = ordered_pairs

    # Prepare hover data
    hover_data = {pair: [] for pair in cell_pairs}

    # Populate data points and hover information
    for interaction in top_interactions:
        interaction_data = plot_df[plot_df["interaction"] == interaction]

        # Scale sizes relative to the full dataset
        scaled_sizes = (
            (interaction_data["magnitude_rank"] - magnitude_rank_min)
            / (magnitude_rank_max - magnitude_rank_min)
        ) * DOT_SIZE

        for _, row in interaction_data.iterrows():
            cell_pair = row["cell_pair"]
            # If cell pair not in cell_pairs, something is wrong
            if cell_pair not in cell_pairs:
                print(f"Warning: {cell_pair} not found in cell_pairs!")
                continue
            size = scaled_sizes[row.name]
            #if nan set to 0
            if size == np.nan:
                size = 0.1
            #if its any kind of nan
            if math.isnan(size):
                size = 0.1
            print(f"size: {size}")
            hover_data[cell_pair].append(
                {
                    "interaction": interaction,
                    "specificity_rank": row["specificity_rank"],
                    "magnitude_rank": row["magnitude_rank"],
                    "size":  size
                }
            )

    
    #
    # Prepare traces for each cell pair
    for pair in cell_pairs:
        # Skip if no data for this pair
        if not hover_data[pair]:
            fig.add_trace(
                go.Scatter(
                    x=[pair],
                    y=(
                        ["No Data"]
                        if "No Data" not in fig.data
                        else [f"No Data {len(fig.data)}"]
                    ),
                    mode="markers",
                    marker=dict(color="lightgray", size=10),
                    showlegend=False,
                )
            )
            continue

        # Prepare data for this pair
        pair_data = hover_data[pair]

        print(pair_data)
        
        x_coords = [pair] * len(pair_data)
        y_coords = [data["interaction"] for data in pair_data]
        sizes = [data["size"] for data in pair_data]
        # change nans to 0
        sizes = [0.1 if math.isnan(size) else size for size in sizes]
        specificity_ranks = [data["specificity_rank"] for data in pair_data]


        fig.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="markers",
                marker=dict(
                    size=sizes,
                    color=specificity_ranks,
                    colorscale=[[0, "lightgrey"], [1, "red"]],
                    showscale=pair
                    == cell_pairs[0],  # Only show colorbar for first trace
                    colorbar=dict(
                        title=dict(
                            text="Specificity (-log10)",
                            side="right",
                            font=dict(size=16),
                        ),
                        title_side="right",
                        thickness=20,
                        len=0.75,
                        tickfont=dict(size=14),
                    ),
                    cmin=color_min,
                    cmax=color_max,
                    sizemode="area",
                ),
                hovertemplate=(
                    "Cell Pair: "
                    + pair
                    + "<br>"
                    + "<b>%{y}</b><br>"
                    + "Specificity (-log10): %{customdata[0]:.3f}<br>"
                    + "Magnitude (-log10): %{customdata[1]:.3f}<br>"
                    + "<extra></extra>"
                ),
                customdata=[
                    [data["specificity_rank"], data["magnitude_rank"]]
                    for data in pair_data
                ],
                showlegend=False,
            )
        )

    # Create magnitude legend with values spanning the full range
    num_legend_points = 6
    legend_magnitude_ranks = np.linspace(
        magnitude_rank_min, magnitude_rank_max, num_legend_points
    )
    
    legend_magnitude_ranks[0] = 5  # Set the first value to 5 as requested
    legend_magnitude_ranks = np.round(legend_magnitude_ranks / 5) * 5

    # Scale sizes for legend relative to full dataset
    legend_sizes = (
        (legend_magnitude_ranks - magnitude_rank_min)
        / (magnitude_rank_max - magnitude_rank_min)
    ) * DOT_SIZE

    legend_df = pd.DataFrame(
        {
            "x": np.zeros(len(legend_magnitude_ranks)),
            "y": [val for val in legend_magnitude_ranks],
            "size": legend_sizes,
        }
    )

    #only keep last instance of each y value
    legend_df = legend_df.drop_duplicates(subset=["y"], keep="last")
    legend_df = legend_df.sort_values(by=["y"], ascending=True)

    legend_df["y"] = [f"Mag.: {val:.2f}" for val in legend_df["y"]]

    print(legend_df)
    # Add magnitude legend dots
    fig.add_trace(
        go.Scatter(
            x=legend_df["x"],
            y=legend_df["y"],
            mode="markers",
            marker=dict(size=legend_df["size"], color="gray", sizemode="area"),
            xaxis="x2",
            yaxis="y2",
            showlegend=False,
        )
    )

    # Update layout
    fig.update_layout(
        height=plot_height,
        width=1700,  # Slightly wider to accommodate all components
        title=f"Top {len(top_interactions)} Ligand-Receptor Interactions (sorted by {sort_by})",
        showlegend=False,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(
            title="Cell Pairs",
            tickangle=45,
            showgrid=True,
            gridcolor="lightgray",
            gridwidth=1,
            tickfont=dict(size=10),  # Slightly smaller font
            title_font=dict(size=20),
            automargin=True,
            categoryorder="array",
            categoryarray=cell_pairs,
            domain=[0, 0.9],  # Main plot takes 0-0.80
        ),
        yaxis=dict(
            title="Ligand-Receptor Pairs",
            tickfont=dict(size=14),
            title_font=dict(size=20),
            automargin=True,
            showticklabels=True,
            constrain="domain",
            tickmode="array",
            ticktext=top_interactions
            + (["No Data"] if any(not data for data in hover_data.values()) else []),
            tickvals=list(
                range(
                    len(top_interactions)
                    + (1 if any(not data for data in hover_data.values()) else 0)
                )
            ),
            range=[-0.5, len(top_interactions) + 0.5],
        ),
        xaxis2=dict(
            domain=[0.9, 0.95], visible=False  # Dot size legend takes 0.80-0.90
        ),
        yaxis2=dict(
            anchor="x2",
            overlaying="y",
            side="right",
            showgrid=False,
            scaleanchor="x2",
            scaleratio=1,
            tickfont=dict(size=12),
        ),
        # Specificity colorbar domain
        coloraxis_colorbar=dict(x=1, len=1),  # Slightly adjusted to fit within 0.90-1.0
        plot_bgcolor="white",
    )

    # Configure download options
    config = {
        "toImageButtonOptions": {
            "format": "svg",
            "filename": f"ligand_receptor_interactions_{sort_by}",
            "height": plot_height + 100,
            "width": 1700,
            "scale": 2,
        },
        "displayModeBar": True,
        "scrollZoom": False,  # Disable zoom
        "staticPlot": False,  # Allow hover while keeping plot static
        "modeBarButtonsToAdd": [],  # Remove additional mode bar buttons
    }

    return fig, config, plot_df
