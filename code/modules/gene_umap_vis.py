import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pathlib import Path



from modules.gene_gene_corr import (
    load_gene_data
)

def create_gene_umap_plot(
    umap_path,
    gene,
    base_path,
    meta_data,
    selected_samples=None,
    selected_cell_types=None,
    color_map="viridis",
    sort_order=False,
    metadata_col="assignments"
):
    """
    Create a scatter plot showing correlation between two genes.

    Parameters:
    -----------
    gene1_name : str
        Name of first gene
    gene2_name : str
        Name of second gene
    base_path : str
        Base path to gene data directory
    meta_data : pandas.DataFrame
        Metadata containing SRA_ID and cell type information
    selected_samples : list, optional
        List of specific sample names to include
    color_by_celltype : bool
        Whether to color points by cell type
    selected_cell_types : list, optional
        List of cell types to include in the plot
    """
    try:
        # Load gene expression data
        gene_data = load_gene_data(gene, base_path)

        data = pd.read_parquet(umap_path)

        # Create plot dataframe
        plot_df = pd.DataFrame(
            {   
                "Gene": gene_data[gene].values,
                "UMAP_1": data["UMAP1"].values,
                "UMAP_2": data["UMAP2"].values,
                "SRA_ID": meta_data["SRA_ID"].values,
                "Cell_Type": meta_data["assignments"].values, #maybe change this
                "second_col" : meta_data[metadata_col].values,
            }
        )




        #check if Cat_col is numerical or categorical
        categorical = False
        if (plot_df["second_col"].dtype == "object" or
            plot_df["second_col"].dtype.name == "category" or
            plot_df["second_col"].dtype.name == "string"):
            plot_df["second_col"] = plot_df["second_col"].astype("category")
            categorical = True

        # Filter by selected samples if provided
        if selected_samples is not None and len(selected_samples) > 0:
            mask = plot_df["SRA_ID"].isin(selected_samples)
            plot_df = plot_df[mask]

        # Filter by selected cell types if provided
        if selected_cell_types is not None and len(selected_cell_types) > 0:
            mask = plot_df["Cell_Type"].isin(selected_cell_types)
            plot_df = plot_df[mask]

        if len(plot_df) == 0:
            return None, None, None, "No data available for selected samples"

        total_cells = len(plot_df)

        # Scale marker size and opacity based on total cells
        base_size = 9
        base_opacity = 0.8
        size_scale = min(1.0, 2000 / total_cells)
        opacity_scale = min(1.0, 2000 / total_cells)
        marker_size = max(base_size * size_scale, 3)
        marker_opacity = max(base_opacity * opacity_scale, 0.3)

        # Gene expression plot
        gene_fig = go.Figure()
        expression = plot_df["Gene"].values
        umap_coords = plot_df[["UMAP_1", "UMAP_2"]].values
        color_values = expression.flatten()
        plot_coords = umap_coords.copy()

        if sort_order:
            sort_indices = np.argsort(color_values)
            plot_coords = plot_coords[sort_indices]
            color_values = color_values[sort_indices]

        gene_fig.add_trace(
            go.Scatter(
                x=plot_coords[:, 0],
                y=plot_coords[:, 1],
                mode="markers",
                marker=dict(
                    color=color_values,
                    colorscale=color_map,
                    colorbar=dict(title=f"log1p counts {gene}"),
                    size=marker_size,
                    opacity=marker_opacity,
                ),
                text=[f"Expression: {val:.2f}" for val in color_values],
                hoverinfo="text",
            )
        )

        gene_fig.update_layout(
            title=f"Gene Expression: {gene}",
            xaxis_title="",
            yaxis_title="",
            height=600,
            width=800,
            showlegend=False,
        )

        # Cell type plot
        cell_type_fig = go.Figure()
        cell_types = sorted(plot_df["Cell_Type"].unique())
        second_values = sorted(plot_df["second_col"].unique())

        if categorical:
            colors = px.colors.qualitative.Set3[: len(second_values)]
            color_dict = dict(zip(second_values, colors))
        else:
            #if numeric just set colors to viridis
            colors = "Viridis"

        
        plot_df["color"] = plot_df["second_col"].map(color_dict)

        cell_type_fig.add_trace(
            go.Scatter(
                x=plot_df["UMAP_1"],
                y=plot_df["UMAP_2"],
                mode="markers",
                marker=dict(
                    color=plot_df["color"],
                    size=marker_size,
                    opacity=marker_opacity

                ),
                name=metadata_col,
                #hover to show SRA_ID and cell type
                text=[f"SRA_ID: {sra_id}<br>Cell Type: {cell_type}" for sra_id, cell_type in zip(plot_df["SRA_ID"], plot_df["Cell_Type"])],
            )
        )

        cell_type_fig.update_layout(
            title="Cell Types",
            xaxis_title="",
            yaxis_title="",
            height=600,
            width=800,
            showlegend=True,
            legend=dict(font=dict(size=14), itemsizing="constant"),
        )

        return gene_fig, cell_type_fig
    except Exception as e:
        print(f"Error in plot_sc_dataset: {str(e)}")
        raise