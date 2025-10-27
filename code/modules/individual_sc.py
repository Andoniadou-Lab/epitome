import os
import scanpy as sc
import anndata
import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
import scipy.sparse
from .utils import create_color_mapping

def plot_sc_dataset(adata, selected_gene, sort_order=False, color_map="viridis", download_as="png"):
    """
    Create two interactive UMAP plots - gene expression and cell types
    """
    try:
        umap_coords = adata.obsm["X_umap"]
        total_cells = len(adata)

        # Scale marker size and opacity based on total cells
        base_size = 9
        base_opacity = 0.8
        size_scale = min(1.0, 2000 / total_cells)
        opacity_scale = min(1.0, 2000 / total_cells)
        marker_size = max(base_size * size_scale, 3)
        marker_opacity = max(base_opacity * opacity_scale, 0.3)

        # Gene expression plot
        gene_fig = go.Figure()
        selected_gene_index = np.where(adata.var_names == selected_gene)[0]
        expression = adata[:, selected_gene_index].X
        color_values = (
            expression.toarray().flatten()
            if scipy.sparse.issparse(expression)
            else expression.flatten()
        )
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
                    colorbar=dict(title=f"log1p counts {selected_gene}"),
                    size=marker_size,
                    opacity=marker_opacity,
                ),
                text=[f"Expression: {val:.2f}" for val in color_values],
                hoverinfo="text",
            )
        )

        gene_fig.update_layout(
            title=f"Gene Expression: {selected_gene}",
            xaxis_title="",
            yaxis_title="",
            height=600,
            width=800,
            showlegend=False,
        )

        # Cell type plot - Fixed to restore legend click functionality
        cell_type_fig = go.Figure()
        cell_types = sorted(adata.obs["new_cell_type"].unique())
        color_dict = create_color_mapping()

        for cell_type in cell_types:
            mask = adata.obs["new_cell_type"] == cell_type
            cell_coords = umap_coords[mask]

            # Scale size and opacity by cell type count
            cell_type_count = len(cell_coords)
            type_size = marker_size
            type_opacity = marker_opacity

            # Add the actual data trace (no legend)
            cell_type_fig.add_trace(
                go.Scatter(
                    x=cell_coords[:, 0],
                    y=cell_coords[:, 1],
                    mode="markers",
                    marker=dict(
                        color=color_dict[cell_type],
                        size=type_size,
                        opacity=type_opacity,
                        line=dict(width=0)
                    ),
                    name=cell_type,
                    showlegend=False,  # Hide legend for data trace
                    legendgroup=cell_type,  # Group for legend control
                    hovertemplate=f'<b>Cell Type:</b> {cell_type}<br>' +
                                '<b>UMAP_1:</b> %{x}<br>' +
                                '<b>UMAP_2:</b> %{y}<extra></extra>'
                )
            )
            
            # Add a bright legend-only trace with no data points
            cell_type_fig.add_trace(
                go.Scatter(
                    x=[None],  # No data points, just for legend
                    y=[None],
                    mode="markers",
                    marker=dict(
                        color=color_dict[cell_type],
                        size=12,  # Fixed size for legend
                        opacity=1.0,  # Full opacity for bright legend
                        line=dict(width=0)
                    ),
                    name=cell_type,
                    showlegend=True,  # Show bright legend
                    legendgroup=cell_type,  # Same group as data trace
                    hoverinfo='skip'
                )
            )

        cell_type_fig.update_layout(
            title="Cell Types",
            xaxis_title="",
            yaxis_title="",
            height=600,
            width=800,
            showlegend=True,
            legend=dict(
                font=dict(size=14), 
                itemsizing="constant",
                tracegroupgap=0,
                bgcolor="rgba(255,255,255,0.8)"
            ),
        )
        
        config = {
        "toImageButtonOptions": {
            "format": download_as,
            "filename": f"{selected_gene}_epitome_umap",
            "scale": 4
        }
    }
        
        return gene_fig, cell_type_fig, config
    except Exception as e:
        print(f"Error in plot_sc_dataset: {str(e)}")
        raise

def list_available_datasets(BASE_PATH, base_path, version="v_0.01"):
    """List available single-cell datasets with metadata"""
    try:
        # Load curation data
        curation_data = pd.read_parquet(
            f"{BASE_PATH}/data/curation/{version}/cpa.parquet"
        )

        # List all .h5ad files in the directory
        datasets = [
            f
            for f in os.listdir(os.path.join(base_path, version, "epitome_h5_files"))
            if f.endswith(".h5ad")
        ]
        # Remove the _processed.h5ad from each string to get SRA_IDs
        #if any of them contain processed
        if any("_processed" in f for f in datasets):
            sra_ids = [f.replace("_processed.h5ad", "") for f in datasets]
        else:
            sra_ids = [f.replace(".h5ad", "") for f in datasets]

        # Create display names using curation data
        display_names = []
        for sra_id in sra_ids:
            dataset_info = curation_data[
                        curation_data["SRA_ID"].str.contains(sra_id, na=False)
                ]
            if dataset_info.empty:
                dataset_info = curation_data[curation_data["GEO"].str.contains(sra_id, na=False)]
    
            if not dataset_info.empty:
                author = dataset_info.iloc[0]["Author"]
                name = dataset_info.iloc[0]["Name"]
                display_names.append(f"{author} - {name} - {sra_id}")
            else:
                display_names.append(sra_id)

        # Create dictionary mapping display names to SRA_IDs
        return dict(zip(display_names, sra_ids))
    except Exception as e:
        print(f"Error listing datasets: {e}")
        return {}


def get_dataset_info(adata):
    """
    Extract basic information about the dataset

    Parameters:
    -----------
    adata : anndata.AnnData
        Loaded single-cell dataset

    Returns:
    --------
    dict
        Dictionary containing dataset information
    """
    try:
        info = {
            "Total Cells": adata.shape[0],
            "Total Genes": adata.shape[1],
            "Cell Types": adata.obs["new_cell_type"].unique().tolist(),
            "Cell Type Counts": adata.obs["new_cell_type"].value_counts().to_dict(),
        }
        return info
    except Exception as e:
        print(f"Error extracting dataset info: {e}")
        return {}