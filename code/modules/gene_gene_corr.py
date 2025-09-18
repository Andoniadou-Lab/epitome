import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pathlib import Path

def load_gene_data(gene_name, base_path):
    """
    Load expression data for a single gene from parquet file.
    """
    gene_path = Path(base_path) / "genes_parquet" / f"{gene_name}.parquet"
    if not gene_path.exists():
        raise ValueError(f"Gene data not found for {gene_name}")
    return pd.read_parquet(gene_path)


def load_total_counts(base_path):
    total_counts  = f"{base_path}/total_counts.parquet"
    return pd.read_parquet(total_counts)

def create_gene_correlation_plot(
    gene1_name,
    gene2_name,
    base_path,
    meta_data,
    selected_samples=None,
    color_by_celltype=True,
    selected_cell_types=None,
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
        gene1_data = load_gene_data(gene1_name, base_path)
        gene2_data = load_gene_data(gene2_name, base_path)

        # Ensure we're using the first column of each gene's data
        gene1_col = gene1_data.columns[0]
        gene2_col = gene2_data.columns[0]

        # Create plot dataframe
        plot_df = pd.DataFrame(
            {
                "Gene1": gene1_data[gene1_col].values,
                "Gene2": gene2_data[gene2_col].values,
                "SRA_ID": meta_data["SRA_ID"].values,
                "Cell_Type": meta_data["new_cell_type"].values,
            }
        )

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

        # Calculate correlation statistics using spearman
        correlation = plot_df["Gene1"].corr(plot_df["Gene2"], method="spearman")
        pct_coexpression = (plot_df["Gene1"] > 0) & (plot_df["Gene2"] > 0)
        pct_coexpression = 100 * pct_coexpression.sum() / len(plot_df)

        # Calculate additional statistics for each cell type
        cell_type_stats = (
            plot_df.groupby("Cell_Type")
            .agg({"Gene1": ["count", lambda x: x.corr(plot_df.loc[x.index, "Gene2"])]})
            .round(3)
        )
        cell_type_stats.columns = ["count", "correlation"]

        stats = {
            "correlation": correlation,
            "sample_count": len(plot_df),
            "cell_type_count": len(plot_df["Cell_Type"].unique()),
            "cell_type_stats": cell_type_stats,
            "pct_coexpression": pct_coexpression,
        }

        # Calculate dynamic opacity based on number of points
        opacity = min(0.7, max(0.1, 5000 / len(plot_df)))
        
        # Subsample for visualization
        # Only subsample if dataset is large
        if len(plot_df) > 50000:
            # Use stratified sampling if coloring by cell type
            if color_by_celltype:
                sample_indices = []
                max_points = 50000
                
                # Calculate needed points from each cell type proportionally
                for cell_type in plot_df["Cell_Type"].unique():
                    cell_indices = plot_df[plot_df["Cell_Type"] == cell_type].index
                    cell_sample_size = max(1, int(len(cell_indices) * max_points / len(plot_df)))
                    
                    if len(cell_indices) > cell_sample_size:
                        cell_sample = np.random.choice(cell_indices, size=cell_sample_size, replace=False)
                        sample_indices.extend(cell_sample)
                    else:
                        sample_indices.extend(cell_indices)
                
                plot_sample = plot_df.loc[sample_indices]
                
                # Add note about subsampling to title
                subsample_note = f" (showing {len(plot_sample):,} of {len(plot_df):,} cells)"
            else:
                # Simple random sampling if not coloring by cell type
                plot_sample = plot_df.sample(min(50000, len(plot_df)))
                subsample_note = f" (showing {len(plot_sample):,} of {len(plot_df):,} cells)"
        else:
            # Use all data if not too large
            plot_sample = plot_df
            subsample_note = ""

        # Create scatter plot (using sampled data for visual representation)
        if color_by_celltype:
            fig = px.scatter(
                plot_sample,
                x="Gene1",
                y="Gene2",
                color="Cell_Type",
                title=f"Gene Correlation: {gene1_name} vs {gene2_name}{subsample_note}",
                labels={
                    "Gene1": f"{gene1_name} Expression",
                    "Gene2": f"{gene2_name} Expression",
                },
                hover_data=["SRA_ID"],
            )
        else:
            fig = px.scatter(
                plot_sample,
                x="Gene1",
                y="Gene2",
                title=f"Gene Correlation: {gene1_name} vs {gene2_name}{subsample_note}",
                labels={
                    "Gene1": f"{gene1_name} Expression",
                    "Gene2": f"{gene2_name} Expression",
                },
                hover_data=["SRA_ID", "Cell_Type"],
            )

        # Update marker properties for visual quality
        # Adjust marker size based on number of points
        marker_size = max(3, min(8, 50000 / len(plot_sample)))
        
        # Update marker parameters for all traces
        for trace in fig.data:
            trace.marker.opacity = opacity
            trace.marker.size = marker_size
            trace.marker.line = dict(width=0)  # Remove marker outlines for cleaner look

        # Add correlation line
        x_min, x_max = plot_df["Gene1"].min(), plot_df["Gene1"].max()
        y_min, y_max = plot_df["Gene2"].min(), plot_df["Gene2"].max()

        # Calculate slope and intercept for correlation line
        slope = correlation * (y_max - y_min) / (x_max - x_min)
        intercept = plot_df["Gene2"].mean() - slope * plot_df["Gene1"].mean()

        fig.add_trace(
            go.Scatter(
                x=[x_min, x_max],
                y=[x_min * slope + intercept, x_max * slope + intercept],
                mode="lines",
                line=dict(color="red", dash="dash"),
                name=f"Correlation (r={correlation:.3f})",
            )
        )

        # Update layout
        fig.update_layout(
            height=600,
            width=800,
            showlegend=True,
            legend_title="Cell Types" if color_by_celltype else None,
        )

        # Configure download options
        config = {
            "toImageButtonOptions": {
                "format": "svg",
                "filename": f"correlation_{gene1_name}_{gene2_name}",
                "height": 800,
                "width": 1000,
                "scale": 2,
            }
        }

        return fig, config, stats, None

    except Exception as e:
        import traceback
        print(f"Error in create_gene_correlation_plot: {str(e)}")
        print(traceback.format_exc())
        return None, None, None, str(e)


def get_available_genes(base_path):
    """
    Get list of available genes from parquet directory.
    """
    gene_dir = Path(base_path) / "genes_parquet"
    if not gene_dir.exists():
        raise ValueError(f"Gene directory not found: {gene_dir}")

    genes = [f.stem for f in gene_dir.glob("*.parquet")]
    return sorted(genes)