import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pathlib import Path

import numpy as np
import pandas as pd

import plotly.graph_objects as go
import numpy as np
import plotly.express as px


def load_gene_data(gene_name, base_path):
    """
    Load expression data for a single gene from parquet file.
    """
    gene_path = Path(base_path) / "genes_parquet" / f"{gene_name}.parquet"
    if not gene_path.exists():
        raise ValueError(f"Gene data not found for {gene_name}")
    return pd.read_parquet(gene_path)


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

        # Calculate dynamic opacity based on number of points
        opacity = min(0.1, 20000 / len(plot_df))

        # Create scatter plot
        if color_by_celltype:
            fig = px.scatter(
                plot_df,
                x="Gene1",
                y="Gene2",
                color="Cell_Type",
                title=f"Gene Correlation: {gene1_name} vs {gene2_name}",
                labels={
                    "Gene1": f"{gene1_name} Expression",
                    "Gene2": f"{gene2_name} Expression",
                },
                hover_data=["SRA_ID"],
            )
        else:
            fig = px.scatter(
                plot_df,
                x="Gene1",
                y="Gene2",
                title=f"Gene Correlation: {gene1_name} vs {gene2_name}",
                labels={
                    "Gene1": f"{gene1_name} Expression",
                    "Gene2": f"{gene2_name} Expression",
                },
                hover_data=["SRA_ID", "Cell_Type"],
            )

        # Update marker opacity for all traces
        for trace in fig.data:
            trace.marker.opacity = opacity

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

        return fig, config, stats, None

    except Exception as e:
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
