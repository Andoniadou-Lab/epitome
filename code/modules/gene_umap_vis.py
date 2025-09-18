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
    total_counts,
    selected_samples=None,
    selected_cell_types=None,
    color_map="viridis",
    sort_order=False,
    metadata_col="assignments",
    download_as="png"

):
    """
    Create a rasterized scatter plot showing gene expression on UMAP coordinates.
    Uses datashader for efficient rendering of large datasets.

    Parameters:
    -----------
    umap_path : str
        Path to UMAP coordinates parquet file
    gene : str
        Name of gene to visualize
    base_path : str
        Base path to gene data directory
    meta_data : pandas.DataFrame
        Metadata containing SRA_ID and cell type information
    selected_samples : list, optional
        List of specific sample names to include
    selected_cell_types : list, optional
        List of cell types to include in the plot
    color_map : str
        Colormap to use for expression visualization
    sort_order : bool
        Whether to sort points by expression value
    metadata_col : str
        Column name in meta_data to use for secondary coloring
    """
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    import plotly.express as px
    import io
    import base64
    
    try:
        #set numpy seed as 42
        np.random.seed(42)
        # Load gene expression data
        gene_data = load_gene_data(gene, base_path)
        
        
        # Load UMAP data
        data = pd.read_parquet(umap_path)
        
        # Create plot dataframe
        plot_df = pd.DataFrame({
            "Gene": gene_data[gene].values,
            "UMAP_1": data["UMAP1"].values,
            "UMAP_2": data["UMAP2"].values,
            "SRA_ID": meta_data["SRA_ID"].values,
            "Cell_Type": meta_data["assignments"].values,
            metadata_col: meta_data[metadata_col].values,
        })
        #total counts take its second column
        if isinstance(total_counts, pd.DataFrame):
            total_counts = total_counts.iloc[:, 1].values
        elif isinstance(total_counts, np.ndarray):
            total_counts = total_counts[:, 1]
            
        plot_df["Gene"] = np.log1p((plot_df["Gene"] / total_counts) * 10000)

        # Filter by selected samples if provided
        if selected_samples is not None and len(selected_samples) > 0:
            mask = plot_df["SRA_ID"].isin(selected_samples)
            plot_df = plot_df[mask]
        
        # Filter by selected cell types if provided
        if selected_cell_types is not None and len(selected_cell_types) > 0:
            mask = plot_df["Cell_Type"].isin(selected_cell_types)
            plot_df = plot_df[mask]
        
        if len(plot_df) == 0:
            return None, None
        
        # Create gene expression plot
        def create_gene_expression_plot():
            num_points = len(plot_df)
            
            if num_points < 1000:
                marker_size = 8
                opacity = 0.9
            elif num_points < 10000:
                marker_size = 6
                opacity = 0.7
            elif num_points < 50000:
                marker_size = 4
                opacity = 0.5
            else:
                marker_size = 3
                opacity = 0.3
            
            # For very large datasets, use sampling
            if num_points > 50000:
                sample_size = 50000
                sampled_df = plot_df.sample(sample_size)
            else:
                sampled_df = plot_df

            if sort_order:
                sampled_df = sampled_df.sort_values(by="Gene")
            
            # Create the figure using Plotly's scatter plot
            fig = px.scatter(
                sampled_df,
                x='UMAP_1',
                y='UMAP_2',
                color='Gene',  # Color by gene expression
                color_continuous_scale=color_map,
                title=f"Gene Expression: {gene} ({len(plot_df):,} cells)",
                height=600,
                width=800
            )
            
            # Update marker properties
            fig.update_traces(
                marker=dict(
                    size=marker_size,
                    opacity=opacity,
                    line=dict(width=0)
                )
            )
            
            # Additional layout settings
            fig.update_layout(
                xaxis=dict(
                    showgrid=False,
                    zeroline=False
                ),
                yaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    scaleanchor="x",
                    scaleratio=1
                ),
                coloraxis_colorbar=dict(
                    title=f"log1p counts {gene}",
                    thickness=20,
                    len=0.7
                )
            )
            
            return fig
        # Create cell type plot
        def create_celltype_plot():
            # Check if second_col is categorical
            
            categorical = False
            if (plot_df[metadata_col].dtype == "object" or
                plot_df[metadata_col].dtype.name == "category" or
                plot_df[metadata_col].dtype.name == "string"):
                categorical = True
                # Convert to category dtype if not already
                if not pd.api.types.is_categorical_dtype(plot_df[metadata_col]):
                    plot_df[metadata_col] = plot_df[metadata_col].astype("category")
            
            # Determine marker size and opacity based on number of points
            num_points = len(plot_df)
            
            if num_points < 1000:
                marker_size = 5
                opacity = 0.8
            elif num_points < 10000:
                marker_size = 4
                opacity = 0.6
            elif num_points < 50000:
                marker_size = 3
                opacity = 0.4
            else:
                marker_size = 2
                opacity = 0.25
            
            # For very large datasets, use sampling
            max_points = 50000
            if num_points > max_points:
                if categorical:
                    # Stratified sampling for categorical data
                    sample_indices = []
                    categories = plot_df[metadata_col].cat.categories
                    
                    for cat in categories:
                        cat_indices = plot_df[plot_df[metadata_col] == cat].index
                        cat_ratio = len(cat_indices) / num_points
                        cat_sample_size = max(1, int(max_points * cat_ratio))
                        
                        if len(cat_indices) > cat_sample_size:
                            cat_sample = np.random.choice(cat_indices, size=cat_sample_size, replace=False)
                            sample_indices.extend(cat_sample)
                        else:
                            sample_indices.extend(cat_indices)
                    
                    sampled_df = plot_df.loc[sample_indices]
                else:
                    # Random sampling for numeric data
                    sampled_df = plot_df.sample(max_points)
            else:
                sampled_df = plot_df
            
            

            # Create the figure
            if categorical:
                fig = px.scatter(
                    sampled_df,
                    x='UMAP_1',
                    y='UMAP_2',
                    color=metadata_col,
                    title=f"{metadata_col} ({len(plot_df):,} cells)",
                    color_discrete_sequence=px.colors.qualitative.Light24,
                    height=600,
                    width=800
                )
            else:
                fig = px.scatter(
                    sampled_df,
                    x='UMAP_1',
                    y='UMAP_2',
                    color=metadata_col,
                    title=f"{metadata_col} ({len(plot_df):,} cells)",
                    color_continuous_scale='viridis',
                    height=600,
                    width=800
                )
            
            # Update marker properties for plot points
            for trace in fig.data:
                trace.marker.size = marker_size
                trace.marker.opacity = opacity
                trace.marker.line = dict(width=0)
            
            # Fix legend opacity for categorical plots only
            if categorical:
                # Create separate legend-only traces with full opacity
                legend_traces = []
                for trace in fig.data:
                    # Create a legend-only trace with full opacity
                    legend_trace = go.Scatter(
                        x=[None],  # No actual data points
                        y=[None],
                        mode="markers",
                        marker=dict(
                            color=trace.marker.color,
                            size=12,  # Fixed size for legend
                            opacity=1.0,  # Full opacity for legend
                            line=dict(width=0)
                        ),
                        name=trace.name,
                        showlegend=True,
                        hoverinfo='skip'
                    )
                    legend_traces.append(legend_trace)
                    
                    # Hide legend for original trace (but keep the data points visible)
                    trace.showlegend = False
                
                # Add the legend-only traces to the figure
                for legend_trace in legend_traces:
                    fig.add_trace(legend_trace)
            
            # Additional layout settings
            fig.update_layout(
                xaxis=dict(
                    showgrid=False,
                    zeroline=False
                ),
                yaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    scaleanchor="x",
                    scaleratio=1
                ),
                showlegend=categorical,
                legend=dict(
                    itemsizing="constant",
                    font=dict(size=10),
                    bgcolor="rgba(255,255,255,0.9)",  # Semi-transparent white background
                    bordercolor="rgba(0,0,0,0.2)",
                    borderwidth=1
                )
            )
            
            return fig
        
        # Create the figures
        gene_fig = create_gene_expression_plot()
        cell_type_fig = create_celltype_plot()


        config = {
        "toImageButtonOptions": {
            "format": download_as,
            "filename": f"{gene}_epitome_umap",
            "scale": 4
        }
    }
        
        return gene_fig, cell_type_fig, config
        
    except Exception as e:
        import traceback
        print(f"Error in create_gene_umap_plot: {str(e)}")
        print(traceback.format_exc())
        raise