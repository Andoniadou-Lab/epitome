import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl
import streamlit as st

from config import Config
from modules.analytics import add_activity
from modules.cached_loaders import (
    AVAILABLE_VERSIONS,
    load_cached_curation_data,
    load_cached_data,
    load_cached_dotplot_data,
    load_cached_gene_curation,
    load_cached_isoform_data,
    load_cached_ligand_receptor_data,
    load_cached_marker_data,
    load_cached_proportion_data,
    load_cached_sex_dim_data,
)
from modules.age_correlation import create_age_correlation_plot
from modules.display_tables import (
    display_aging_genes_table,
    display_ligand_receptor_table,
    display_marker_table,
    display_sex_dimorphism_table,
)
from modules.dotplot import (
    create_dotplot,
    create_ligand_receptor_plot,
    filter_dotplot_data,
)
from modules.expression import create_expression_plot
from modules.gene_gene_corr import (
    create_gene_correlation_plot,
    get_available_genes,
    load_gene_data,
)
from modules.gene_umap_vis import create_gene_umap_plot
from modules.isoforms import create_isoform_plot, filter_isoform_data
from modules.proportion_plot import create_proportion_plot
from modules.ui.plot_settings import plot_settings_panel
from modules.ui.plot_summary import plot_summary_caption
from modules.utils import (
    create_cell_type_stats_display,
    create_filter_ui,
    create_gene_selector,
    filter_data,
    to_array,
)

BASE_PATH = Config.BASE_PATH

col1, col2 = st.columns([5, 1])
with col1:
    st.header("Gene-Gene Relationships")
    st.markdown("Explore correlations between the expression levels of two genes across cell types in the mouse pituitary. Each point represents a cell.")
with col2:
    selected_version = st.selectbox(
        "Version",
        options=AVAILABLE_VERSIONS,
        key="version_select_gene_corr",
        label_visibility="collapsed",
    )

try:
    # Get available genes
    base_path = f"{BASE_PATH}/data/large_umap/{selected_version}/"
    available_genes = get_available_genes(base_path)

    # Load metadata
    obs_data = pd.read_parquet(f"{base_path}/obs.parquet")

    valid_sra_ids = obs_data["SRA_ID"].unique().tolist()
    curation = load_cached_curation_data(
        version=selected_version
    )
    filtered_meta = curation[
        curation["SRA_ID"].isin(valid_sra_ids)
    ].copy()

    with plot_settings_panel("Plot settings"):
        col1, col2 = st.columns(2)
        with col1:
            gene1 = st.selectbox(
                "Select First Gene",
                options=available_genes,
                index=(
                    available_genes.index("Sox2")
                    if "Sox2" in available_genes
                    else 0
                ),
                key="gene1_select",
                width=250,
            )
        with col2:
            gene2 = st.selectbox(
                "Select Second Gene",
                options=available_genes,
                index=(
                    available_genes.index("Sox9")
                    if "Sox9" in available_genes
                    else 0
                ),
                key="gene2_select",
                width=250,
            )

        (
            filter_type,
            selected_samples,
            selected_authors,
            age_range,
            only_normal,
            modality,
        ) = create_filter_ui(filtered_meta, key_suffix="gene_corr")

        all_cell_types = sorted(obs_data["new_cell_type"].unique())
        selected_cell_types = st.multiselect(
            "Select Cell Types",
            options=all_cell_types,
            default=(
                ["Stem_cells"]
                if "Stem_cells" in all_cell_types
                else None
            ),
            key="selected_cell_types_gene_gene",
        )

        color_by_celltype = st.checkbox(
            "Color by Cell Type",
            value=True,
            help="Color points by cell type or show all points in a single color",
        )

    if filter_type == "Sample":
        filtered_meta = filtered_meta[
            filtered_meta["Name"].isin(selected_samples)
        ]
    elif filter_type == "Author":
        filtered_meta = filtered_meta[
            filtered_meta["Author"].isin(selected_authors)
        ]
    elif filter_type == "Age" and age_range:
        age_mask = (
            filtered_meta["Age_numeric"].notna()
            & (filtered_meta["Age_numeric"] >= age_range[0])
            & (filtered_meta["Age_numeric"] <= age_range[1])
        )
        filtered_meta = filtered_meta[age_mask]

    elif filter_type =="Modality" and modality:
        filtered_meta = filtered_meta[
            filtered_meta["Modality"].isin(modality)
        ]

    if only_normal:
        filtered_meta = filtered_meta[
            filtered_meta["Normal"] == 1
        ]

    filtered_sra_ids = filtered_meta["SRA_ID"].unique().tolist()

    selected_cell_types_show = [
        cell_type.split("_")[0] for cell_type in selected_cell_types
    ]
    create_cell_type_stats_display(
        version=selected_version,
        sra_ids=filtered_sra_ids,
        display_title="Cell Counts in Current Selection",
        column_count=6,
        size="small",
        cell_types=(
            "all"
            if selected_cell_types_show is None
            else selected_cell_types_show
        ),
        atac_rna="rna",
    )

    add_activity(value=[gene1, gene2], analysis="Gene-Gene Correlation",
            user=st.session_state.session_id,time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Create plot
    fig, config, stats, error = create_gene_correlation_plot(
        gene1,
        gene2,
        base_path,
        obs_data,
        selected_samples=filtered_sra_ids,
        color_by_celltype=color_by_celltype,
        selected_cell_types=selected_cell_types,
    )
    #gc.collect()

    if error:
        st.error(f"Error creating plot: {error}")
    elif fig:
        st.plotly_chart(
            fig, use_container_width=True, config=config
        )
        plot_summary_caption(
            f"{gene1} vs {gene2}",
            f"{len(obs_data)} pseudobulk samples",
            f"r = {stats['correlation']:.3f}",
        )

        # Display overall statistics
        st.subheader("Overall Spearman Correlation (r)")
        st.metric("Correlation", f"{stats['correlation']:.3f}")
        # Pct coexpression
        st.subheader("Percentage of Coexpression")
        st.metric(
            "Percentage", f"{stats['pct_coexpression']:.2f}%"
        )

        # Create download data
        gene1_data = load_gene_data(gene1, base_path)
        gene2_data = load_gene_data(gene2, base_path)
        download_df = pd.DataFrame(
            {
                gene1: gene1_data[gene1_data.columns[0]],
                gene2: gene2_data[gene2_data.columns[0]],
                "Cell_Type": obs_data["new_cell_type"],
                "SRA_ID": obs_data["SRA_ID"],
            }
        )

        if valid_sra_ids is not None:
            download_df = download_df[
                download_df["SRA_ID"].isin(valid_sra_ids)
            ]

        if selected_cell_types is not None:
            download_df = download_df[
                download_df["Cell_Type"].isin(
                    selected_cell_types
                )
            ]

        st.download_button(
            on_click="ignore",
            label="Download Correlation Data",
            data=download_df.to_csv(index=False),
            file_name=f"correlation_{gene1}_{gene2}.csv",
            mime="text/csv",
            key="download_button_gene_corr",
            help="Download the current correlation dataset",
        )

        with st.container():
            st.markdown(
                """
                This plot shows the correlation between single-cell expression levels of two selected genes.

                **X-axis**: Expression level of first selected gene
                **Y-axis**: Expression level of second selected gene

                The visualization includes:
                - Scatter plot of gene expression values
                - Optional coloring by cell type
                - Correlation line (dashed red)
                - Hover information showing sample details
                - Dynamic point opacity based on total number of points

                Note: This feature was requested by a user. The analysis may not be very informative, and any gene-gene relationship should not be assumed to be a result of regulation or the lack thereof. Gene counts here are simply log1p(counts_per_10k) values and batch effects are not accounted for in any way.
            """
            )

except Exception as e:
    st.error(f"Error in gene correlation analysis: {str(e)}")
    if st.checkbox("Show detailed error"):
        st.exception(e)
        traceback.print_exc() 
        # Capture the full traceback
        tb = traceback.format_exc()

        # Display it in a collapsible section
        with st.expander("Show full error traceback"):
            st.code(tb, language='python')
