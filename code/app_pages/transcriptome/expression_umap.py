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
from modules.ui.plot_settings import download_format_select, plot_settings_panel
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
    st.header("UMAP visualisation")
    st.markdown("Generate UMAP plots showing gene expression across the entire atlas. Each dot is a cell.")
with col2:
    selected_version = st.selectbox(
        "Version",
        options=AVAILABLE_VERSIONS,
        key="version_select_umap",
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
        (
            filter_type,
            selected_samples,
            selected_authors,
            age_range,
            only_normal,
            modality,
        ) = create_filter_ui(filtered_meta, key_suffix="umap")

        all_cell_types = sorted(obs_data["new_cell_type"].unique())
        selected_cell_types = st.multiselect(
            "Select Cell Types",
            options=all_cell_types,
            default=all_cell_types,
            key="selected_cell_types_umap",
        )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            selected_gene = create_gene_selector(
                gene_list=available_genes,
                key_suffix="umap_gene_select",
            )
        with col2:
            color_map = st.selectbox(
                "Color Map",
                [
                    "blues",
                    "reds",
                    "plasma",
                    "inferno",
                    "magma",
                    "viridis",
                    "greens",
                    "YlOrRd",
                ],
                key="color_map_select_datasets1",
            )
        with col3:
            sort_order = st.checkbox(
                "Sort plotted cells by expression", value=False, key="sort3"
            )
        with col4:
            metadata_cols = [
                "Cell type",
                "Sex",
                "10X version",
                "Assay modality",
                "pct_counts_mt",
                "pct_counts_ribo",
                "pct_counts_malat",
                "Normal",
            ]
            metadata_col = st.selectbox(
                "Color second plot by",
                options=metadata_cols,
                index=0,
                key="color_by_select",
            )

        download_as = download_format_select(
            "download_as_umap", formats=("png", "jpeg", "svg")
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
        #convert to float
        #change , to .
        filtered_meta["Age_numeric"] = pd.to_numeric(
            filtered_meta["Age_numeric"].str.replace(",", "."),
            errors="coerce",
        )
        filtered_meta["Age_numeric"] = filtered_meta["Age_numeric"].astype(float)
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

    selected_cell_types_display = [ct.split("_")[0] for ct in selected_cell_types]

    create_cell_type_stats_display(
        version=selected_version,
        sra_ids=filtered_sra_ids,
        display_title="Cell Counts in Current Selection",
        column_count=6,
        size="small",
        cell_types=(
            selected_cell_types_display 
        ),
        atac_rna="rna",
    )

    if metadata_col == "Sex":
        metadata_col = "Comp_sex"

    if metadata_col == "Cell type":
        metadata_col = "new_cell_type"

    if metadata_col in ["Assay modality"]:
        metadata_col = "Modality"

    add_activity(
        value=selected_gene,
        analysis="UMAP Plot",
        user=st.session_state.session_id,
        time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    gene_fig, cell_type_fig, config = create_gene_umap_plot(
        selected_gene,
        base_path,
        obs_data,
        selected_samples=filtered_sra_ids,
        selected_cell_types=selected_cell_types,
        color_map=color_map,
        sort_order=sort_order,
        metadata_col=metadata_col,
        download_as=download_as
    )
    #gc.collect()

    # Display plots side by side
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(gene_fig, use_container_width=True, config=config)
    with col2:
        st.plotly_chart(cell_type_fig, use_container_width=True, config=config)
    plot_summary_caption(
        f"{len(obs_data):,} cells",
        f"{obs_data['new_cell_type'].nunique()} cell types",
        f"gene: {selected_gene}",
    )
    #gc.collect()
    # Add explanation in a container
    with st.container():
        st.markdown(
            """
            This plot shows the expression of a selected gene across different cell types in the mouse pituitary.
            Datasets were integrated using scVI, and the latent space was used for identifying nearest neighbours and generating a UMAP plot.

            **X-axis**: Arbitrary UMAP axis 1
            **Y-axis**: Arbitrary UMAP axis 2

            The visualization includes:
            - Scatter plot of gene expression values
            - Hover information showing sample details
            - Dynamic point opacity based on total number of points

            Note: For any statistically robust visualisation use the box plots or dot plots. The UMAP coordinates are arbitrary and do not necessarily represent or relate to anything biological. For concerns on the use of UMAPs, read: doi.org/10.1371/journal.pcbi.1011288
        """
        )


except Exception as e:
    st.error(f"Error creating plots: {str(e)}")
    traceback.print_exc() 
    # Capture the full traceback
    tb = traceback.format_exc()

    # Display it in a collapsible section
    with st.expander("Show full error traceback"):
        st.code(tb, language='python')
