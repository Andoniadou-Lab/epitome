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
    st.header("Age Correlation Analysis")
    st.markdown("Analyze how gene expression changes in mouse pituitary cell types across different ages. Each dot is a pseudobulk sample.")
with col2:
    selected_version = st.selectbox(
        "Version",
        options=AVAILABLE_VERSIONS,
        key="version_select_tab2",
        label_visibility="collapsed",
    )

if "matrix" not in st.session_state:
    with st.spinner("Loading expression data..."):
        matrix, genes, meta_data = load_cached_data(
            version=selected_version
        )

with plot_settings_panel("Plot settings"):
    (
        filter_type,
        selected_samples,
        selected_authors,
        age_range,
        only_normal,
        modality,
    ) = create_filter_ui(meta_data, age_analysis=True, key_suffix="age_corr")

    gene_list = sorted(genes[0].unique())
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        selected_gene = create_gene_selector(
            gene_list=gene_list,
            key_suffix="gene_select_tab2",
        )
    with col2:
        data_type_options = [
            "All Data Types",
            "Single Cell Only (sc)",
            "Single Nucleus Only (sn)",
            "Multi-modal RNA Only",
        ]
        if filter_type == "Reproduce age-dependent analysis":
            data_type_options = ["Single Cell Only (sc)"]

        selected_data_type = st.selectbox(
            "Data Type Filter:",
            options=data_type_options,
            index=0,
            help="Filter to show only specific data types",
            width=250,
        )
    with col3:
        color_by = st.selectbox(
            "Color points by:",
            ["None", "Comp_sex", "Modality"],
            key="color_select_age_correlation",
            help="Choose a variable to color the points by",
            width=250,
        )
    with col4:
        download_as = download_format_select(
            "download_as_age_correlation", formats=("png", "jpeg", "svg")
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        use_log_age = st.checkbox(
            "Use log10 scale for age", value=True
        )
    with col2:
        show_trendline = st.checkbox(
            "Show trendline",
            value=True,
            help="Display linear regression trendline",
        )
    with col3:
        remove_zeros = st.checkbox(
            "Remove zero values",
            value=False,
            help="Remove cells with expression values < 0.01. Some highly contaminating (ambient RNA) transcripts might have been overcorrected in some datasets.",
        )

    data_type_filter = None
    if selected_data_type == "Single Cell Only (sc)":
        data_type_filter = "sc"
    elif selected_data_type == "Single Nucleus Only (sn)":
        data_type_filter = "sn"
    elif selected_data_type == "Multi-modal RNA Only":
        data_type_filter = "multi_rna"

    show_days = False
    if use_log_age:
        show_days = st.checkbox(
            "Show age in days instead of log10 values",
            value=True,
            key="show_days_checkbox",
        )

if only_normal:
    total_sra_ids = len(set(meta_data["SRA_ID"].unique()))
    normal_sra_ids = len(
        set(
            meta_data[meta_data["Normal"] == 1][
                "SRA_ID"
            ].unique()
        )
    )
    st.info(
        f"Samples remaining after wild-type filter: {normal_sra_ids} ({total_sra_ids} without filter)"
    )

# Apply filters to get filtered data
filtered_meta = meta_data.copy()
filtered_matrix = matrix

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

elif filter_type == "Reproduce age-dependent analysis":
    age_mask = (
        filtered_meta["Age_numeric"].notna()
        & (filtered_meta["Age_numeric"] >= 0)
        & (filtered_meta["Age_numeric"] <= 2000)
    )
    filtered_meta = filtered_meta[age_mask]

if only_normal:
    filtered_meta = filtered_meta[filtered_meta["Normal"] == 1]

filtered_meta["new_cell_type"] = [
    ct.split("_")[0] for ct in filtered_meta["new_cell_type"]
]
cell_types = sorted(filtered_meta["new_cell_type"].unique())
selected_cell_type = st.selectbox(
    "Select Cell Type",
    cell_types,
    index=(
        cell_types.index("Stem")
        if "Stem" in cell_types
        else 0
    ),
    width=250,
    key="cell_type_select_age_correlation",
)

if data_type_filter == "sc":
    filtered_meta = filtered_meta[
        filtered_meta["Modality"].isin(["sc"])
    ]
elif data_type_filter == "sn":
    filtered_meta = filtered_meta[
        filtered_meta["Modality"].isin(["sn"])
    ]
elif data_type_filter == "multi_rna":
    filtered_meta = filtered_meta[
        filtered_meta["Modality"].isin(["multi_rna"])
    ]

filtered_sra_ids = filtered_meta["SRA_ID"].unique().tolist()

# Update matrix to match filtered metadata
filtered_matrix = matrix[:, filtered_meta.index].copy()

create_cell_type_stats_display(
    version=selected_version,
    sra_ids=filtered_sra_ids,
    display_title="Cell Counts in Current Selection",
    cell_types=[selected_cell_type],
    size="small",
    column_count=1,
    atac_rna="rna",
)

add_activity(
    value=selected_gene,
    analysis="Age Correlation",
    user=st.session_state.session_id,
    time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
)

# Create the plot with log and color options
fig, config, r_squared, p_value, aging_genes_df = (
    create_age_correlation_plot(
        matrix=filtered_matrix,
        genes=genes,
        meta_data=filtered_meta,
        gene_name=selected_gene,
        cell_type=selected_cell_type,
        use_log_age=use_log_age,
        show_days=show_days,
        remove_zeros=remove_zeros,
        color_by=None if color_by == "None" else color_by,
        show_trendline=show_trendline,
        data_type_filter=data_type_filter,
        version = selected_version,
        download_as=download_as
    )
)

st.plotly_chart(fig, use_container_width=True, config=config)
cell_mask = filtered_meta["new_cell_type"] == selected_cell_type
plot_summary_caption(
    selected_gene,
    selected_cell_type,
    f"{cell_mask.sum()} pseudobulk samples",
)
#gc.collect()

with st.container():
    st.markdown(
        """
        This plot visualizes the correlation between gene expression and age for a specific cell type.

        **X-axis**: Age in days (can be log10-transformed)
        **Y-axis**: Log10-transformed counts per million* values of the selected gene

        The plot includes:
        - Scatter points representing individual samples
        - Trend line showing the linear correlation
        - Optional coloring by sex or data type
        - R-squared value and p-value statistics - Note these do not exactly match the statistical results from Limma-voom used in the publication

        *These values are first normalised using TMM within the Limma-voom workflow.
    """
    )

# Correlation Statistics
st.subheader("Correlation Statistics (based on selected/displayed data)")
col1, col2 = st.columns(2)
with col1:
    st.metric("R-squared", f"{r_squared:.3f}")
with col2:
    st.metric("P-value", f"{p_value:.3e}")

# Create download data for tab 2
gene_idx = genes[genes[0] == selected_gene].index[0]
#expression_values = (
#    matrix[gene_idx, :].A1
#    if hasattr(matrix[gene_idx, :], "A1")
#    else matrix[gene_idx, :]
#)
expression_values = to_array(matrix[gene_idx, :])

cell_type_mask = (
    meta_data["new_cell_type"] == selected_cell_type
)

download_df = meta_data[cell_type_mask].copy()
download_df["Expression"] = expression_values[cell_type_mask]
download_df["R_squared"] = r_squared
download_df["P_value"] = p_value

st.download_button(
    on_click="ignore",
    label="Download Age Correlation Data",
    data=download_df.to_csv(index=False),
    file_name=f"{selected_gene}_{selected_cell_type}_age_correlation.csv",
    mime="text/csv",
    key="download_button_tab2",
    help="Download the current age correlation dataset",
)

# Add Aging Genes Table
st.subheader("Age-dependent Genes")
filtered_df = display_aging_genes_table(aging_genes_df, "aging")
