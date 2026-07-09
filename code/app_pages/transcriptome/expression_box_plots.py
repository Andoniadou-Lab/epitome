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
    st.header("Expression Box Plots")
    st.markdown("Generate box plots showing the distribution of gene expression across cell types in the mouse pituitary. Each dot is a pseudobulk sample.")
with col2:
    selected_version = st.selectbox(
        "Version",
        options=AVAILABLE_VERSIONS,
        key="version_select_tab1",
        label_visibility="collapsed",
    )

with st.spinner("Loading data..."):
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
    ) = create_filter_ui(meta_data, sex_analysis=True)

    gene_list = sorted(genes[0].unique())
    col1, col2 = st.columns(2)
    with col1:
        selected_gene = create_gene_selector(
            gene_list=gene_list,
            key_suffix="gene_select_tab1",
        )
    with col2:
        if filter_type == "Reproduce sex-specific analysis":
            additional_groups = ["Comp_sex"]
        else:
            additional_groups = ["None", "Modality", "Comp_sex"]
        additional_group = st.selectbox(
            "Additional Grouping Variable",
            additional_groups,
            key="additional_group_select",
            width=250,
        )

    col1, col2 = st.columns(2)
    with col1:
        download_as = download_format_select(
            "download_as_expr_boxplot", formats=("png", "jpeg", "svg")
        )
    with col2:
        connect_dots = st.checkbox(
            "Connect Dots",
            value=False,
            help="Connect dots with the same SRA_ID (e.g., to visualise if outlier samples across cell types are from the same study)",
            key="connect_dots_tab1",
        )

# Apply filters to get filtered data
filtered_meta = meta_data.copy()

filtered_meta, filtered_matrix = filter_data(
    meta_data=filtered_meta,
    age_range=age_range,
    selected_samples=selected_samples,
    selected_authors=selected_authors,
    matrix=matrix,
    only_normal=only_normal,
    modality=modality,
)

#reorder meta and matrix based on alphabetical order of cell types
filtered_meta = filtered_meta.reset_index(drop=True)
sort_order = np.argsort(filtered_meta["new_cell_type"].values)
filtered_matrix = filtered_matrix[:, sort_order]
filtered_meta = filtered_meta.iloc[sort_order].reset_index(drop=True)

filtered_sra_ids = filtered_meta["SRA_ID"].unique().tolist()

create_cell_type_stats_display(
    version=selected_version,
    sra_ids=filtered_sra_ids,
    display_title="Cell Counts in Current Selection",
    column_count=6,
    size="small",
    atac_rna="rna",
)

st.markdown("<br>", unsafe_allow_html=True)

all_cell_types = sorted(filtered_meta["new_cell_type"].unique())
cell_type_selection = st.radio(
    "Cell Type Selection",
    ["All Cell Types", "Select Specific Cell Types"],
    key="cell_type_selection_expression",
    horizontal=True,
)

selected_cell_types = None
if cell_type_selection == "Select Specific Cell Types":
    selected_cell_types = st.multiselect(
        "Select Cell Types to Display",
        options=all_cell_types,
        default=(
            [all_cell_types[0]]
            if len(all_cell_types) > 0
            else []
        ),
        key="selected_cell_types_expression",
    )

add_activity(
    value=selected_gene,
    analysis="Expression Box Plots",
    user=st.session_state.session_id,
    time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
)



# Create plot with filtered data and cell type selection
fig, config = create_expression_plot(
    matrix=filtered_matrix,
    genes=genes,
    meta_data=filtered_meta,
    gene_name=selected_gene,
    additional_group=(
        None if additional_group == "None" else additional_group
    ),
    connect_dots=connect_dots,
    selected_cell_types=selected_cell_types,
    download_as=download_as
)
st.plotly_chart(fig, use_container_width=True, config=config)
n_studies = (
    filtered_meta["Author"].nunique()
    if "Author" in filtered_meta.columns
    else None
)
plot_summary_caption(
    selected_gene,
    f"{len(filtered_meta)} pseudobulk samples",
    f"{filtered_meta['new_cell_type'].nunique()} cell types",
    f"{n_studies} studies" if n_studies is not None else None,
)
#gc.collect()

with st.container():
    st.markdown(
        f"""
        This box plot shows the distribution of gene expression across different cell types in the mouse pituitary.

        **X-axis**: Cell types present in the selected samples
        **Y-axis**: Log10-transformed counts per million* values of the selected gene


        **Box:** centre line = **median**; top and bottom edges = **75th** and **25th** percentiles

        **Whiskers:** extend to the **5th** and **95th** percentiles

        **Points:** each dot is one sample; hover a dot for sample-level detail

        The box plot also enables:
        - Optional grouping by additional variables (e.g., sex, data type)
        - Optional connecting lines between samples from the same source
        - Cell type filtering to focus on specific cell populations



        *These values are first normalised using TMM within the Limma-voom workflow.
    """
    )

# Add download button for tab 1
gene_idx = genes[genes[0] == selected_gene].index[0]
#expression_values = (
#    filtered_matrix[gene_idx, :].A1
#    if hasattr(filtered_matrix[gene_idx, :], "A1")
#    else filtered_matrix[gene_idx, :]
#)
expression_values = to_array(filtered_matrix[gene_idx, :])
download_df = filtered_meta.copy()
download_df["Expression"] = expression_values

st.download_button(
    on_click="ignore",
            label="Download Plotting Data",
            data=lambda: download_df.to_csv(index=False),
            file_name=f"{selected_gene}_expression_data.csv",
            mime="text/csv",
            key="download_button_tab1",
            help="Download the current filtered dataset used for plotting",
        )

#add separator line
st.markdown("---")
# Add marker browser section
col1, col2 = st.columns([5, 1])
with col1:
    st.subheader("Marker Gene Browser")
with col2:
    selected_version_marker_rna = st.selectbox(
        "Version",
        options=AVAILABLE_VERSIONS,
        key="version_select_marker_browser",
        label_visibility="collapsed",
    )

filtered_data = display_marker_table(
    selected_version_marker_rna, load_cached_marker_data, "expression"
)

# Add sexually dimorphic genes section
st.markdown("---")
col1, col2 = st.columns([5, 1])
with col1:
    st.subheader("Sex-biased Genes")
with col2:
    selected_version_sex_dim = st.selectbox(
        'Version',
        options=AVAILABLE_VERSIONS,
        key='version_select_sex_dim',
        label_visibility="collapsed"
    )

filtered_sex_dim_data = display_sex_dimorphism_table(sex_dim_data=load_cached_sex_dim_data(version=selected_version_sex_dim), key_prefix="sex_dimorphism")
