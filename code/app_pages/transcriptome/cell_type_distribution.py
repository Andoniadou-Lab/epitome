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
    st.header("Cell Type Distribution")
    st.markdown("Visualize the distribution of cell type abundance across samples in the mouse pituitary.")
with col2:
    selected_version = st.selectbox(
        "Version",
        options=AVAILABLE_VERSIONS,
        key="version_select_proportion",
        label_visibility="collapsed",
    )

with st.spinner("Loading proportion data..."):
    proportion_matrix, proportion_rows, proportion_cols = (
        load_cached_proportion_data(version=selected_version)
    )

with plot_settings_panel("Plot settings"):
    filter_type = st.radio(
        "Filter data by:",
        ["No filter", "Sample", "Author"],
        key="filter_type_proportion",
    )
    meta_data = load_cached_curation_data(version=selected_version)
    selected_samples = all_samples = sorted(
        meta_data["Name"].unique()
    )
    selected_authors = all_authors = sorted(
        meta_data["Author"].unique()
    )

    if filter_type == "Sample":
        selected_samples = st.multiselect(
            "Select Samples",
            all_samples,
            default=[all_samples[0]],
            help="Choose which samples to include in the analysis",
            key="samples_multiselect_proportion",
        )
    elif filter_type == "Author":
        selected_authors = st.multiselect(
            "Select Authors",
            all_authors,
            default=[all_authors[0]],
            help="Choose which authors' data to include",
            key="authors_multiselect_proportion",
        )

    col1, col2 = st.columns(2)
    with col1:
        only_normal = st.checkbox(
            "Show only control samples",
            value=False,
            help="Samples that are wild-type, untreated etc. (In curation, Normal == 1)",
            key="only_normal_proportion",
        )
        only_whole = st.checkbox(
            "Show only whole-pituitary samples",
            value=False,
            help="Samples not sorted to enrich for a given sub-population (In curation, Sorted == 0)",
            key="only_whole_proportion",
        )
        show_mean = st.checkbox(
            "Show mean proportions",
            value=False,
            help="Show average cell type proportions across selected samples",
            key="show_mean_proportion",
        )
    with col2:
        group_by_sex = st.checkbox(
            "Group by Sex",
            value=False,
            help="Create separate plots for male and female samples",
            key="group_by_sex_proportion",
        )
        order_by_age = st.checkbox(
            "Order by Age",
            value=False,
            help="Order samples by age",
            key="order_by_age_proportion",
        )
        download_as = download_format_select(
            "download_as_atac_proportion", formats=("png", "jpeg", "svg")
        )

    use_log_age = False

filtered_meta = meta_data.copy()
if only_normal:
    filtered_meta = filtered_meta[filtered_meta["Normal"] == 1]
if only_whole:
    filtered_meta = filtered_meta[filtered_meta["Sorted"] == 0]
if filter_type == "Author":
    filtered_meta = filtered_meta[
        filtered_meta["Author"].isin(selected_authors)
    ]
if filter_type == "Sample":
    filtered_meta = filtered_meta[
        filtered_meta["Name"].isin(selected_samples)
    ]

create_cell_type_stats_display(
    version=selected_version,
    # make it selected samples if empty then use  all samples
    sra_ids=filtered_meta["SRA_ID"].unique().tolist(),
    display_title="Cell Counts in Current Selection",
    column_count=6,
    size="small",
    atac_rna="rna",
)

add_activity(value="NA", analysis="Cell Type Proportions",
            user=st.session_state.session_id,time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Create plot
fig_male, fig_female, config, error_message = (
    create_proportion_plot(
        matrix=proportion_matrix,
        rows=proportion_rows,
        columns=proportion_cols,
        meta_data=meta_data,
        selected_samples=(
            selected_samples
            if filter_type == "Sample"
            else None
        ),
        selected_authors=(
            selected_authors
            if filter_type == "Author"
            else None
        ),
        only_normal=only_normal,
        only_whole=only_whole,
        group_by_sex=group_by_sex,
        order_by_age=order_by_age,
        show_mean=show_mean,
        use_log_age=use_log_age,
        download_as=download_as
    )
)

if error_message:
    st.warning(error_message)
elif group_by_sex:
    if fig_male is not None:
        st.plotly_chart(
            fig_male, use_container_width=True, config=config
        )
    if fig_female is not None:
        st.plotly_chart(
            fig_female, use_container_width=True, config=config
        )
else:
    if (
        fig_male is not None
    ):  # Using fig_male as the main figure
        st.plotly_chart(
            fig_male, use_container_width=True, config=config
        )

if not error_message:
    n_samples = len(filtered_meta)
    n_cell_types = len(proportion_cols)
    n_studies = (
        filtered_meta["Author"].nunique()
        if "Author" in filtered_meta.columns
        else None
    )
    plot_summary_caption(
        f"{n_samples} scRNA-seq samples",
        f"{n_cell_types} cell types",
        f"{n_studies} studies" if n_studies is not None else None,
    )

#gc.collect()

with st.container():
    st.markdown(
        """
        This plot shows the relative proportions of different cell types across samples.

        **X-axis**: Samples (can be ordered by age)
        **Y-axis**: Percentage of each cell type

        Features:
        - Stacked bar chart showing relative proportions
        - Optional grouping by sex
        - Option to show mean proportions
        - Age-based ordering and log-transformation
        - Smooth visualization option for age-based trends
    """
    )

# Add download button for data
if hasattr(proportion_matrix, "toarray"):
    prop_data = proportion_matrix.toarray()
else:
    prop_data = proportion_matrix

prop_df = pd.DataFrame(
    prop_data,
    index=proportion_rows.iloc[:, 0],
    columns=proportion_cols.iloc[:, 0],
)

st.download_button(
    on_click="ignore",
    label="Download Proportion Data",
    data=prop_df.to_csv(index=True),
    file_name="cell_type_proportions.csv",
    mime="text/csv",
    help="Download the cell type proportion data",
    key="download_button_proportion",
)
