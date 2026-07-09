import traceback
from datetime import datetime

import streamlit as st

from modules.analytics import add_activity
from modules.proportion_plot import create_proportion_plot
from modules.pta.data_loader import load_pta_proportion_data, load_pta_scrna_curation
from modules.pta.page_layout import pta_page_header
from modules.pta.stats import pta_rna_stats_path
from modules.ui.plot_settings import download_format_select, plot_settings_panel
from modules.ui.plot_summary import plot_summary_caption
from modules.utils import create_cell_type_stats_display

selected_version = pta_page_header(
    "Cell Type Abundance",
    "Distribution of inferred cell-type proportions across human pituitary tumour scRNA-seq samples.",
    "version_select_tumor_proportion",
)

with st.spinner("Loading proportion data..."):
    proportion_matrix, proportion_rows, proportion_cols = load_pta_proportion_data(
        version=selected_version
    )
    meta_data = load_pta_scrna_curation(version=selected_version)

with plot_settings_panel("Plot settings"):
    filter_type = st.radio(
        "Filter data by:",
        ["No filter", "Sample", "Author"],
        key="filter_type_tumor_proportion",
    )
    all_samples = sorted(meta_data["Name"].unique())
    all_authors = sorted(meta_data["Author"].unique())
    selected_samples = all_samples
    selected_authors = all_authors

    if filter_type == "Sample":
        selected_samples = st.multiselect(
            "Select Samples", all_samples, default=[all_samples[0]], key="samples_tumor_proportion"
        )
    elif filter_type == "Author":
        selected_authors = st.multiselect(
            "Select Authors", all_authors, default=[all_authors[0]], key="authors_tumor_proportion"
        )

    col1, col2 = st.columns(2)
    with col1:
        group_by_sex = st.checkbox("Group by Sex", value=False, key="group_by_sex_tumor_proportion")
        show_mean = st.checkbox("Show mean proportions", value=False, key="show_mean_tumor_proportion")
    with col2:
        download_as = download_format_select(
            "download_tumor_proportion", formats=("png", "jpeg", "svg")
        )

filtered_meta = meta_data.copy()
if filter_type == "Author":
    filtered_meta = filtered_meta[filtered_meta["Author"].isin(selected_authors)]
if filter_type == "Sample":
    filtered_meta = filtered_meta[filtered_meta["Name"].isin(selected_samples)]

create_cell_type_stats_display(
    version=selected_version,
    sra_ids=filtered_meta["SRA_ID"].unique().tolist(),
    display_title="Cell Counts in Current Selection",
    column_count=6,
    size="small",
    atac_rna="rna",
    rna_stats_path=pta_rna_stats_path(selected_version),
)

add_activity(
    value="NA",
    analysis="Tumor Cell Type Proportions",
    user=st.session_state.session_id,
    time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
)

try:
    fig_male, fig_female, config, error_message = create_proportion_plot(
        matrix=proportion_matrix,
        rows=proportion_rows,
        columns=proportion_cols,
        meta_data=meta_data,
        selected_samples=selected_samples if filter_type == "Sample" else None,
        selected_authors=selected_authors if filter_type == "Author" else None,
        only_normal=False,
        only_whole=False,
        group_by_sex=group_by_sex,
        order_by_age=False,
        show_mean=show_mean,
        use_log_age=False,
        download_as=download_as,
    )

    if error_message:
        st.warning(error_message)
    elif group_by_sex:
        if fig_male is not None:
            st.plotly_chart(fig_male, use_container_width=True, config=config)
        if fig_female is not None:
            st.plotly_chart(fig_female, use_container_width=True, config=config)
    elif fig_male is not None:
        st.plotly_chart(fig_male, use_container_width=True, config=config)

    n_samples = len(filtered_meta)
    n_cell_types = len(proportion_cols)
    n_studies = (
        filtered_meta["Author"].nunique()
        if "Author" in filtered_meta.columns
        else None
    )
    summary = [
        f"{n_samples} scRNA-seq samples",
        f"{n_cell_types} cell types",
    ]
    if n_studies is not None:
        summary.append(f"{n_studies} studies")
    plot_summary_caption(*summary)

except Exception as exc:
    st.error(f"An error occurred: {exc}")
    with st.expander("Show full traceback"):
        st.code(traceback.format_exc(), language="python")
