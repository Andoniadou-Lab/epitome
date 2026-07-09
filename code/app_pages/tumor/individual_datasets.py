import traceback
from datetime import datetime

import streamlit as st

from modules.analytics import add_activity
from modules.pta.individual_sc import (
    get_pta_dataset_info,
    list_pta_datasets,
    load_pta_single_cell_dataset_cached,
    plot_pta_sc_dataset,
)
from modules.pta.page_layout import pta_page_header
from modules.pta.stats import pta_rna_stats_path
from modules.ui.plot_settings import download_format_select, plot_settings_panel
from modules.ui.plot_summary import plot_summary_caption
from modules.utils import create_cell_type_stats_display, create_gene_selector

selected_version = pta_page_header(
    "Individual Datasets",
    "Visualize gene expression and cell types in published human pituitary tumour scRNA-seq datasets.",
    "version_select_tumor_datasets",
)

available_datasets = list_pta_datasets(selected_version)
if not available_datasets:
    st.warning("No individual scRNA-seq datasets found for this version.")
    st.stop()

sorted_names = sorted(available_datasets.keys())
default_name = next(
    (n for n in sorted_names if "HRS1408776" in n),
    sorted_names[0],
)
selected_display_name = st.selectbox(
    "Select a dataset",
    options=sorted_names,
    index=sorted_names.index(default_name),
    key="dataset_select_tumor_rna",
)

selected_dataset = available_datasets[selected_display_name]

with st.spinner("Loading dataset (computing UMAP if not present)..."):
    adata = load_pta_single_cell_dataset_cached(selected_dataset, selected_version)
    available_genes = [str(g) for g in adata.var_names.tolist()]

if adata is None:
    st.stop()

dataset_info = get_pta_dataset_info(adata)
st.write("Dataset Information")
st.metric("Total Cells", dataset_info["Total Cells"])
st.metric("Total Genes", dataset_info["Total Genes"])
create_cell_type_stats_display(
    version=selected_version,
    sra_ids=[selected_dataset],
    display_title="Cell Counts in Current Selection",
    column_count=6,
    size="small",
    atac_rna="rna",
    rna_stats_path=pta_rna_stats_path(selected_version),
)

with plot_settings_panel("Plot settings"):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        selected_gene = create_gene_selector(
            gene_list=available_genes,
            key_suffix="gene_select_tumor_datasets",
        )
    with col2:
        color_map = st.selectbox(
            "Color Map",
            ["reds", "plasma", "inferno", "magma", "blues", "viridis", "greens", "YlOrRd"],
            key="color_map_tumor_datasets",
        )
    with col3:
        sort_order = st.checkbox("Sort plotted cells by expression", value=False, key="sort_tumor_datasets")
    with col4:
        download_as = download_format_select(
            "download_tumor_datasets", formats=("png", "jpeg", "svg")
        )

try:
    gene_fig, cell_type_fig, config = plot_pta_sc_dataset(
        adata, selected_gene, sort_order, color_map, download_as=download_as
    )
    add_activity(
        value=[selected_dataset, selected_gene],
        analysis="Tumor Individual Dataset RNA",
        user=st.session_state.session_id,
        time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(gene_fig, use_container_width=True, config=config)
    with col2:
        st.plotly_chart(cell_type_fig, use_container_width=True, config=config)
    plot_summary_caption(
        f"{dataset_info['Total Cells']} cells",
        f"{len(dataset_info['Cell Types'])} cell types",
        selected_dataset,
        f"gene: {selected_gene}",
    )
except Exception as exc:
    st.error(f"Error creating plots: {exc}")
    with st.expander("Show full traceback"):
        st.code(traceback.format_exc(), language="python")
