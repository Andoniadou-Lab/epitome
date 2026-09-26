import traceback
from datetime import datetime

import streamlit as st

from modules.analytics import add_activity
from modules.other.page_layout import other_page_header
from modules.other.species_atlases import (
    get_other_species_info,
    list_other_species,
    load_other_species_atlas_cached,
    plot_other_species_atlas,
    preferred_gene,
)
from modules.ui.plot_settings import download_format_select, plot_settings_panel
from modules.ui.plot_summary import plot_summary_caption
from modules.utils import create_gene_selector

selected_version = other_page_header(
    "Species Atlases",
    "Gene expression and cell types in pituitary single-cell atlases across species. "
    "Each pair of plots is one species: expression on the left, cell types on the right.",
    "version_select_other_species",
)

available = list_other_species(selected_version)
if not available:
    st.warning(
        f"No species atlases found for version {selected_version} or any earlier version. "
        "Expected folders under other_data/species_atlases/."
    )
    st.stop()

sorted_names = sorted(available.keys())
default_name = next((n for n in sorted_names if n.startswith("Danio rerio")), sorted_names[0])
selected_display = st.selectbox(
    "Select a species",
    options=sorted_names,
    index=sorted_names.index(default_name),
    key="other_species_select",
)
scientific_name = available[selected_display]

with st.spinner(f"Loading {scientific_name}..."):
    adata = load_other_species_atlas_cached(scientific_name, selected_version)

dataset_info = get_other_species_info(adata)
st.write("Atlas information")
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.metric("Total cells", f"{dataset_info['Total Cells']:,}")
with col_b:
    st.metric("Total genes", f"{dataset_info['Total Genes']:,}")
with col_c:
    st.metric("Cell types", len(dataset_info["Cell Types"]))

if dataset_info["Cell Type Counts"]:
    count_cols = st.columns(min(6, len(dataset_info["Cell Type Counts"])))
    for i, (label, count) in enumerate(
        sorted(dataset_info["Cell Type Counts"].items(), key=lambda item: (-item[1], item[0]))
    ):
        with count_cols[i % len(count_cols)]:
            st.caption(label.replace("_", " "))
            st.markdown(f"**{count:,}**")

if not dataset_info["Has annotations"]:
    st.warning(
        f"{scientific_name} has a UMAP but no cell-type annotations in obs "
        "(no broad_cluster / cell_type column). The right-hand plot is empty until "
        "those labels are added."
    )

available_genes = [str(g) for g in adata.var_names.tolist()]
suggested = preferred_gene(available_genes)
if (
    suggested
    and st.session_state.get("selected_gene") not in available_genes
):
    st.session_state["selected_gene"] = suggested

with plot_settings_panel("Plot settings"):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        selected_gene = create_gene_selector(
            gene_list=available_genes,
            key_suffix="other_species_gene",
        )
    with col2:
        color_map = st.selectbox(
            "Color Map",
            ["reds", "plasma", "inferno", "magma", "blues", "viridis", "greens", "YlOrRd"],
            key="color_map_other_species",
        )
    with col3:
        sort_order = st.checkbox(
            "Sort plotted cells by expression",
            value=False,
            key="sort_other_species",
        )
    with col4:
        download_as = download_format_select(
            "download_other_species", formats=("png", "jpeg", "svg")
        )

try:
    gene_fig, cell_type_fig, config = plot_other_species_atlas(
        adata, selected_gene, sort_order, color_map, download_as=download_as
    )
    add_activity(
        value=[scientific_name, selected_gene],
        analysis="Other Species Atlas",
        user=st.session_state.session_id,
        time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(gene_fig, use_container_width=True, config=config)
    with col2:
        st.plotly_chart(cell_type_fig, use_container_width=True, config=config)
    plot_summary_caption(
        f"{dataset_info['Total Cells']:,} cells",
        f"{len(dataset_info['Cell Types'])} cell types",
        scientific_name,
        f"gene: {selected_gene}",
        version=selected_version,
        loader_key="other_species_atlas",
    )
    st.markdown(
        """
        **X-axis / Y-axis**: arbitrary UMAP coordinates from the species integration.
        Cell-type colours are shared across species wherever the same label appears.
        Empty or missing labels are omitted. For statistically robust comparisons use
        the other analysis pages once they are available.
        """
    )
except Exception as exc:
    st.error(f"Error creating plots: {exc}")
    with st.expander("Show full traceback"):
        st.code(traceback.format_exc(), language="python")
