import traceback
from datetime import datetime

import streamlit as st

from modules.analytics import add_activity
from modules.pta.large_umap import (
    COLOR_BY_OPTIONS,
    cell_type_labels,
    create_pta_umap_plots,
    large_umap_dir,
    list_large_umap_genes,
    load_gene_counts,
    load_large_umap_obs,
    visible_cell_mask,
)
from modules.pta.page_layout import pta_page_header
from modules.ui.plot_settings import download_format_select, plot_settings_panel
from modules.ui.plot_summary import plot_summary_caption
from modules.utils import create_gene_selector

selected_version = pta_page_header(
    "UMAP visualisation",
    "Gene expression across the entire human pituitary tumour atlas. Each dot is a cell.",
    "version_select_tumor_umap",
)

try:
    directory, _resolved = large_umap_dir(selected_version)
    if directory is None:
        st.warning(
            f"UMAP data is not available for version {selected_version} "
            "or any earlier version"
        )
        st.stop()
    obs = load_large_umap_obs(str(directory), selected_version)
    available_genes = list_large_umap_genes(str(directory))

    with plot_settings_panel("Plot settings"):
        col1, col2, col3 = st.columns(3)
        with col1:
            all_studies = sorted(obs["Author"].unique())
            studies = st.multiselect(
                f"Studies ({len(all_studies)})",
                options=all_studies,
                default=all_studies,
                key="tumor_umap_studies",
            )
        with col2:
            all_tumour_types = sorted(obs["Tumor_pta"].astype(str).unique())
            tumour_types = st.multiselect(
                "Tumour type",
                options=all_tumour_types,
                default=all_tumour_types,
                key="tumor_umap_tumour_types",
            )
        with col3:
            status = st.radio(
                "Samples",
                ["All", "Tumour only", "Healthy only"],
                horizontal=True,
                key="tumor_umap_status",
            )
            merge_immune = st.checkbox(
                "Merge immune cell types",
                value=False,
                key="tumor_umap_merge_immune",
                help=(
                    "Collapse immune populations (B cells, plasma cells, T-cell subsets, "
                    "NK cells, ILCs, monocytes, mast cells, dendritic cells, neutrophils, "
                    "and pDC) into Immune_cells. Macrophages stay separate."
                ),
            )

        cell_types = cell_type_labels(obs, merge_immune=merge_immune)
        visible = visible_cell_mask(obs, cell_types, merge_immune=merge_immune)
        all_cell_types = sorted(cell_types[visible].unique())
        selected_cell_types = st.multiselect(
            "Select Cell Types",
            options=all_cell_types,
            default=all_cell_types,
            key=f"tumor_umap_cell_types_{int(merge_immune)}",
        )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            selected_gene = create_gene_selector(
                gene_list=available_genes,
                key_suffix="tumor_umap_gene_select",
            )
        with col2:
            color_map = st.selectbox(
                "Color Map",
                ["blues", "reds", "plasma", "inferno", "magma", "viridis", "greens", "YlOrRd"],
                key="tumor_umap_color_map",
            )
        with col3:
            sort_order = st.checkbox(
                "Sort plotted cells by expression", value=False, key="tumor_umap_sort"
            )
        with col4:
            color_by_label = st.selectbox(
                "Color second plot by",
                options=list(COLOR_BY_OPTIONS),
                index=0,
                key="tumor_umap_color_by",
            )

        download_as = download_format_select(
            "tumor_umap_download", formats=("png", "jpeg", "svg")
        )

    if not studies or not tumour_types or not selected_cell_types:
        st.warning("Select at least one study, tumour type, and cell type.")
        st.stop()

    keep = (
        visible
        & obs["Author"].isin(studies)
        & obs["Tumor_pta"].astype(str).isin(tumour_types)
        & cell_types.isin(selected_cell_types)
    )
    if status == "Tumour only":
        keep &= obs["Normal"].astype(str) == "Tumour"
    elif status == "Healthy only":
        keep &= obs["Normal"].astype(str) == "Healthy"
    if not keep.any():
        st.warning("No cells match the current selection.")
        st.stop()

    add_activity(
        value=selected_gene,
        analysis="Tumor UMAP Plot",
        user=st.session_state.session_id,
        time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    with st.spinner(f"Loading {selected_gene}..."):
        counts = load_gene_counts(directory, selected_gene)
    gene_fig, meta_fig, config = create_pta_umap_plots(
        selected_gene,
        counts,
        obs,
        keep,
        cell_types=cell_types,
        color_by=COLOR_BY_OPTIONS[color_by_label],
        color_map=color_map,
        sort_order=sort_order,
        download_as=download_as,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(gene_fig, use_container_width=True, config=config)
    with col2:
        st.plotly_chart(meta_fig, use_container_width=True, config=config)
    plot_summary_caption(
        f"{int(keep.sum()):,} of {len(obs):,} cells",
        f"{obs.loc[keep, 'SRA_ID'].nunique()} datasets",
        f"{cell_types[keep].nunique()} cell types",
        f"gene: {selected_gene}",
        version=selected_version,
        loader_key="pta_large_umap",
    )

    with st.container():
        st.markdown(
            """
            This plot shows the expression of a selected gene across cell types in human pituitary
            tumours and healthy pituitary. Datasets were integrated using scVI, and the latent space
            was used for identifying nearest neighbours and generating a UMAP plot. Expression is
            shown as log1p counts per 10,000. Up to 50,000 cells are drawn per plot; the metadata
            plot samples each category in proportion to its size.

            **X-axis**: Arbitrary UMAP axis 1
            **Y-axis**: Arbitrary UMAP axis 2

            Note: For any statistically robust visualisation use the box plots or dot plots. The UMAP coordinates are arbitrary and do not necessarily represent or relate to anything biological. For concerns on the use of UMAPs, read: doi.org/10.1371/journal.pcbi.1011288
            """
        )

except Exception as exc:
    st.error(f"Error creating plots: {exc}")
    with st.expander("Show full error traceback"):
        st.code(traceback.format_exc(), language="python")
