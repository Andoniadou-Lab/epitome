import traceback
from datetime import datetime

import streamlit as st

from modules.analytics import add_activity
from modules.dotplot import create_dotplot, filter_dotplot_data
from modules.pta.data_loader import load_pta_dotplot_data, load_pta_scrna_curation
from modules.pta.page_layout import pta_page_header
from modules.pta.stats import pta_rna_stats_path
from modules.ui.plot_settings import download_format_select, plot_settings_panel
from modules.ui.plot_summary import plot_summary_caption
from modules.utils import create_cell_type_stats_display, create_filter_ui, parse_row_info

selected_version = pta_page_header(
    "Gene Expression Dot Plot",
    "Compare gene expression patterns across cell types in human pituitary tumours. "
    "Each dot summarises pseudobulk profiles across samples.",
    "version_select_tumor_dotplot",
)

with st.spinner("Loading data..."):
    (
        proportion_matrix,
        genes1,
        rows1,
        expression_matrix,
        genes2,
        rows2,
    ) = load_pta_dotplot_data(version=selected_version)
    curation_data = load_pta_scrna_curation(version=selected_version)
    row_info = parse_row_info(rows1)
    valid_sra_ids = set(row_info["SRA_ID"].unique())
    filtered_curation = curation_data[curation_data["SRA_ID"].isin(valid_sra_ids)].copy()

with plot_settings_panel("Plot settings"):
    (
        filter_type,
        selected_samples,
        selected_authors,
        age_range,
        only_normal,
        modality,
    ) = create_filter_ui(filtered_curation, key_suffix="tumor_dotplot")

    available_genes = sorted(set(genes1.iloc[:, 0].astype(str).unique()))
    default_genes = ["GH1", "PRL", "POMC"] if "GH1" in available_genes else [available_genes[0]]
    default_genes = [g for g in default_genes if g in available_genes] or [available_genes[0]]

    selected_genes = st.multiselect(
        f"Select Genes for Dot Plot ({len(available_genes)} genes)",
        available_genes,
        default=default_genes,
        max_selections=60,
        key="tumor_dotplot_genes",
    )

    col1, col2 = st.columns(2)
    with col1:
        chosen_color_scheme = st.selectbox(
            "Select Color Scheme",
            options=["Blue", "Red", "Viridis", "Cividis"],
            index=0,
            key="color_scheme_tumor_dotplot",
        )
    with col2:
        download_as = download_format_select(
            "download_as_tumor_dotplot", formats=("png", "jpeg", "svg")
        )

(
    filtered_curation,
    filtered_prop_matrix,
    filtered_expr_matrix,
    filtered_rows1,
    filtered_rows2,
) = filter_dotplot_data(
    proportion_matrix,
    expression_matrix,
    rows1,
    rows2,
    filtered_curation,
    selected_samples if filter_type == "Sample" else None,
    selected_authors if filter_type == "Author" else None,
    age_range if filter_type == "Age" else None,
    only_normal,
)

try:
    if selected_genes:
        all_cell_types = sorted(
            {
                (ct.split("_", 1)[1] if "_" in ct else ct)
                for ct in filtered_rows1[filtered_rows1.columns[0]]
                if (ct.split("_", 1)[1] if "_" in ct else ct).strip().lower() != "other"
                and (ct.split("_", 1)[1] if "_" in ct else ct).strip().lower() not in {"intermediate", "intermediate_lobe"}
            }
        )
        cell_type_selection = st.radio(
            "Cell Type Selection",
            ["All Cell Types", "Select Specific Cell Types"],
            key="cell_type_selection_tumor_dotplot",
        )
        selected_cell_types = None
        if cell_type_selection == "Select Specific Cell Types":
            selected_cell_types = st.multiselect(
                "Select Cell Types",
                options=all_cell_types,
                default=[all_cell_types[0]],
                key="cell_type_multiselect_tumor_dotplot",
            )

        group_by_extras = st.multiselect(
            "Group by (in addition to Cell type)",
            options=[
                c
                for c in ["Normal", "Comp_sex", "Modality", "Tumor_pta", "Lineage"]
                if c in filtered_curation.columns
            ],
            default=[],
            key="group_by_extras_tumor_dotplot",
        )
        remove_unknown_extras = st.checkbox(
            "Remove Unknown/Unclear in additional groupings",
            value=False,
            key="tumor_dotplot_remove_unknown_extras",
            help="When additional grouping is enabled, drop rows where any selected extra is Unknown/Unclear/NA.",
        )

        create_cell_type_stats_display(
            version=selected_version,
            sra_ids=filtered_curation["SRA_ID"].unique().tolist(),
            display_title="Cell Counts in Current Selection",
            cell_types=selected_cell_types,
            column_count=6,
            size="small",
            atac_rna="rna",
            rna_stats_path=pta_rna_stats_path(selected_version),
        )

        add_activity(
            value=selected_genes,
            analysis="Tumor Dot Plot",
            user=st.session_state.session_id,
            time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        effective_cell_types = (
            selected_cell_types
            if selected_cell_types
            else all_cell_types
        )

        fig, config = create_dotplot(
            filtered_prop_matrix,
            filtered_expr_matrix,
            genes1,
            genes2,
            filtered_rows1,
            filtered_rows2,
            selected_genes,
            selected_cell_types=effective_cell_types,
            color_scheme=chosen_color_scheme,
            download_as=download_as,
            meta_data=filtered_curation,
            group_by_extras=group_by_extras,
            remove_unknown_extras=remove_unknown_extras,
        )
        st.plotly_chart(fig, use_container_width=True, config=config)
        n_cell_types = len(effective_cell_types)
        plot_summary_caption(
            f"{len(filtered_rows1)} pseudobulk profiles",
            f"{len(selected_genes)} genes",
            f"{n_cell_types} cell types",
        )

except Exception as exc:
    st.error(f"An error occurred: {exc}")
    with st.expander("Show full traceback"):
        st.code(traceback.format_exc(), language="python")
