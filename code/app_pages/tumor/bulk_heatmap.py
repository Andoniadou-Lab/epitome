import traceback

import streamlit as st

from modules.pta.cell_type_labels import (
    annotation_color_maps_for_columns,
    apply_pta_bulk_metadata_labels,
)
from modules.pta.config import PtaConfig
from modules.pta.data_loader import (
    align_bulk_samples,
    filter_by_author,
    load_pta_expression,
    load_pta_metadata,
)
from modules.pta.heatmap import build_matrix, create_heatmap, select_genes
from modules.pta.page_layout import pta_page_header
from modules.ui.plot_settings import download_format_select, plot_settings_panel
from modules.ui.plot_summary import heatmap_shape_caption

selected_version = pta_page_header(
    "Bulk RNA Heatmap",
    "Heatmap of bulk tumour RNA-seq expression across samples or metadata groups. "
    "Expression is log1p counts-per-million.",
    "version_select_tumor_bulk_heatmap",
)

try:
    meta = load_pta_metadata(version=selected_version)
    expr = load_pta_expression(version=selected_version)
    expr, meta = align_bulk_samples(expr, meta)
except FileNotFoundError as exc:
    st.error(
        "Bulk expression data not found. Expected files under "
        f"`pta_data/bulk_expression/{selected_version}/`."
    )
    st.code(str(exc))
    st.stop()

try:
    with plot_settings_panel("Plot settings"):
        col1, col2 = st.columns(2)
        with col1:
            if PtaConfig.AUTHOR_COL in meta.columns:
                all_studies = sorted(meta[PtaConfig.AUTHOR_COL].unique())
                heat_studies = st.multiselect(
                    f"Studies ({len(all_studies)})",
                    options=all_studies,
                    default=all_studies,
                    key="tumor_heat_studies",
                )
            else:
                heat_studies = None

            group_col_1 = st.selectbox(
                "Group / annotate by",
                options=PtaConfig.GROUPING_COLS,
                index=PtaConfig.GROUPING_COLS.index("Cell_type_pta")
                if "Cell_type_pta" in PtaConfig.GROUPING_COLS
                else 0,
                key="tumor_heat_group1",
            )
            second_options = ["(none)"] + [c for c in PtaConfig.GROUPING_COLS if c != group_col_1]
            group_col_2 = st.selectbox(
                "Second grouping (optional)",
                options=second_options,
                index=0,
                key="tumor_heat_group2",
            )
            group_cols = [group_col_1]
            if group_col_2 != "(none)":
                group_cols.append(group_col_2)

        with col2:
            gene_mode = st.radio(
                "Gene selection",
                options=["Choose genes", "Top variable genes"],
                horizontal=True,
                key="tumor_heat_genemode",
            )
            gene_list = None
            top_variable = None
            if gene_mode == "Choose genes":
                defaults = [g for g in ["GH1", "PRL", "POMC", "FSHB", "TSHB"] if g in expr.index]
                gene_list = st.multiselect(
                    f"Select genes ({len(expr.index)} available)",
                    options=sorted(expr.index.tolist()),
                    default=defaults,
                    max_selections=80,
                    key="tumor_heat_genes",
                )
            else:
                top_variable = st.slider(
                    "Number of top-variable genes", 5, 100, 30, step=5,
                    key="tumor_heat_topvar",
                )

            per_group = st.toggle(
                "Aggregate per group (mean) instead of per sample",
                value=False,
                key="tumor_heat_pergroup",
            )
            zscore = st.toggle("Z-score per gene", value=True, key="tumor_heat_zscore")
            merge_mixed = st.checkbox(
                "Merge mixed pitnets",
                value=True,
                key="tumor_heat_merge_mixed",
                help="Collapse plurihormonal and mixed cell types into a single Mixed group.",
            )
            heat_download = download_format_select("tumor_heat_download")

    if heat_studies is not None:
        if not heat_studies:
            st.warning("No studies selected.")
            st.stop()
        keep = filter_by_author(meta, heat_studies)
        meta = meta.loc[keep]
        expr = expr[[s for s in expr.columns if s in keep]]

    meta = apply_pta_bulk_metadata_labels(meta, merge_mixed=merge_mixed)

    genes = select_genes(expr, gene_list=gene_list, top_variable=top_variable)
    if not genes:
        st.warning("No matching genes found.")
        st.stop()

    matrix_df, annotations = build_matrix(
        expr,
        meta,
        genes,
        group_cols,
        per_group=per_group,
        zscore=zscore,
        merge_mixed=merge_mixed,
    )
    ann_colors = annotation_color_maps_for_columns(
        meta, group_cols, merge_mixed=merge_mixed
    )
    fig, config = create_heatmap(
        matrix_df,
        annotations,
        group_cols,
        zscore=zscore,
        download_as=heat_download,
        annotation_color_maps=ann_colors,
        merge_mixed=merge_mixed,
    )
    st.plotly_chart(fig, use_container_width=True, config=config)
    heatmap_shape_caption(matrix_df.shape[0], matrix_df.shape[1], per_group=per_group)

except Exception as exc:
    st.error(f"An error occurred: {exc}")
    with st.expander("Show full traceback"):
        st.code(traceback.format_exc(), language="python")
