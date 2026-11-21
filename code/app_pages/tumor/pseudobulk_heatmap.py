import traceback
from datetime import datetime

import pandas as pd
import streamlit as st

from modules.analytics import add_activity
from modules.pta.cell_type_labels import (
    annotation_color_maps_for_columns,
    apply_pta_pseudobulk_metadata_labels,
    drop_hidden_cluster_rows,
    drop_other_cell_type_rows,
    filter_to_selected_categories,
    merge_pseudobulk_immune_cell_types,
    with_healthy_colour,
)
from modules.pta.config import PtaConfig
from modules.pta.data_loader import load_pta_pseudobulk_tables, pta_log10_expression
from modules.publication_export import plot_downloads
from modules.pta.heatmap import (
    build_matrix,
    create_heatmap,
    heatmap_download_frame,
    heatmap_publication_figure,
    select_genes,
)
from modules.pta.page_layout import (
    grouping_category_multiselect,
    healthy_colour_select,
    pta_page_header,
)
from modules.ui.plot_settings import download_format_select, plot_settings_panel
from modules.ui.plot_summary import heatmap_shape_caption

DEFAULT_GENES = ["SOX2", "GH1", "PRL", "TSHB", "FSHB", "POMC", "PECAM1", "COL1A1", "PTPRC"]
COLOUR_MAPS = {
    "Blue-white-red": "RdBu_r",
    "Viridis": "Viridis",
    "Reds": "Reds",
    "Magma": "Magma",
    "Inferno": "Inferno",
    "Cividis": "Cividis",
}

selected_version = pta_page_header(
    "Pseudobulk Heatmap",
    "Heatmap of pseudobulk RNA-seq expression across inferred cell clusters and sample "
    "metadata. Each column is a pseudobulk profile (or a group mean). "
    "Expression is log10(CPM + 1).",
    "version_select_tumor_pseudobulk_heatmap",
)

try:
    expr, meta = load_pta_pseudobulk_tables(version=selected_version)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.code(str(PtaConfig.pseudobulk_dir(selected_version)))
    st.stop()

try:
    grouping_cols = [c for c in PtaConfig.PSEUDOBULK_GROUPING_COLS if c in meta.columns]
    if "Normal" in meta.columns and "Normal" not in grouping_cols:
        grouping_cols.insert(1, "Normal")
    gene_universe = sorted(expr.index.tolist())

    with plot_settings_panel("Plot settings"):
        col1, col2 = st.columns(2)
        with col1:
            if PtaConfig.AUTHOR_COL in meta.columns:
                all_studies = sorted(meta[PtaConfig.AUTHOR_COL].dropna().unique())
                studies = st.multiselect(
                    f"Studies ({len(all_studies)})",
                    options=all_studies,
                    default=all_studies,
                    key="tumor_pseudo_heat_studies",
                )
            else:
                studies = None

            group_col_1 = st.selectbox(
                "Group / annotate by",
                options=grouping_cols,
                index=grouping_cols.index("broad_cluster_final")
                if "broad_cluster_final" in grouping_cols
                else 0,
                key="tumor_pseudo_heat_group1",
            )
            second_options = ["(none)"] + [c for c in grouping_cols if c != group_col_1]
            group_col_2 = st.selectbox(
                "Second grouping (optional)",
                options=second_options,
                index=second_options.index("Normal") if "Normal" in second_options else 0,
                key="tumor_pseudo_heat_group2",
            )
            group_cols = [group_col_1]
            if group_col_2 != "(none)":
                group_cols.append(group_col_2)

            merge_immune = st.checkbox(
                "Merge immune cell types",
                value=True,
                key="tumor_pseudo_heat_merge_immune",
                help=(
                    "Collapse immune populations (B cells, plasma cells, T-cell subsets, "
                    "NK cells, macrophages, monocytes, dendritic cells, neutrophils, pDC, "
                    "and residual Immune_cells) into Immune_cells."
                ),
            )
            remove_unknown = st.checkbox(
                "Remove Unknown/Unclear",
                value=False,
                key="tumor_pseudo_heat_remove_unknown",
            )

        with col2:
            gene_mode = st.radio(
                "Gene selection",
                options=["Choose genes", "Top variable genes"],
                horizontal=True,
                key="tumor_pseudo_heat_genemode",
            )
            gene_list = None
            top_variable = None
            if gene_mode == "Choose genes":
                defaults = [g for g in DEFAULT_GENES if g in gene_universe]
                gene_list = st.multiselect(
                    f"Select genes ({len(gene_universe)} available)",
                    options=gene_universe,
                    default=defaults,
                    max_selections=80,
                    key="tumor_pseudo_heat_genes",
                )
            else:
                top_variable = st.slider(
                    "Number of top-variable genes", 5, 100, 30, step=5,
                    key="tumor_pseudo_heat_topvar",
                )

            per_group = st.toggle(
                "Aggregate per group (mean) instead of per profile",
                value=False,
                key="tumor_pseudo_heat_pergroup",
            )
            zscore = st.toggle("Z-score per gene", value=True, key="tumor_pseudo_heat_zscore")
            zscore_cap = None
            if zscore:
                zscore_cap = st.number_input(
                    "Z-score colour cap",
                    min_value=0.5,
                    max_value=10.0,
                    value=3.0,
                    step=0.5,
                    key="tumor_pseudo_heat_zscore_cap",
                    help="Colour scale is clipped to −cap … +cap. Hover still shows the uncapped z-score.",
                )
            colour_map_label = st.selectbox(
                "Colour map",
                options=list(COLOUR_MAPS),
                index=0 if zscore else 1,
                key=f"tumor_pseudo_heat_cmap_{'z' if zscore else 'raw'}",
            )
            grey_zeros = st.toggle(
                "Show zero expression in light grey",
                value=False,
                key="tumor_pseudo_heat_grey_zeros",
                help="Cells whose expression is exactly 0 (before z-scoring; for group means, 0 in every profile) are drawn in light grey.",
            )
            download_as = download_format_select("tumor_pseudo_heat_download")
            healthy_colour = healthy_colour_select("tumor_pseudo_heat_healthy_colour")

        preview_meta = meta
        if studies is not None:
            preview_meta = meta.loc[meta[PtaConfig.AUTHOR_COL].isin(studies)]
        preview_meta = apply_pta_pseudobulk_metadata_labels(preview_meta.copy())
        preview_meta = drop_hidden_cluster_rows(preview_meta)
        preview_meta = merge_pseudobulk_immune_cell_types(
            preview_meta, merge_immune=merge_immune
        )
        if remove_unknown:
            drop_labels = {"Unknown", "Unclear"}
            keep_preview = pd.Series(True, index=preview_meta.index)
            for col in group_cols:
                keep_preview &= ~preview_meta[col].astype(str).isin(drop_labels)
            preview_meta = preview_meta.loc[keep_preview]
        selected_levels = grouping_category_multiselect(
            preview_meta,
            group_cols,
            key_prefix="tumor_pseudo_heat_levels",
            merge_mixed=False,
            key_suffix=str(int(merge_immune)),
        )

    if studies is not None:
        if not studies:
            st.warning("No studies selected.")
            st.stop()
        keep = meta.index[meta[PtaConfig.AUTHOR_COL].isin(studies)]
        meta = meta.loc[keep]
        expr = expr[[s for s in expr.columns if s in keep]]

    meta = apply_pta_pseudobulk_metadata_labels(meta)
    meta, expr = drop_other_cell_type_rows(meta, expr, cell_type_col="broad_cluster_final")
    meta = merge_pseudobulk_immune_cell_types(meta, merge_immune=merge_immune)
    if remove_unknown:
        drop_labels = {"Unknown", "Unclear"}
        keep = pd.Series(True, index=meta.index)
        for col in group_cols:
            keep &= ~meta[col].astype(str).isin(drop_labels)
        meta = meta.loc[keep]
        expr = expr[[s for s in expr.columns if s in meta.index]]
        if expr.shape[1] == 0:
            st.warning("No pseudobulk profiles remain after removing Unknown/Unclear.")
            st.stop()

    if any(not values for values in selected_levels.values()):
        st.warning("Select at least one category for each grouping.")
        st.stop()
    n_before = len(meta)
    meta = filter_to_selected_categories(meta, selected_levels)
    expr = expr[[s for s in expr.columns if s in meta.index]]
    meta = meta.loc[expr.columns]
    if expr.shape[1] == 0:
        st.warning("No pseudobulk profiles remain after category filtering.")
        st.stop()
    if n_before - len(meta):
        st.caption(
            f"{n_before - len(meta)} profiles outside the selected grouping categories were omitted."
        )

    # log1p and log10(x + 1) differ by a constant factor, so top-variable ranking is unchanged.
    genes = select_genes(expr, gene_list=gene_list, top_variable=top_variable)
    if not genes:
        st.warning("No matching genes found.")
        st.stop()
    sub = expr.loc[genes]
    expr_log10 = pd.DataFrame(
        pta_log10_expression(sub.values, "pseudobulk"), index=sub.index, columns=sub.columns
    )

    matrix_df, annotations = build_matrix(
        expr_log10, meta, genes, group_cols, per_group=per_group, zscore=zscore,
        merge_mixed=False,
    )
    zero_mask = None
    if grey_zeros:
        unscaled, _ = build_matrix(
            expr_log10, meta, genes, group_cols, per_group=per_group, zscore=False,
            merge_mixed=False,
        )
        zero_mask = unscaled.eq(0)
    ann_colors = annotation_color_maps_for_columns(meta, group_cols, merge_mixed=False)
    ann_colors = [
        with_healthy_colour(cmap, meta[col], healthy_colour)
        for cmap, col in zip(ann_colors, group_cols)
    ]
    fig, config = create_heatmap(
        matrix_df,
        annotations,
        group_cols,
        zscore=zscore,
        zscore_cap=zscore_cap,
        download_as=download_as,
        annotation_color_maps=ann_colors,
        merge_mixed=False,
        colorscale=COLOUR_MAPS[colour_map_label],
        zero_mask=zero_mask,
        value_label="log10(CPM + 1)",
    )
    add_activity(
        value=genes,
        analysis="Tumor Pseudobulk Heatmap",
        user=st.session_state.session_id,
        time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    st.plotly_chart(fig, use_container_width=True, config=config)
    export_fig, export_w, export_h = heatmap_publication_figure(fig)
    plot_downloads(
        export_fig, export_w, export_h,
        filename="pseudobulk_heatmap",
        download_as=download_as,
        data=heatmap_download_frame(matrix_df, annotations, group_cols),
        data_index=True,
        key="tumor_pseudobulk_heatmap",
    )
    heatmap_shape_caption(
        matrix_df.shape[0], matrix_df.shape[1], per_group=per_group,
        version=selected_version,
        loader_keys=("pta_pseudobulk_tables",),
        column_label="profiles",
    )

except Exception as exc:
    st.error(f"An error occurred: {exc}")
    with st.expander("Show full traceback"):
        st.code(traceback.format_exc(), language="python")
