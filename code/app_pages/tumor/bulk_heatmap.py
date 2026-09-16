import traceback

import streamlit as st

from modules.pta.cell_type_labels import (
    annotation_color_maps_for_columns,
    apply_pta_bulk_metadata_labels,
    filter_to_selected_categories,
)
from modules.pta.config import PtaConfig
from modules.pta.data_loader import (
    align_bulk_samples,
    filter_by_author,
    load_pta_bulk_gene_universe,
    load_pta_metadata,
    resolve_bulk_expression_for_genes,
)
from modules.pta.heatmap import build_matrix, create_heatmap, select_genes
from modules.pta.page_layout import (
    bulk_matrix_caption,
    grouping_category_multiselect,
    pta_bulk_cohort_toggle,
    pta_page_header,
)
from modules.ui.plot_settings import download_format_select, plot_settings_panel
from modules.ui.plot_summary import heatmap_shape_caption

selected_version = pta_page_header(
    "Bulk RNA Heatmap",
    "Heatmap of bulk tumour RNA-seq expression across samples or metadata groups. "
    "The main cohort is log1p counts-per-million; Zhang and Jotanovic validation "
    "cohorts are log2(TPM + 1).",
    "version_select_tumor_bulk_heatmap",
)

try:
    meta = load_pta_metadata(version=selected_version)
except FileNotFoundError as exc:
    st.error(
        "Bulk expression data not found. Expected files under "
        f"`pta_data/bulk_expression/{selected_version}/`."
    )
    st.code(str(exc))
    st.stop()

try:
    with plot_settings_panel("Plot settings"):
        cohort = pta_bulk_cohort_toggle("tumor_heat_cohort")
        gene_universe = sorted(
            load_pta_bulk_gene_universe(version=selected_version, cohort=cohort)
        )
        if not gene_universe:
            raise FileNotFoundError(f"No genes in the {cohort} expression matrix.")
        preview_expr, _, _ = resolve_bulk_expression_for_genes(
            selected_version, None, cohort=cohort
        )
        preview_expr, meta = align_bulk_samples(preview_expr, meta)
        col1, col2 = st.columns(2)
        with col1:
            if PtaConfig.AUTHOR_COL in meta.columns:
                all_studies = sorted(meta[PtaConfig.AUTHOR_COL].unique())
                heat_studies = st.multiselect(
                    f"Studies ({len(all_studies)})",
                    options=all_studies,
                    default=all_studies,
                    key=f"tumor_heat_studies_{cohort}",
                )
            else:
                heat_studies = None

            grouping_cols = [c for c in PtaConfig.GROUPING_COLS if c in meta.columns]
            group_col_1 = st.selectbox(
                "Group / annotate by",
                options=grouping_cols,
                index=grouping_cols.index("Cell_type_pta")
                if "Cell_type_pta" in grouping_cols
                else 0,
                key="tumor_heat_group1",
            )
            second_options = ["(none)"] + [c for c in grouping_cols if c != group_col_1]
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
                defaults = [g for g in ["GH1", "PRL", "POMC", "FSHB", "TSHB"] if g in gene_universe]
                gene_list = st.multiselect(
                    f"Select genes ({len(gene_universe)} available)",
                    options=gene_universe,
                    default=defaults,
                    max_selections=80,
                    key=f"tumor_heat_genes_{cohort}",
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
            zscore_cap = None
            if zscore:
                zscore_cap = st.number_input(
                    "Z-score colour cap",
                    min_value=0.5,
                    max_value=10.0,
                    value=3.0,
                    step=0.5,
                    key="tumor_heat_zscore_cap",
                    help="Colour scale is clipped to −cap … +cap. Hover still shows the uncapped z-score.",
                )
            merge_mixed = st.checkbox(
                "Merge mixed pitnets",
                value=True,
                key="tumor_heat_merge_mixed",
                help="Collapse plurihormonal and other mixed cell types into a single Mixed group. Somatotroph / Lactotroph is kept separate.",
            )
            heat_download = download_format_select("tumor_heat_download")

        preview_meta = meta
        if heat_studies is not None:
            preview_meta = meta.loc[filter_by_author(meta, heat_studies)]
        preview_meta = apply_pta_bulk_metadata_labels(
            preview_meta.copy(), merge_mixed=merge_mixed
        )
        selected_levels = grouping_category_multiselect(
            preview_meta,
            group_cols,
            key_prefix="tumor_heat_levels",
            merge_mixed=merge_mixed,
            key_suffix=cohort,
        )

    requested_genes = gene_list if gene_mode == "Choose genes" else None
    expr, matrix_id, missing_genes = resolve_bulk_expression_for_genes(
        selected_version, requested_genes, cohort=cohort
    )
    expr, meta = align_bulk_samples(expr, meta)
    if expr.shape[1] == 0:
        st.warning("No bulk samples overlap the selected expression matrix and metadata.")
        st.stop()

    if heat_studies is not None:
        if not heat_studies:
            st.warning("No studies selected.")
            st.stop()
        keep = filter_by_author(meta, heat_studies)
        meta = meta.loc[keep]
        expr = expr.reindex(columns=[s for s in meta.index if s in expr.columns])
        if expr.shape[1] == 0:
            st.warning("No samples remain after study filtering.")
            st.stop()
        meta = meta.loc[expr.columns]

    if cohort == "main_cohort" and matrix_id == "just_aligned":
        st.caption(
            "One or more selected genes are absent from the shared-gene matrix, "
            "so this heatmap uses the just-aligned matrix (fewer samples: only "
            "datasets processed from raw reads)."
        )
    if missing_genes:
        st.warning(
            "Not in this cohort's expression matrix, so omitted: " + ", ".join(missing_genes)
        )

    meta = apply_pta_bulk_metadata_labels(meta, merge_mixed=merge_mixed)
    if any(not values for values in selected_levels.values()):
        st.warning("Select at least one category for each grouping.")
        st.stop()
    n_before = len(meta)
    meta = filter_to_selected_categories(meta, selected_levels)
    expr = expr.reindex(columns=[s for s in meta.index if s in expr.columns])
    meta = meta.loc[expr.columns]
    if expr.shape[1] == 0:
        st.warning("No samples remain after category filtering.")
        st.stop()
    if n_before - len(meta):
        st.caption(
            f"{n_before - len(meta)} samples outside the selected grouping categories were omitted."
        )

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
        zscore_cap=zscore_cap,
        download_as=heat_download,
        annotation_color_maps=ann_colors,
        merge_mixed=merge_mixed,
    )
    st.plotly_chart(fig, use_container_width=True, config=config)
    heatmap_shape_caption(matrix_df.shape[0], matrix_df.shape[1], per_group=per_group,
        version=selected_version,
        loader_keys=("pta_expression", "pta_metadata"),
    )
    st.caption(bulk_matrix_caption(cohort, matrix_id))

except Exception as exc:
    st.error(f"An error occurred: {exc}")
    with st.expander("Show full traceback"):
        st.code(traceback.format_exc(), language="python")
