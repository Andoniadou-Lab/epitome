import traceback

import pandas as pd
import streamlit as st

from modules.pta.boxplot import add_comparison_column, create_pta_boxplot
from modules.pta.cell_type_labels import (
    apply_pta_bulk_metadata_labels,
    filter_to_selected_categories,
    group_color_map_for_column,
    sort_pta_categories,
)
from modules.pta.config import PtaConfig
from modules.pta.data_loader import (
    align_bulk_samples,
    filter_by_author,
    load_pta_bulk_gene_universe,
    load_pta_metadata,
    resolve_bulk_expression_for_genes,
)
from modules.pta.page_layout import (
    bulk_expression_unit,
    bulk_matrix_caption,
    grouping_category_multiselect,
    pta_bulk_cohort_toggle,
    pta_page_header,
)
from modules.ui.plot_settings import download_format_select, plot_settings_panel
from modules.ui.plot_summary import boxplot_sample_caption

selected_version = pta_page_header(
    "Bulk Boxplot",
    "Distribution of bulk RNA-seq gene expression across tumour sample metadata. "
    "Each dot is a sample. The main cohort is log1p counts-per-million; "
    "Zhang and Jotanovic validation cohorts are log2(TPM + 1).",
    "version_select_tumor_bulk",
)

try:
    meta = load_pta_metadata(version=selected_version)
except FileNotFoundError as exc:
    st.error(
        "Tumour atlas data not found. Expected files under "
        f"`pta_data/bulk_expression/{selected_version}/` and "
        f"`pta_data/bulk_curation/{selected_version}/`."
    )
    st.code(str(exc))
    st.stop()

try:
    with plot_settings_panel("Plot settings"):
        cohort = pta_bulk_cohort_toggle("tumor_bulk_cohort")
        unit = bulk_expression_unit(cohort)
        all_genes = sorted(load_pta_bulk_gene_universe(version=selected_version, cohort=cohort))
        if not all_genes:
            raise FileNotFoundError(f"No genes in the {cohort} expression matrix.")
        col1, col2, col3 = st.columns(3)
        with col1:
            default_gene = "GH1" if "GH1" in all_genes else all_genes[0]
            gene = st.selectbox(
                "Gene",
                options=all_genes,
                index=all_genes.index(default_gene),
                key=f"tumor_bulk_gene_{cohort}",
            )
        grouping_cols = [c for c in PtaConfig.GROUPING_COLS if c in meta.columns]
        with col2:
            group_col = st.selectbox(
                "Group by",
                options=grouping_cols,
                index=grouping_cols.index("Cell_type_pta")
                if "Cell_type_pta" in grouping_cols
                else 0,
                key="tumor_bulk_group",
            )
        with col3:
            sec_options = ["None"] + [c for c in grouping_cols if c != group_col]
            secondary = st.selectbox(
                "Additional grouping",
                options=sec_options,
                index=0,
                key="tumor_bulk_secondary",
            )

        expr, matrix_id, missing_genes = resolve_bulk_expression_for_genes(
            selected_version, [gene], cohort=cohort
        )
        expr, meta = align_bulk_samples(expr, meta)
        if expr.shape[1] == 0:
            st.warning("No bulk samples overlap the selected expression matrix and metadata.")
            st.stop()
        if gene not in expr.index:
            st.warning(f"{gene} is not present in this cohort's expression matrix.")
            st.stop()

        col4, col5 = st.columns(2)
        with col4:
            if PtaConfig.AUTHOR_COL in meta.columns:
                all_studies = sorted(meta[PtaConfig.AUTHOR_COL].unique())
                studies = st.multiselect(
                    f"Studies ({len(all_studies)})",
                    options=all_studies,
                    default=all_studies,
                    key=f"tumor_bulk_studies_{cohort}",
                )
            else:
                studies = None
        with col5:
            merge_mixed = st.checkbox(
                "Merge mixed pitnets",
                value=True,
                key="tumor_bulk_merge_mixed",
                help="Collapse plurihormonal and other mixed cell types into a single Mixed group. Somatotroph / Lactotroph is kept separate.",
            )
            remove_unknown = st.checkbox(
                "Remove Unknown/Unclear",
                value=False,
                key="tumor_bulk_remove_unknown",
            )
            download_as = download_format_select("tumor_bulk_download")

        preview_meta = meta
        if studies is not None:
            preview_meta = meta.loc[filter_by_author(meta, studies)]
        preview_meta = apply_pta_bulk_metadata_labels(
            preview_meta.copy(), merge_mixed=merge_mixed
        )
        if remove_unknown:
            drop_labels = {"Unknown", "Unclear"}
            keep_preview = ~preview_meta[group_col].astype(str).isin(drop_labels)
            if secondary != "None":
                keep_preview &= ~preview_meta[secondary].astype(str).isin(drop_labels)
            preview_meta = preview_meta.loc[keep_preview]
        selected_levels = grouping_category_multiselect(
            preview_meta,
            [group_col, secondary],
            key_prefix="tumor_bulk_levels",
            merge_mixed=merge_mixed,
            key_suffix=cohort,
        )

        comparison_on = st.checkbox(
            "Comparison",
            value=False,
            key="tumor_bulk_comparison",
            help="Recode the current X-axis groups into a left-hand and right-hand comparison.",
        )
        left_groups, right_groups = [], []
        left_name, right_name = "Left", "Right"
        if comparison_on:
            with st.expander("Comparison", expanded=True):
                axis_values = selected_levels.get(group_col) or sort_pta_categories(
                    preview_meta[group_col].unique(), group_col, merge_mixed=merge_mixed
                )
                st.caption(
                    "Assign the current X-axis groups to a left-hand and right-hand comparison. "
                    "Samples in unassigned groups are omitted."
                )
                left_col, right_col = st.columns(2)
                with left_col:
                    left_name = st.text_input(
                        "Left-hand name",
                        value="Left",
                        key="tumor_bulk_cmp_left_name",
                    )
                    left_groups = st.multiselect(
                        "Left-hand groups",
                        options=axis_values,
                        default=[],
                        key=f"tumor_bulk_cmp_left_{group_col}",
                    )
                with right_col:
                    right_name = st.text_input(
                        "Right-hand name",
                        value="Right",
                        key="tumor_bulk_cmp_right_name",
                    )
                    right_groups = st.multiselect(
                        "Right-hand groups",
                        options=axis_values,
                        default=[],
                        key=f"tumor_bulk_cmp_right_{group_col}",
                    )

    if cohort == "main_cohort" and matrix_id == "just_aligned":
        st.caption(
            f"{gene} is absent from the shared-gene matrix, so this plot uses the "
            "just-aligned matrix (fewer samples: only datasets processed from raw reads)."
        )

    if studies is not None:
        if not studies:
            st.warning("No studies selected. Pick at least one study.")
            st.stop()
        keep = filter_by_author(meta, studies)
        overlap = [s for s in keep if s in expr.columns]
        meta = meta.loc[overlap]
        expr = expr[overlap]
        if expr.shape[1] == 0:
            st.warning("No samples remain after study filtering.")
            st.stop()

    meta = apply_pta_bulk_metadata_labels(meta, merge_mixed=merge_mixed)
    if remove_unknown:
        drop_labels = {"Unknown", "Unclear"}
        keep = pd.Series(True, index=meta.index)
        keep &= ~meta[group_col].astype(str).isin(drop_labels)
        if secondary != "None":
            keep &= ~meta[secondary].astype(str).isin(drop_labels)
        meta = meta.loc[keep]
        overlap = [s for s in meta.index if s in expr.columns]
        meta = meta.loc[overlap]
        expr = expr[overlap]
        if expr.shape[1] == 0:
            st.warning("No samples remain after removing Unknown/Unclear.")
            st.stop()

    if any(not values for values in selected_levels.values()):
        st.warning("Select at least one category for each grouping.")
        st.stop()
    n_before = len(meta)
    meta = filter_to_selected_categories(meta, selected_levels)
    overlap = [s for s in meta.index if s in expr.columns]
    meta = meta.loc[overlap]
    expr = expr[overlap]
    if expr.shape[1] == 0:
        st.warning("No samples remain after category filtering.")
        st.stop()
    n_dropped = n_before - len(meta)
    if n_dropped:
        st.caption(f"{n_dropped} samples outside the selected grouping categories were omitted.")

    plot_df = meta.copy()
    plot_df["Expression"] = expr.loc[gene, plot_df.index].to_numpy()
    plot_group_col = group_col
    comparison_order = None
    if comparison_on:
        if not left_groups or not right_groups:
            st.warning("Comparison needs at least one group on each side.")
            st.stop()
        n_before = len(plot_df)
        plot_df = add_comparison_column(
            plot_df,
            group_col,
            left_groups,
            right_groups,
            left_name,
            right_name,
        )
        n_dropped = n_before - len(plot_df)
        if plot_df.empty:
            st.warning("No samples remain in the selected comparison groups.")
            st.stop()
        if n_dropped:
            st.caption(f"{n_dropped} samples in unassigned X-axis groups were omitted.")
        plot_group_col = "Comparison"
        comparison_order = [
            (left_name or "Left").strip() or "Left",
            (right_name or "Right").strip() or "Right",
        ]
        overlap = [s for s in plot_df.index if s in expr.columns]
        plot_df = plot_df.loc[overlap]
        expr = expr[overlap]
        meta = meta.loc[overlap]

    color_dimension = secondary if secondary != "None" else plot_group_col
    color_map = group_color_map_for_column(
        color_dimension,
        plot_df[color_dimension],
        merge_mixed=merge_mixed,
    )

    fig, config = create_pta_boxplot(
        plot_df,
        gene,
        plot_group_col,
        secondary_group=None if secondary == "None" else secondary,
        hover_columns=plot_df.columns,
        color_map=color_map,
        merge_mixed=merge_mixed,
        download_as=download_as,
        y_label=f"{gene} expression ({unit})",
        primary_order=comparison_order,
    )
    st.plotly_chart(fig, use_container_width=True, config=config)
    n_studies = (
        meta[PtaConfig.AUTHOR_COL].nunique()
        if PtaConfig.AUTHOR_COL in meta.columns
        else None
    )
    boxplot_sample_caption(
        gene, expr.shape[1], sample_label="bulk samples", n_studies=n_studies,
        version=selected_version,
        loader_keys=("pta_expression", "pta_metadata"),
    )
    st.caption(bulk_matrix_caption(cohort, matrix_id))

except Exception as exc:
    st.error(f"An error occurred: {exc}")
    with st.expander("Show full traceback"):
        st.code(traceback.format_exc(), language="python")
