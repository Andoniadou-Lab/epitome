import traceback

import pandas as pd
import streamlit as st

from modules.pta.boxplot import create_pta_boxplot
from modules.pta.cell_type_labels import (
    apply_pta_bulk_metadata_labels,
    group_color_map_for_column,
)
from modules.pta.config import PtaConfig
from modules.pta.data_loader import (
    align_bulk_samples,
    filter_by_author,
    load_pta_expression,
    load_pta_metadata,
)
from modules.pta.page_layout import pta_page_header
from modules.ui.plot_settings import download_format_select, plot_settings_panel
from modules.ui.plot_summary import boxplot_sample_caption

selected_version = pta_page_header(
    "Bulk Boxplot",
    "Distribution of bulk RNA-seq gene expression across tumour sample metadata. "
    "Each dot is a sample. Expression is log1p counts-per-million.",
    "version_select_tumor_bulk",
)

try:
    meta = load_pta_metadata(version=selected_version)
    expr = load_pta_expression(version=selected_version)
    expr, meta = align_bulk_samples(expr, meta)
except FileNotFoundError as exc:
    st.error(
        "Tumour atlas data not found. Expected files under "
        f"`pta_data/bulk_expression/{selected_version}/` and "
        f"`pta_data/bulk_curation/{selected_version}/`."
    )
    st.code(str(exc))
    st.stop()

try:
    all_genes = sorted(expr.index.tolist())

    with plot_settings_panel("Plot settings"):
        col1, col2, col3 = st.columns(3)
        with col1:
            default_gene = "GH1" if "GH1" in expr.index else all_genes[0]
            gene = st.selectbox(
                "Gene",
                options=all_genes,
                index=all_genes.index(default_gene),
                key="tumor_bulk_gene",
            )
        with col2:
            group_col = st.selectbox(
                "Group by",
                options=PtaConfig.GROUPING_COLS,
                index=PtaConfig.GROUPING_COLS.index("Cell_type_pta")
                if "Cell_type_pta" in PtaConfig.GROUPING_COLS
                else 0,
                key="tumor_bulk_group",
            )
        with col3:
            sec_options = ["None"] + [c for c in PtaConfig.GROUPING_COLS if c != group_col]
            secondary = st.selectbox(
                "Additional grouping",
                options=sec_options,
                index=0,
                key="tumor_bulk_secondary",
            )

        col4, col5 = st.columns(2)
        with col4:
            if PtaConfig.AUTHOR_COL in meta.columns:
                all_studies = sorted(meta[PtaConfig.AUTHOR_COL].unique())
                studies = st.multiselect(
                    f"Studies ({len(all_studies)})",
                    options=all_studies,
                    default=all_studies,
                    key="tumor_bulk_studies",
                )
            else:
                studies = None
        with col5:
            merge_mixed = st.checkbox(
                "Merge mixed pitnets",
                value=True,
                key="tumor_bulk_merge_mixed",
                help="Collapse plurihormonal and mixed cell types into a single Mixed group.",
            )
            remove_unknown = st.checkbox(
                "Remove Unknown/Unclear",
                value=False,
                key="tumor_bulk_remove_unknown",
            )
            download_as = download_format_select("tumor_bulk_download")

    if studies is not None:
        if not studies:
            st.warning("No studies selected. Pick at least one study.")
            st.stop()
        keep = filter_by_author(meta, studies)
        meta = meta.loc[keep]
        expr = expr[[s for s in expr.columns if s in keep]]
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
        expr = expr[[s for s in expr.columns if s in meta.index]]
        if expr.shape[1] == 0:
            st.warning("No samples remain after removing Unknown/Unclear.")
            st.stop()

    plot_df = meta.copy()
    plot_df["Expression"] = expr.loc[gene].values
    color_dimension = secondary if secondary != "None" else group_col
    color_map = group_color_map_for_column(
        color_dimension,
        plot_df[color_dimension],
        merge_mixed=merge_mixed,
    )

    fig, config = create_pta_boxplot(
        plot_df,
        gene,
        group_col,
        secondary_group=None if secondary == "None" else secondary,
        hover_columns=meta.columns,
        color_map=color_map,
        merge_mixed=merge_mixed,
        download_as=download_as,
    )
    st.plotly_chart(fig, use_container_width=True, config=config)
    n_studies = (
        meta[PtaConfig.AUTHOR_COL].nunique()
        if PtaConfig.AUTHOR_COL in meta.columns
        else None
    )
    boxplot_sample_caption(
        gene, expr.shape[1], sample_label="bulk samples", n_studies=n_studies
    )

except Exception as exc:
    st.error(f"An error occurred: {exc}")
    with st.expander("Show full traceback"):
        st.code(traceback.format_exc(), language="python")
