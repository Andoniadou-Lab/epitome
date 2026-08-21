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
    load_pta_bulk_gene_universe,
    load_pta_metadata,
    resolve_bulk_expression_for_genes,
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
    all_genes = sorted(load_pta_bulk_gene_universe(version=selected_version))
    if not all_genes:
        raise FileNotFoundError("No genes in either bulk expression matrix.")
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
        col1, col2, col3 = st.columns(3)
        with col1:
            default_gene = "GH1" if "GH1" in all_genes else all_genes[0]
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

        expr, matrix_id, missing_genes = resolve_bulk_expression_for_genes(
            selected_version, [gene]
        )
        expr, meta = align_bulk_samples(expr, meta)
        if expr.shape[1] == 0:
            st.warning("No bulk samples overlap the selected expression matrix and metadata.")
            st.stop()
        if gene not in expr.index:
            st.warning(
                f"{gene} is not present in the just-aligned matrix either, so it cannot be plotted."
            )
            st.stop()

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

    if matrix_id == "just_aligned":
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

    plot_df = meta.copy()
    plot_df["Expression"] = expr.loc[gene, plot_df.index].to_numpy()
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
        gene, expr.shape[1], sample_label="bulk samples", n_studies=n_studies,
        version=selected_version,
        loader_keys=("pta_expression", "pta_metadata"),
    )
    st.caption(
        "shared-gene matrix (all datasets)"
        if matrix_id == "shared"
        else "just-aligned matrix (raw-aligned datasets only)"
    )

except Exception as exc:
    st.error(f"An error occurred: {exc}")
    with st.expander("Show full traceback"):
        st.code(traceback.format_exc(), language="python")
