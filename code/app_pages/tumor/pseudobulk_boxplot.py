import traceback

import pandas as pd
import streamlit as st

from modules.pta.boxplot import create_pta_boxplot
from modules.pta.cell_type_labels import (
    apply_pta_pseudobulk_metadata_labels,
    drop_other_cell_type_rows,
    group_color_map_for_column,
    merge_pseudobulk_immune_cell_types,
)
from modules.pta.config import PtaConfig
from modules.pta.data_loader import load_pta_pseudobulk_tables
from modules.pta.page_layout import pta_page_header
from modules.ui.plot_settings import download_format_select, plot_settings_panel
from modules.ui.plot_summary import boxplot_sample_caption

selected_version = pta_page_header(
    "Pseudobulk Boxplot",
    "Distribution of pseudobulk RNA-seq expression across inferred cell clusters "
    "and sample metadata. Each dot is a pseudobulk profile. Expression is log1p CPM.",
    "version_select_tumor_pseudobulk",
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
        grouping_cols.append("Normal")
    if "Normal" in grouping_cols:
        grouping_cols = ["Normal"] + [c for c in grouping_cols if c != "Normal"]
    all_genes = sorted(expr.index.tolist())

    with plot_settings_panel("Plot settings"):
        col1, col2, col3 = st.columns(3)
        with col1:
            default_gene = "GH1" if "GH1" in expr.index else all_genes[0]
            gene = st.selectbox(
                "Gene",
                options=all_genes,
                index=all_genes.index(default_gene),
                key="tumor_pseudo_gene",
            )
        with col2:
            default_group = (
                "broad_cluster_final"
                if "broad_cluster_final" in grouping_cols
                else grouping_cols[0]
            )
            group_col = st.selectbox(
                "Group by",
                options=grouping_cols,
                index=grouping_cols.index(default_group),
                key="tumor_pseudo_group",
            )
        with col3:
            sec_options = ["None"] + [c for c in grouping_cols if c != group_col]
            secondary = st.selectbox(
                "Additional grouping",
                options=sec_options,
                index=0,
                key="tumor_pseudo_secondary",
            )

        col4, col5 = st.columns(2)
        with col4:
            if PtaConfig.AUTHOR_COL in meta.columns:
                all_studies = sorted(meta[PtaConfig.AUTHOR_COL].dropna().unique())
                studies = st.multiselect(
                    f"Studies ({len(all_studies)})",
                    options=all_studies,
                    default=all_studies,
                    key="tumor_pseudo_studies",
                )
            else:
                studies = None
        with col5:
            merge_immune = st.checkbox(
                "Merge immune cell types",
                value=False,
                key="tumor_pseudo_merge_immune",
                help="Collapse B_cells, Immune_cells, Neutrophil, pDC_cells, and T_cells into Immune_cells.",
            )
            remove_unknown = st.checkbox(
                "Remove Unknown/Unclear",
                value=False,
                key="tumor_pseudo_remove_unknown",
            )
            download_as = download_format_select("tumor_pseudo_download")

    if studies is not None:
        if not studies:
            st.warning("No studies selected.")
            st.stop()
        keep = meta.index[meta[PtaConfig.AUTHOR_COL].isin(studies)]
        meta = meta.loc[keep]
        expr = expr[[s for s in expr.columns if s in keep]]

    meta = apply_pta_pseudobulk_metadata_labels(meta)
    meta = merge_pseudobulk_immune_cell_types(meta, merge_immune=merge_immune)
    meta, expr = drop_other_cell_type_rows(meta, expr, cell_type_col="broad_cluster_final")
    if remove_unknown:
        drop_labels = {"Unknown", "Unclear"}
        keep = pd.Series(True, index=meta.index)
        keep &= ~meta[group_col].astype(str).isin(drop_labels)
        if secondary != "None":
            keep &= ~meta[secondary].astype(str).isin(drop_labels)
        meta = meta.loc[keep]
        expr = expr[[s for s in expr.columns if s in meta.index]]
        if expr.shape[1] == 0:
            st.warning("No pseudobulk profiles remain after removing Unknown/Unclear.")
            st.stop()

    if gene not in expr.index:
        st.warning(f"Gene {gene} not found in pseudobulk matrix.")
        st.stop()

    plot_df = meta.copy()
    plot_df["Expression"] = expr.loc[gene].values
    color_dimension = secondary if secondary != "None" else group_col
    color_map = group_color_map_for_column(color_dimension, plot_df[color_dimension])

    fig, config = create_pta_boxplot(
        plot_df,
        gene,
        group_col,
        secondary_group=None if secondary == "None" else secondary,
        hover_columns=meta.columns,
        color_map=color_map,
        download_as=download_as,
    )
    st.plotly_chart(fig, use_container_width=True, config=config)
    n_studies = (
        meta[PtaConfig.AUTHOR_COL].nunique()
        if PtaConfig.AUTHOR_COL in meta.columns
        else None
    )
    boxplot_sample_caption(
        gene, expr.shape[1], sample_label="pseudobulk profiles", n_studies=n_studies
    )

except Exception as exc:
    st.error(f"An error occurred: {exc}")
    with st.expander("Show full traceback"):
        st.code(traceback.format_exc(), language="python")
