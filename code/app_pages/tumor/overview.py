import traceback

import pandas as pd
import streamlit as st

from config import Config
from modules.pta.data_loader import load_pta_scrna_curation
from modules.pta.page_layout import pta_page_header
from modules.pta.stats import pta_rna_stats_path
from modules.utils import create_cell_type_stats_display

BASE_PATH = Config.BASE_PATH
ACCENT = "#cc0000"

selected_version = pta_page_header(
    "Overview",
    "Summary of human pituitary tumour atlas data — single-cell RNA-seq, bulk RNA-seq, and pseudobulk profiles.",
    "version_select_tumor_overview",
)

try:
    curation = load_pta_scrna_curation(version=selected_version)
    rna_samples = len(curation[curation["Modality"].isin(["sn", "sc", "multi_rna"])])
    unique_papers = curation["Author"].nunique()
    total_cells = int(curation["n_cells"].sum())

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
            <div style="text-align:center;padding:20px;background:#f8f9fa;border-radius:10px;">
                <h3 style="color:#666;font-size:20px;">scRNA-seq Samples</h3>
                <div style="font-size:40px;font-weight:bold;color:{ACCENT};">{rna_samples:,}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div style="text-align:center;padding:20px;background:#f8f9fa;border-radius:10px;">
                <h3 style="color:#666;font-size:20px;">Publications</h3>
                <div style="font-size:40px;font-weight:bold;color:{ACCENT};">{unique_papers:,}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div style="text-align:center;padding:20px;background:#f8f9fa;border-radius:10px;">
                <h3 style="color:#666;font-size:20px;">Total Cells (scRNA)</h3>
                <div style="font-size:40px;font-weight:bold;color:{ACCENT};">{total_cells:,}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("Modality breakdown")
    modality_counts = (
        curation["Modality"].value_counts().rename_axis("Modality").reset_index(name="Samples")
    )
    st.dataframe(modality_counts, use_container_width=True, hide_index=True)

    if "Tumor_pta" in curation.columns:
        st.subheader("Tumour type breakdown")
        tumor_counts = (
            curation["Tumor_pta"].value_counts().head(15).rename_axis("Tumour type").reset_index(name="Samples")
        )
        st.dataframe(tumor_counts, use_container_width=True, hide_index=True)

    stats_df = pd.read_parquet(pta_rna_stats_path(selected_version))
    cell_types_no_other = [
        c for c in stats_df.columns if c != "dataset" and c.strip().lower() != "other"
    ]
    create_cell_type_stats_display(
        version=selected_version,
        display_title="Total Cells by Cell Type (scRNA)",
        column_count=4,
        atac_rna="rna",
        cell_types=cell_types_no_other,
        rna_stats_path=pta_rna_stats_path(selected_version),
    )

    st.markdown("### Data included in this release")
    st.markdown(
        "- **Single-cell RNA-seq**: 118 tumour samples across 11 studies, with dot plots, "
        "cell-type abundance, and individual dataset UMAPs.\n"
        "- **Bulk RNA-seq**: ~1,750 tumour samples with expression boxplots and heatmaps.\n"
        "- **Pseudobulk**: inferred cell-cluster profiles from integrated scRNA-seq."
    )

except FileNotFoundError as exc:
    st.error(f"Tumour atlas overview data not found for `{selected_version}`.")
    st.code(str(exc))
except Exception as exc:
    st.error(f"An error occurred: {exc}")
    with st.expander("Show full traceback"):
        st.code(traceback.format_exc(), language="python")
