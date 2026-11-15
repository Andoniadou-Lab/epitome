import pandas as pd
import streamlit as st

from modules.pta.page_layout import pta_page_header
from modules.pta.stats import pta_rna_totals_path

selected_version = pta_page_header(
    "Release Notes",
    "Details of features and datasets in each human pituitary tumour atlas release.",
    "version_select_tumor_release_notes",
)

try:
    rna_totals = pd.read_parquet(pta_rna_totals_path(selected_version))
    total_cells = int(rna_totals.loc[rna_totals["total_cells"] > 0, "total_cells"].sum())
    cells_text = f"{total_cells:,} cells after QC"
except Exception:
    cells_text = "cells after QC"

st.info(
    "**v_0.04**: First integrated release of the human pituitary tumour atlas on epitome.\n\n"
    "- **Single-cell RNA-seq**: 135 samples (106 tumour, 29 healthy) from 14 studies "
    f"({cells_text}), with dot plots, cell-type abundance, the integrated atlas UMAP, "
    "individual dataset UMAPs, and sample curation.\n"
    "- **Bulk RNA-seq**: 2,003 samples from 23 studies. The main cohort has 1,732 samples "
    "(1,366 tumour, 366 healthy) from 21 studies; the Zhang et al., 2022 (194) and "
    "Jotanovic et al., 2024 (77) tumour cohorts are shown separately. Expression boxplots "
    "and RNA heatmaps.\n"
    "- **Pseudobulk**: 659 cell-type profiles from 130 scRNA-seq samples (106 tumour, "
    "24 healthy) for cross-sample comparison.\n\n"
    "Bulk and scRNA metadata are both available under Curation. "
    "Only `v_0.04` and later versions are supported on the tumour site."
)
