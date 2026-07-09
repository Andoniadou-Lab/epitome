import streamlit as st

from modules.pta.page_layout import pta_page_header

pta_page_header(
    "Release Notes",
    "Details of features and datasets in each human pituitary tumour atlas release.",
    "version_select_tumor_release_notes",
)

st.info(
    "**v_0.04**: First integrated release of the human pituitary tumour atlas on epitome.\n\n"
    "- **Single-cell RNA-seq**: 118 samples from 11 studies (~768k cells), with dot plots, "
    "cell-type abundance, individual dataset UMAPs, and sample curation.\n"
    "- **Bulk RNA-seq**: ~1,750 tumour samples with expression boxplots and RNA heatmaps.\n"
    "- **Pseudobulk**: integrated pseudobulk profiles for cross-sample comparison.\n"
    "when embeddings are not precomputed.\n\n"
    "Bulk and scRNA metadata are both available under Curation. "
    "Only `v_0.04` and later versions are supported on the tumour site."
)
