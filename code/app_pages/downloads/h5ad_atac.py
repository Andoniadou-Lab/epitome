import streamlit as st

from config import Config
from modules.cached_loaders import AVAILABLE_VERSIONS
from modules.download import create_downloads_ui_with_metadata_atac

BASE_PATH = Config.BASE_PATH

col1, col2 = st.columns([5, 1])
with col1:
    st.header("Dataset Files (h5ad) - ATAC")
    st.markdown("Download processed single-cell ATAC-seq datasets in `.h5ad` format.")
with col2:
    selected_version = st.selectbox(
        "Version",
        options=AVAILABLE_VERSIONS,
        key="version_select_download_h5ad_atac",
        label_visibility="collapsed",
    )

create_downloads_ui_with_metadata_atac(BASE_PATH, version=selected_version)
