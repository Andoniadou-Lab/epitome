import streamlit as st

from config import Config
from modules.cached_loaders import AVAILABLE_VERSIONS
from modules.download import create_bulk_data_downloads_ui

BASE_PATH = Config.BASE_PATH

col1, col2 = st.columns([5, 1])
with col1:
    st.header("Analysis Data Files")
    st.markdown("Download additional processed analysis files and integrated objects.")
with col2:
    selected_version = st.selectbox(
        "Version",
        options=AVAILABLE_VERSIONS,
        key="version_select_download_analysis_files",
        label_visibility="collapsed",
    )

create_bulk_data_downloads_ui(BASE_PATH, version=selected_version)
