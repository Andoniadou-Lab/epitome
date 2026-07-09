import traceback

import streamlit as st

from modules.cached_loaders import AVAILABLE_VERSIONS, load_cached_curation_data
from modules.display_tables import display_curation_table

col1, col2 = st.columns([5, 1])
with col1:
    st.header("Data Curation Information")
    st.markdown("Browse detailed metadata for all samples included in the epitome.")
with col2:
    selected_version = st.selectbox(
        "Version",
        options=AVAILABLE_VERSIONS,
        key="version_select_tab9",
        label_visibility="collapsed",
    )

try:
    # Load curation data
    curation_data = load_cached_curation_data(version=selected_version)
    # do not expose the following columns passed_qc, species, pseudoaligned, filtering_junk, median_cellassign_prob, passed_qc_tcc
    curation_data = curation_data.drop(
        columns=[
            "passed_qc",
            "species",
            "pseudoaligned",
            "filtering_junk",
            "median_cellassign_prob",
            "passed_qc_tcc",
        ]
    )

    filtered_data = display_curation_table(
        curation_data, key_prefix="curation"
    )
except FileNotFoundError:
    st.error("Curation data file not found. Please check the file path.")
except Exception as e:
    st.error(f"Error loading curation data: {str(e)}")
    traceback.print_exc() 
    # Capture the full traceback
    tb = traceback.format_exc()

    # Display it in a collapsible section
    with st.expander("Show full error traceback"):
        st.code(tb, language='python')
