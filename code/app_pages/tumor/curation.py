import traceback

import streamlit as st

from modules.display_tables import display_curation_table
from modules.pta.data_loader import load_pta_bulk_curation, load_pta_scrna_curation
from modules.pta.page_layout import pta_page_header

selected_version = pta_page_header(
    "Curation",
    "Browse metadata for single-cell and bulk samples in the human pituitary tumour atlas.",
    "version_select_tumor_curation",
)

tab_sc, tab_bulk = st.tabs(["Single-cell RNA-seq", "Bulk RNA-seq"])

with tab_sc:
    try:
        sc_curation = load_pta_scrna_curation(version=selected_version)
        hide = [
            c
            for c in [
                "passed_qc",
                "species",
                "pseudoaligned",
                "filtering_junk",
                "median_cellassign_prob",
                "passed_qc_tcc",
            ]
            if c in sc_curation.columns
        ]
        if hide:
            sc_curation = sc_curation.drop(columns=hide)
        st.caption(f"{len(sc_curation):,} scRNA-seq samples · version {selected_version}")
        display_curation_table(sc_curation, key_prefix="tumor_scrna_curation")
    except FileNotFoundError as exc:
        st.error(f"scRNA curation not found for `{selected_version}`.")
        st.code(str(exc))
    except Exception as exc:
        st.error(f"An error occurred: {exc}")
        with st.expander("Show full traceback"):
            st.code(traceback.format_exc(), language="python")

with tab_bulk:
    try:
        bulk_curation = load_pta_bulk_curation(version=selected_version)
        st.caption(f"{len(bulk_curation):,} bulk samples · version {selected_version}")
        st.dataframe(bulk_curation, use_container_width=True, height=700)
    except FileNotFoundError as exc:
        st.error(
            f"Bulk curation not found at `pta_data/bulk_curation/{selected_version}/`."
        )
        st.code(str(exc))
    except Exception as exc:
        st.error(f"An error occurred: {exc}")
        with st.expander("Show full traceback"):
            st.code(traceback.format_exc(), language="python")
