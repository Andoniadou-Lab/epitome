"""Shared layout helpers for tumour atlas pages."""

import streamlit as st

from modules.pta.config import list_pta_versions


def pta_page_header(title: str, subtitle: str, version_key: str) -> str:
    col1, col2 = st.columns([5, 1])
    with col1:
        st.header(title)
        st.markdown(subtitle)
    with col2:
        versions = list_pta_versions()
        if not versions:
            st.caption("No PTA data")
            return "v_0.04"
        return st.selectbox(
            "Version",
            options=versions,
            key=version_key,
            label_visibility="collapsed",
        )
