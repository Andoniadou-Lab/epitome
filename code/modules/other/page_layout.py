"""Shared layout helpers for the Other Atlas site."""

from modules.other.config import list_other_versions
import streamlit as st


def other_page_header(title: str, subtitle: str, version_key: str) -> str:
    col1, col2 = st.columns([5, 1])
    with col1:
        st.header(title)
        st.markdown(subtitle)
    with col2:
        versions = list_other_versions()
        if not versions:
            st.caption("No Other Atlas data")
            return "v_0.05"
        return st.selectbox(
            "Version",
            options=versions,
            key=version_key,
            label_visibility="collapsed",
        )
