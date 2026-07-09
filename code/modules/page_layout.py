import streamlit as st

from modules.cached_loaders import AVAILABLE_VERSIONS


def page_header(title: str, subtitle: str, version_key: str) -> str:
    col1, col2 = st.columns([5, 1])
    with col1:
        st.header(title)
        st.markdown(subtitle)
    with col2:
        return st.selectbox(
            "Version",
            options=AVAILABLE_VERSIONS,
            key=version_key,
            label_visibility="collapsed",
        )
