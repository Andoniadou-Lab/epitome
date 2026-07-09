"""Compact plot-settings panels shared across epitome sites."""

from __future__ import annotations

from contextlib import contextmanager

import streamlit as st


@contextmanager
def plot_settings_panel(title: str = "Plot settings", *, expanded: bool = True):
    """Collapse plot controls into an expander (open by default)."""
    with st.expander(title, expanded=expanded):
        yield


def download_format_select(
    key: str,
    formats: tuple[str, ...] = ("png", "svg"),
    label: str = "Download as:",
) -> str:
    return st.selectbox(label, options=list(formats), index=0, key=key)
