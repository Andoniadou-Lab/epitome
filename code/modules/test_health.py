"""Render the latest smoke-test health summary on the Release Notes tab."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

_RESULTS_FILE = Path(__file__).resolve().parent.parent / "test_results" / "latest.json"


def load_latest_results() -> dict | None:
    if not _RESULTS_FILE.is_file():
        return None
    try:
        return json.loads(_RESULTS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def render_test_health_bar() -> None:
    """Show a green progress bar for the most recent daily smoke-test run."""
    data = load_latest_results()
    if data is None:
        st.caption("Smoke test status is not yet available.")
        return

    passed = int(data.get("passed", 0))
    total = int(data.get("total", 0))
    date = data.get("date", "unknown")
    pct = (passed / total) if total else 0.0

    st.markdown(f"**{passed}/{total} code tests passed on {date}**")
    st.markdown(
        f"""
        <div style="background:#e8e8e8;border-radius:6px;height:22px;width:100%;overflow:hidden;">
          <div style="background:#28a745;width:{pct * 100:.1f}%;height:22px;border-radius:6px;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
