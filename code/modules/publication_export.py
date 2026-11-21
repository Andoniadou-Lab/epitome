"""Publication-styled figure and data downloads shown below a plot.

The on-screen chart and its Plotly camera button stay unchanged. The figure button
renders a separate, publication-styled copy in the visitor's browser with plotly.js,
so the server needs no image export engine.
"""

from __future__ import annotations

import html
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.offline import get_plotlyjs_version

TEXT_COLOUR = "black"
FONT_FAMILY = "Arial, Helvetica, sans-serif"
_PLOTLY_JS = f"https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"
_FORMATS = {"png": "png", "jpg": "jpeg", "jpeg": "jpeg", "webp": "webp", "svg": "svg"}


def clamp(value, low, high):
    return max(low, min(high, value))


def text_width(label: str, font_px: float) -> float:
    """Rough rendered width of ``label`` in Arial (capitals and digits are wider)."""
    text = str(label)
    wide = sum(1 for ch in text if ch.isupper() or ch.isdigit())
    return (wide * 0.68 + (len(text) - wide) * 0.52) * font_px


def base_publication_layout(fig: go.Figure) -> None:
    """White template with black Arial text. Streamlit's theme colours must not leak in."""
    fig.update_layout(
        template="plotly_white",
        font=dict(family=FONT_FAMILY, color=TEXT_COLOUR),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )


def journal_axes(fig: go.Figure, line_width: float = 1.5) -> None:
    axis = dict(
        showline=True,
        linecolor=TEXT_COLOUR,
        linewidth=line_width,
        ticks="outside",
        tickcolor=TEXT_COLOUR,
        ticklen=6,
        showgrid=False,
        zeroline=False,
    )
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)


def _set_family(font: dict) -> None:
    font["family"] = FONT_FAMILY


def _font_everywhere(node) -> None:
    if isinstance(node, list):
        for item in node:
            _font_everywhere(item)
        return
    if not isinstance(node, dict):
        return
    if isinstance(node.get("title"), dict):
        _set_family(node["title"].setdefault("font", {}))
    if "colorbar" in node and isinstance(node["colorbar"], dict):
        _set_family(node["colorbar"].setdefault("tickfont", {}))
    for key, value in node.items():
        if key.endswith("font") and isinstance(value, dict):
            _set_family(value)
        _font_everywhere(value)


def publication_fonts(fig: go.Figure) -> go.Figure:
    """Copy of ``fig`` with Arial (Cell Press style) set on every text element.

    Explicit per-element families keep SVG exports in Arial even when an editor
    ignores Plotly's figure-wide font inheritance.
    """
    spec = fig.to_plotly_json()
    layout = spec.setdefault("layout", {})
    _set_family(layout.setdefault("font", {}))
    for key, axis in layout.items():
        if key.startswith(("xaxis", "yaxis")) and isinstance(axis, dict):
            _set_family(axis.setdefault("tickfont", {}))
            title = axis.get("title")
            if isinstance(title, str):
                axis["title"] = title = {"text": title}
            if isinstance(title, dict):
                title.setdefault("font", {})
    _set_family(layout.setdefault("legend", {}).setdefault("font", {}))
    for annotation in layout.get("annotations", []) or []:
        _set_family(annotation.setdefault("font", {}))
    _font_everywhere(spec)
    return go.Figure(spec)


def _figure_button_html(fig: go.Figure, width: int, height: int, fmt: str, filename: str, label: str) -> str:
    fig_json = publication_fonts(fig).to_json().replace("</", "<\\/")
    opts = json.dumps(
        {
            "format": fmt,
            "width": int(width),
            "height": int(height),
            "scale": 1 if fmt == "svg" else 3,
            "filename": filename,
        }
    )
    return f"""
<style>
  body {{ margin: 0; font-family: Arial, sans-serif; }}
  button {{
    font: inherit; font-size: 15px; padding: 6px 12px; border-radius: 8px; cursor: pointer;
    border: 1px solid rgba(49, 51, 63, 0.2); background: white; color: rgb(49, 51, 63);
  }}
  button:hover {{ border-color: #0202ff; color: #0202ff; }}
  button:disabled {{ cursor: progress; opacity: 0.6; }}
</style>
<button id="dl">{html.escape(label)}</button>
<script src="{_PLOTLY_JS}"></script>
<script>
  const fig = {fig_json};
  const opts = {opts};
  const btn = document.getElementById("dl");
  const label = btn.textContent;
  btn.onclick = async () => {{
    btn.disabled = true;
    btn.textContent = "Preparing figure...";
    try {{
      await Plotly.downloadImage({{data: fig.data, layout: fig.layout}}, opts);
    }} catch (err) {{
      console.error(err);
      alert("Figure export failed: " + err);
    }} finally {{
      btn.disabled = false;
      btn.textContent = label;
    }}
  }};
</script>
"""


def plot_downloads(
    export_fig: go.Figure,
    width: int,
    height: int,
    *,
    filename: str,
    download_as: str = "png",
    data: pd.DataFrame | None = None,
    data_index: bool = False,
    key: str,
) -> None:
    """Publication figure button and, when ``data`` is given, a CSV button for the plotted data."""
    fmt = _FORMATS.get(str(download_as).lower(), "png")
    figure_label = f"Download publication figure ({fmt.upper()})"
    cols = st.columns([1, 1, 2]) if data is not None else st.columns([1, 3])
    button_html = _figure_button_html(export_fig, width, height, fmt, filename, figure_label)
    with cols[0]:
        if hasattr(st, "iframe"):
            st.iframe(button_html, height=44)
        else:
            components.html(button_html, height=44)
    if data is not None:
        with cols[1]:
            st.download_button(
                "Download plot data (CSV)",
                data=data.to_csv(index=data_index),
                file_name=f"{filename}_data.csv",
                mime="text/csv",
                key=f"{key}_data",
                on_click="ignore",
            )
