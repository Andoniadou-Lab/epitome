"""Other Atlas phylogeny: load marker tables and draw Fitch reconstructions.

The reconstruction itself lives in ``evo_markers``. This module only finds the
v_0.05 tables, caches trees, and turns ``node.x`` / ``node.y`` into a Plotly
cladogram so the website does not reimplement parsimony.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules.other import evo_markers as em
from modules.other.config import OtherConfig, other_version_candidates
from modules.versioning import record_resolved_version

CELL_TYPE_LABELS: dict[str, str] = {
    "sc": "Stem cells",
    "lac": "Lactotrophs",
    "thy": "Thyrotrophs",
    "cor": "Corticotrophs",
    "mel": "Melanotrophs",
    "gon": "Gonadotrophs",
    "som": "Somatotrophs",
}
CELL_TYPE_ORDER = ("sc", "lac", "thy", "cor", "mel", "gon", "som")
DEFAULT_GENES: dict[str, str] = {
    "sc": "NFIA",
    "lac": "GH1",
    "thy": "TSHB",
    "cor": "POMC",
    "mel": "POMC",
    "gon": "CGA",
    "som": "GH1",
}


def phylogeny_dir(version: str) -> tuple[Path | None, str | None]:
    for candidate in other_version_candidates(version):
        directory = OtherConfig.phylogeny_dir(candidate)
        if directory.is_dir() and (directory / "background_table.csv").is_file():
            record_resolved_version("other_phylogeny", version, candidate)
            return directory, candidate
    return None, None


def dge_path(directory: Path, cell_type: str) -> Path:
    return directory / f"all_species_{cell_type}_dge_gene_tree.csv"


def available_cell_types(directory: Path) -> list[str]:
    return [ct for ct in CELL_TYPE_ORDER if dge_path(directory, ct).is_file()]


@st.cache_data(show_spinner="Loading marker background...")
def load_background_table(directory: str) -> pd.DataFrame:
    return em.load_background(Path(directory) / "background_table.csv")


@st.cache_data(show_spinner="Loading marker table...")
def load_dge_table(directory: str, cell_type: str) -> pd.DataFrame:
    return em.load_dge(dge_path(Path(directory), cell_type))


@st.cache_resource(show_spinner="Building species tree...")
def load_cell_type_tree(directory: str, cell_type: str):
    df = load_dge_table(directory, cell_type)
    background = load_background_table(directory)
    return em.cell_type_tree(df, background)


def orthogroup_genes(df: pd.DataFrame) -> list[str]:
    return sorted(df["human_symbol"].dropna().astype(str).unique())


def tip_reason_table(tree, gene: str) -> pd.DataFrame:
    rows = []
    for leaf in tree:
        rows.append(
            {
                "species": leaf.name,
                "state": leaf.state,
                "reason": leaf.reason,
                "paralogs": ", ".join(leaf.instances.get(gene, [])),
                "change": leaf.change,
            }
        )
    return pd.DataFrame(rows)


def change_table(tree) -> pd.DataFrame:
    rows = []
    for node in tree.traverse():
        if node.change:
            rows.append(
                {
                    "node": node.name,
                    "clade": getattr(node, "sci_name", node.name),
                    "change": node.change,
                    "state": node.state,
                }
            )
    return pd.DataFrame(rows)


def reconstruct_gene(tree, gene: str, root_state: bool = False):
    """Copy the cached tree, run Fitch, ladderize, and write plot coordinates."""
    work = tree.copy("deepcopy")
    em.fitch(work, gene, root_state=root_state)
    work.ladderize()
    xmax = em._layout(work)
    return work, xmax


def plot_gene_tree_plotly(tree, gene: str, *, title: str | None = None, download_as: str = "png"):
    """Rectangular cladogram from the coordinates written by ``evo_markers._layout``."""
    work, xmax = reconstruct_gene(tree, gene)
    n_tips = len(work)
    height = max(420, 36 * n_tips + 80)

    xs, ys = [], []
    for node in work.traverse():
        for child in node.children:
            xs.extend([node.x, node.x, None, node.x, child.x, None])
            ys.extend([node.y, child.y, None, child.y, child.y, None])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color="#444444", width=1.2),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    counts = {
        "marker": 0,
        "not significant": 0,
        "orthogroup missing": 0,
        "no data": 0,
    }
    for node in work.traverse():
        colour = em._node_color(node)
        if node.is_leaf() and node.reason in counts:
            counts[node.reason] += 1
        hover = node.name if node.is_leaf() else getattr(node, "sci_name", node.name)
        if node.is_leaf():
            hover = f"<b>{node.name}</b><br>{node.reason}"
            paralogs = node.instances.get(gene, [])
            if paralogs:
                hover += f"<br>{', '.join(paralogs)}"
        elif node.state is True:
            hover += "<br>reconstructed marker"
        elif node.state is False:
            hover += "<br>reconstructed absent"
        if node.change:
            hover += f"<br>{node.change}"
        fig.add_trace(
            go.Scatter(
                x=[node.x],
                y=[node.y],
                mode="markers",
                marker=dict(
                    size=12 if node.is_leaf() else 9,
                    color=colour,
                    line=dict(color="white", width=0.8),
                ),
                hovertemplate=hover + "<extra></extra>",
                showlegend=False,
            )
        )

    annotations = []
    for node in work.traverse():
        if not node.is_leaf() and getattr(node, "sci_name", None):
            annotations.append(
                dict(
                    x=node.x - 0.07,
                    y=node.y + 0.09,
                    text=node.sci_name,
                    showarrow=False,
                    xanchor="right",
                    yanchor="bottom",
                    font=dict(size=10, color="#555555"),
                )
            )
    pad = 0.12
    for leaf in work:
        if leaf.reason == "orthogroup missing":
            color, sub, sub_color = em.MISSING_COLOR, "orthogroup missing", em.MISSING_COLOR
        elif leaf.reason == "no data":
            color, sub, sub_color = em.MISSING_COLOR, "no data", em.MISSING_COLOR
        else:
            color = em.MARKER_COLOR if leaf.state else "#2c2c2c"
            sub = ", ".join(leaf.instances.get(gene, []))
            sub_color = em.MARKER_COLOR if leaf.state else "#999999"
        annotations.append(
            dict(
                x=leaf.x + pad,
                y=leaf.y,
                text=f"<i>{leaf.name}</i>",
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font=dict(size=13, color=color),
            )
        )
        if sub:
            annotations.append(
                dict(
                    x=leaf.x + pad,
                    y=leaf.y - 0.33,
                    text=sub,
                    showarrow=False,
                    xanchor="left",
                    yanchor="middle",
                    font=dict(size=10, color=sub_color),
                )
            )

    legend_items = [
        ("marker", em.MARKER_COLOR, counts["marker"]),
        ("not a marker", em.OTHER_COLOR, counts["not significant"]),
        ("orthogroup missing", em.MISSING_COLOR, counts["orthogroup missing"]),
        ("no data", em.MISSING_COLOR, counts["no data"]),
    ]
    for label, colour, n in legend_items:
        if not n:
            continue
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color=colour, line=dict(color="white", width=0.8)),
                name=f"{label} (n={n})",
            )
        )

    fig.update_layout(
        title=title or gene,
        annotations=annotations,
        height=height,
        width=900,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=12)),
        xaxis=dict(visible=False, range=[-1.8, xmax + 3.0]),
        yaxis=dict(visible=False, range=[-n_tips + 0.2, 0.9]),
        plot_bgcolor="white",
        margin=dict(l=40, r=20, t=70, b=20),
    )
    config = {
        "toImageButtonOptions": {
            "format": download_as,
            "filename": f"{gene}_marker_phylogeny",
            "height": height,
            "width": 1100,
            "scale": 3,
        }
    }
    return fig, config, work
