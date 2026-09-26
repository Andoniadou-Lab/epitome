"""Atlas-wide UMAP for the human pituitary tumour atlas (per-gene parquet export)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules.pta.cell_type_labels import (
    GROUPING_COLORS,
    HIDDEN_CLUSTER_LABELS,
    MIXED_SUBTYPE_COLORS,
    NORMAL_STATUS_COLOR_MAP,
    PSEUDOBULK_IMMUNE_TERMS,
    PTA_LINEAGE_COLORS,
    SEX_COLOR_MAP,
)
from modules.pta.config import PtaConfig, pta_version_candidates
from modules.utils import create_color_mapping
from modules.versioning import record_resolved_version

MAX_PLOTTED_CELLS = 50_000

OBS_COLUMNS = [
    "SRA_ID",
    "Author",
    "Tumor_pta",
    "Lineage",
    "Cell type",
    "Subtype",
    "Sex_numeric",
    "Normal",
    "Modality",
    "10X version",
    "cell_type",
    "UMAP1",
    "UMAP2",
    "ncounts",
]

# Label shown in the UI -> column in the prepared obs table.
COLOR_BY_OPTIONS = {
    "Cell type (single-cell)": "cell_type",
    "Tumour type": "Tumor_pta",
    "Lineage": "Lineage",
    "Clinical cell type": "Cell type",
    "Subtype": "Subtype",
    "Normal vs tumour": "Normal",
    "Study": "Author",
    "Sex": "Sex",
    "Modality": "Modality",
    "10X version": "10X version",
}

TUMOR_PTA_COLORS: dict[str, str] = {
    "Somatotroph_pitnet": "#1E90FF",
    "Lactotroph_pitnet": "#00BFFF",
    "Mammosomatotroph_pitnet": "#1e00ff",
    "Thyrotroph_pitnet": "#87CEEB",
    "Corticotroph_pitnet": "#f4f748",
    "Gonadotroph_pitnet": "#FF0000",
    "Gonadotroph_NFPA": "#a50f15",
    "Null_cell_pitnet": "#fcb258",
    "Healthy": "#6fe339",
}


def large_umap_dir(version: str) -> tuple[Path | None, str | None]:
    """Export directory for ``version``, falling back to earlier PTA versions."""
    for candidate in pta_version_candidates(version):
        directory = PtaConfig.large_umap_dir(candidate)
        if directory is not None:
            record_resolved_version("pta_large_umap", version, candidate)
            return directory, candidate
    return None, None


def _healthy(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    return "Healthy" if text in {"Normal", "Residual_Normal"} else (text or "Unclear")


def prepare_obs(obs: pd.DataFrame) -> pd.DataFrame:
    """Readable labels for filtering and colouring; row order is kept aligned with gene files."""
    out = pd.DataFrame(index=obs.index)
    out["SRA_ID"] = obs["SRA_ID"].astype(str)
    out["Author"] = obs["Author"].astype(str)
    out["Tumor_pta"] = obs["Tumor_pta"].map(_healthy).astype("category")
    out["Lineage"] = obs["Lineage"].map(_healthy).astype("category")
    out["Cell type"] = (
        obs["Cell type"]
        .map(_healthy)
        .astype(str)
        .replace({"Lactotroph / Somatotroph": "Somatotroph / Lactotroph", "Null": "Null_cell"})
        .astype("category")
    )
    out["Subtype"] = (
        obs["Subtype"].map(_healthy).astype(str).replace({"Null": "Null_cell"}).astype("category")
    )
    out["Sex"] = (
        obs["Sex_numeric"]
        .map(lambda v: {"0.0": "Female", "1.0": "Male"}.get(str(v), "Unknown"))
        .astype("category")
    )
    out["Normal"] = obs["Normal"].map({0: "Tumour", 1: "Healthy"}).fillna("Unclear").astype("category")
    out["Modality"] = obs["Modality"].astype(str).astype("category")
    out["10X version"] = obs["10X version"].astype(str).astype("category")
    out["cell_type"] = obs["cell_type"].astype(str)
    out["UMAP1"] = obs["UMAP1"].astype("float32")
    out["UMAP2"] = obs["UMAP2"].astype("float32")
    out["ncounts"] = obs["ncounts"].astype("float32")
    return out


@st.cache_resource(show_spinner="Loading tumour UMAP metadata...")
def load_large_umap_obs(directory: str) -> pd.DataFrame:
    return prepare_obs(pd.read_parquet(Path(directory) / "obs.parquet", columns=OBS_COLUMNS))


@st.cache_data(show_spinner=False)
def list_large_umap_genes(directory: str) -> list[str]:
    return sorted(p.stem for p in (Path(directory) / "genes_parquet").glob("*.parquet"))


def load_gene_counts(directory: str | Path, gene: str) -> np.ndarray:
    path = Path(directory) / "genes_parquet" / f"{gene}.parquet"
    if not path.is_file():
        raise ValueError(f"Gene data not found for {gene}")
    return pd.read_parquet(path)[gene].to_numpy(dtype="float32")


def cell_type_labels(obs: pd.DataFrame, *, merge_immune: bool) -> pd.Series:
    labels = obs["cell_type"]
    if merge_immune:
        labels = labels.where(~labels.isin(PSEUDOBULK_IMMUNE_TERMS), "Immune_cells")
    return labels


def visible_cell_mask(obs: pd.DataFrame, labels: pd.Series, *, merge_immune: bool) -> pd.Series:
    hidden = set(HIDDEN_CLUSTER_LABELS)
    if merge_immune:
        hidden.discard("immune_cells")
    return ~labels.str.strip().str.lower().isin(hidden)


def color_map_for(column: str, labels) -> dict[str, str] | None:
    values = [str(v) for v in pd.unique(pd.Series(labels).astype(str))]
    if column == "cell_type":
        return create_color_mapping(values)
    if column == "Tumor_pta":
        base = dict(TUMOR_PTA_COLORS)
    elif column == "Lineage":
        base = {**GROUPING_COLORS["Lineage_pta"], "Null": "#fcb258"}
    elif column == "Cell type":
        base = {**MIXED_SUBTYPE_COLORS, **PTA_LINEAGE_COLORS}
    elif column == "Subtype":
        base = {
            **GROUPING_COLORS["Subtype_pta"],
            "Healthy": "#6fe339",
            "Silent Corticotroph": "#c7c100",
            "Silent Gonadotroph": "#a50f15",
            "Silent Thyrotroph": "#4682B4",
        }
    elif column == "Normal":
        base = dict(NORMAL_STATUS_COLOR_MAP)
    elif column == "Sex":
        base = dict(SEX_COLOR_MAP)
    else:
        base = {}
    palette = px.colors.qualitative.Light24
    missing = [v for v in sorted(values) if v not in base]
    for i, value in enumerate(missing):
        base[value] = palette[i % len(palette)]
    return base


def _marker_style(n: int, *, gene: bool) -> tuple[int, float]:
    if n < 1000:
        return (8, 0.9) if gene else (5, 0.8)
    if n < 10000:
        return (6, 0.7) if gene else (4, 0.6)
    if n < 50000:
        return (4, 0.5) if gene else (3, 0.4)
    return (3, 0.3) if gene else (2, 0.25)


def _stratified_sample(df: pd.DataFrame, column: str, rng: np.random.Generator) -> pd.DataFrame:
    if len(df) <= MAX_PLOTTED_CELLS:
        return df
    picks = []
    for _, group in df.groupby(column, observed=True, sort=False):
        size = max(1, int(MAX_PLOTTED_CELLS * len(group) / len(df)))
        if len(group) > size:
            picks.append(group.index[rng.choice(len(group), size=size, replace=False)])
        else:
            picks.append(group.index)
    return df.loc[np.concatenate(picks)]


def _axes_layout() -> dict:
    return dict(
        xaxis=dict(showgrid=False, zeroline=False, title="UMAP 1"),
        yaxis=dict(showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1, title="UMAP 2"),
        height=600,
        width=800,
        plot_bgcolor="white",
    )


def create_pta_umap_plots(
    gene: str,
    counts: np.ndarray,
    obs: pd.DataFrame,
    keep: pd.Series,
    *,
    cell_types: pd.Series,
    color_by: str,
    color_map: str = "blues",
    sort_order: bool = False,
    download_as: str = "png",
):
    """Gene-expression UMAP plus a metadata-coloured UMAP for the kept cells.

    ``counts`` are raw counts aligned with ``obs``; they are shown as log1p
    counts per 10k using ``obs['ncounts']``, as on the mouse UMAP page.
    """
    rng = np.random.default_rng(42)
    keep = keep.to_numpy()
    expression = np.log1p(counts[keep] / obs["ncounts"].to_numpy()[keep] * 10_000)
    plot_df = pd.DataFrame(
        {
            "UMAP_1": obs["UMAP1"].to_numpy()[keep],
            "UMAP_2": obs["UMAP2"].to_numpy()[keep],
            "Expression": expression,
            "cell_type": cell_types.to_numpy()[keep],
            "SRA_ID": obs["SRA_ID"].to_numpy()[keep],
        }
    )
    if color_by != "cell_type":
        plot_df[color_by] = obs[color_by].astype(str).to_numpy()[keep]
    n_cells = len(plot_df)
    if n_cells == 0:
        return None, None, None

    gene_df = plot_df.sample(MAX_PLOTTED_CELLS, random_state=42) if n_cells > MAX_PLOTTED_CELLS else plot_df
    if sort_order:
        gene_df = gene_df.sort_values("Expression")
    size, opacity = _marker_style(n_cells, gene=True)
    gene_fig = px.scatter(
        gene_df,
        x="UMAP_1",
        y="UMAP_2",
        color="Expression",
        color_continuous_scale=color_map,
        hover_data={"cell_type": True, "SRA_ID": True, "UMAP_1": False, "UMAP_2": False},
        title=f"Gene Expression: {gene} ({n_cells:,} cells)",
    )
    gene_fig.update_traces(marker=dict(size=size, opacity=opacity, line=dict(width=0)))
    gene_fig.update_layout(
        **_axes_layout(),
        coloraxis_colorbar=dict(title=f"log1p counts {gene}", thickness=20, len=0.7),
    )

    meta_df = _stratified_sample(plot_df, color_by, rng)
    colours = color_map_for(color_by, meta_df[color_by])
    order = sorted(meta_df[color_by].astype(str).unique())
    size, opacity = _marker_style(n_cells, gene=False)
    meta_fig = go.Figure()
    for label in order:
        part = meta_df[meta_df[color_by].astype(str) == label]
        colour = colours.get(label, "#888888")
        meta_fig.add_trace(
            go.Scatter(
                x=part["UMAP_1"],
                y=part["UMAP_2"],
                mode="markers",
                marker=dict(color=colour, size=size, opacity=opacity, line=dict(width=0)),
                name=label,
                legendgroup=label,
                showlegend=False,
                hovertemplate=f"<b>{label}</b><extra></extra>",
            )
        )
        meta_fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(color=colour, size=12, opacity=1.0),
                name=label,
                legendgroup=label,
                showlegend=True,
                hoverinfo="skip",
            )
        )
    title = next((k for k, v in COLOR_BY_OPTIONS.items() if v == color_by), color_by)
    meta_fig.update_layout(
        **_axes_layout(),
        title=f"{title} ({n_cells:,} cells)",
        legend=dict(
            itemsizing="constant",
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
        ),
    )

    config = {
        "toImageButtonOptions": {
            "format": download_as,
            "filename": f"{gene}_epitome_tumour_umap",
            "scale": 4,
        }
    }
    return gene_fig, meta_fig, config
