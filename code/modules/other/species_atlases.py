"""Load and plot cross-species pituitary atlases for the Other site."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules.other.cell_type_labels import (
    is_hidden_other_cell_type,
    normalize_other_cell_type,
    other_cell_type_color_map,
    species_display_name,
)
from modules.other.config import OtherConfig, other_version_candidates
from modules.versioning import record_resolved_version

_CELL_TYPE_COLUMNS = ("broad_cluster", "cell_type", "new_cell_type", "cell_type_final")
_PREFERRED_GENES = (
    "GH1",
    "Gh1",
    "PRL",
    "Prl",
    "POMC",
    "Pomc",
    "TSHB",
    "Tshb",
    "FSHB",
    "Fshb",
    "LHB",
    "Lhb",
    "SOX2",
    "Sox2",
    "POU1F1",
    "Pou1f1",
)


def other_cell_type_column(obs) -> str | None:
    for column in _CELL_TYPE_COLUMNS:
        if column in obs.columns:
            return column
    return None


def prepare_species_adata(adata):
    """Normalise cell-type labels and ensure a UMAP exists."""
    source = other_cell_type_column(adata.obs)
    if source is None:
        adata.obs["cell_type"] = "Unclear"
    else:
        adata.obs["cell_type"] = adata.obs[source].map(normalize_other_cell_type).astype(str)
    if "X_umap" not in adata.obsm:
        import scanpy as sc

        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.pca(adata, n_comps=min(30, adata.n_vars - 1, adata.n_obs - 1))
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
    return adata


def list_other_species(version: str = "v_0.05") -> dict[str, str]:
    """Display name → scientific folder name, for the current or a lower version."""
    for candidate in other_version_candidates(version):
        root = OtherConfig.species_atlases_dir(candidate)
        if not root.is_dir():
            continue
        species: dict[str, str] = {}
        for path in sorted(root.iterdir()):
            if not path.is_dir():
                continue
            if not any(path.glob("*.h5ad")):
                continue
            species[species_display_name(path.name)] = path.name
        if species:
            record_resolved_version("other_species_atlas_list", version, candidate)
            return species
    return {}


def _species_h5ad(root, scientific_name: str):
    folder = root / scientific_name
    for name in ("adata.h5ad", "adata_final_annotated.h5ad", "adata_final.h5ad"):
        path = folder / name
        if path.is_file():
            return path
    matches = sorted(folder.glob("*.h5ad"))
    return matches[0] if matches else None


def load_other_species_atlas(scientific_name: str, version: str = "v_0.05"):
    import anndata as ad

    errors: list[str] = []
    for candidate in other_version_candidates(version):
        root = OtherConfig.species_atlases_dir(candidate)
        path = _species_h5ad(root, scientific_name)
        if path is None:
            errors.append(f"{candidate}: no h5ad under {root / scientific_name}")
            continue
        try:
            adata = ad.read_h5ad(path)
            adata.obs_names_make_unique()
            adata = prepare_species_adata(adata)
            record_resolved_version("other_species_atlas", version, candidate)
            return adata
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{candidate}: {exc}")
    raise FileNotFoundError(
        f"No atlas found for {scientific_name} (requested {version}). "
        f"Attempts: {'; '.join(errors)}"
    )


@st.cache_resource(show_spinner="Loading species atlas...")
def _load_other_species_atlas_cached_pair(scientific_name: str, version: str = "v_0.05"):
    adata = load_other_species_atlas(scientific_name, version)
    from modules.versioning import get_resolved_version

    return adata, get_resolved_version(version, "other_species_atlas")


def load_other_species_atlas_cached(scientific_name: str, version: str = "v_0.05"):
    adata, resolved = _load_other_species_atlas_cached_pair(scientific_name, version)
    record_resolved_version("other_species_atlas", version, resolved)
    return adata


def get_other_species_info(adata) -> dict:
    labels = adata.obs["cell_type"].astype(str)
    visible = ~labels.map(is_hidden_other_cell_type)
    shown = labels[visible]
    return {
        "Total Cells": adata.shape[0],
        "Total Genes": adata.shape[1],
        "Cell Types": shown.unique().tolist(),
        "Cell Type Counts": shown.value_counts().to_dict(),
        "Has annotations": bool(shown.nunique()),
    }


def preferred_gene(gene_list) -> str | None:
    genes = set(gene_list)
    for name in _PREFERRED_GENES:
        if name in genes:
            return name
    return gene_list[0] if gene_list else None


def _gene_expression_values(adata, gene: str) -> np.ndarray:
    import scipy.sparse as sp

    if gene not in adata.var_names:
        raise ValueError(f"Gene {gene!r} not found in dataset")
    values = adata[:, gene].X
    if sp.issparse(values):
        return np.asarray(values.todense()).ravel()
    return np.asarray(values).ravel()


def _plotly_colorscale(name: str) -> str:
    return {
        "reds": "Reds",
        "blues": "Blues",
        "viridis": "Viridis",
        "plasma": "Plasma",
        "inferno": "Inferno",
        "magma": "Magma",
        "greens": "Greens",
        "ylorrd": "YlOrRd",
    }.get(name.lower(), name)


def plot_other_species_atlas(
    adata,
    selected_gene,
    sort_order=False,
    color_map="viridis",
    download_as="png",
    hide_unclear: bool = True,
):
    """Gene-expression UMAP on the left, cell-type UMAP on the right."""
    umap_coords = np.asarray(adata.obsm["X_umap"])
    labels = adata.obs["cell_type"].astype(str)
    type_keep = np.ones(len(adata), dtype=bool)
    if hide_unclear:
        type_keep &= ~labels.map(is_hidden_other_cell_type).to_numpy()
    plot_coords = umap_coords[type_keep]
    plot_labels = labels.to_numpy()[type_keep]
    total_cells = len(adata)
    marker_size = max(9 * min(1.0, 2000 / max(total_cells, 1)), 3)
    marker_opacity = max(0.8 * min(1.0, 2000 / max(total_cells, 1)), 0.3)

    color_values = _gene_expression_values(adata, selected_gene)
    gene_coords = umap_coords.copy()
    if sort_order:
        order = np.argsort(color_values)
        gene_coords = gene_coords[order]
        color_values = color_values[order]

    cmin = float(np.min(color_values)) if total_cells else 0.0
    cmax = float(np.max(color_values)) if total_cells else 0.0
    if total_cells == 0 or cmin == cmax:
        marker = dict(color="#c0c0c0", size=marker_size, opacity=marker_opacity)
    else:
        marker = dict(
            color=color_values,
            colorscale=_plotly_colorscale(color_map),
            cmin=cmin,
            cmax=cmax,
            colorbar=dict(title=f"counts {selected_gene}"),
            size=marker_size,
            opacity=marker_opacity,
        )

    gene_fig = go.Figure()
    gene_fig.add_trace(
        go.Scatter(
            x=gene_coords[:, 0],
            y=gene_coords[:, 1],
            mode="markers",
            marker=marker,
            text=[f"Expression: {val:.2f}" for val in color_values],
            hoverinfo="text",
        )
    )
    gene_fig.update_layout(
        title=f"Gene Expression: {selected_gene}",
        height=600,
        width=800,
        showlegend=False,
        xaxis_title="",
        yaxis_title="",
        plot_bgcolor="white",
    )

    cell_types = sorted(pd.unique(plot_labels))
    color_dict = other_cell_type_color_map(cell_types)
    cell_type_fig = go.Figure()
    for cell_type in cell_types:
        mask = plot_labels == cell_type
        cell_coords = plot_coords[mask]
        colour = color_dict.get(cell_type, "#888888")
        cell_type_fig.add_trace(
            go.Scatter(
                x=cell_coords[:, 0],
                y=cell_coords[:, 1],
                mode="markers",
                marker=dict(color=colour, size=marker_size, opacity=marker_opacity, line=dict(width=0)),
                name=cell_type,
                showlegend=False,
                legendgroup=cell_type,
                hovertemplate=f"<b>Cell Type:</b> {cell_type}<br>"
                + "<b>UMAP_1:</b> %{x}<br><b>UMAP_2:</b> %{y}<extra></extra>",
            )
        )
        cell_type_fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(color=colour, size=12, opacity=1.0),
                name=cell_type,
                showlegend=True,
                legendgroup=cell_type,
                hoverinfo="skip",
            )
        )
    cell_type_fig.update_layout(
        title="Cell Types",
        height=600,
        width=800,
        showlegend=True,
        xaxis_title="",
        yaxis_title="",
        plot_bgcolor="white",
        legend=dict(
            font=dict(size=14),
            itemsizing="constant",
            tracegroupgap=0,
            bgcolor="rgba(255,255,255,0.8)",
        ),
    )

    config = {
        "toImageButtonOptions": {
            "format": download_as,
            "filename": f"{selected_gene}_other_atlas_umap",
            "scale": 4,
        }
    }
    return gene_fig, cell_type_fig, config
