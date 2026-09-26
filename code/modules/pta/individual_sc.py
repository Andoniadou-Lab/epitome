"""Individual single-cell dataset helpers for the human pituitary tumour atlas."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import streamlit as st

from modules.pta.config import PtaConfig, pta_version_candidates
from modules.utils import create_color_mapping
from modules.versioning import record_resolved_version


def _ensure_umap(adata):
    """Brief default processing when UMAP is not precomputed."""
    import scanpy as sc

    if "X_umap" not in adata.obsm:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.pca(adata, n_comps=min(30, adata.n_vars - 1, adata.n_obs - 1))
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
    return adata


def pta_cell_type_column(obs) -> str | None:
    """Observation column that holds tumour single-cell identities.

    Current releases store identities in ``cell_type``. Older files used
    ``new_cell_type`` or ``broad_cluster_final``.
    """
    for column in ("cell_type", "new_cell_type", "broad_cluster_final", "cell_type_final"):
        if column in obs.columns:
            return column
    return None


def _ensure_cell_type_column(adata):
    source = pta_cell_type_column(adata.obs)
    if source == "cell_type":
        adata.obs["new_cell_type"] = adata.obs["cell_type"].astype(str)
    elif source is None:
        adata.obs["new_cell_type"] = "Unknown"
    elif "new_cell_type" not in adata.obs.columns:
        adata.obs["new_cell_type"] = adata.obs[source].astype(str)
    return adata


def load_pta_single_cell_dataset(dataset_id: str, version: str = "v_0.04"):
    import scanpy as sc

    errors: list[str] = []
    for candidate in pta_version_candidates(version):
        root = PtaConfig.sc_datasets_dir(candidate)
        try:
            for name in (f"{dataset_id}_processed.h5ad", f"{dataset_id}.h5ad"):
                path = root / name
                if path.is_file():
                    adata = sc.read(path)
                    adata = _ensure_cell_type_column(_ensure_umap(adata))
                    record_resolved_version("pta_sc_dataset", version, candidate)
                    return adata
            errors.append(f"{candidate}: no h5ad under {root}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{candidate}: {exc}")
    raise FileNotFoundError(
        f"No h5ad found for {dataset_id} (requested {version}). Attempts: {'; '.join(errors)}"
    )


@st.cache_resource(show_spinner="Loading single-cell dataset (UMAP computed if needed)...")
def _load_pta_single_cell_dataset_cached_pair(dataset_id: str, version: str = "v_0.04"):
    adata = load_pta_single_cell_dataset(dataset_id, version)
    from modules.versioning import get_resolved_version

    return adata, get_resolved_version(version, "pta_sc_dataset")


def load_pta_single_cell_dataset_cached(dataset_id: str, version: str = "v_0.04"):
    adata, resolved = _load_pta_single_cell_dataset_cached_pair(dataset_id, version)
    record_resolved_version("pta_sc_dataset", version, resolved)
    return adata


def list_pta_datasets(version: str = "v_0.04") -> dict[str, str]:
    for candidate in pta_version_candidates(version):
        root = PtaConfig.sc_datasets_dir(candidate)
        if not root.is_dir():
            continue
        try:
            curation = pd.read_parquet(PtaConfig.curation_path(candidate))
        except Exception:
            continue
        datasets = [f for f in os.listdir(root) if f.endswith(".h5ad")]
        if not datasets:
            continue
        sra_ids = [
            f.replace("_processed.h5ad", "").replace(".h5ad", "") for f in datasets
        ]
        display_names = []
        for sra_id in sra_ids:
            info = curation[curation["SRA_ID"].astype(str).str.contains(sra_id, na=False)]
            if info.empty and "GEO" in curation.columns:
                info = curation[curation["GEO"].astype(str).str.contains(sra_id, na=False)]
            if not info.empty:
                display_names.append(
                    f"{info.iloc[0]['Author']} - {info.iloc[0]['Name']} - {sra_id}"
                )
            else:
                display_names.append(sra_id)
        record_resolved_version("pta_sc_dataset_list", version, candidate)
        return dict(zip(display_names, sra_ids))
    return {}


def get_pta_dataset_info(adata) -> dict:
    column = pta_cell_type_column(adata.obs)
    if column is None:
        cell_types: list = []
        counts: dict = {}
    else:
        labels = adata.obs[column].astype(str)
        cell_types = labels.unique().tolist()
        counts = labels.value_counts().to_dict()
    return {
        "Total Cells": adata.shape[0],
        "Total Genes": adata.shape[1],
        "Cell Types": cell_types,
        "Cell Type Counts": counts,
    }


def _gene_expression_values(adata, gene: str) -> np.ndarray:
    """Extract per-cell expression for one gene as a 1D float array."""
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


def plot_pta_sc_dataset(adata, selected_gene, sort_order=False, color_map="viridis", download_as="png"):
    """Reuse mouse individual-sc plotting with PTA-prepared AnnData."""
    import plotly.graph_objects as go

    umap_coords = adata.obsm["X_umap"]
    total_cells = len(adata)
    marker_size = max(9 * min(1.0, 2000 / total_cells), 3)
    marker_opacity = max(0.8 * min(1.0, 2000 / total_cells), 0.3)

    gene_fig = go.Figure()
    color_values = _gene_expression_values(adata, selected_gene)
    plot_coords = umap_coords.copy()
    if sort_order:
        sort_indices = np.argsort(color_values)
        plot_coords = plot_coords[sort_indices]
        color_values = color_values[sort_indices]

    cmin = float(np.min(color_values))
    cmax = float(np.max(color_values))
    if cmin == cmax:
        marker = dict(
            color="#c0c0c0",
            size=marker_size,
            opacity=marker_opacity,
        )
    else:
        marker = dict(
            color=color_values,
            colorscale=_plotly_colorscale(color_map),
            cmin=cmin,
            cmax=cmax,
            colorbar=dict(title=f"log1p counts {selected_gene}"),
            size=marker_size,
            opacity=marker_opacity,
        )

    gene_fig.add_trace(
        go.Scatter(
            x=plot_coords[:, 0],
            y=plot_coords[:, 1],
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
    )

    cell_type_fig = go.Figure()
    annotation_column = pta_cell_type_column(adata.obs)
    if annotation_column is None:
        raise KeyError(
            "No cell type annotation in obs (expected cell_type)"
        )
    cell_types = sorted(adata.obs[annotation_column].astype(str).unique())
    color_dict = create_color_mapping(cell_types)
    for cell_type in cell_types:
        mask = adata.obs[annotation_column].astype(str).to_numpy() == cell_type
        cell_coords = umap_coords[mask]
        cell_type_fig.add_trace(
            go.Scatter(
                x=cell_coords[:, 0],
                y=cell_coords[:, 1],
                mode="markers",
                marker=dict(color=color_dict.get(cell_type, "#888888"), size=marker_size, opacity=marker_opacity),
                name=cell_type,
                showlegend=False,
                legendgroup=cell_type,
            )
        )
        cell_type_fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(color=color_dict.get(cell_type, "#888888"), size=12, opacity=1.0),
                name=cell_type,
                showlegend=True,
                legendgroup=cell_type,
                hoverinfo="skip",
            )
        )
    cell_type_fig.update_layout(title="Cell Types", height=600, width=800, showlegend=True)

    config = {"toImageButtonOptions": {"format": download_as, "filename": f"{selected_gene}_pta_umap", "scale": 4}}
    return gene_fig, cell_type_fig, config
