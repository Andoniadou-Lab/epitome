"""Centralised Streamlit cache wrappers for epitome data loaders.

Each loader is defined explicitly (not via a factory) so Streamlit cache keys
stay unique — a shared factory body caused cache collisions and wrong return
tuple sizes at runtime.

Fallback across versions happens *outside* the cache: we only cache exact-version
loads. Caching ``(fallback_data, "v_0.02")`` under key ``"v_0.03"`` permanently
served stale fallbacks after data was fixed.
"""

from __future__ import annotations

import streamlit as st

from modules.accessibility import preprocess_features
from modules.data_loader import (
    load_accessibility_data,
    load_aging_genes,
    load_and_transform_data,
    load_annotation_data,
    load_atac_proportion_data,
    load_chromvar_data,
    load_curation_data,
    load_dotplot_data,
    load_enhancer_data,
    load_enrichment_results,
    load_gene_curation,
    load_heatmap_data,
    load_isoform_data,
    load_ligand_receptor_data,
    load_marker_data,
    load_marker_data_atac,
    load_motif_data,
    load_proportion_data,
    load_sex_dim_data,
    load_single_cell_dataset,
)
from modules.versioning import (
    MOUSE_AVAILABLE_VERSIONS,
    record_resolved_version,
    version_candidates,
)

AVAILABLE_VERSIONS = MOUSE_AVAILABLE_VERSIONS
DEFAULT_VERSION = AVAILABLE_VERSIONS[0]


def _is_no_data(item) -> bool:
    if item is None:
        return True
    # Table loaders signal "nothing for this version" with an empty frame.
    empty = getattr(item, "empty", None)
    return bool(empty) if isinstance(empty, bool) else False


def _is_empty_result(result) -> bool:
    """True for loaders that report failure by returning empty/``None`` results.

    Treated as a failed version so ``_load_with_fallback`` keeps looking at
    lower versions instead of showing an empty table.
    """
    if isinstance(result, tuple):
        return all(_is_no_data(item) for item in result)
    return _is_no_data(result)


def _load_with_fallback(loader_key: str, requested: str, cached_exact_loader):
    """Try exact cached loads for requested then lower versions."""
    errors: list[str] = []
    for candidate in version_candidates(requested, AVAILABLE_VERSIONS):
        try:
            result = cached_exact_loader(candidate)
            if _is_empty_result(result):
                errors.append(f"{candidate}: loader returned no data")
                continue
            record_resolved_version(loader_key, requested, candidate)
            return result
        except Exception as exc:  # noqa: BLE001 — soft fallback across versions
            errors.append(f"{candidate}: {exc}")
    detail = "; ".join(errors) if errors else "no candidates"
    raise FileNotFoundError(
        f"No data for {loader_key} version {requested} (or lower). Attempts: {detail}"
    )


@st.cache_resource()
def _cached_data_exact(version: str):
    return load_and_transform_data(version)


def load_cached_data(version=DEFAULT_VERSION):
    return _load_with_fallback("expression", version, _cached_data_exact)


@st.cache_resource()
def _cached_chromvar_exact(version: str):
    return load_chromvar_data(version)


def load_cached_chromvar_data(version=DEFAULT_VERSION):
    return _load_with_fallback("chromvar", version, _cached_chromvar_exact)


@st.cache_resource()
def _cached_isoform_exact(version: str):
    return load_isoform_data(version)


def load_cached_isoform_data(version=DEFAULT_VERSION):
    return _load_with_fallback("isoforms", version, _cached_isoform_exact)


@st.cache_resource()
def _cached_dotplot_exact(version: str):
    return load_dotplot_data(version)


def load_cached_dotplot_data(version=DEFAULT_VERSION):
    return _load_with_fallback("dotplot", version, _cached_dotplot_exact)


@st.cache_resource()
def _cached_accessibility_exact(version: str):
    return load_accessibility_data(version)


def load_cached_accessibility_data(version=DEFAULT_VERSION):
    return _load_with_fallback("accessibility", version, _cached_accessibility_exact)


@st.cache_data()
def _cached_curation_exact(version: str):
    return load_curation_data(version)


def load_cached_curation_data(version=DEFAULT_VERSION):
    return _load_with_fallback("curation", version, _cached_curation_exact)


@st.cache_data()
def _cached_annotation_exact(version: str):
    return load_annotation_data(version)


def load_cached_annotation_data(version=DEFAULT_VERSION):
    return _load_with_fallback("annotation", version, _cached_annotation_exact)


@st.cache_data()
def _cached_sex_dim_exact(version: str):
    return load_sex_dim_data(version)


def load_cached_sex_dim_data(version=DEFAULT_VERSION):
    return _load_with_fallback("sex_dim", version, _cached_sex_dim_exact)


@st.cache_data()
def _cached_motif_exact(version: str):
    return load_motif_data(version)


def load_cached_motif_data(version=DEFAULT_VERSION):
    return _load_with_fallback("motif", version, _cached_motif_exact)


@st.cache_data()
def _cached_enhancer_exact(version: str):
    return load_enhancer_data(version)


def load_cached_enhancer_data(version=DEFAULT_VERSION):
    return _load_with_fallback("enhancer", version, _cached_enhancer_exact)


@st.cache_data()
def _cached_marker_exact(version: str):
    return load_marker_data(version)


def load_cached_marker_data(version=DEFAULT_VERSION):
    return _load_with_fallback("markers", version, _cached_marker_exact)


@st.cache_data()
def _cached_marker_atac_exact(version: str):
    return load_marker_data_atac(version)


def load_cached_marker_data_atac(version=DEFAULT_VERSION):
    return _load_with_fallback("markers_atac", version, _cached_marker_atac_exact)


@st.cache_data()
def _cached_proportion_exact(version: str):
    return load_proportion_data(version)


def load_cached_proportion_data(version=DEFAULT_VERSION):
    return _load_with_fallback("proportion", version, _cached_proportion_exact)


@st.cache_data()
def _cached_ligand_receptor_exact(version: str):
    return load_ligand_receptor_data(version)


def load_cached_ligand_receptor_data(version=DEFAULT_VERSION):
    return _load_with_fallback("lig_rec", version, _cached_ligand_receptor_exact)


@st.cache_data()
def _cached_enrichment_exact(version: str):
    return load_enrichment_results(version)


def load_cached_enrichment_data(version=DEFAULT_VERSION):
    return _load_with_fallback("enrichment", version, _cached_enrichment_exact)


@st.cache_data()
def _cached_atac_proportion_exact(version: str):
    return load_atac_proportion_data(version)


def load_cached_atac_proportion_data(version=DEFAULT_VERSION):
    return _load_with_fallback("proportion_atac", version, _cached_atac_proportion_exact)


@st.cache_resource()
def _cached_heatmap_exact(version: str):
    return load_heatmap_data(version)


def load_cached_heatmap_data(version=DEFAULT_VERSION):
    return _load_with_fallback("heatmap", version, _cached_heatmap_exact)


@st.cache_resource(ttl=600)
def _cached_single_cell_exact(dataset, version: str, rna_atac: str):
    return load_single_cell_dataset(dataset, version, rna_atac)


def load_cached_single_cell_dataset(dataset, version=DEFAULT_VERSION, rna_atac="rna"):
    def _exact(candidate: str):
        return _cached_single_cell_exact(dataset, candidate, rna_atac)

    return _load_with_fallback("sc_dataset", version, _exact)


@st.cache_data()
def _cached_gene_curation_exact(version: str):
    return load_gene_curation(version)


def load_cached_gene_curation(version=DEFAULT_VERSION):
    return _load_with_fallback("gene_curation", version, _cached_gene_curation_exact)


@st.cache_data()
def _cached_aging_genes_exact(version: str):
    return load_aging_genes(version)


def load_cached_aging_genes(version=DEFAULT_VERSION):
    """Aging-genes table, falling back to lower versions when absent."""
    return _load_with_fallback("aging", version, _cached_aging_genes_exact)


@st.cache_data()
def preprocess_features_cached(features):
    return preprocess_features(features)


@st.cache_data()
def load_all_cached_data(version=DEFAULT_VERSION):
    """Warm all caches once per session (router calls this on first visit)."""
    load_cached_data(version=version)
    load_cached_chromvar_data(version=version)
    load_cached_isoform_data(version=version)
    load_cached_dotplot_data(version=version)
    load_cached_accessibility_data(version=version)
    load_cached_curation_data(version=version)
    load_cached_annotation_data(version=version)
    load_cached_sex_dim_data(version=version)
    load_cached_motif_data(version=version)
    load_cached_enhancer_data(version=version)
    load_cached_marker_data(version=version)
    load_cached_marker_data_atac(version=version)
    load_cached_proportion_data(version=version)
    load_cached_ligand_receptor_data(version=version)
    load_cached_enrichment_data(version=version)
    load_cached_atac_proportion_data(version=version)
    load_cached_heatmap_data(version=version)
    load_cached_gene_curation(version=version)
