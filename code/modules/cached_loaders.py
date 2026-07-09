"""Centralised Streamlit cache wrappers for epitome data loaders.

Each loader is defined explicitly (not via a factory) so Streamlit cache keys
stay unique — a shared factory body caused cache collisions and wrong return
tuple sizes at runtime.
"""

import streamlit as st

from modules.accessibility import preprocess_features
from modules.data_loader import (
    load_accessibility_data,
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

AVAILABLE_VERSIONS = ["v_0.02", "v_0.01"]
_FALLBACK = "v_0.01"


@st.cache_resource()
def load_cached_data(version="v_0.02"):
    try:
        return load_and_transform_data(version)
    except Exception:
        return load_and_transform_data(_FALLBACK)


@st.cache_resource()
def load_cached_chromvar_data(version="v_0.02"):
    try:
        return load_chromvar_data(version)
    except Exception:
        return load_chromvar_data(_FALLBACK)


@st.cache_resource()
def load_cached_isoform_data(version="v_0.02"):
    try:
        return load_isoform_data(version)
    except Exception:
        return load_isoform_data(_FALLBACK)


@st.cache_resource()
def load_cached_dotplot_data(version="v_0.02"):
    try:
        return load_dotplot_data(version)
    except Exception:
        return load_dotplot_data(_FALLBACK)


@st.cache_resource()
def load_cached_accessibility_data(version="v_0.02"):
    try:
        return load_accessibility_data(version)
    except Exception:
        return load_accessibility_data(_FALLBACK)


@st.cache_data()
def load_cached_curation_data(version="v_0.02"):
    try:
        return load_curation_data(version)
    except Exception:
        return load_curation_data(_FALLBACK)


@st.cache_data()
def load_cached_annotation_data(version="v_0.02"):
    try:
        return load_annotation_data(version)
    except Exception:
        return load_annotation_data(_FALLBACK)


@st.cache_data()
def load_cached_sex_dim_data(version="v_0.02"):
    try:
        return load_sex_dim_data(version)
    except Exception:
        return load_sex_dim_data(_FALLBACK)


@st.cache_data()
def load_cached_motif_data(version="v_0.02"):
    try:
        return load_motif_data(version)
    except Exception:
        return load_motif_data(_FALLBACK)


@st.cache_data()
def load_cached_enhancer_data(version="v_0.02"):
    try:
        return load_enhancer_data(version)
    except Exception:
        return load_enhancer_data(_FALLBACK)


@st.cache_data()
def load_cached_marker_data(version="v_0.02"):
    try:
        return load_marker_data(version)
    except Exception:
        return load_marker_data(_FALLBACK)


@st.cache_data()
def load_cached_marker_data_atac(version="v_0.02"):
    try:
        return load_marker_data_atac(version)
    except Exception:
        return load_marker_data_atac(_FALLBACK)


@st.cache_data()
def load_cached_proportion_data(version="v_0.02"):
    try:
        return load_proportion_data(version)
    except Exception:
        return load_proportion_data(_FALLBACK)


@st.cache_data()
def load_cached_ligand_receptor_data(version="v_0.02"):
    try:
        return load_ligand_receptor_data(version)
    except Exception:
        return load_ligand_receptor_data(_FALLBACK)


@st.cache_data()
def load_cached_enrichment_data(version="v_0.02"):
    try:
        return load_enrichment_results(version)
    except Exception:
        return load_enrichment_results(_FALLBACK)


@st.cache_data()
def load_cached_atac_proportion_data(version="v_0.02"):
    try:
        return load_atac_proportion_data(version)
    except Exception:
        return load_atac_proportion_data(_FALLBACK)


@st.cache_resource()
def load_cached_heatmap_data(version="v_0.02"):
    try:
        return load_heatmap_data(version)
    except Exception:
        return load_heatmap_data(_FALLBACK)


@st.cache_resource(ttl=600)
def load_cached_single_cell_dataset(dataset, version="v_0.02", rna_atac="rna"):
    try:
        return load_single_cell_dataset(dataset, version, rna_atac)
    except Exception:
        return load_single_cell_dataset(dataset, _FALLBACK, rna_atac)


@st.cache_data()
def load_cached_gene_curation(version="v_0.02"):
    try:
        return load_gene_curation(version)
    except Exception:
        return load_gene_curation(_FALLBACK)


@st.cache_data()
def preprocess_features_cached(features):
    return preprocess_features(features)


@st.cache_data()
def load_all_cached_data(version="v_0.02"):
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
