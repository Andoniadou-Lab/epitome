"""PTA data loading — scRNA, bulk, pseudobulk, dotplot, and proportion data."""

from __future__ import annotations

import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
import streamlit as st

from modules.pta.config import PtaConfig
from modules.pta.gene_annotation import apply_pta_gene_annotations
from modules.pta.mtx_io import load_mtx_cached


def _read_raw_matrix(path: os.PathLike | str) -> pd.DataFrame:
    path = str(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext in (".tsv", ".txt"):
        df = pd.read_csv(path, sep="\t")
    else:
        df = pd.read_csv(path)
    df = df.set_index(df.columns[0])
    df.index.name = "gene"
    return df


def _read_index_file(path: os.PathLike | str) -> pd.DataFrame:
    path = Path(path) if not isinstance(path, os.PathLike) else path
    path = str(path)
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, sep="\t", header=None)
    if len(df.columns) == 1:
        df.columns = [0]
    return df


def _normalise_counts(counts: pd.DataFrame) -> pd.DataFrame:
    lib_size = counts.sum(axis=0).replace(0, np.nan)
    cpm = counts.divide(lib_size, axis=1) * 1e6
    return np.log1p(cpm.fillna(0.0))


def _clean_grouping_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    if "Sex_pta" in out.columns:
        out["Sex_pta"] = out["Sex_pta"].map(PtaConfig.SEX_MAP).fillna(PtaConfig.NA_LABEL)
    for col in cols:
        if col == "Sex_pta" or col not in out.columns:
            continue
        out[col] = (
            out[col]
            .astype(str)
            .replace({"nan": PtaConfig.NA_LABEL, "Null": PtaConfig.NA_LABEL, "": PtaConfig.NA_LABEL})
        )
    return out


@st.cache_data(show_spinner="Loading tumour scRNA curation...")
def load_pta_scrna_curation(version: str = "v_0.04") -> pd.DataFrame:
    df = pd.read_parquet(PtaConfig.curation_path(version))
    df["Name"] = df["Name"].fillna(df["SRA_ID"])
    sex_from_sex_col = None
    if "Sex" in df.columns:
        sex_from_sex_col = (
            df["Sex"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace(
                {
                    "f": "Female",
                    "female": "Female",
                    "0": "Female",
                    "0.0": "Female",
                    "m": "Male",
                    "male": "Male",
                    "1": "Male",
                    "1.0": "Male",
                    "none": "Unknown",
                    "nan": "Unknown",
                    "": "Unknown",
                    "<na>": "Unknown",
                }
            )
        )
    if "Comp_sex" in df.columns:
        comp = df["Comp_sex"].astype(str).str.strip()
        df["Comp_sex"] = comp.replace(
            {
                "1": "Male",
                "1.0": "Male",
                "0": "Female",
                "0.0": "Female",
            }
        )
        df["Comp_sex"] = df["Comp_sex"].replace(
            {"nan": "Unknown", "": "Unknown", "<NA>": "Unknown"}
        )
        # Dotplot curation can have Comp_sex fully missing; fall back to Sex labels where available.
        if sex_from_sex_col is not None:
            missing = df["Comp_sex"].isin(["Unknown", "nan", ""])
            df.loc[missing, "Comp_sex"] = sex_from_sex_col.loc[missing]
    if "Normal" in df.columns:
        normal = df["Normal"].astype(str).str.strip().str.lower()
        df["Normal"] = normal.replace(
            {
                "1": "Healthy",
                "1.0": "Healthy",
                "0": "Tumour",
                "0.0": "Tumour",
                "true": "Healthy",
                "false": "Tumour",
            }
        )
    return df


@st.cache_data(show_spinner="Loading tumour bulk metadata...")
def load_pta_metadata(version: str = "v_0.04") -> pd.DataFrame:
    df = pd.read_excel(PtaConfig.metadata_path(version))
    keep = [PtaConfig.SAMPLE_ID_COL, PtaConfig.AUTHOR_COL] + PtaConfig.GROUPING_COLS
    if "Name" in df.columns:
        keep.append("Name")
    df = df[[c for c in keep if c in df.columns]].copy()
    df = _clean_grouping_columns(df, PtaConfig.GROUPING_COLS)
    return df.set_index(PtaConfig.SAMPLE_ID_COL)


@st.cache_data(show_spinner="Loading tumour bulk curation table...")
def load_pta_bulk_curation(version: str = "v_0.04") -> pd.DataFrame:
    df = pd.read_excel(PtaConfig.metadata_path(version))
    if PtaConfig.SAMPLE_ID_COL in df.columns:
        df = df.set_index(PtaConfig.SAMPLE_ID_COL)
    if "Sex_pta" in df.columns:
        df["Sex_pta"] = df["Sex_pta"].map(PtaConfig.SEX_MAP).fillna(PtaConfig.NA_LABEL)
    return df


@st.cache_data(show_spinner="Loading bulk expression...")
def load_pta_expression(version: str = "v_0.04") -> pd.DataFrame:
    cache = PtaConfig.normalised_cache_path(version)
    if cache.is_file():
        return pd.read_parquet(cache)
    counts = _read_raw_matrix(PtaConfig.expression_path(version))
    logcpm = _normalise_counts(counts)
    try:
        logcpm.to_parquet(cache)
    except OSError as exc:
        print(f"Could not write PTA normalised cache: {exc}")
    return logcpm


@st.cache_resource(show_spinner="Loading dotplot matrices...")
def load_pta_dotplot_data(version: str = "v_0.04"):
    root = PtaConfig.dotplot_dir(version)
    proportion_matrix = load_mtx_cached(root / "matrix2.mtx", repair=True)
    expression_matrix = load_mtx_cached(root / "matrix1.mtx", repair=False)
    genes1 = _read_index_file(root / "matrix1_genes.tsv")
    genes2 = _read_index_file(root / "matrix2_genes.tsv")
    rows1 = _read_index_file(root / "matrix1_rows.tsv")
    rows2 = _read_index_file(root / "matrix2_rows.tsv")
    return proportion_matrix, genes1, rows1, expression_matrix, genes2, rows2


@st.cache_resource(show_spinner="Loading cell proportion data...")
def load_pta_proportion_data(version: str = "v_0.04"):
    root = PtaConfig.cell_proportion_dir(version)
    abundance_matrix = scipy.io.mmread(root / "abundance.mtx")
    abundance_rows = pd.read_csv(root / "abundance_rows.tsv", sep="\t", header=None)
    abundance_cols = pd.read_csv(root / "abundance_cols.tsv", sep="\t", header=None)
    return abundance_matrix, abundance_rows, abundance_cols


@st.cache_resource(show_spinner="Loading pseudobulk data...")
def load_pta_pseudobulk(version: str = "v_0.04") -> ad.AnnData:
    path = PtaConfig.pseudobulk_path(version)
    if not path.is_file():
        raise FileNotFoundError(
            f"Pseudobulk h5ad not found at {path}. "
            f"Add `pdatas.h5ad` (or `pdatas_2026_05_07.h5ad`) under `{path.parent}`."
        )
    return ad.read_h5ad(path)


@st.cache_data(show_spinner="Preparing pseudobulk expression...")
def load_pta_pseudobulk_tables(version: str = "v_0.04") -> tuple[pd.DataFrame, pd.DataFrame]:
    adata = load_pta_pseudobulk(version)
    return pseudobulk_expression_matrix(adata), pseudobulk_metadata(adata)


def align_bulk_samples(
    expr: pd.DataFrame, meta: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    shared = [s for s in expr.columns if s in meta.index]
    return expr[shared], meta.loc[shared]


def pseudobulk_expression_matrix(adata: ad.AnnData) -> pd.DataFrame:
    x = adata.X
    if hasattr(x, "toarray"):
        x = x.toarray()
    counts = pd.DataFrame(x.T, index=adata.var_names, columns=adata.obs_names)
    return _normalise_counts(counts)


def pseudobulk_metadata(adata: ad.AnnData) -> pd.DataFrame:
    meta = adata.obs.copy()
    if "Sex" in meta.columns:
        meta["Sex"] = meta["Sex"].astype(str)
    for col in PtaConfig.PSEUDOBULK_GROUPING_COLS:
        if col in meta.columns and col != "Sex":
            meta[col] = (
                meta[col]
                .astype(str)
                .replace({"nan": PtaConfig.NA_LABEL, "Null": PtaConfig.NA_LABEL, "": PtaConfig.NA_LABEL})
            )
    return meta.set_index(adata.obs_names)


def filter_by_author(meta: pd.DataFrame, studies: list[str] | None) -> pd.Index:
    if studies is None or PtaConfig.AUTHOR_COL not in meta.columns:
        return meta.index
    return meta.index[meta[PtaConfig.AUTHOR_COL].isin(studies)]


@st.cache_data(show_spinner="Loading volcano comparisons...")
def load_volcano_manifest(version: str = "v_0.04") -> list[dict]:
    path = PtaConfig.volcano_manifest_path(version)
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data["comparisons"]


@st.cache_data(show_spinner="Loading volcano results...")
def load_volcano_results(version: str, comparison_id: str) -> pd.DataFrame:
    for entry in load_volcano_manifest(version):
        if entry["id"] == comparison_id:
            csv_path = PtaConfig.volcano_dir(version) / entry["file"]
            df = pd.read_csv(csv_path)
            return apply_pta_gene_annotations(df, version)
    raise FileNotFoundError(f"Unknown volcano comparison: {comparison_id}")
