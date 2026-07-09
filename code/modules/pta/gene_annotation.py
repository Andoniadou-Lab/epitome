"""Human tumour atlas gene category annotations (Recon2 metabolism, clinical targets, etc.)."""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
import streamlit as st

from modules.pta.config import PtaConfig

_GENE_COLUMN_CANDIDATES = ("gene", "symbol", "gene_symbol", "Gene", "SYMBOL", "hgnc")

# Most advanced clinical stage wins when a gene has multiple target records.
_CLINICAL_STAGE_RANK = {
    "PRECLINICAL": 0,
    "PREAPPROVAL": 1,
    "IND": 2,
    "EARLY_PHASE_1": 3,
    "PHASE_1": 4,
    "PHASE_1_2": 5,
    "PHASE_2": 6,
    "PHASE_2_3": 7,
    "PHASE_3": 8,
    "UNKNOWN": 9,
    "APPROVAL": 10,
}


@lru_cache(maxsize=8)
def _read_metabolism_gene_table(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def _gene_column(df: pd.DataFrame) -> str:
    lower_map = {str(c).lower(): c for c in df.columns}
    for candidate in _GENE_COLUMN_CANDIDATES:
        if candidate.lower() in lower_map:
            return str(lower_map[candidate.lower()])
    return str(df.columns[0])


def format_clinical_approval_stage(stage: object) -> str:
    """Human-readable approval stage for hover text and tables."""
    if stage is None or (isinstance(stage, float) and pd.isna(stage)):
        return ""
    text = str(stage).strip()
    if not text or text.lower() == "nan":
        return ""
    return text.replace("_", " ")


@st.cache_data(show_spinner=False)
def load_pta_metabolism_genes(version: str = "v_0.04") -> frozenset[str]:
    """Gene symbols from the Recon2 metabolism store list."""
    path = PtaConfig.metabolism_genes_path(version)
    if not path.is_file():
        return frozenset()
    df = _read_metabolism_gene_table(str(path))
    col = _gene_column(df)
    genes = (
        df[col]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s != ""]
    )
    return frozenset(genes.tolist())


def format_clinical_target_drugs(drugs: object) -> str:
    """Pipe-separated drug names for hover text and tables."""
    if drugs is None or (isinstance(drugs, float) and pd.isna(drugs)):
        return ""
    text = str(drugs).strip()
    if not text or text.lower() == "nan":
        return ""
    return text


@st.cache_data(show_spinner=False)
def _load_pta_clinical_targets_table() -> pd.DataFrame:
    path = PtaConfig.clinical_targets_path()
    if not path.is_file():
        return pd.DataFrame(columns=["geneName", "maxClinicalStage", "drugName"])
    df = pd.read_parquet(path, columns=["geneName", "maxClinicalStage", "drugName"])
    df = df.dropna(subset=["geneName"])
    df["geneName"] = df["geneName"].astype(str).str.strip()
    df = df[df["geneName"] != ""]
    if "maxClinicalStage" in df.columns:
        df["maxClinicalStage"] = df["maxClinicalStage"].astype(str).str.strip()
        df.loc[df["maxClinicalStage"].isin(("", "nan")), "maxClinicalStage"] = pd.NA
    if "drugName" in df.columns:
        df["drugName"] = df["drugName"].astype(str).str.strip()
        df.loc[df["drugName"].isin(("", "nan")), "drugName"] = pd.NA
    return df


def _aggregate_clinical_target_drugs(df: pd.DataFrame) -> dict[str, str]:
    if df.empty or "drugName" not in df.columns:
        return {}
    drugs = df.dropna(subset=["drugName"])
    if drugs.empty:
        return {}

    def _join_unique(names: pd.Series) -> str:
        unique = sorted({str(name).strip() for name in names if str(name).strip()})
        return " | ".join(unique)

    return drugs.groupby("geneName")["drugName"].apply(_join_unique).to_dict()


@st.cache_data(show_spinner=False)
def load_pta_clinical_target_annotations() -> pd.DataFrame:
    """Per-gene clinical approval stage and pipe-separated drug names."""
    df = _load_pta_clinical_targets_table()
    if df.empty:
        return pd.DataFrame(columns=["clinical_approval_stage", "clinical_target_drugs"])

    stage_df = df.dropna(subset=["maxClinicalStage"])
    if stage_df.empty:
        stages = pd.Series(dtype=object, name="clinical_approval_stage")
    else:
        stage_df = stage_df.copy()
        stage_df["rank"] = stage_df["maxClinicalStage"].map(_CLINICAL_STAGE_RANK).fillna(-1)
        stages = (
            stage_df.sort_values(["geneName", "rank"])
            .drop_duplicates("geneName", keep="last")
            .set_index("geneName")["maxClinicalStage"]
            .rename("clinical_approval_stage")
        )

    drugs = pd.Series(
        _aggregate_clinical_target_drugs(df),
        name="clinical_target_drugs",
        dtype=object,
    )
    return pd.concat([stages, drugs], axis=1)


@st.cache_data(show_spinner=False)
def load_pta_clinical_target_stages() -> dict[str, str]:
    """Map gene symbol to the most advanced known clinical approval stage."""
    annotations = load_pta_clinical_target_annotations()
    if annotations.empty or "clinical_approval_stage" not in annotations.columns:
        return {}
    return annotations["clinical_approval_stage"].dropna().astype(str).to_dict()


def apply_pta_gene_annotations(df: pd.DataFrame, version: str = "v_0.04") -> pd.DataFrame:
    """Add boolean gene-category columns used in volcano tables and plots."""
    if "gene" not in df.columns:
        return df
    out = df.copy()
    genes = out["gene"].astype(str)

    metabolism_genes = load_pta_metabolism_genes(version)
    out["is_metabolism"] = genes.isin(metabolism_genes)

    clinical = load_pta_clinical_target_annotations()
    if clinical.empty:
        out["clinical_approval_stage"] = pd.NA
        out["clinical_target_drugs"] = pd.NA
    else:
        out["clinical_approval_stage"] = genes.map(clinical["clinical_approval_stage"])
        out["clinical_target_drugs"] = genes.map(clinical["clinical_target_drugs"])
    out["is_clinical_target"] = out["clinical_approval_stage"].notna()

    return out
