"""Tests for PTA Recon2 metabolism gene annotations."""

from __future__ import annotations

import pandas as pd

from modules.pta.config import PtaConfig
from modules.pta.gene_annotation import (
    _load_pta_clinical_targets_table,
    apply_pta_gene_annotations,
    format_clinical_target_drugs,
    load_pta_clinical_target_annotations,
    load_pta_clinical_target_stages,
    load_pta_metabolism_genes,
)


def _clear_clinical_target_caches() -> None:
    _load_pta_clinical_targets_table.clear()
    load_pta_clinical_target_annotations.clear()
    load_pta_clinical_target_stages.clear()


def test_load_metabolism_genes_from_fixture(monkeypatch, tmp_path):
    tsv = tmp_path / "recon2_metabolism_genes.tsv"
    tsv.write_text("gene\nLDHA\nG6PC\n", encoding="utf-8")

    monkeypatch.setattr(
        PtaConfig,
        "metabolism_genes_path",
        classmethod(lambda cls, version: tsv),
    )
    monkeypatch.setattr(
        PtaConfig,
        "clinical_targets_path",
        classmethod(lambda cls: tmp_path / "missing_clinical.parquet"),
    )
    load_pta_metabolism_genes.clear()
    _clear_clinical_target_caches()

    genes = load_pta_metabolism_genes("v_0.04")
    assert genes == frozenset({"LDHA", "G6PC"})


def test_apply_pta_gene_annotations(monkeypatch, tmp_path):
    tsv = tmp_path / "recon2_metabolism_genes.tsv"
    tsv.write_text("gene\nLDHA\n", encoding="utf-8")
    monkeypatch.setattr(
        PtaConfig,
        "metabolism_genes_path",
        classmethod(lambda cls, version: tsv),
    )
    monkeypatch.setattr(
        PtaConfig,
        "clinical_targets_path",
        classmethod(lambda cls: tmp_path / "missing_clinical.parquet"),
    )
    load_pta_metabolism_genes.clear()
    _clear_clinical_target_caches()

    df = pd.DataFrame({"gene": ["LDHA", "GH1"], "logFC": [1.0, -1.0]})
    annotated = apply_pta_gene_annotations(df, "v_0.04")
    assert annotated["is_metabolism"].tolist() == [True, False]


def test_clinical_target_annotations_from_parquet(monkeypatch, tmp_path):
    clinical = tmp_path / "clinical_target_enriched.parquet"
    pd.DataFrame(
        {
            "geneName": ["MAP2K1", "MAP2K1", "GH1"],
            "maxClinicalStage": ["PHASE_2", "APPROVAL", "PHASE_1"],
            "drugName": ["DRUG-A", "DRUG-B", "DRUG-C"],
        }
    ).to_parquet(clinical)

    monkeypatch.setattr(
        PtaConfig,
        "metabolism_genes_path",
        classmethod(lambda cls, version: tmp_path / "missing_metabolism.tsv"),
    )
    monkeypatch.setattr(
        PtaConfig,
        "clinical_targets_path",
        classmethod(lambda cls: clinical),
    )
    load_pta_metabolism_genes.clear()
    _clear_clinical_target_caches()

    stages = load_pta_clinical_target_stages()
    assert stages["MAP2K1"] == "APPROVAL"
    assert stages["GH1"] == "PHASE_1"

    annotations = load_pta_clinical_target_annotations()
    assert annotations.loc["MAP2K1", "clinical_target_drugs"] == "DRUG-A | DRUG-B"

    df = pd.DataFrame({"gene": ["MAP2K1", "NOGENE"], "logFC": [1.0, 0.0]})
    annotated = apply_pta_gene_annotations(df, "v_0.04")
    assert annotated["is_clinical_target"].tolist() == [True, False]
    assert annotated.loc[0, "clinical_approval_stage"] == "APPROVAL"
    assert annotated.loc[0, "clinical_target_drugs"] == "DRUG-A | DRUG-B"
    assert pd.isna(annotated.loc[1, "clinical_approval_stage"])
    assert pd.isna(annotated.loc[1, "clinical_target_drugs"])
