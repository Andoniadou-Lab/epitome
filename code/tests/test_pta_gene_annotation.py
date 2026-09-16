"""Tests for PTA Recon2 metabolism gene annotations."""

from __future__ import annotations

import pandas as pd

from modules.pta.config import PtaConfig
from modules.pta.gene_annotation import (
    _load_pta_clinical_targets_table,
    apply_pta_gene_annotations,
    format_clinical_drug_with_stage,
    format_clinical_target_drugs,
    load_pta_clinical_target_annotations,
    load_pta_clinical_target_stages,
    load_pta_druggability_annotations,
    load_pta_metabolism_genes,
    load_pta_tf_genes,
)


def _clear_clinical_target_caches() -> None:
    _load_pta_clinical_targets_table.clear()
    load_pta_clinical_target_annotations.clear()
    load_pta_clinical_target_stages.clear()
    load_pta_druggability_annotations.clear()
    load_pta_tf_genes.clear()


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
    monkeypatch.setattr(
        PtaConfig,
        "druggability_path",
        classmethod(lambda cls: tmp_path / "missing_druggability.parquet"),
    )
    monkeypatch.setattr(
        PtaConfig,
        "tf_annotation_path",
        classmethod(lambda cls: tmp_path / "missing_tfs.parquet"),
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
    monkeypatch.setattr(
        PtaConfig,
        "druggability_path",
        classmethod(lambda cls: tmp_path / "missing_druggability.parquet"),
    )
    monkeypatch.setattr(
        PtaConfig,
        "tf_annotation_path",
        classmethod(lambda cls: tmp_path / "missing_tfs.parquet"),
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
    monkeypatch.setattr(
        PtaConfig,
        "druggability_path",
        classmethod(lambda cls: tmp_path / "missing_druggability.parquet"),
    )
    monkeypatch.setattr(
        PtaConfig,
        "tf_annotation_path",
        classmethod(lambda cls: tmp_path / "missing_tfs.parquet"),
    )
    load_pta_metabolism_genes.clear()
    _clear_clinical_target_caches()

    stages = load_pta_clinical_target_stages()
    assert stages["MAP2K1"] == "APPROVAL"
    assert stages["GH1"] == "PHASE_1"

    annotations = load_pta_clinical_target_annotations()
    assert annotations.loc["MAP2K1", "clinical_target_drugs"] == (
        "DRUG-B (Approval) | DRUG-A (Phase 2)"
    )
    assert annotations.loc["GH1", "clinical_target_drugs"] == "DRUG-C (Phase 1)"

    df = pd.DataFrame({"gene": ["MAP2K1", "NOGENE"], "logFC": [1.0, 0.0]})
    annotated = apply_pta_gene_annotations(df, "v_0.04")
    assert annotated["is_clinical_target"].tolist() == [True, False]
    assert annotated.loc[0, "clinical_approval_stage"] == "APPROVAL"
    assert annotated.loc[0, "clinical_target_drugs"] == (
        "DRUG-B (Approval) | DRUG-A (Phase 2)"
    )
    assert pd.isna(annotated.loc[1, "clinical_approval_stage"])
    assert pd.isna(annotated.loc[1, "clinical_target_drugs"])


def test_druggability_annotations_from_parquet(monkeypatch, tmp_path):
    path = tmp_path / "target_prioritisation_druggable.parquet"
    pd.DataFrame(
        {
            "geneName": ["GH1", "GH1", "PRL", "MAP2K1"],
            "druggable": [True, False, False, True],
            "priority": [0.4, 0.8, 0.2, 0.9],
        }
    ).to_parquet(path)
    monkeypatch.setattr(
        PtaConfig,
        "metabolism_genes_path",
        classmethod(lambda cls, version: tmp_path / "missing_metabolism.tsv"),
    )
    monkeypatch.setattr(
        PtaConfig,
        "clinical_targets_path",
        classmethod(lambda cls: tmp_path / "missing_clinical.parquet"),
    )
    monkeypatch.setattr(PtaConfig, "druggability_path", classmethod(lambda cls: path))
    monkeypatch.setattr(
        PtaConfig,
        "tf_annotation_path",
        classmethod(lambda cls: tmp_path / "missing_tfs.parquet"),
    )
    load_pta_metabolism_genes.clear()
    _clear_clinical_target_caches()

    annotated = apply_pta_gene_annotations(
        pd.DataFrame({"gene": ["GH1", "PRL", "NOGENE"]}), "v_0.04"
    )
    assert annotated.loc[0, "druggable"] == True
    assert annotated.loc[0, "priority"] == 0.8
    assert annotated.loc[1, "druggable"] == False
    assert annotated.loc[1, "priority"] == 0.2
    assert pd.isna(annotated.loc[2, "druggable"])
    assert pd.isna(annotated.loc[2, "priority"])


def test_is_tf_keeps_only_is_tf_true_rows(monkeypatch, tmp_path):
    path = tmp_path / "lambert_human_tfs.parquet"
    pd.DataFrame(
        {
            "geneName": ["POU1F1", "AATF", "NR5A1", "AFF1"],
            "Is.TF": [True, False, True, False],
        }
    ).to_parquet(path)
    monkeypatch.setattr(
        PtaConfig,
        "metabolism_genes_path",
        classmethod(lambda cls, version: tmp_path / "missing_metabolism.tsv"),
    )
    monkeypatch.setattr(
        PtaConfig,
        "clinical_targets_path",
        classmethod(lambda cls: tmp_path / "missing_clinical.parquet"),
    )
    monkeypatch.setattr(
        PtaConfig,
        "druggability_path",
        classmethod(lambda cls: tmp_path / "missing_druggability.parquet"),
    )
    monkeypatch.setattr(PtaConfig, "tf_annotation_path", classmethod(lambda cls: path))
    load_pta_metabolism_genes.clear()
    _clear_clinical_target_caches()

    annotated = apply_pta_gene_annotations(
        pd.DataFrame(
            {
                "gene": ["POU1F1", "AATF", "NR5A1", "GH1"],
                "is_tf": [True, True, True, False],
            }
        ),
        "v_0.04",
    )
    assert annotated["is_tf"].tolist() == [True, False, True, False]


def test_format_clinical_drug_with_stage():
    assert format_clinical_drug_with_stage("TRAMETINIB", "APPROVAL") == "TRAMETINIB (Approval)"
    assert format_clinical_drug_with_stage("DRUG-A", "PHASE_1_2") == "DRUG-A (Phase 1/2)"
    assert format_clinical_target_drugs("DRUG-B (Approval) | DRUG-A (Phase 2)") == (
        "DRUG-B (Approval) | DRUG-A (Phase 2)"
    )
