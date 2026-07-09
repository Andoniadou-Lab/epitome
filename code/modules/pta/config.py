"""Configuration for pituitary tumour atlas (PTA) data under ``pta_data/``."""

from __future__ import annotations

from pathlib import Path

from config import Config

PTA_MIN_VERSION = "v_0.04"


class PtaConfig:
    BASE_PATH = Config.BASE_PATH
    PTA_ROOT = BASE_PATH / "pta_data"

    SAMPLE_ID_COL = "Internal_ID"
    AUTHOR_COL = "Author"
    NA_LABEL = "Unknown"

    SEX_MAP = {1: "Male", 0: "Female", -1: "Unknown"}

    GROUPING_COLS = [
        "Sex_pta",
        "Lineage_pta",
        "Cell_type_pta",
        "Subtype_pta",
        "Secretion_pta",
        "Disease_pta",
        "Invasion_pta",
        "USP8_geno_pta",
        "GNAS_geno_pta",
    ]

    PSEUDOBULK_GROUPING_COLS = [
        "broad_cluster_final",
        "Sex",
        "Lineage",
        "Cell type",
        "Subtype",
        "Secretion",
        "Disease",
        "Invasion",
    ]

    @classmethod
    def curation_dir(cls, version: str) -> Path:
        return cls.PTA_ROOT / "curation" / version

    @classmethod
    def dotplot_dir(cls, version: str) -> Path:
        return cls.PTA_ROOT / "dotplot" / version

    @classmethod
    def cell_proportion_dir(cls, version: str) -> Path:
        return cls.PTA_ROOT / "cell_proportion" / version

    @classmethod
    def overview_dir(cls, version: str) -> Path:
        return cls.PTA_ROOT / "overview" / version

    @classmethod
    def sc_datasets_dir(cls, version: str) -> Path:
        return cls.PTA_ROOT / "sc_data" / "datasets" / version / "epitome_h5_files"

    @classmethod
    def bulk_curation_dir(cls, version: str) -> Path:
        return cls.PTA_ROOT / "bulk_curation" / version

    @classmethod
    def bulk_expression_dir(cls, version: str) -> Path:
        return cls.PTA_ROOT / "bulk_expression" / version

    @classmethod
    def pseudobulk_dir(cls, version: str) -> Path:
        return cls.PTA_ROOT / "pseudobulk" / version

    @classmethod
    def volcano_dir(cls, version: str) -> Path:
        return cls.PTA_ROOT / "epitome_volcanos" / version

    @classmethod
    def gene_group_annotation_dir(cls) -> Path:
        return cls.PTA_ROOT / "gene_group_annotation"

    @classmethod
    def metabolism_genes_path(cls, _version: str = "v_0.04") -> Path:
        return cls.gene_group_annotation_dir() / "recon2_metabolism_genes.tsv"

    @classmethod
    def clinical_targets_path(cls) -> Path:
        return cls.gene_group_annotation_dir() / "clinical_target_enriched.parquet"

    @classmethod
    def volcano_manifest_path(cls, version: str) -> Path:
        return cls.volcano_dir(version) / "volcanos.json"

    @classmethod
    def curation_path(cls, version: str) -> Path:
        return cls.curation_dir(version) / "cpa.parquet"

    @classmethod
    def metadata_path(cls, version: str) -> Path:
        return cls.bulk_curation_dir(version) / "pituitary_tumor_atlas_bulk_updated_final.xlsx"

    @classmethod
    def expression_path(cls, version: str) -> Path:
        return cls.bulk_expression_dir(version) / "concatted_matrix_shared.csv"

    @classmethod
    def normalised_cache_path(cls, version: str) -> Path:
        return cls.bulk_expression_dir(version) / "expression_log1p_cpm.parquet"

    @classmethod
    def pseudobulk_path(cls, version: str) -> Path:
        directory = cls.pseudobulk_dir(version)
        for name in ("pdatas_2026_05_07.h5ad", "pdatas.h5ad"):
            path = directory / name
            if path.is_file():
                return path
        candidates = sorted(directory.glob("*.h5ad"))
        if candidates:
            return candidates[0]
        return directory / "pdatas_2026_05_07.h5ad"


def _version_key(version: str) -> tuple[int, ...]:
    normalized = version.removeprefix("v_").replace(".", "_")
    return tuple(int(p) for p in normalized.split("_") if p)


def list_pta_versions(min_version: str = PTA_MIN_VERSION) -> list[str]:
    root = PtaConfig.PTA_ROOT / "curation"
    if not root.is_dir():
        return []
    min_key = _version_key(min_version)
    versions = [
        p.name
        for p in root.iterdir()
        if p.is_dir() and p.name.startswith("v_") and _version_key(p.name) >= min_key
    ]
    return sorted(versions, key=_version_key, reverse=True)
