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
        "Granulation_pta",
        "KI67_pta",
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
    def druggability_path(cls) -> Path:
        return cls.gene_group_annotation_dir() / "target_prioritisation_druggable.parquet"

    @classmethod
    def tf_annotation_path(cls) -> Path:
        return cls.gene_group_annotation_dir() / "lambert_human_tfs.parquet"

    @classmethod
    def volcano_manifest_path(cls, version: str) -> Path:
        return cls.volcano_dir(version) / "volcanos.json"

    @classmethod
    def curation_path(cls, version: str) -> Path:
        return cls.curation_dir(version) / "cpa.parquet"

    @classmethod
    def metadata_path(cls, version: str) -> Path:
        return cls.bulk_curation_dir(version) / "pituitary_tumor_atlas_bulk_updated_final.xlsx"

    BULK_COHORTS = {
        "main_cohort": {
            "label": "Main cohort",
            "author": None,
            "unit": "log1p CPM",
            "kind": "counts_log1p_cpm",
            "matrices": ("shared", "just_aligned"),
        },
        "zhang_cohort": {
            "label": "Zhang et al., 2022",
            "author": "Zhang et al., 2022",
            "unit": "log2(TPM + 1)",
            "kind": "tpm_log2p1",
            "matrices": ("zhang",),
        },
        "jotanovic_cohort": {
            "label": "Jotanovic et al., 2024",
            "author": "Jotanovic et al., 2024",
            "unit": "log2(TPM + 1)",
            "kind": "tpm_log2p1",
            "matrices": ("jotanovic",),
        },
    }
    TPM_LOG2P1_MATRICES = frozenset({"zhang", "jotanovic"})

    BULK_MATRIX_FILES = {
        "shared": "concatted_matrix_shared.csv",
        "just_aligned": "concatted_matrix_just_aligned.csv",
        "zhang": "zhang.csv",
        "jotanovic": "jotanovic.csv",
    }
    BULK_CACHE_FILES = {
        "shared": "expression_log1p_cpm_shared.parquet",
        "just_aligned": "expression_log1p_cpm_just_aligned.parquet",
        "zhang": "expression_log2p1_tpm_zhang.parquet",
        "jotanovic": "expression_log2p1_tpm_jotanovic.parquet",
    }

    @classmethod
    def expression_path(cls, version: str, matrix: str = "shared") -> Path:
        filename = cls.BULK_MATRIX_FILES.get(matrix, cls.BULK_MATRIX_FILES["shared"])
        directory = cls.bulk_expression_dir(version)
        parquet = directory / Path(filename).with_suffix(".parquet")
        csv_path = directory / filename
        if parquet.is_file():
            return parquet
        return csv_path

    @classmethod
    def normalised_cache_path(cls, version: str, matrix: str = "shared") -> Path:
        filename = cls.BULK_CACHE_FILES.get(matrix, cls.BULK_CACHE_FILES["shared"])
        return cls.bulk_expression_dir(version) / filename

    @classmethod
    def large_umap_dir(cls, version: str) -> Path | None:
        """Newest ``adata_export_large_umap*`` export that has per-gene parquet files."""
        root = cls.PTA_ROOT / "large_umap" / version
        if not root.is_dir():
            return None
        exports = sorted(
            (
                p
                for p in root.glob("adata_export_large_umap*")
                if p.is_dir() and (p / "genes_parquet").is_dir() and (p / "obs.parquet").is_file()
            ),
            key=lambda p: p.name,
            reverse=True,
        )
        return exports[0] if exports else None

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


def is_current_pta_version(version: str) -> bool:
    """True for the newest PTA release (or anything newer than what is on disk)."""
    available = list_pta_versions()
    return not available or _version_key(version) >= _version_key(available[0])


def pta_version_candidates(requested: str) -> list[str]:
    """Requested PTA version first, then any lower available PTA versions."""
    from modules.versioning import version_candidates

    available = list_pta_versions()
    if not available:
        available = [requested]
    return version_candidates(requested, available)
