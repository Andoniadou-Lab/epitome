"""PTA data loading — scRNA, bulk, pseudobulk, dotplot, and proportion data."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import streamlit as st

from modules.expression_transform import log10p1
from modules.pta.config import PtaConfig, is_current_pta_version, pta_version_candidates
from modules.pta.gene_annotation import apply_pta_gene_annotations
from modules.pta.mtx_io import load_mtx_cached
from modules.versioning import OLD_VERSION_TTL_SECONDS, record_resolved_version


@st.cache_data(ttl=OLD_VERSION_TTL_SECONDS, show_spinner="Loading older data version...")
def _old_version_data(loader_name: str, version: str, args: tuple, _loader):
    return _loader(version, *args)


@st.cache_resource(ttl=OLD_VERSION_TTL_SECONDS, show_spinner="Loading older data version...")
def _old_version_resource(loader_name: str, version: str, args: tuple, _loader):
    return _loader(version, *args)


def _versioned(current_loader, uncached_loader, version: str, *args, resource: bool = False):
    """Permanent cache for the current release, short-lived cache for older ones.

    ``resource`` must match the decorator on ``current_loader`` so callers get
    the same sharing semantics (shared object vs. per-call copy) either way.
    """
    if is_current_pta_version(version):
        return current_loader(version, *args)
    old_cache = _old_version_resource if resource else _old_version_data
    return old_cache(uncached_loader.__name__, version, args, uncached_loader)


def _pta_try(loader_key: str, requested: str, call):
    errors: list[str] = []
    for candidate in pta_version_candidates(requested):
        try:
            result = call(candidate)
            record_resolved_version(loader_key, requested, candidate)
            return result, candidate
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{candidate}: {exc}")
    detail = "; ".join(errors) if errors else "no candidates"
    raise FileNotFoundError(
        f"No PTA data for {loader_key} version {requested} (or lower). Attempts: {detail}"
    )


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


def _read_name_list(path: Path) -> list[str]:
    """One name per line; prefers the ``.parquet`` sibling written by efficient_files."""
    parquet = path.with_suffix(".parquet")
    source = parquet if parquet.is_file() else path
    return _read_index_file(source).iloc[:, 0].astype(str).tolist()


def _read_log1p_mtx(directory: Path, mtx_name: str, genes_name: str, samples_name: str) -> pd.DataFrame:
    """Genes x samples log1p matrix from normalised MTX plus gene/sample name lists."""
    mtx_path = directory / mtx_name
    for path in (mtx_path, directory / genes_name, directory / samples_name):
        if not path.is_file() and not path.with_suffix(".parquet").is_file():
            raise FileNotFoundError(path)
    matrix = load_mtx_cached(mtx_path).astype(np.float32)
    matrix.data = np.log1p(matrix.data)
    genes = _read_name_list(directory / genes_name)
    samples = _read_name_list(directory / samples_name)
    if matrix.shape != (len(genes), len(samples)):
        raise ValueError(
            f"{mtx_path.name} is {matrix.shape[0]} x {matrix.shape[1]} but lists have "
            f"{len(genes)} genes and {len(samples)} samples"
        )
    df = pd.DataFrame(matrix.toarray(), index=pd.Index(genes, name="gene"), columns=samples)
    return _collapse_duplicate_genes(df)


def _collapse_duplicate_genes(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the highest-mean row when a gene symbol appears more than once."""
    if not df.index.has_duplicates:
        return df
    ranked = df.assign(_mean=df.mean(axis=1)).sort_values("_mean", ascending=False)
    return ranked.loc[~ranked.index.duplicated(keep="first")].drop(columns="_mean")


def _log2p1(normalised: pd.DataFrame) -> pd.DataFrame:
    numeric = normalised.apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(lower=0)
    return np.log2(numeric + 1.0)


def pta_log10_expression(values, matrix_id: str) -> np.ndarray:
    """Shared log10(x + 1) scale for values read from a cached PTA matrix.

    Cached matrices are stored as log2(x + 1) (validation cohorts) or log1p (bulk and
    pseudobulk MTX); the linear values are recovered before the shared transform.
    """
    stored = np.asarray(values, dtype=np.float64)
    if matrix_id in PtaConfig.LOG2P1_MATRICES:
        linear = np.exp2(stored) - 1.0
    else:
        linear = np.expm1(stored)
    return log10p1(np.clip(linear, 0.0, None))


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


def _scrna_curation_pair(version: str) -> tuple[pd.DataFrame, str]:
    def _load(v: str) -> pd.DataFrame:
        df = pd.read_parquet(PtaConfig.curation_path(v))
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
        if "n_cells" in df.columns:
            df["n_cells"] = pd.to_numeric(df["n_cells"], errors="coerce")
        if "Age_numeric" in df.columns:
            df["Age_numeric"] = pd.to_numeric(
                df["Age_numeric"].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )
        return df

    return _pta_try("pta_scrna_curation", version, _load)


@st.cache_data(show_spinner="Loading tumour scRNA curation...")
def _load_pta_scrna_curation_pair(version: str) -> tuple[pd.DataFrame, str]:
    return _scrna_curation_pair(version)


def load_pta_scrna_curation(version: str = "v_0.04") -> pd.DataFrame:
    df, resolved = _versioned(_load_pta_scrna_curation_pair, _scrna_curation_pair, version)
    record_resolved_version("pta_scrna_curation", version, resolved)
    return df


def _metadata_pair(version: str) -> tuple[pd.DataFrame, str]:
    def _load(v: str) -> pd.DataFrame:
        df = pd.read_excel(PtaConfig.metadata_path(v))
        keep = [PtaConfig.SAMPLE_ID_COL, PtaConfig.AUTHOR_COL] + list(PtaConfig.GROUPING_COLS)
        # Granulation_pta (DG / SG / NG) and KI67_pta (high / low) are bulk grouping columns.
        if "Name" in df.columns:
            keep.append("Name")
        df = df[[c for c in keep if c in df.columns]].copy()
        df = _clean_grouping_columns(df, PtaConfig.GROUPING_COLS)
        return df.set_index(PtaConfig.SAMPLE_ID_COL)

    return _pta_try("pta_metadata", version, _load)


@st.cache_data(show_spinner="Loading tumour bulk metadata...")
def _load_pta_metadata_pair(version: str) -> tuple[pd.DataFrame, str]:
    return _metadata_pair(version)


def load_pta_metadata(version: str = "v_0.04") -> pd.DataFrame:
    df, resolved = _versioned(_load_pta_metadata_pair, _metadata_pair, version)
    record_resolved_version("pta_metadata", version, resolved)
    return df


def _bulk_curation_pair(version: str) -> tuple[pd.DataFrame, str]:
    def _load(v: str) -> pd.DataFrame:
        df = pd.read_excel(PtaConfig.metadata_path(v))
        if PtaConfig.SAMPLE_ID_COL in df.columns:
            df = df.set_index(PtaConfig.SAMPLE_ID_COL)
        if "Sex_pta" in df.columns:
            df["Sex_pta"] = df["Sex_pta"].map(PtaConfig.SEX_MAP).fillna(PtaConfig.NA_LABEL)
        return df

    return _pta_try("pta_bulk_curation", version, _load)


@st.cache_data(show_spinner="Loading tumour bulk curation table...")
def _load_pta_bulk_curation_pair(version: str) -> tuple[pd.DataFrame, str]:
    return _bulk_curation_pair(version)


def load_pta_bulk_curation(version: str = "v_0.04") -> pd.DataFrame:
    df, resolved = _versioned(_load_pta_bulk_curation_pair, _bulk_curation_pair, version)
    record_resolved_version("pta_bulk_curation", version, resolved)
    return df


def _validation_matrix(version: str, matrix: str) -> pd.DataFrame:
    cache = PtaConfig.normalised_cache_path(version, matrix)
    if cache.is_file():
        return pd.read_parquet(cache)
    raw = _collapse_duplicate_genes(_read_raw_matrix(PtaConfig.expression_path(version, matrix)))
    transformed = _log2p1(raw)
    try:
        transformed.to_parquet(cache)
    except OSError as exc:
        print(f"Could not write PTA normalised cache ({matrix}): {exc}")
    return transformed


def _bulk_matrix(version: str, matrix: str) -> pd.DataFrame:
    if matrix in PtaConfig.BULK_MTX_FILES:
        return _read_log1p_mtx(
            PtaConfig.bulk_expression_dir(version), *PtaConfig.BULK_MTX_FILES[matrix]
        )
    return _validation_matrix(version, matrix)


def _expression_pair(version: str, matrix: str) -> tuple[pd.DataFrame, str]:
    def _load(v: str) -> pd.DataFrame:
        return _bulk_matrix(v, matrix)

    return _pta_try(f"pta_expression_{matrix}", version, _load)


# Shared across sessions, not copied per call: callers must not modify it in place.
@st.cache_resource(show_spinner="Loading bulk expression...")
def _load_pta_expression_pair(version: str, matrix: str) -> tuple[pd.DataFrame, str]:
    return _expression_pair(version, matrix)


def load_pta_expression(version: str = "v_0.04", matrix: str = "shared") -> pd.DataFrame:
    """Bulk expression matrix. Main cohort is log1p-CPM; Zhang is log2(FPKM+1); Jotanovic is log2(TPM+1)."""
    df, resolved = _versioned(
        _load_pta_expression_pair, _expression_pair, version, matrix, resource=True
    )
    record_resolved_version("pta_expression", version, resolved)
    record_resolved_version(f"pta_expression_{matrix}", version, resolved)
    return df


def _cohort_primary_matrix(cohort: str) -> str:
    spec = PtaConfig.BULK_COHORTS.get(cohort, PtaConfig.BULK_COHORTS["main_cohort"])
    return spec["matrices"][0]


def load_pta_bulk_gene_universe(
    version: str = "v_0.04", cohort: str = "main_cohort"
) -> list[str]:
    """Gene symbols for ``cohort``. Main cohort: shared first, then just-aligned extras."""
    if cohort == "main_cohort":
        shared = load_pta_expression(version, "shared")
        aligned = load_pta_expression(version, "just_aligned")
        extras = [g for g in aligned.index if g not in shared.index]
        return list(shared.index) + extras
    matrix = _cohort_primary_matrix(cohort)
    return list(load_pta_expression(version, matrix).index)


def resolve_bulk_expression_for_genes(
    version: str,
    genes: list[str] | None = None,
    cohort: str = "main_cohort",
) -> tuple[pd.DataFrame, str, list[str]]:
    """Pick the bulk matrix for ``genes`` within ``cohort``.

    Main cohort prefers the shared-gene matrix (more samples). If any requested
    gene is missing there, use the just-aligned matrix instead. Validation
    cohorts have a single matrix (Zhang FPKM, Jotanovic TPM). Returns
    ``(expression, matrix_id, missing_genes)``.
    """
    if cohort != "main_cohort":
        matrix = _cohort_primary_matrix(cohort)
        expr = load_pta_expression(version, matrix)
        if not genes:
            return expr, matrix, []
        wanted = [g for g in genes if g]
        missing = [g for g in wanted if g not in expr.index]
        return expr, matrix, missing

    shared = load_pta_expression(version, "shared")
    if not genes:
        return shared, "shared", []
    wanted = [g for g in genes if g]
    missing_shared = [g for g in wanted if g not in shared.index]
    if not missing_shared:
        return shared, "shared", []
    aligned = load_pta_expression(version, "just_aligned")
    missing_aligned = [g for g in wanted if g not in aligned.index]
    return aligned, "just_aligned", missing_aligned


def _dotplot_pair(version: str):
    def _load(v: str):
        root = PtaConfig.dotplot_dir(v)
        proportion_matrix = load_mtx_cached(root / "matrix2.mtx", repair=True)
        expression_matrix = load_mtx_cached(root / "matrix1.mtx", repair=False)
        genes1 = _read_index_file(root / "matrix1_genes.tsv")
        genes2 = _read_index_file(root / "matrix2_genes.tsv")
        rows1 = _read_index_file(root / "matrix1_rows.tsv")
        rows2 = _read_index_file(root / "matrix2_rows.tsv")
        return proportion_matrix, genes1, rows1, expression_matrix, genes2, rows2

    return _pta_try("pta_dotplot", version, _load)


@st.cache_resource(show_spinner="Loading dotplot matrices...")
def _load_pta_dotplot_pair(version: str):
    return _dotplot_pair(version)


def load_pta_dotplot_data(version: str = "v_0.04"):
    data, resolved = _versioned(_load_pta_dotplot_pair, _dotplot_pair, version, resource=True)
    record_resolved_version("pta_dotplot", version, resolved)
    return data


def _proportion_pair(version: str):
    def _load(v: str):
        root = PtaConfig.cell_proportion_dir(v)
        abundance_matrix = scipy.io.mmread(root / "abundance.mtx")
        abundance_rows = pd.read_csv(root / "abundance_rows.tsv", sep="\t", header=None)
        abundance_cols = pd.read_csv(root / "abundance_cols.tsv", sep="\t", header=None)
        return abundance_matrix, abundance_rows, abundance_cols

    return _pta_try("pta_proportion", version, _load)


@st.cache_resource(show_spinner="Loading cell proportion data...")
def _load_pta_proportion_pair(version: str):
    return _proportion_pair(version)


def load_pta_proportion_data(version: str = "v_0.04"):
    data, resolved = _versioned(_load_pta_proportion_pair, _proportion_pair, version, resource=True)
    record_resolved_version("pta_proportion", version, resolved)
    return data


def _pseudobulk_tables_pair(version: str) -> tuple[tuple[pd.DataFrame, pd.DataFrame], str]:
    def _load(v: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        expr = _read_log1p_mtx(PtaConfig.pseudobulk_dir(v), *PtaConfig.PSEUDOBULK_MTX_FILES)
        curation = pd.read_parquet(PtaConfig.curation_path(v))
        return expr, pseudobulk_metadata(expr.columns, curation)

    return _pta_try("pta_pseudobulk_tables", version, _load)


# Shared across sessions, not copied per call: callers must not modify it in place.
@st.cache_resource(show_spinner="Preparing pseudobulk expression...")
def _load_pta_pseudobulk_tables_pair(version: str) -> tuple[tuple[pd.DataFrame, pd.DataFrame], str]:
    return _pseudobulk_tables_pair(version)


def load_pta_pseudobulk_tables(version: str = "v_0.04") -> tuple[pd.DataFrame, pd.DataFrame]:
    tables, resolved = _versioned(
        _load_pta_pseudobulk_tables_pair, _pseudobulk_tables_pair, version, resource=True
    )
    record_resolved_version("pta_pseudobulk_tables", version, resolved)
    return tables


def align_bulk_samples(
    expr: pd.DataFrame, meta: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    shared = [s for s in expr.columns if s in meta.index]
    return expr[shared], meta.loc[shared]


def pseudobulk_metadata(profiles: pd.Index, curation: pd.DataFrame) -> pd.DataFrame:
    """Per-profile metadata: cell type from the profile name, sample fields from curation."""
    parts = pd.Series(profiles, index=profiles).str.split(PtaConfig.PSEUDOBULK_NAME_SEP, n=1)
    unparsed = parts.str.len() != 2
    if unparsed.any():
        raise ValueError(
            f"Pseudobulk names must be <Internal_ID>{PtaConfig.PSEUDOBULK_NAME_SEP}<cell type>; "
            f"got e.g. {list(profiles[unparsed.to_numpy()][:3])}"
        )
    sample_info = curation.drop_duplicates(PtaConfig.SAMPLE_ID_COL).set_index(
        PtaConfig.SAMPLE_ID_COL
    )
    meta = pd.DataFrame(
        {
            PtaConfig.SAMPLE_ID_COL: parts.str[0],
            "broad_cluster_final": parts.str[1],
        },
        index=profiles,
    )
    meta = meta.join(sample_info, on=PtaConfig.SAMPLE_ID_COL)
    if "SRA_ID" in meta.columns:
        meta.insert(0, "sample", meta["SRA_ID"])
    if "Sex" in meta.columns:
        meta["Sex"] = meta["Sex"].astype(str)
    missing = {label: PtaConfig.NA_LABEL for label in ("nan", "None", "Null", "")}
    for col in PtaConfig.PSEUDOBULK_GROUPING_COLS:
        if col in meta.columns and col != "Sex":
            meta[col] = meta[col].astype(str).replace(missing)
    return meta


def filter_by_author(meta: pd.DataFrame, studies: list[str] | None) -> pd.Index:
    if studies is None or PtaConfig.AUTHOR_COL not in meta.columns:
        return meta.index
    return meta.index[meta[PtaConfig.AUTHOR_COL].isin(studies)]


def _parse_volcano_manifest(data: dict) -> list[dict]:
    """Families with nested comparisons, wrapping a legacy flat list if needed."""
    if "families" in data:
        return data["families"]
    return [
        {
            "id": "lineage",
            "name": "Lineages",
            "description": "",
            "comparisons": data.get("comparisons", []),
        }
    ]


def flatten_volcano_comparisons(families: list[dict]) -> list[dict]:
    """Flat pairwise comparison dicts, each carrying ``family_id`` / ``family_name``."""
    return flatten_volcano_entries(families, "comparisons")


def flatten_volcano_markers(families: list[dict]) -> list[dict]:
    """Flat one-group marker dicts, each carrying ``family_id`` / ``family_name``."""
    return flatten_volcano_entries(families, "markers")


def flatten_volcano_entries(families: list[dict], key: str) -> list[dict]:
    out: list[dict] = []
    for family in families:
        for entry in family.get(key) or []:
            out.append(
                {
                    **entry,
                    "family_id": family.get("id"),
                    "family_name": family.get("name"),
                    "family_description": family.get("description", ""),
                    "kind": entry.get("kind") or ("markers" if key == "markers" else "contrast"),
                }
            )
    return out


def _read_volcano_table(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "gene" in df.columns:
        df = df.drop_duplicates("gene", keep="first")
    return df


def _load_marker_volcano_table(entry: dict, volcano_dir) -> pd.DataFrame:
    frames = []
    for key in ("pos_file", "neg_file"):
        relative = entry.get(key)
        if not relative:
            continue
        path = volcano_dir / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        frames.append(_read_volcano_table(path))
    if not frames:
        raise FileNotFoundError(f"No marker files for {entry.get('id')}")
    combined = pd.concat(frames, ignore_index=True)
    if "gene" in combined.columns:
        if "adj.P.Val" in combined.columns:
            combined = combined.sort_values("adj.P.Val")
        combined = combined.drop_duplicates("gene", keep="first")
    return combined.reset_index(drop=True)


def _volcano_manifest_pair(version: str) -> tuple[list[dict], str]:
    def _load(v: str) -> list[dict]:
        path = PtaConfig.volcano_manifest_path(v)
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return _parse_volcano_manifest(data)

    return _pta_try("pta_volcano_manifest", version, _load)


@st.cache_data(show_spinner="Loading volcano comparisons...")
def _load_volcano_manifest_pair(version: str) -> tuple[list[dict], str]:
    return _volcano_manifest_pair(version)


def load_volcano_manifest(version: str = "v_0.04") -> list[dict]:
    data, resolved = _versioned(_load_volcano_manifest_pair, _volcano_manifest_pair, version)
    record_resolved_version("pta_volcano_manifest", version, resolved)
    return data


def _volcano_results_pair(version: str, comparison_id: str) -> tuple[pd.DataFrame, str]:
    def _load(v: str) -> pd.DataFrame:
        path = PtaConfig.volcano_manifest_path(v)
        with open(path, encoding="utf-8") as fh:
            families = _parse_volcano_manifest(json.load(fh))
        for entry in flatten_volcano_comparisons(families) + flatten_volcano_markers(families):
            if entry["id"] == comparison_id:
                if entry.get("kind") == "markers" or entry.get("pos_file"):
                    df = _load_marker_volcano_table(entry, PtaConfig.volcano_dir(v))
                else:
                    csv_path = PtaConfig.volcano_dir(v) / entry["file"]
                    df = _read_volcano_table(csv_path)
                return apply_pta_gene_annotations(df, v)
        raise FileNotFoundError(f"Unknown volcano comparison: {comparison_id}")

    return _pta_try("pta_volcano_results", version, _load)


@st.cache_data(show_spinner="Loading volcano results...")
def _load_volcano_results_pair(version: str, comparison_id: str) -> tuple[pd.DataFrame, str]:
    return _volcano_results_pair(version, comparison_id)


def load_volcano_results(version: str, comparison_id: str) -> pd.DataFrame:
    df, resolved = _versioned(
        _load_volcano_results_pair, _volcano_results_pair, version, comparison_id
    )
    record_resolved_version("pta_volcano_results", version, resolved)
    return df