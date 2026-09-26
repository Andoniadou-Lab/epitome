"""Cell-type label normalisation and colours for PTA bulk plots."""

from __future__ import annotations

import pandas as pd

from modules.pta.config import PtaConfig

MIXED_PITNET_TERMS = frozenset(
    {
        "Somatotroph / Thyrotroph",
        "Somatotroph / Gonadotroph",
        "Somatotroph / Corticotroph",
        "Plurihormonal POU1F1",
        "Plurihormonal POU1F1+",
        "Silent POU1F1",
        "Silent POU1F1+",
        "Mixed_GH-PRL",
        "Mixed GH-PRL",
        "Undefined TBX19+ / POU1F1+",
        "Undefined POU1F1+ / NR5A1+",
        "Lactotroph / Corticotroph",
        "Corticotroph / Gonadotroph",
        "Somatotroph / Lactotroph / Thyrotroph",
        "Lactotroph / Gonadotroph",
        "Somatotroph / Lactotroph / Gonadotroph / Corticotroph",
        "Lactotroph / Thyrotroph",
        "Plurihormonal",
    }
)

# Cell_type_pta colours. Somatotroph / Lactotroph is a distinct group, not Mixed.
PTA_LINEAGE_COLORS: dict[str, str] = {
    "Somatotroph": "#1E90FF",
    "Lactotroph": "#00BFFF",
    "Somatotroph / Lactotroph": "#1e00ff",
    "Thyrotroph": "#87CEEB",
    "Corticotroph": "#f4f748",
    "Gonadotroph": "#FF0000",
    "Null-cell": "#fcb258",
    "Null_cell": "#fcb258",
    "Mixed": "#9370DB",
    "Unclear": "#bfbdbd",
    "Healthy": "#6fe339",
}

# Individual mixed subtypes when "Merge mixed pitnets" is off — update as colours are confirmed.
MIXED_SUBTYPE_COLORS: dict[str, str] = {
    "Somatotroph / Lactotroph": "#1e00ff",
    "Somatotroph / Thyrotroph": "#6CA6D9",
    "Somatotroph / Gonadotroph": "#99d96c",
    "Somatotroph / Corticotroph": "#d9996c",
    "Plurihormonal POU1F1": "#d96c7c",
    "Plurihormonal POU1F1+": "#d96c7c",
    "Silent POU1F1": "#d96c7c",
    "Silent POU1F1+": "#d96c7c",
    "Mixed_GH-PRL": "#6cd992",
    "Mixed GH-PRL": "#c3d96c",
    "Undefined TBX19+ / POU1F1+": "#6cd9c3",
    "Undefined POU1F1+ / NR5A1+": "#6ca3d9",
    "Lactotroph / Corticotroph": "#d96cd7",
    "Corticotroph / Gonadotroph": "#DC143C",
    "Somatotroph / Lactotroph / Thyrotroph": "#4682B4",
    "Lactotroph / Gonadotroph": "#8fd96c",
    "Somatotroph / Lactotroph / Gonadotroph / Corticotroph": "#71667a",
    "Lactotroph / Thyrotroph": "#40E0D0",
    "Plurihormonal": "#351e47",
}

_FALLBACK_PALETTE = [
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#dbdb8d", "#9edae5", "#393b79",
]

LINEAGE_PTA_ORDER = ["Mixed", "NR5A1", "TBX19", "POU1F1", "Healthy"]
CELL_TYPE_PTA_ORDER = [
    "Mixed",
    "Gonadotroph",
    "Corticotroph",
    "Thyrotroph",
    "Somatotroph / Lactotroph",
    "Lactotroph",
    "Somatotroph",
    "Healthy",
]
CELL_TYPE_PTA_PURE_ORDER = [
    "Gonadotroph",
    "Corticotroph",
    "Thyrotroph",
    "Somatotroph / Lactotroph",
    "Lactotroph",
    "Somatotroph",
]
CELL_TYPE_PTA_TAIL_ORDER = ["Null_cell", "Unclear", "Healthy"]
GRANULATION_PTA_ORDER = ["DG", "SG", "NG", "Unclear"]
KI67_PTA_ORDER = ["high", "low", "Unclear"]
MUTATION_PTA_ORDER = ["Mut", "WT", "Unclear"]

MUTATION_COLOR_MAP: dict[str, str] = {
    "Mut": "#ff000d",
    "Mutant": "#ff000d",
    "WT": "#5ca1fa",
    "nan": "#cccaca",
    "Unclear": "#cccaca",
    "Unknown": "#cccaca",
}

SEX_COLOR_MAP: dict[str, str] = {
    "Female": "#FFA500",
    "Male": "#63B3ED",
    "Unknown": "#B0B0B0",
}

NORMAL_STATUS_COLOR_MAP: dict[str, str] = {
    "Tumour": "#cc0000",
    "Healthy": "#6fe339",
    "Unclear": "#bfbdbd",
}

# Shared by pseudobulk boxplots, dot-plot row labels, and individual-dataset UMAPs.
# Endocrine and stromal colours are unchanged. New immune subsets are tints of the
# previous immune colours: T_cells, B_cells, Macrophages, Neutrophil, pDC_cells, Immune_cells.
PSEUDOBULK_CLUSTER_COLORS: dict[str, str] = {
    "Corticotrophs": "#1f77b4",
    "Endothelial_cells": "#ff7f0e",
    "Mesenchymal_cells": "#7f7f7f",
    "Pituicytes": "#bcbd22",
    "Stem_cells": "#aec7e8",
    "Somatotrophs": "#17becf",
    "Lactotrophs": "#8c564b",
    "Thyrotrophs": "#ffbb78",
    "Gonadotrophs": "#d62728",
    "Erythrocytes": "#2ca02c",
    "Melanotrophs": "#e377c2",
    "Intermediate_lobe": "#19d3f3",
    "Low-quality": "#bdbdbd",
    "other": "#B6B6B6",
    # Immune. Parent cluster stays purple; subsets are extrapolated from the old palette.
    "Immune_cells": "#9467bd",
    "B_cells": "#636efa",
    "Plasma_cells": "#3d4db8",
    "T_cells": "#00cc96",
    "CD4_T_cells": "#12b886",
    "CD8_T_cells": "#0e6655",
    "CD4_T_regs": "#76d7c4",
    "NK_cells": "#117a65",
    "ILCs": "#48c9b0",
    "Macrophages": "#EF553B",
    "Monocytes": "#ffab91",
    "Mast_cells": "#b23c17",
    "Dendritic_cells": "#e65100",
    "Neutrophil": "#ab63fa",
    "Neutrophils": "#ab63fa",
    "pDC_cells": "#FFA15A",
    "pDC": "#FFA15A",
}

PSEUDOBULK_IMMUNE_TERMS = frozenset(
    {
        "B_cells",
        "Plasma_cells",
        "Immune_cells",
        "Neutrophil",
        "Neutrophils",
        "pDC_cells",
        "pDC",
        "T_cells",
        "CD4_T_cells",
        "CD4_T_regs",
        "CD8_T_cells",
        "NK_cells",
        "ILCs",
        "Monocytes",
        "Mast_cells",
        "Dendritic_cells",
    }
)

GROUPING_COLORS: dict[str, dict[str, str]] = {
    "Lineage_pta": {
        "POU1F1": "#0000FF",
        "TBX19": "#dfe300",
        "NR5A1": "#DC143C",
        "Mixed": "#9370DB",
        "Unclear": "#bfbdbd",
        "Healthy": "#41cc00",
        "POU1F1 / TBX19": "#6CA6D9",
        "POU1F1 / NR5A1": "#6ca3d9",
        "NR5A1 / POU1F1": "#6ca3d9",
        "TBX19 / NR5A1": "#d96cd7",
        "NR5A1 / TBX19": "#d96cd7",
        "POU1F1 / NR5A1 / TBX19": "#71667a",
    },
    "Subtype_pta": {
        "Lactotroph": "#00BFFF",
        "Somatotroph": "#1E90FF",
        "Thyrotroph": "#87CEEB",
        "Corticotroph": "#c7c100",
        "Gonadotroph": "#FF0000",
        "Mixed GH-PRL": "#c3d96c",
        "Plurihormonal": "#351e47",
        "NF": "#fcb258",
        "Normal": "#6fe339",
        "Null_cell": "#fcb258",
        "Unclear": "#bfbdbd",
    },
    "Secretion_pta": {
        "GH": "#1E90FF",
        "PRL": "#00BFFF",
        "TSH": "#87CEEB",
        "ACTH": "#c7c100",
        "FSH/LH": "#FF0000",
        "Mixed": "#9370DB",
        "None": "#fcb258",
        "Unclear": "#bfbdbd",
    },
    "Disease_pta": {
        "Acromegaly": "#1E90FF",
        "HyperPRL": "#00BFFF",
        "HyperTSH": "#87CEEB",
        "Cushings_disease": "#c7c100",
        "Gonadotrophin": "#FF0000",
        "Unclear": "#bfbdbd",
    },
    "Invasion_pta": {
        "Yes": "#cc0000",
        "No": "#6fe339",
        "Unclear": "#bfbdbd",
    },
    "USP8_geno_pta": dict(MUTATION_COLOR_MAP),
    "GNAS_geno_pta": dict(MUTATION_COLOR_MAP),
    "Granulation_pta": {
        "DG": "#0d47a1",
        "SG": "#64b5f6",
        "NG": "#ffb74d",
        "Unclear": "#bfbdbd",
        "Unknown": "#bfbdbd",
    },
    "KI67_pta": {
        "high": "#c62828",
        "low": "#1565c0",
        "Unclear": "#cccaca",
        "Unknown": "#cccaca",
        "nan": "#cccaca",
    },
    "Lineage": {
        "Normal": "#6fe339",
        "PIT1": "#1E90FF",
        "TPIT": "#c7c100",
        "SF1": "#FF0000",
        "PIT1 / TPIT": "#6CA6D9",
        "PIT1 / SF1": "#6ca3d9",
        "SF1 / TPIT": "#d96cd7",
        "PIT1 / SF1 / TPIT": "#71667a",
        "Unclear": "#bfbdbd",
    },
    "Cell type": {
        "Lactotroph": "#00BFFF",
        "Somatotroph": "#1E90FF",
        "Thyrotroph": "#87CEEB",
        "Corticotroph": "#c7c100",
        "Gonadotroph": "#FF0000",
        "Normal": "#6fe339",
        "Unclear": "#bfbdbd",
    },
    "Subtype": {
        "Lactotroph": "#00BFFF",
        "Somatotroph": "#1E90FF",
        "Thyrotroph": "#87CEEB",
        "Corticotroph": "#c7c100",
        "Gonadotroph": "#FF0000",
        "Normal": "#6fe339",
        "Silent Corticotroph": "#d9996c",
        "Silent Gonadotroph": "#8fd96c",
        "Plurihormonal (Somatotroph / Lactotroph)": "#351e47",
        "Plurihormonal (Lactotroph / Somatotroph)": "#d96c7c",
        "Unclear": "#bfbdbd",
    },
}


def normalize_pta_category(value: object) -> str:
    """Map missing/unknown metadata values to Unclear."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "Unclear"
    text = str(value).strip()
    if not text or text.lower() in {"nan", "unknown", "na", "<na>"}:
        return "Unclear"
    if text == "Null-cell":
        return "Null_cell"
    return text


def normalize_sex_label(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "Unknown"
    text = str(value).strip().lower()
    if text in {"female", "f", "0"}:
        return "Female"
    if text in {"male", "m", "1"}:
        return "Male"
    return "Unknown"


def normalize_normal_status(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "Unclear"
    if isinstance(value, (int, float)):
        if float(value) == 1.0:
            return "Healthy"
        if float(value) == 0.0:
            return "Tumour"
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "healthy", "normal"}:
        return "Healthy"
    if text in {"0", "0.0", "false", "tumour", "tumor"}:
        return "Tumour"
    return "Unclear"


def _normalize_bulk_group_label(value: object, col_name: str) -> str:
    if col_name in {"Sex", "Sex_pta"}:
        return normalize_sex_label(value)
    if col_name == "Normal":
        return normalize_normal_status(value)
    label = normalize_pta_category(value)
    if col_name in {"Lineage_pta", "Cell_type_pta"} and label == "Normal":
        return "Healthy"
    return label


def apply_cell_type_pta(series: pd.Series, *, merge_mixed: bool = False) -> pd.Series:
    labels = series.map(lambda value: _normalize_bulk_group_label(value, "Cell_type_pta"))
    if merge_mixed:
        labels = labels.where(~labels.isin(MIXED_PITNET_TERMS), "Mixed")
    return labels


def apply_lineage_pta(series: pd.Series, *, merge_mixed: bool = False) -> pd.Series:
    labels = series.map(lambda value: _normalize_bulk_group_label(value, "Lineage_pta"))
    if merge_mixed:
        labels = labels.where(~labels.astype(str).str.contains(r" / ", regex=True), "Mixed")
    return labels


def _canonical_sort_tuple(label: str, canonical: list[str], extra_after: str = "Mixed") -> tuple:
    """Named order, with unknown levels inserted just after ``extra_after``."""
    split = canonical.index(extra_after) if extra_after in canonical else -1
    if label in canonical:
        idx = canonical.index(label)
        if idx <= split:
            return (0, idx, "")
        return (2, idx, "")
    return (1, 0, label)


def cell_type_pta_category_order(*, merge_mixed: bool = True) -> list[str]:
    """Canonical left-to-right order shared by bulk boxplot and heatmap."""
    del merge_mixed
    return list(CELL_TYPE_PTA_ORDER)


def lineage_pta_category_order() -> list[str]:
    return list(LINEAGE_PTA_ORDER)


def sort_pta_categories(
    categories,
    col_name: str,
    *,
    merge_mixed: bool = True,
) -> list[str]:
    """Sort metadata categories for bulk boxplot and heatmap axes."""
    unique = list(dict.fromkeys(_normalize_bulk_group_label(c, col_name) for c in categories))

    if col_name == "Cell_type_pta":
        canonical = cell_type_pta_category_order(merge_mixed=merge_mixed)
        return sorted(unique, key=lambda label: _canonical_sort_tuple(label, canonical))
    if col_name == "Lineage_pta":
        canonical = lineage_pta_category_order()
        return sorted(unique, key=lambda label: _canonical_sort_tuple(label, canonical))
    if col_name == "Granulation_pta":
        return sorted(
            unique, key=lambda label: _canonical_sort_tuple(label, GRANULATION_PTA_ORDER, extra_after="NG")
        )
    if col_name == "KI67_pta":
        return sorted(
            unique, key=lambda label: _canonical_sort_tuple(label, KI67_PTA_ORDER, extra_after="low")
        )
    if col_name in {"USP8_geno_pta", "GNAS_geno_pta"}:
        return sorted(
            unique, key=lambda label: _canonical_sort_tuple(label, MUTATION_PTA_ORDER, extra_after="WT")
        )

    if "Healthy" in unique:
        return ["Healthy"] + sorted(label for label in unique if label != "Healthy")
    return sorted(unique)


def _category_sort_key(value: object, col_name: str, *, merge_mixed: bool) -> tuple:
    label = _normalize_bulk_group_label(value, col_name)
    if col_name == "Cell_type_pta":
        return _canonical_sort_tuple(label, cell_type_pta_category_order(merge_mixed=merge_mixed))
    if col_name == "Lineage_pta":
        return _canonical_sort_tuple(label, lineage_pta_category_order())
    if col_name == "Granulation_pta":
        return _canonical_sort_tuple(label, GRANULATION_PTA_ORDER, extra_after="NG")
    if col_name == "KI67_pta":
        return _canonical_sort_tuple(label, KI67_PTA_ORDER, extra_after="low")
    if col_name in {"USP8_geno_pta", "GNAS_geno_pta"}:
        return _canonical_sort_tuple(label, MUTATION_PTA_ORDER, extra_after="WT")
    if label == "Healthy":
        return (0, 0, label)
    return (1, 0, label)


def ordered_sample_index(
    meta: pd.DataFrame,
    group_cols: list[str],
    *,
    merge_mixed: bool = True,
) -> list:
    """Sample order for heatmap columns using the shared bulk category order."""
    sort_cols: list[str] = []
    work = meta.copy()
    for i, col in enumerate(group_cols):
        rank_col = f"__pta_rank_{i}"
        work[rank_col] = work[col].map(
            lambda value, column=col: _category_sort_key(value, column, merge_mixed=merge_mixed)
        )
        sort_cols.append(rank_col)
    return work.sort_values(sort_cols).index.tolist()


def ordered_group_keys(
    keys,
    group_cols: list[str],
    *,
    merge_mixed: bool = True,
) -> list:
    """Sort aggregated heatmap column keys with the same rules as per-sample plots."""

    def sort_key(key) -> tuple:
        values = (key,) if len(group_cols) == 1 else tuple(key)
        parts = []
        for col_name, value in zip(group_cols, values):
            parts.append(_category_sort_key(value, col_name, merge_mixed=merge_mixed))
        return tuple(parts)

    return sorted(keys, key=sort_key)


def filter_to_selected_categories(meta: pd.DataFrame, selections: dict[str, list] | None) -> pd.DataFrame:
    """Keep rows whose labels are in the chosen levels for each grouping column."""
    if not selections:
        return meta
    keep = pd.Series(True, index=meta.index)
    for col, values in selections.items():
        if col not in meta.columns or values is None:
            continue
        allowed = {str(v) for v in values}
        keep &= meta[col].astype(str).isin(allowed)
    return meta.loc[keep]


def apply_pta_bulk_metadata_labels(
    meta: pd.DataFrame, *, merge_mixed: bool = False
) -> pd.DataFrame:
    """Normalise grouping metadata for bulk boxplot/heatmap (does not alter expression)."""
    out = meta.copy()
    for col in PtaConfig.GROUPING_COLS:
        if col in out.columns:
            out[col] = out[col].map(normalize_pta_category)
    if "Cell_type_pta" in out.columns:
        out["Cell_type_pta"] = apply_cell_type_pta(out["Cell_type_pta"], merge_mixed=merge_mixed)
    if "Lineage_pta" in out.columns:
        out["Lineage_pta"] = apply_lineage_pta(out["Lineage_pta"], merge_mixed=merge_mixed)
    return out


def apply_pta_pseudobulk_metadata_labels(meta: pd.DataFrame) -> pd.DataFrame:
    out = meta.copy()
    if "Sex" in out.columns:
        out["Sex"] = out["Sex"].map(normalize_sex_label)
    if "Normal" in out.columns:
        out["Normal"] = out["Normal"].map(normalize_normal_status)
    for col in ("Lineage", "Cell type", "Subtype", "Secretion", "Disease", "Invasion"):
        if col in out.columns:
            out[col] = out[col].map(normalize_pta_category)
    return out


# Coarse or unusable cluster labels omitted from pseudobulk plots and tumour dot plots.
HIDDEN_CLUSTER_LABELS = frozenset({"other", "low-quality", "immune_cells"})


def hidden_cluster_mask(labels) -> pd.Series:
    return labels.astype(str).str.strip().str.lower().isin(HIDDEN_CLUSTER_LABELS)


def drop_hidden_cluster_rows(meta: pd.DataFrame, *, cell_type_col: str = "broad_cluster_final") -> pd.DataFrame:
    if cell_type_col not in meta.columns:
        return meta
    return meta.loc[~hidden_cluster_mask(meta[cell_type_col])]


def drop_other_cell_type_rows(
    meta: pd.DataFrame,
    expr: pd.DataFrame,
    *,
    cell_type_col: str = "broad_cluster_final",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cell_type_col not in meta.columns:
        return meta, expr
    filtered_meta = drop_hidden_cluster_rows(meta, cell_type_col=cell_type_col)
    filtered_expr = expr[[c for c in expr.columns if c in filtered_meta.index]]
    return filtered_meta, filtered_expr


def merge_pseudobulk_immune_cell_types(
    meta: pd.DataFrame,
    *,
    merge_immune: bool = False,
    cell_type_col: str = "broad_cluster_final",
) -> pd.DataFrame:
    if not merge_immune or cell_type_col not in meta.columns:
        return meta
    out = meta.copy()
    out[cell_type_col] = out[cell_type_col].astype(str).where(
        ~out[cell_type_col].astype(str).isin(PSEUDOBULK_IMMUNE_TERMS),
        "Immune_cells",
    )
    return out


def cell_type_color_map(labels, *, merge_mixed: bool = True) -> dict[str, str]:
    """Colour map for Cell_type_pta annotation or boxplot grouping."""
    unique = sort_pta_categories(labels, "Cell_type_pta", merge_mixed=merge_mixed)
    colours: dict[str, str] = {}
    fallback_i = 0
    for label in unique:
        if label in PTA_LINEAGE_COLORS:
            colours[label] = PTA_LINEAGE_COLORS[label]
        elif label in MIXED_SUBTYPE_COLORS:
            colours[label] = MIXED_SUBTYPE_COLORS[label]
        else:
            colours[label] = _FALLBACK_PALETTE[fallback_i % len(_FALLBACK_PALETTE)]
            fallback_i += 1
    return colours


def annotation_color_maps_for_columns(
    meta: pd.DataFrame,
    group_cols: list[str],
    *,
    merge_mixed: bool = True,
) -> list[dict[str, str] | None]:
    maps: list[dict[str, str] | None] = []
    for col in group_cols:
        if col == "Cell_type_pta":
            maps.append(cell_type_color_map(meta[col], merge_mixed=merge_mixed))
        elif col in {"Sex_pta", "Sex"}:
            maps.append(SEX_COLOR_MAP)
        elif col == "broad_cluster_final":
            from modules.utils import create_color_mapping

            maps.append(create_color_mapping(meta[col]))
        else:
            maps.append(group_color_map_for_column(col, meta[col]))
    return maps


def group_color_map_for_column(
    col_name: str,
    labels,
    *,
    merge_mixed: bool = True,
) -> dict[str, str] | None:
    if col_name == "Cell_type_pta":
        return cell_type_color_map(labels, merge_mixed=merge_mixed)
    if col_name in {"Sex_pta", "Sex"}:
        return SEX_COLOR_MAP
    if col_name == "broad_cluster_final":
        from modules.utils import create_color_mapping

        return create_color_mapping(labels)
    if col_name == "Normal":
        return NORMAL_STATUS_COLOR_MAP
    base = GROUPING_COLORS.get(col_name)
    if not base:
        return None
    out = dict(base)
    values = sorted({str(v).strip() for v in labels if str(v).strip()})
    fallback_i = 0
    for value in values:
        if value not in out:
            out[value] = _FALLBACK_PALETTE[fallback_i % len(_FALLBACK_PALETTE)]
            fallback_i += 1
    return out
