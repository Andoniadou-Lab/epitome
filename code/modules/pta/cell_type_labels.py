"""Cell-type label normalisation and colours for PTA bulk plots."""

from __future__ import annotations

import pandas as pd

from modules.pta.config import PtaConfig

MIXED_PITNET_TERMS = frozenset(
    {
        "Somatotroph / Lactotroph",
        "Somatotroph / Thyrotroph",
        "Somatotroph / Gonadotroph",
        "Somatotroph / Corticotroph",
        "Plurihormonal POU1F1",
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

PTA_LINEAGE_COLORS: dict[str, str] = {
    "Lactotroph": "#00BFFF",
    "Mixed": "#9370DB",
    "Somatotroph": "#1E90FF",
    "Thyrotroph": "#87CEEB",
    "Corticotroph": "#c7c100",
    "Gonadotroph": "#FF0000",
    "Null_cell": "#fcb258",
    "Unclear": "#bfbdbd",
    "Healthy": "#6fe339",
}

# Individual mixed subtypes when "Merge mixed pitnets" is off — update as colours are confirmed.
MIXED_SUBTYPE_COLORS: dict[str, str] = {
    "Somatotroph / Lactotroph": "#5B9BD5",
    "Somatotroph / Thyrotroph": "#6CA6D9",
    "Somatotroph / Gonadotroph": "#99d96c",
    "Somatotroph / Corticotroph": "#d9996c",
    "Plurihormonal POU1F1": "#d96c7c",
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

CELL_TYPE_PTA_PURE_ORDER = [
    "Healthy",
    "Lactotroph",
    "Somatotroph",
    "Thyrotroph",
    "Corticotroph",
    "Gonadotroph",
]

CELL_TYPE_PTA_TAIL_ORDER = ["Null_cell", "Unclear"]

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

PSEUDOBULK_CLUSTER_COLORS: dict[str, str] = {
    "Corticotrophs": "#1f77b4",
    "Endothelial_cells": "#ff7f0e",
    "Immune_cells": "#9467bd",
    "Mesenchymal_cells": "#7f7f7f",
    "Pituicytes": "#bcbd22",
    "Stem_cells": "#aec7e8",
    "Somatotrophs": "#17becf",
    "Lactotrophs": "#8c564b",
    "Thyrotrophs": "#ffbb78",
    "Gonadotrophs": "#d62728",
    "B_cells": "#636efa",
    "Macrophages": "#EF553B",
    "T_cells": "#00cc96",
    "Neutrophil": "#ab63fa",
    "pDC_cells": "#FFA15A",
}

PSEUDOBULK_IMMUNE_TERMS = frozenset(
    {"B_cells", "Immune_cells", "Neutrophil", "pDC_cells", "T_cells"}
)

GROUPING_COLORS: dict[str, dict[str, str]] = {
    "Lineage_pta": {
        "Healthy": "#6fe339",
        "POU1F1": "#1E90FF",
        "TBX19": "#c7c100",
        "NR5A1": "#FF0000",
        "POU1F1 / TBX19": "#6CA6D9",
        "POU1F1 / NR5A1": "#6ca3d9",
        "TBX19 / NR5A1": "#d96cd7",
        "POU1F1 / NR5A1 / TBX19": "#71667a",
        "Unclear": "#bfbdbd",
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
    "USP8_geno_pta": {
        "Mutant": "#cc0000",
        "WT": "#6fe339",
        "Unclear": "#bfbdbd",
    },
    "GNAS_geno_pta": {
        "Mutant": "#cc0000",
        "WT": "#6fe339",
        "Unclear": "#bfbdbd",
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


def apply_cell_type_pta(series: pd.Series, *, merge_mixed: bool = False) -> pd.Series:
    labels = series.map(normalize_pta_category)
    if merge_mixed:
        labels = labels.where(~labels.isin(MIXED_PITNET_TERMS), "Mixed")
    return labels


def cell_type_pta_category_order(*, merge_mixed: bool) -> list[str]:
    """Canonical left-to-right order shared by bulk boxplot and heatmap."""
    if merge_mixed:
        return [*CELL_TYPE_PTA_PURE_ORDER, "Mixed", *CELL_TYPE_PTA_TAIL_ORDER]
    return [
        *CELL_TYPE_PTA_PURE_ORDER,
        *MIXED_SUBTYPE_COLORS.keys(),
        *CELL_TYPE_PTA_TAIL_ORDER,
    ]


def sort_pta_categories(
    categories,
    col_name: str,
    *,
    merge_mixed: bool = True,
) -> list[str]:
    """Sort metadata categories with Healthy first; shared across bulk plots."""
    if col_name in {"Sex", "Sex_pta"}:
        normalized = [normalize_sex_label(c) for c in categories]
    elif col_name == "Normal":
        normalized = [normalize_normal_status(c) for c in categories]
    else:
        normalized = [normalize_pta_category(c) for c in categories]
    unique = list(dict.fromkeys(normalized))

    if col_name == "Cell_type_pta":
        canonical = cell_type_pta_category_order(merge_mixed=merge_mixed)
        rank = {label: i for i, label in enumerate(canonical)}
        default_rank = len(canonical)
        return sorted(unique, key=lambda label: (rank.get(label, default_rank), label))

    if "Healthy" in unique:
        return ["Healthy"] + sorted(label for label in unique if label != "Healthy")
    return sorted(unique)


def _category_sort_key(value: object, col_name: str, *, merge_mixed: bool) -> tuple:
    if col_name in {"Sex", "Sex_pta"}:
        label = normalize_sex_label(value)
    elif col_name == "Normal":
        label = normalize_normal_status(value)
    else:
        label = normalize_pta_category(value)
    if col_name == "Cell_type_pta":
        order = cell_type_pta_category_order(merge_mixed=merge_mixed)
        rank = {name: i for i, name in enumerate(order)}
        return (rank.get(label, len(order)), label)
    if label == "Healthy":
        return (0, label)
    return (1, label)


def ordered_sample_index(
    meta: pd.DataFrame,
    group_cols: list[str],
    *,
    merge_mixed: bool = True,
) -> list:
    """Sample order for heatmap columns: Healthy left, then canonical cell types."""
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


def drop_other_cell_type_rows(
    meta: pd.DataFrame,
    expr: pd.DataFrame,
    *,
    cell_type_col: str = "broad_cluster_final",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cell_type_col not in meta.columns:
        return meta, expr
    keep = meta[cell_type_col].astype(str).str.strip().str.lower() != "other"
    filtered_meta = meta.loc[keep]
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
            maps.append(PSEUDOBULK_CLUSTER_COLORS)
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
        return PSEUDOBULK_CLUSTER_COLORS
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
