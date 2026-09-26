"""Shared cell-type labels and colours for Other Atlas species UMAPs.

Most objects store identities in ``obs['broad_cluster']``. A few aliases
(``Endothelial``, ``Immune``, empty / NaN) are normalised so the same type
keeps the same colour across species. Shared endocrine and stromal colours
match the mouse and tumour sites; fish-specific gonadotroph and somatolactotroph
splits are tints of those parents.
"""

from __future__ import annotations

# Display names used in the species selector.
SPECIES_COMMON_NAMES: dict[str, str] = {
    "Anguilla japonica": "Japanese eel",
    "Bubalus bubalis": "water buffalo",
    "Ctenopharyngodon idella": "grass carp",
    "Danio rerio": "zebrafish",
    "Felis catus": "cat",
    "Gallus gallus": "chicken",
    "Gasterosteus aculeatus": "three-spined stickleback",
    "Larimichthys crocea": "large yellow croaker",
    "Macaca fascicularis": "crab-eating macaque",
    "Oryzias latipes": "medaka",
    "Ovis aries": "sheep",
    "Panthera tigris altaica": "Amur tiger",
    "Rattus norvegicus": "rat",
    "Sus scrofa": "pig",
    "Tachysurus fulvidraco": "yellow catfish",
}

# Canonical colours, aligned with mouse / tumour cluster palettes.
OTHER_CLUSTER_COLORS: dict[str, str] = {
    "Corticotrophs": "#1f77b4",
    "Somatotrophs": "#17becf",
    "Lactotrophs": "#8c564b",
    "Thyrotrophs": "#ffbb78",
    "Gonadotrophs": "#d62728",
    "Melanotrophs": "#e377c2",
    "Stem_cells": "#aec7e8",
    "Endothelial_cells": "#ff7f0e",
    "Mesenchymal_cells": "#7f7f7f",
    "Pituicytes": "#bcbd22",
    "Erythrocytes": "#2ca02c",
    "Macrophages": "#EF553B",
    "Immune_cells": "#9467bd",
    "Other": "#B6B6B6",
    "Unclear": "#bfbdbd",
    # Fish / mixed endocrine — extrapolated from Gonadotrophs and POU1F1 types.
    "FSHB+ Gonadotrophs": "#ff7f7f",
    "LHB+ Gonadotrophs": "#8b0000",
    "SMTLA+ Somatolactotrophs": "#2e8b7a",
    "SMTLB+ Somatolactotrophs": "#5c7a4a",
    "Somatotrophs/Somatolactotrophs": "#3d9b8f",
    "Somatotrophs/Thyrotrophs/Somatolactotrophs": "#6ba3a0",
    "CGA+ cells": "#e07a5f",
    "NR5A1+ progenitors": "#c44e52",
    "PIT1 lineage": "#1E90FF",
}

_LABEL_ALIASES: dict[str, str] = {
    "endothelial": "Endothelial_cells",
    "immune": "Immune_cells",
    "nan": "Unclear",
    "none": "Unclear",
    "unknown": "Unclear",
    "": "Unclear",
}

_HIDDEN_LABELS = frozenset({"unclear"})
_FALLBACK_PALETTE = (
    "#636efa",
    "#EF553B",
    "#00cc96",
    "#ab63fa",
    "#FFA15A",
    "#19d3f3",
    "#FF6692",
    "#FECB52",
    "#7A5195",
    "#c5b0d5",
)


def normalize_other_cell_type(value: object) -> str:
    if value is None:
        return "Unclear"
    try:
        if value != value:  # NaN
            return "Unclear"
    except Exception:
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "<na>", "none", "unknown"}:
        return "Unclear"
    return _LABEL_ALIASES.get(text.lower(), text)


def is_hidden_other_cell_type(label: object) -> bool:
    return normalize_other_cell_type(label).strip().lower() in _HIDDEN_LABELS


def other_cell_type_color_map(labels) -> dict[str, str]:
    colours = dict(OTHER_CLUSTER_COLORS)
    missing = [
        str(label)
        for label in labels
        if str(label) not in colours
    ]
    seen: set[str] = set()
    fallback_i = 0
    for label in missing:
        if label in seen:
            continue
        seen.add(label)
        colours[label] = _FALLBACK_PALETTE[fallback_i % len(_FALLBACK_PALETTE)]
        fallback_i += 1
    return colours


def species_display_name(scientific_name: str) -> str:
    common = SPECIES_COMMON_NAMES.get(scientific_name)
    if common:
        return f"{scientific_name} ({common})"
    return scientific_name
