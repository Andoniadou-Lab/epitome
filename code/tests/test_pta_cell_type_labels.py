"""Tests for PTA cell-type label normalisation and colours."""

from __future__ import annotations

import pandas as pd

from modules.pta.cell_type_labels import (
    MIXED_PITNET_TERMS,
    apply_cell_type_pta,
    apply_pta_bulk_metadata_labels,
    cell_type_color_map,
    cell_type_pta_category_order,
    normalize_pta_category,
    ordered_sample_index,
    sort_pta_categories,
)


def test_normalize_pta_category_maps_missing_to_unclear():
    assert normalize_pta_category(None) == "Unclear"
    assert normalize_pta_category(float("nan")) == "Unclear"
    assert normalize_pta_category("Unknown") == "Unclear"
    assert normalize_pta_category("Null-cell") == "Null_cell"
    assert normalize_pta_category("Lactotroph") == "Lactotroph"


def test_merge_mixed_pitnets():
    series = pd.Series(["Somatotroph / Lactotroph", "Lactotroph", "Plurihormonal"])
    merged = apply_cell_type_pta(series, merge_mixed=True)
    assert merged.tolist() == ["Mixed", "Lactotroph", "Mixed"]


def test_apply_pta_bulk_metadata_labels_does_not_touch_other_columns():
    meta = pd.DataFrame(
        {
            "Cell_type_pta": ["Unknown", "Somatotroph / Lactotroph"],
            "Lineage_pta": [None, "POU1F1"],
            "Disease_pta": ["Acromegaly", "Acromegaly"],
        }
    )
    out = apply_pta_bulk_metadata_labels(meta, merge_mixed=True)
    assert out.loc[0, "Cell_type_pta"] == "Unclear"
    assert out.loc[1, "Cell_type_pta"] == "Mixed"
    assert out.loc[0, "Lineage_pta"] == "Unclear"
    assert out.loc[1, "Disease_pta"] == "Acromegaly"


def test_cell_type_color_map_uses_lineage_palette():
    colours = cell_type_color_map(["Lactotroph", "Mixed", "Unclear"])
    assert colours["Lactotroph"] == "#00BFFF"
    assert colours["Mixed"] == "#9370DB"
    assert colours["Unclear"] == "#bfbdbd"


def test_all_mixed_terms_have_subtype_colours():
    assert MIXED_PITNET_TERMS.issubset(set(__import__(
        "modules.pta.cell_type_labels", fromlist=["MIXED_SUBTYPE_COLORS"]
    ).MIXED_SUBTYPE_COLORS))


def test_cell_type_order_puts_healthy_first():
    order = sort_pta_categories(
        ["Mixed", "Healthy", "Lactotroph", "Somatotroph"],
        "Cell_type_pta",
        merge_mixed=True,
    )
    assert order[0] == "Healthy"
    assert order.index("Mixed") > order.index("Lactotroph")


def test_ordered_sample_index_matches_boxplot_and_heatmap():
    meta = pd.DataFrame(
        {
            "Cell_type_pta": ["Lactotroph", "Healthy", "Mixed", "Somatotroph"],
        },
        index=["s3", "s1", "s4", "s2"],
    )
    meta["Cell_type_pta"] = apply_cell_type_pta(meta["Cell_type_pta"], merge_mixed=True)
    ordered = ordered_sample_index(meta, ["Cell_type_pta"], merge_mixed=True)
    assert ordered == ["s1", "s3", "s2", "s4"]


def test_merged_default_order_is_shared():
    merged = cell_type_pta_category_order(merge_mixed=True)
    assert merged[0] == "Healthy"
    assert "Mixed" in merged
