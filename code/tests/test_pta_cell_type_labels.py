"""Tests for PTA cell-type label normalisation and colours."""

from __future__ import annotations

import pandas as pd

from modules.pta.cell_type_labels import (
    MIXED_PITNET_TERMS,
    apply_cell_type_pta,
    apply_lineage_pta,
    apply_pta_bulk_metadata_labels,
    cell_type_color_map,
    cell_type_pta_category_order,
    group_color_map_for_column,
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


def test_merge_mixed_pitnets_keeps_somatotroph_lactotroph():
    series = pd.Series(["Somatotroph / Lactotroph", "Lactotroph", "Plurihormonal", "Silent POU1F1+"])
    merged = apply_cell_type_pta(series, merge_mixed=True)
    assert merged.tolist() == ["Somatotroph / Lactotroph", "Lactotroph", "Mixed", "Mixed"]
    assert "Somatotroph / Lactotroph" not in MIXED_PITNET_TERMS


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
    assert out.loc[1, "Cell_type_pta"] == "Somatotroph / Lactotroph"
    assert out.loc[0, "Lineage_pta"] == "Unclear"
    assert out.loc[1, "Disease_pta"] == "Acromegaly"


def test_merge_mixed_lineages_collapses_slash_labels():
    series = pd.Series(["POU1F1", "NR5A1 / TBX19", "Normal"])
    merged = apply_lineage_pta(series, merge_mixed=True)
    assert merged.tolist() == ["POU1F1", "Mixed", "Healthy"]


def test_cell_type_color_map_uses_requested_palette():
    colours = cell_type_color_map(
        ["Lactotroph", "Mixed", "Unclear", "Corticotroph", "Somatotroph / Lactotroph"]
    )
    assert colours["Lactotroph"] == "#00BFFF"
    assert colours["Mixed"] == "#9370DB"
    assert colours["Unclear"] == "#bfbdbd"
    assert colours["Corticotroph"] == "#f4f748"
    assert colours["Somatotroph / Lactotroph"] == "#1e00ff"


def test_all_mixed_terms_have_subtype_colours():
    assert MIXED_PITNET_TERMS.issubset(set(__import__(
        "modules.pta.cell_type_labels", fromlist=["MIXED_SUBTYPE_COLORS"]
    ).MIXED_SUBTYPE_COLORS))


def test_cell_type_order_puts_mixed_first_and_healthy_last():
    order = sort_pta_categories(
        ["Mixed", "Healthy", "Lactotroph", "Somatotroph", "Unclear", "Gonadotroph"],
        "Cell_type_pta",
        merge_mixed=True,
    )
    assert order[0] == "Mixed"
    assert order[-1] == "Healthy"
    assert order.index("Gonadotroph") < order.index("Lactotroph")
    assert order.index("Unclear") == 1


def test_lineage_order_inserts_extras_between_mixed_and_nr5a1():
    order = sort_pta_categories(
        ["POU1F1", "Healthy", "Mixed", "Unclear", "NR5A1", "TBX19"],
        "Lineage_pta",
    )
    assert order == ["Mixed", "Unclear", "NR5A1", "TBX19", "POU1F1", "Healthy"]
    colours = group_color_map_for_column("Lineage_pta", order)
    assert colours["POU1F1"] == "#0000FF"
    assert colours["TBX19"] == "#dfe300"
    assert colours["NR5A1"] == "#DC143C"
    assert colours["Healthy"] == "#41cc00"
    assert colours["Mixed"] == "#9370DB"


def test_ordered_sample_index_matches_boxplot_and_heatmap():
    meta = pd.DataFrame(
        {
            "Cell_type_pta": ["Lactotroph", "Healthy", "Mixed", "Somatotroph"],
        },
        index=["s3", "s1", "s4", "s2"],
    )
    meta["Cell_type_pta"] = apply_cell_type_pta(meta["Cell_type_pta"], merge_mixed=True)
    ordered = ordered_sample_index(meta, ["Cell_type_pta"], merge_mixed=True)
    assert ordered == ["s4", "s3", "s2", "s1"]


def test_merged_default_order_is_shared():
    merged = cell_type_pta_category_order(merge_mixed=True)
    assert merged[0] == "Mixed"
    assert merged[-1] == "Healthy"
    assert "Somatotroph / Lactotroph" in merged


def test_granulation_pta_is_a_grouping_column():
    from modules.pta.config import PtaConfig

    assert "Granulation_pta" in PtaConfig.GROUPING_COLS
    order = sort_pta_categories(
        ["Unclear", "SG", "NG", "DG"],
        "Granulation_pta",
    )
    assert order == ["DG", "SG", "NG", "Unclear"]
    colours = group_color_map_for_column("Granulation_pta", order)
    assert colours["DG"] == "#0d47a1"
    assert colours["SG"] == "#64b5f6"
    assert colours["NG"] == "#ffb74d"
    assert colours["Unclear"] == "#bfbdbd"


def test_ki67_and_mutation_grouping_colours():
    from modules.pta.config import PtaConfig

    assert "KI67_pta" in PtaConfig.GROUPING_COLS
    ki67_order = sort_pta_categories(["low", "Unclear", "high"], "KI67_pta")
    assert ki67_order == ["high", "low", "Unclear"]
    ki67_colours = group_color_map_for_column("KI67_pta", ki67_order)
    assert ki67_colours["high"] == "#c62828"
    assert ki67_colours["low"] == "#1565c0"

    mut_order = sort_pta_categories(["WT", "Unclear", "Mut"], "GNAS_geno_pta")
    assert mut_order == ["Mut", "WT", "Unclear"]
    for col in ("GNAS_geno_pta", "USP8_geno_pta"):
        colours = group_color_map_for_column(col, mut_order)
        assert colours["Mut"] == "#ff000d"
        assert colours["WT"] == "#5ca1fa"
        assert colours["Unclear"] == "#cccaca"


def test_filter_to_selected_categories_keeps_chosen_levels():
    from modules.pta.cell_type_labels import filter_to_selected_categories

    meta = pd.DataFrame(
        {
            "Cell_type_pta": ["Lactotroph", "Somatotroph", "Lactotroph", "Gonadotroph"],
            "Lineage_pta": ["POU1F1", "POU1F1", "POU1F1", "NR5A1"],
        },
        index=["s1", "s2", "s3", "s4"],
    )
    out = filter_to_selected_categories(
        meta, {"Cell_type_pta": ["Lactotroph", "Gonadotroph"]}
    )
    assert list(out.index) == ["s1", "s3", "s4"]
    out = filter_to_selected_categories(
        meta, {"Cell_type_pta": ["Lactotroph"], "Lineage_pta": ["POU1F1"]}
    )
    assert list(out.index) == ["s1", "s3"]
