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


def test_cluster_palette_assigns_every_new_immune_population():
    from modules.pta.cell_type_labels import (
        PSEUDOBULK_CLUSTER_COLORS,
        PSEUDOBULK_IMMUNE_TERMS,
        merge_pseudobulk_immune_cell_types,
    )
    from modules.utils import create_color_mapping

    expected = {
        "Immune_cells": "#9467bd",
        "B_cells": "#636efa",
        "Plasma_cells": "#3d4db8",
        "T_cells": "#00cc96",
        "CD4_T_cells": "#12b886",
        "CD8_T_cells": "#0e6655",
        "CD4_T_regs": "#76d7c4",
        "NK_cells": "#117a65",
        "Macrophages": "#EF553B",
        "Monocytes": "#ffab91",
        "Dendritic_cells": "#e65100",
        "Neutrophil": "#ab63fa",
        "Neutrophils": "#ab63fa",
        "pDC_cells": "#FFA15A",
        "pDC": "#FFA15A",
        "Corticotrophs": "#1f77b4",
        "Somatotrophs": "#17becf",
        "Erythrocytes": "#2ca02c",
        "Low-quality": "#bdbdbd",
    }
    for label, colour in expected.items():
        assert PSEUDOBULK_CLUSTER_COLORS[label] == colour
        assert create_color_mapping([label])[label] == colour

    colours = group_color_map_for_column("broad_cluster_final", list(expected))
    assert colours["CD4_T_cells"] == "#12b886"
    assert "Macrophages" not in PSEUDOBULK_IMMUNE_TERMS
    meta = pd.DataFrame(
        {"broad_cluster_final": ["CD4_T_cells", "Macrophages", "Somatotrophs"]}
    )
    merged = merge_pseudobulk_immune_cell_types(meta, merge_immune=True)
    assert merged["broad_cluster_final"].tolist() == [
        "Immune_cells",
        "Macrophages",
        "Somatotrophs",
    ]


def test_pta_datasets_read_cell_type_and_share_cluster_colours():
    import numpy as np
    from modules.dotplot import create_dotplot
    from modules.pta.boxplot import create_pta_boxplot
    from modules.pta.cell_type_labels import drop_other_cell_type_rows
    from modules.pta.individual_sc import pta_cell_type_column

    obs = pd.DataFrame(
        {"new_cell_type": ["Immune_cells"], "cell_type": ["CD4_T_cells"]}
    )
    assert pta_cell_type_column(obs) == "cell_type"

    rows = pd.DataFrame({0: ["SRX1_CD4_T_cells", "SRX1_CD8_T_cells"]})
    genes = pd.DataFrame({0: ["GH1"]})
    matrix = np.array([[0.5], [0.2]])
    fig, _config = create_dotplot(matrix, matrix, genes, genes, rows, rows, ["GH1"])
    assert fig.layout.yaxis.tickfont.color is None
    assert fig.layout.yaxis.tickfont.size == 25
    assert "CD4_T_cells" in set(fig.data[0].y)

    merged_rows = pd.DataFrame(
        {0: ["SRX1_CD4_T_cells", "SRX1_CD8_T_cells", "SRX1_Somatotrophs", "SRX1_Macrophages"]}
    )
    merged_matrix = np.array([[0.5], [0.2], [0.8], [0.1]])
    merged_fig, _config = create_dotplot(
        merged_matrix,
        merged_matrix,
        genes,
        genes,
        merged_rows,
        merged_rows,
        ["GH1"],
        merge_immune=True,
    )
    assert set(merged_fig.data[0].y) == {"Immune_cells", "Somatotrophs", "Macrophages"}

    many_rows = pd.DataFrame({0: [f"SRX1_Type_{i}" for i in range(15)]})
    many_matrix = np.ones((15, 1))
    fitted, _config = create_dotplot(
        many_matrix,
        many_matrix,
        genes,
        genes,
        many_rows,
        many_rows,
        ["GH1"],
        fit_cell_type_labels=True,
    )
    assert fitted.layout.yaxis.tickfont.size == 17
    assert fitted.layout.height == 860
    assert fitted.layout.yaxis.ticklabeloverflow == "allow"

    meta = pd.DataFrame(
        {
            "broad_cluster_final": [
                "CD4_T_cells",
                "Immune_cells",
                "Low-quality",
                "Somatotrophs",
                "other",
            ]
        },
        index=["a", "b", "c", "d", "e"],
    )
    expr = pd.DataFrame([[1, 1, 1, 1, 1]], index=["GH1"], columns=list("abcde"))
    kept, kept_expr = drop_other_cell_type_rows(meta, expr)
    assert list(kept["broad_cluster_final"]) == ["CD4_T_cells", "Somatotrophs"]
    assert list(kept_expr.columns) == ["a", "d"]

    plot_df = pd.DataFrame(
        {
            "broad_cluster_final": ["CD4_T_cells", "Somatotrophs"] * 4,
            "Expression": list(range(8)),
        }
    )
    colours = group_color_map_for_column(
        "broad_cluster_final", plot_df["broad_cluster_final"]
    )
    box_fig, _ = create_pta_boxplot(
        plot_df, "GH1", "broad_cluster_final", color_map=colours
    )
    percentile_boxes = [
        trace
        for trace in box_fig.data
        if trace.type == "box" and not trace.boxpoints
    ]
    by_name = {trace.name: str(trace.line.color).replace(" ", "") for trace in percentile_boxes}
    assert "rgb(18,184,134)" in by_name["CD4_T_cells"]
    assert "rgb(23,190,207)" in by_name["Somatotrophs"]
    assert all(trace.fillcolor and "0.45" in str(trace.fillcolor) for trace in percentile_boxes)
