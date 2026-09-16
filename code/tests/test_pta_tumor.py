"""Smoke tests for human pituitary tumour atlas (PTA) modules."""

from __future__ import annotations

from pathlib import Path

import pytest

from config import Config

CODE_DIR = Path(__file__).resolve().parent.parent
TUMOR_PAGES = list((CODE_DIR / "app_pages" / "tumor").glob("*.py"))
V = "v_0.04"
PTA = Config.BASE_PATH / "pta_data"


def test_pta_data_v004_present():
    required = [
        PTA / "curation" / V / "cpa.parquet",
        PTA / "dotplot" / V / "matrix1.mtx",
        PTA / "dotplot" / V / "matrix2.mtx",
        PTA / "cell_proportion" / V / "abundance.mtx",
        PTA / "overview" / V / "rna_cell_type_counts.parquet",
        PTA / "bulk_curation" / V / "pituitary_tumor_atlas_bulk_updated_final.xlsx",
        PTA / "bulk_expression" / V / "concatted_matrix_shared.csv",
        PTA / "bulk_expression" / V / "concatted_matrix_just_aligned.csv",
        PTA / "bulk_expression" / V / "zhang.parquet",
        PTA / "bulk_expression" / V / "jotanovic.parquet",
        PTA / "sc_data" / "datasets" / V / "epitome_h5_files" / "HRS1408776.h5ad",
        PTA / "epitome_volcanos" / V / "volcanos.json",
        PTA / "epitome_volcanos" / V / "dream_outputs_merged" / "01_lineage" / "contrasts" / "dream_NR5A1_vs_POU1F1.csv",
        PTA / "gene_group_annotation" / "target_prioritisation_druggable.parquet",
        PTA / "gene_group_annotation" / "lambert_human_tfs.parquet",
    ]
    missing = [str(p.relative_to(PTA)) for p in required if not p.is_file()]
    pseudobulk_h5ad = list((PTA / "pseudobulk" / V).glob("*.h5ad"))
    if not pseudobulk_h5ad:
        missing.append(f"pseudobulk/{V}/*.h5ad")
    assert not missing, f"Missing PTA data files: {missing}"


def test_list_pta_versions():
    from modules.pta.config import list_pta_versions

    versions = list_pta_versions()
    assert versions == ["v_0.04"]


def test_tumor_pages_import_cleanly():
    from legacy_parity_support import _BUILTIN_CALLABLES, imported_names_from_source, local_function_defs, bare_function_calls

    failures = []
    for path in TUMOR_PAGES:
        full = path.read_text()
        imported = imported_names_from_source(full)
        local = local_function_defs(full)
        for fn in bare_function_calls(full):
            if fn in _BUILTIN_CALLABLES:
                continue
            if fn not in imported and fn not in local:
                failures.append(f"{path.name}: {fn}")
    assert not failures, failures


def test_pseudobulk_path_resolves():
    from modules.pta.config import PtaConfig

    path = PtaConfig.pseudobulk_path(V)
    assert path.is_file(), path


@pytest.mark.parametrize("module_path", [
    "modules.pta.config",
    "modules.pta.data_loader",
    "modules.pta.boxplot",
    "modules.pta.heatmap",
    "modules.pta.individual_sc",
    "modules.pta.stats",
    "modules.pta.gene_annotation",
    "modules.pta.volcano",
    "modules.ui.plot_settings",
])
def test_pta_modules_import(module_path: str):
    __import__(module_path)


def test_pta_scrna_curation_loads():
    import pandas as pd

    from modules.pta.data_loader import load_pta_scrna_curation

    df = load_pta_scrna_curation(V)
    assert len(df) >= 118
    assert "SRA_ID" in df.columns
    assert pd.api.types.is_numeric_dtype(df["n_cells"])
    assert int(df["n_cells"].fillna(0).sum()) > 0
    assert pd.api.types.is_numeric_dtype(df["Age_numeric"])
    assert df["Age_numeric"].notna().any()


def test_coerce_age_numeric_handles_none_and_commas():
    import pandas as pd

    from modules.utils import coerce_age_numeric

    out = coerce_age_numeric(pd.Series(["49", None, "50,5", "None", "nan"]))
    assert list(out.isna()) == [False, True, False, True, True]
    assert float(out.iloc[0]) == 49.0
    assert abs(float(out.iloc[2]) - 50.5) < 1e-9


def test_pta_bulk_census_columns():
    import pandas as pd

    from modules.pta.data_loader import load_pta_bulk_curation

    df = load_pta_bulk_curation(V)
    for col in (
        "Lineage_pta",
        "Subtype_pta",
        "Cell_type_pta",
        "Invasion_pta",
        "GNAS_geno_pta",
        "USP8_geno_pta",
    ):
        assert col in df.columns, col
    assert "KI67_pta" in df.columns or "Ki67_pta" in df.columns
    ki67 = "KI67_pta" if "KI67_pta" in df.columns else "Ki67_pta"
    labelled = df["Lineage_pta"].astype("string").fillna("Unknown")
    counts = labelled.value_counts()
    assert counts.sum() == len(df)
    crosstab = pd.crosstab(
        labelled,
        df[ki67].astype("string").fillna("Unknown"),
        margins=True,
        margins_name="Total",
    )
    assert int(crosstab.loc["Total", "Total"]) == len(df)


def test_lambert_is_tf_filter_shrinks_volcano_tf_set():
    import pandas as pd

    from modules.pta.gene_annotation import apply_pta_gene_annotations, load_pta_tf_genes

    load_pta_tf_genes.clear()
    tfs = load_pta_tf_genes()
    assert tfs is not None
    assert len(tfs) == 1639

    path = PTA / "epitome_volcanos" / V / "dream_NR5A1_vs_POU1F1.csv"
    df = pd.read_csv(path)
    before = int(df["is_tf"].sum())
    after = int(apply_pta_gene_annotations(df)["is_tf"].sum())
    assert before == 2360
    assert after == 1347
    assert before - after == 1013


def test_volcano_manifest_and_paths():
    from modules.pta.config import PtaConfig
    from modules.pta.data_loader import flatten_volcano_comparisons, load_volcano_manifest

    families = load_volcano_manifest(V)
    family_ids = [family["id"] for family in families]
    assert "lineage" in family_ids
    assert "pou1f1_cell_types" in family_ids
    assert "secretion" in family_ids
    assert "granulation" in family_ids
    assert "invasion" in family_ids
    assert "mki67" in family_ids
    assert "gnas" in family_ids
    assert "usp8" in family_ids
    comparisons = flatten_volcano_comparisons(families)
    assert len(comparisons) >= 20
    ids = [entry["id"] for entry in comparisons]
    assert "NR5A1_vs_POU1F1" in ids
    assert "Lactotroph_vs_Somatotroph" in ids
    assert "Lactotroph_vs_Somatotroph_mixed_model" in ids
    assert "GNAS_Mut_vs_WT" in ids
    assert "USP8_Mut_vs_WT" in ids
    for entry in comparisons:
        assert "id" in entry and "file" in entry
        path = PtaConfig.volcano_dir(V) / entry["file"]
        assert path.is_file(), f"Missing volcano CSV: {path}"
    from modules.pta.data_loader import flatten_volcano_markers

    markers = flatten_volcano_markers(families)
    assert len(markers) >= 20
    marker_ids = [entry["id"] for entry in markers]
    assert "Lactotroph_markers" in marker_ids
    assert "Lactotroph_markers_mixed_model" in marker_ids
    for entry in markers:
        for key in ("pos_file", "neg_file"):
            path = PtaConfig.volcano_dir(V) / entry[key]
            assert path.is_file(), f"Missing marker CSV: {path}"


def test_volcano_marker_table_combines_pos_and_neg():
    from modules.pta.data_loader import load_volcano_results

    df = load_volcano_results(V, "Lactotroph_markers")
    assert df["gene"].is_unique
    assert (df["logFC"] > 0).any() and (df["logFC"] < 0).any()
    assert len(df) > 1000


def test_volcano_plot_renders():
    from modules.pta.data_loader import load_volcano_results
    from modules.pta.volcano import create_volcano_plot

    df = load_volcano_results(V, "NR5A1_vs_POU1F1")
    fig, config = create_volcano_plot(df, title="test", highlight_genes=["GH1"])
    assert len(fig.data) >= 1
    assert config["toImageButtonOptions"]["width"] == 800
    assert config["toImageButtonOptions"]["height"] == 800


def test_volcano_pou1f1_and_mutation_results_load():
    from modules.pta.data_loader import load_volcano_results

    mixed = load_volcano_results(V, "Lactotroph_vs_Somatotroph_mixed_model")
    three_way = load_volcano_results(V, "Lactotroph_vs_Somatotroph")
    assert len(mixed) > 0 and len(three_way) > 0
    gnas = load_volcano_results(V, "GNAS_Mut_vs_WT")
    assert "gene" in gnas.columns


def test_bulk_expression_prefers_shared_then_just_aligned():
    from modules.pta.data_loader import (
        load_pta_expression,
        resolve_bulk_expression_for_genes,
    )

    shared = load_pta_expression(V, "shared")
    aligned = load_pta_expression(V, "just_aligned")
    assert aligned.shape[1] < shared.shape[1]
    assert aligned.shape[0] > shared.shape[0]
    expr, matrix_id, missing = resolve_bulk_expression_for_genes(V, ["GH1"])
    assert matrix_id == "shared"
    assert not missing
    aligned_only = next(g for g in aligned.index if g not in shared.index)
    expr, matrix_id, missing = resolve_bulk_expression_for_genes(V, [aligned_only])
    assert matrix_id == "just_aligned"
    assert aligned_only in expr.index
    assert set(expr.columns) <= set(shared.columns)


def test_validation_cohorts_are_tpm_log2p1_and_match_curation():
    from modules.pta.data_loader import (
        align_bulk_samples,
        load_pta_expression,
        load_pta_metadata,
        resolve_bulk_expression_for_genes,
    )

    meta = load_pta_metadata(V)
    assert "Granulation_pta" in meta.columns
    assert "KI67_pta" in meta.columns
    zhang = load_pta_expression(V, "zhang")
    assert zhang.shape[1] == 194
    assert zhang.index.is_unique
    assert float(zhang.to_numpy().max()) < 25
    zhang_expr, zhang_meta = align_bulk_samples(zhang, meta)
    assert zhang_expr.shape[1] == 194
    assert set(zhang_meta["Author"].unique()) == {"Zhang et al., 2022"}

    jotanovic = load_pta_expression(V, "jotanovic")
    assert jotanovic.shape[1] == 77
    assert jotanovic.index.is_unique
    jot_expr, jot_meta = align_bulk_samples(jotanovic, meta)
    assert jot_expr.shape[1] == 77
    assert set(jot_meta["Author"].unique()) == {"Jotanovic et al., 2024"}

    expr, matrix_id, missing = resolve_bulk_expression_for_genes(
        V, ["GH1"], cohort="zhang_cohort"
    )
    assert matrix_id == "zhang"
    assert not missing
    assert "GH1" in expr.index
    expr, matrix_id, missing = resolve_bulk_expression_for_genes(
        V, ["GH1"], cohort="jotanovic_cohort"
    )
    assert matrix_id == "jotanovic"
    assert not missing


def test_heatmap_zscore_cap_sets_color_limits():
    import pandas as pd

    from modules.pta.heatmap import create_heatmap

    matrix = pd.DataFrame(
        [[-8.0, 8.0], [0.0, 1.0]],
        index=["GH1", "PRL"],
        columns=["s1", "s2"],
    )
    annotations = [pd.Series(["A", "B"], index=["s1", "s2"])]
    fig, _ = create_heatmap(matrix, annotations, ["Cell_type_pta"], zscore=True, zscore_cap=3)
    heat = [t for t in fig.data if t.type == "heatmap"][-1]
    assert heat.zmin == -3
    assert heat.zmax == 3
    assert heat.z[0][0] == -8.0
    assert heat.z[0][1] == 8.0


def test_comparison_column_maps_left_and_right():
    import pandas as pd

    from modules.pta.boxplot import add_comparison_column

    df = pd.DataFrame(
        {
            "Cell_type_pta": ["Somatotroph", "Corticotroph", "Gonadotroph", "Lactotroph"],
            "Expression": [1, 2, 3, 4],
        }
    )
    out = add_comparison_column(
        df,
        "Cell_type_pta",
        ["Somatotroph", "Corticotroph"],
        ["Gonadotroph"],
        "Somatotroph + Corticotroph",
        "Rest",
    )
    assert list(out["Comparison"]) == [
        "Somatotroph + Corticotroph",
        "Somatotroph + Corticotroph",
        "Rest",
    ]
    assert "Lactotroph" not in set(out["Cell_type_pta"])
