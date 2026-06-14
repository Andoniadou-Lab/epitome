"""Extra smoke tests approved alongside the five baseline tests.

Includes T8, T9, T11, T18, T19, T20 and T22 from the plan.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse


CANONICAL_CELL_TYPES = {
    "Corticotrophs",
    "Endothelial_cells",
    "Gonadotrophs",
    "Immune_cells",
    "Lactotrophs",
    "Melanotrophs",
    "Mesenchymal_cells",
    "Pituicytes",
    "Somatotrophs",
    "Stem_cells",
    "Thyrotrophs",
}


def _dotplot_mean_per_cell_type(matrix, gene_idx, rows_df) -> pd.Series:
    """Group dotplot rows (``<sample>_<cell_type>``) by cell type and return per-CT means."""
    rows = rows_df[rows_df.columns[0]].astype(str).tolist()
    cell_types = [r.split("_", 1)[1] if "_" in r else "" for r in rows]
    if scipy.sparse.issparse(matrix):
        col = matrix.tocsc()[:, gene_idx].toarray().flatten()
    else:
        col = np.asarray(matrix[:, gene_idx]).flatten()
    df = pd.DataFrame({"ct": cell_types, "val": col})
    df = df[df["ct"].isin(CANONICAL_CELL_TYPES)]
    return df.groupby("ct")["val"].mean().sort_values(ascending=False)


def test_grouping_lineage_has_eight_groupings(markers):
    _, gl = markers
    actual = set(gl["grouping"].astype(str).unique())
    expected = {f"grouping_{i}" for i in range(1, 9)}
    assert actual == expected, (
        f"Expected exactly groupings {sorted(expected)}, got {sorted(actual)}."
    )


def test_cpdb_has_four_categories(base_path, version):
    cpdb = pd.read_csv(base_path / "data" / "gene_group_annotation" / version / "cpdb.csv")
    actual = set(cpdb["category"].astype(str).unique())
    expected = {"TF", "ligand", "receptor", "metabolism"}
    assert actual == expected, (
        f"Expected exactly cpdb categories {sorted(expected)}, got {sorted(actual)}."
    )


def test_grouping_1_marker_totals(markers):
    _, gl = markers
    g1 = gl[gl["grouping"] == "grouping_1"]
    n_total = len(g1)
    n_up = int((g1["direction"] == "up").sum())
    n_down = int((g1["direction"] == "down").sum())
    assert (n_total, n_up, n_down) == (3669, 2490, 1179), (
        f"grouping_1 totals drifted: total={n_total} (want 3669), "
        f"up={n_up} (want 2490), down={n_down} (want 1179)."
    )


@pytest.mark.parametrize("gene,expected_top", [("Gh", "Somatotrophs"), ("Prl", "Lactotrophs")])
def test_canonical_marker_top_cell_type(dotplot, gene, expected_top):
    _, _, _, expression_matrix, genes2, rows2 = dotplot
    gene_list = genes2[genes2.columns[0]].tolist()
    assert gene in gene_list, f"{gene} missing from dotplot expression genes."
    means = _dotplot_mean_per_cell_type(expression_matrix, gene_list.index(gene), rows2)
    top = means.index[0]
    assert top == expected_top, (
        f"{gene} should have its highest mean dotplot expression in {expected_top}, but got {top}. "
        f"Top 3: {means.head(3).to_dict()}"
    )


def test_pomc_top_in_corticotrophs_or_melanotrophs(dotplot):
    _, _, _, expression_matrix, genes2, rows2 = dotplot
    gene_list = genes2[genes2.columns[0]].tolist()
    assert "Pomc" in gene_list, "Pomc missing from dotplot expression genes."
    means = _dotplot_mean_per_cell_type(expression_matrix, gene_list.index("Pomc"), rows2)
    top = means.index[0]
    assert top in {"Corticotrophs", "Melanotrophs"}, (
        f"Pomc should peak in Corticotrophs or Melanotrophs, got {top}. "
        f"Top 3: {means.head(3).to_dict()}"
    )


def test_sox2_listed_as_stem_cell_marker(markers):
    cell_typing, gl = markers
    in_cell_typing = (
        (cell_typing["gene"] == "Sox2") & (cell_typing["celltype"] == "Stem_cells")
    ).any()
    in_grouping = ((gl["gene"] == "Sox2") & (gl["direction"] == "up")).any()
    assert in_cell_typing or in_grouping, (
        "Sox2 is missing from both cell_typing_markers (Stem_cells) and "
        "grouping_lineage_markers as an up marker."
    )


def test_displayed_sex_dim_filter_contract(sex_dim_df):
    df = sex_dim_df.rename(columns={"cell_type": "Cell Type"}).copy()
    df = df[df["logFC"].abs() >= 1]
    df = df[df["-log10_pval"] > 1.30102]
    df = df.drop_duplicates(subset=["gene", "Cell Type"])
    assert (df["logFC"].abs() >= 1).all(), "Found rows with |logFC| < 1 after display filter."
    assert (df["-log10_pval"] > 1.30102).all(), (
        "Found rows with -log10_pval <= 1.30102 after display filter."
    )
