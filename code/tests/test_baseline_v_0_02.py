"""Baseline smoke tests for the displayed contents of v_0.02.

These five tests pin user-visible invariants. Each one reproduces the exact
filtering/dedup pipeline used by the corresponding ``display_*`` function in
``code/modules/display_tables.py`` so the assertions reflect what users see in
the Streamlit UI, not just on-disk row counts.
"""

from __future__ import annotations

import numpy as np
import pytest


EXPECTED_AGING_TOTAL = 13_535
EXPECTED_AGING_STEM_CELLS = 2_095
EXPECTED_SEX_BIASED_TOTAL = 6_415

GROUPING_1_GENES = ["Epha1", "Nfix", "Epha2", "Ephb1", "Ephb3", "Ephb4", "Ephb6"]
STEM_CELLS_SEX_BIASED_GENES = ["Csmd1", "Slco1a4", "Tac1", "Ptx4"]


def _aging_displayed(aging_df):
    """Replicate ``display_aging_genes_table`` filtering + dedup on a fresh copy."""
    df = aging_df.rename(
        columns={
            "Logfc": "log2FC",
            "Aveexpr": "AveExpr",
            "T": "t",
            "P.Value": "pvalue",
            "Adj.P.Val": "adj.P.Val",
            "B": "B",
            "Genes": "gene",
        }
    ).copy()
    df["-log10_adj_pval"] = -np.log10(df["adj.P.Val"] + 1e-300)
    df = df.drop(columns=["pvalue", "adj.P.Val"])
    df = df[df["-log10_adj_pval"] > 1.301]
    df = df.drop_duplicates(subset=["gene", "Cell Type"])
    return df


def _sex_dim_displayed(sex_dim_df):
    """Replicate ``display_sex_dimorphism_table`` filtering + dedup on a fresh copy."""
    df = sex_dim_df.rename(columns={"cell_type": "Cell Type"}).copy()
    df = df[df["logFC"].abs() >= 1]
    df = df[df["-log10_pval"] > 1.30102]
    df = df.drop_duplicates(subset=["gene", "Cell Type"])
    return df


def test_aging_table_total_rows(aging_df):
    df = _aging_displayed(aging_df)
    assert len(df) == EXPECTED_AGING_TOTAL, (
        f"Expected {EXPECTED_AGING_TOTAL} displayed age-dependent genes for v_0.02, got {len(df)}."
    )


def test_aging_table_stem_cells_rows(aging_df):
    df = _aging_displayed(aging_df)
    n_stem = int((df["Cell Type"] == "Stem_cells").sum())
    assert n_stem == EXPECTED_AGING_STEM_CELLS, (
        f"Expected {EXPECTED_AGING_STEM_CELLS} Stem_cells age-dependent genes, got {n_stem}."
    )


def test_lhb_in_expression_genes(expression_genes):
    col = expression_genes.columns[0]
    assert (expression_genes[col] == "Lhb").any(), (
        "Lhb is missing from data/expression/v_0.02/genes.parquet "
        "and would therefore be unselectable for expression boxplots."
    )


def test_lhb_in_dotplot_proportion_matrix(dotplot):
    _, genes1, _, _, _, _ = dotplot
    col = genes1.columns[0]
    assert (genes1[col] == "Lhb").any(), (
        "Lhb is missing from dotplot matrix1_genes (proportion) and would be unselectable in the dotplot."
    )


def test_lhb_in_dotplot_expression_matrix(dotplot):
    _, _, _, _, genes2, _ = dotplot
    col = genes2.columns[0]
    assert (genes2[col] == "Lhb").any(), (
        "Lhb is missing from dotplot matrix2_genes (expression) and would be unselectable in the dotplot."
    )


@pytest.mark.parametrize("gene", GROUPING_1_GENES)
def test_grouping_1_lineage_marker(markers, gene):
    _, grouping_lineage_markers = markers
    g1 = grouping_lineage_markers[grouping_lineage_markers["grouping"] == "grouping_1"]
    assert (g1["gene"] == gene).any(), (
        f"Gene '{gene}' is expected as a grouping_1 lineage marker but is missing."
    )


@pytest.mark.parametrize("gene", STEM_CELLS_SEX_BIASED_GENES)
def test_stem_cells_sex_biased_gene(sex_dim_df, gene):
    df = _sex_dim_displayed(sex_dim_df)
    stem = df[df["Cell Type"] == "Stem_cells"]
    assert (stem["gene"] == gene).any(), (
        f"Gene '{gene}' is expected to be sex-biased in Stem_cells but is missing from the displayed table."
    )


def test_sex_biased_total_rows(sex_dim_df):
    df = _sex_dim_displayed(sex_dim_df)
    assert len(df) == EXPECTED_SEX_BIASED_TOTAL, (
        f"Expected {EXPECTED_SEX_BIASED_TOTAL} displayed sex-biased genes for v_0.02, got {len(df)}."
    )
