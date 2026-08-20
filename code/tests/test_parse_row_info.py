"""Row labels may contain underscores in both sample ID and cell type."""

from modules.utils import parse_row_info, parse_sample_cell_type
import pandas as pd


def test_parse_sample_cell_type_simple_sra():
    assert parse_sample_cell_type("SRX8489818_Somatotrophs") == (
        "SRX8489818",
        "Somatotrophs",
    )


def test_parse_sample_cell_type_underscored_cell_type():
    assert parse_sample_cell_type("SRX8489818_Endothelial_cells") == (
        "SRX8489818",
        "Endothelial_cells",
    )


def test_parse_sample_cell_type_underscored_sample_id():
    assert parse_sample_cell_type("698_B6J_10F_01_Somatotrophs") == (
        "698_B6J_10F_01",
        "Somatotrophs",
    )
    assert parse_sample_cell_type("164_B6NODF1J_10F_01_Stem_cells") == (
        "164_B6NODF1J_10F_01",
        "Stem_cells",
    )


def test_parse_row_info_dataframe():
    rows = pd.DataFrame(
        {
            0: [
                "SRX1_Lactotrophs",
                "698_B6J_10F_01_Immune_cells",
            ]
        }
    )
    parsed = parse_row_info(rows)
    assert list(parsed["SRA_ID"]) == ["SRX1", "698_B6J_10F_01"]
    assert list(parsed["cell_type"]) == ["Lactotrophs", "Immune_cells"]
