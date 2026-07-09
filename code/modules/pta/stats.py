"""Helpers for PTA overview statistics paths."""

from config import Config


def pta_rna_stats_path(version: str) -> str:
    return f"{Config.BASE_PATH}/pta_data/overview/{version}/rna_cell_type_counts.parquet"
