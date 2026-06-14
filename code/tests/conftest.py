"""Shared pytest fixtures for the epitome smoke-test suite.

Loaders are session-scoped so every parquet/mtx is read at most once per
pytest invocation, keeping the suite cheap to run inside
``code/pre_deployment_checks.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_CODE_DIR = Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from config import Config  # noqa: E402
from modules.data_loader import (  # noqa: E402
    load_aging_genes,
    load_dotplot_data,
    load_marker_data,
    load_sex_dim_data,
)


@pytest.fixture(scope="session")
def version() -> str:
    return "v_0.02"


@pytest.fixture(scope="session")
def base_path() -> Path:
    return Path(Config.BASE_PATH)


@pytest.fixture(scope="session")
def aging_df(version):
    """Raw output of ``load_aging_genes`` (after cpdb merge + whole-row dedup)."""
    return load_aging_genes(version)


@pytest.fixture(scope="session")
def sex_dim_df(version):
    """Raw output of ``load_sex_dim_data``."""
    return load_sex_dim_data(version)


@pytest.fixture(scope="session")
def markers(version):
    """Tuple ``(cell_typing_markers, grouping_lineage_markers)``."""
    return load_marker_data(version)


@pytest.fixture(scope="session")
def dotplot(version):
    """Tuple ``(proportion_matrix, genes1, rows1, expression_matrix, genes2, rows2)``."""
    return load_dotplot_data(version)


@pytest.fixture(scope="session")
def expression_genes(base_path, version):
    """Gene index from ``data/expression/<version>/genes.parquet``."""
    import pandas as pd

    return pd.read_parquet(base_path / "data" / "expression" / version / "genes.parquet")
