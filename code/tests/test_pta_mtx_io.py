"""Tests for PTA Matrix Market repair loader."""

import pandas as pd
import pytest

from config import Config

DOTPLOT_DIR = Config.BASE_PATH / "pta_data" / "dotplot" / "v_0.04"


def _n_entries(stem: str) -> int:
    parquet = DOTPLOT_DIR / f"{stem}.parquet"
    if parquet.is_file():
        return len(pd.read_parquet(parquet))
    return len(pd.read_csv(DOTPLOT_DIR / f"{stem}.tsv", sep="\t", header=None))


@pytest.mark.parametrize("name,repair", [("matrix1.mtx", False), ("matrix2.mtx", True)])
def test_pta_dotplot_mtx_loads(name, repair):
    from modules.pta.mtx_io import load_mtx_cached

    path = DOTPLOT_DIR / name
    if not path.is_file():
        pytest.skip(f"{name} not present")
    mat = load_mtx_cached(path, repair=repair)
    stem = path.stem
    assert mat.shape == (_n_entries(f"{stem}_rows"), _n_entries(f"{stem}_genes"))
    assert mat.nnz > 1_000_000
