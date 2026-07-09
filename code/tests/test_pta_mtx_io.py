"""Tests for PTA Matrix Market repair loader."""

from pathlib import Path

import pytest

from config import Config


@pytest.mark.parametrize("name,repair", [("matrix1.mtx", False), ("matrix2.mtx", True)])
def test_pta_dotplot_mtx_loads(name, repair):
    from modules.pta.mtx_io import load_mtx_cached

    path = Config.BASE_PATH / "pta_data" / "dotplot" / "v_0.04" / name
    if not path.is_file():
        pytest.skip(f"{name} not present")
    mat = load_mtx_cached(path, repair=repair)
    assert mat.shape[0] == 536
    assert mat.shape[1] == 39005
    assert mat.nnz > 1_000_000
