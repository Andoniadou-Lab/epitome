"""Matrix Market loading with repair for occasionally corrupted PTA exports."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse as sp


def _parse_merged_token(token: str, nrows: int, ncols: int) -> tuple[int, int, float] | None:
    """Recover a triplet from tokens like ``1661904761904E-2`` (newline merge artefact)."""
    m = re.match(r"^(\d{1,3})(\d{4,5})(\d*\.?\d+(?:[eE][+-]?\d+)?)$", token)
    if not m:
        return None
    row, col, val = int(m.group(1)), int(m.group(2)), float(m.group(3))
    if 1 <= row <= nrows and 1 <= col <= ncols:
        return row, col, val
    return None


def read_mtx_repaired(path: str | Path) -> sp.csr_matrix:
    """Read coordinate MTX, recovering a few known merge patterns in damaged files."""
    path = Path(path)
    text = path.read_bytes().decode("latin-1", errors="replace")
    lines = text.splitlines()
    if len(lines) < 3 or not lines[0].startswith("%%MatrixMarket"):
        raise ValueError(f"Not a Matrix Market file: {path}")

    nrows, ncols, _nnz_declared = map(int, lines[2].split())
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for line in lines[3:]:
        parts = line.split()
        if len(parts) == 3:
            try:
                r, c, v = int(parts[0]), int(parts[1]), float(parts[2])
                if 1 <= r <= nrows and 1 <= c <= ncols:
                    rows.append(r - 1)
                    cols.append(c - 1)
                    data.append(v)
            except ValueError:
                continue
        elif len(parts) == 5:
            # Two triplets on one line sharing the same row: r c1 v1 c2 v2
            try:
                r, c1, v1, c2, v2 = (
                    int(parts[0]),
                    int(parts[1]),
                    float(parts[2]),
                    int(parts[3]),
                    float(parts[4]),
                )
                for c, v in ((c1, v1), (c2, v2)):
                    if 1 <= r <= nrows and 1 <= c <= ncols:
                        rows.append(r - 1)
                        cols.append(c - 1)
                        data.append(v)
            except ValueError:
                continue
        elif len(parts) == 1:
            parsed = _parse_merged_token(parts[0], nrows, ncols)
            if parsed:
                r, c, v = parsed
                rows.append(r - 1)
                cols.append(c - 1)
                data.append(v)

    mat = sp.coo_matrix((data, (rows, cols)), shape=(nrows, ncols)).tocsr()
    return mat


def load_mtx_cached(path: str | Path, *, repair: bool = False) -> sp.csr_matrix:
    """Load MTX, using a sidecar ``.npz`` cache when present or after first read."""
    path = Path(path)
    cache = path.with_suffix(path.suffix + ".npz")
    if cache.is_file() and cache.stat().st_mtime >= path.stat().st_mtime:
        try:
            loaded = np.load(cache)
            return sp.csr_matrix(
                (loaded["data"], loaded["indices"], loaded["indptr"]),
                shape=loaded["shape"],
            )
        except (OSError, ValueError):
            cache.unlink(missing_ok=True)

    if repair:
        mat = read_mtx_repaired(path)
    else:
        mat = scipy.io.mmread(path).tocsr()

    sp.save_npz(cache, mat)
    return mat
