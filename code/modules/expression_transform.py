"""Expression scale shared by the box plots of every site: log10(x + 1) of linear values."""

from __future__ import annotations

import numpy as np
import scipy.sparse


def log10p1(values):
    """log10(x + 1) of linear expression (CPM, FPKM, TPM).

    Sparse matrices are converted to CSR and transformed in place on their stored
    entries (zeros stay zero), so pass a copy if the input must be kept.
    """
    if scipy.sparse.issparse(values):
        values = values.tocsr()
        values.data = np.log10(values.data + 1)
        return values
    if isinstance(values, (list, tuple)):
        values = np.asarray(values, dtype=float)
    return np.log10(values + 1)


def log10_axis_label(gene: str) -> str:
    return f"{gene} Expression (log10)"
