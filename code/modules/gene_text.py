"""Gene symbols are set in italics wherever they are drawn in a plot."""

from __future__ import annotations

ITALIC = "italic"


def italic_gene(gene) -> str:
    """Plotly rich text for a gene symbol, e.g. ``<i>Pou1f1</i>``."""
    return f"<i>{gene}</i>"


def italic_genes(genes) -> list[str]:
    return [italic_gene(g) for g in genes]
