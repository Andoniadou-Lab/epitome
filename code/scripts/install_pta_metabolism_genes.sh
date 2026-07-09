#!/usr/bin/env bash
# Copy Recon2 metabolism gene list into the tumour atlas annotation folder.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="${1:-$HOME/Downloads/recon-store-genes-1.tsv}"
DEST="$ROOT/pta_data/gene_group_annotation/recon2_metabolism_genes.tsv"

mkdir -p "$(dirname "$DEST")"
cp "$SRC" "$DEST"
echo "Installed metabolism genes: $DEST ($(wc -l < "$DEST") lines)"
