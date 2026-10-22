import streamlit as st

from modules.pta.page_layout import pta_page_header

pta_page_header(
    "Single-Cell Object Usage Guide",
    "How to work with downloaded `.h5ad` files in Python and R.",
    "tumor_version_select_download_usage_guide",
)

st.subheader("Working with h5ad Files")

st.markdown("### Python (Scanpy)")
st.markdown(
    """
Reading h5ad files in Python is straightforward using the Scanpy library:

```python
import scanpy as sc

# Read the h5ad file
adata = sc.read_h5ad('your_dataset.h5ad')

# Basic operations
print(adata.shape)  # (n_cells, n_genes)
print(adata.obs_names)  # Cell barcodes
print(adata.var_names)  # Gene names

# Access expression matrix
expression_matrix = adata.X

# Access cell metadata
cell_metadata = adata.obs

# Access gene metadata
gene_metadata = adata.var

# Access embeddings (e.g., UMAP)
umap_coords = adata.obsm['X_umap']

# Plotting UMAP
sc.pl.umap(adata, color='cell_type')
sc.pl.umap(adata, color=['SOX2', 'POMC'])  # Feature plots for specific genes
```

Key components in the h5ad files:
- `.X`: Expression matrix
- `.obs`: Cell metadata (cell types, sample information, etc.)
- `.var`: Gene metadata
- `.obsm`: Cell embeddings (UMAP etc.)
"""
)

st.markdown("### R (Seurat v5)")
st.markdown(
    """
The most efficient way to work with h5ad files in R is using `anndataR`, which reads
`.h5ad` natively and converts straight to Seurat:

```r
# Install required packages if needed
if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager")
}
BiocManager::install("anndataR")
BiocManager::install("rhdf5")  # native h5ad reading
install.packages("Seurat")

# Load libraries
library(anndataR)
library(Seurat)

# Read the h5ad file
adata <- read_h5ad("your_dataset.h5ad")

# Convert to a Seurat object
seurat_rna <- adata$as_Seurat(
    assay_name = "RNA",
    layers_mapping = c(counts = "counts")
)

Idents(seurat_rna) <- seurat_rna$cell_type

# Normalize data (if needed)
seurat_rna <- NormalizeData(
    seurat_rna,
    normalization.method = "LogNormalize",
    scale.factor = 10000
)

# Basic operations
dim(seurat_rna)  # View dimensions
head(seurat_rna@meta.data)  # View metadata

# Dimensionality reduction visualization
DimPlot(seurat_rna, group.by = "cell_type")

# Feature plots
FeaturePlot(seurat_rna, features = c("SOX2", "POMC"))

# Find markers
markers <- FindAllMarkers(seurat_rna)
```
"""
)

st.markdown("### File Structure")
st.markdown(
    """
Each h5ad file in the tumour atlas contains:
- Raw counts matrix (in adata.layers["counts"])
- Normalized expression matrix (adata.X)
- Standard cell metadata (in adata.obs)
- Dimensionality reduction coordinates (PCA, UMAP) (in adata.obsm)
- Cell type annotations (in adata.obs["cell_type"])

Some datasets were processed with an earlier export: these store raw counts in `adata.X`,
cell types in `adata.obs["broad_cluster_final"]` and have no embeddings. Check
`adata.layers` and `adata.obs.columns` after loading.
"""
)

st.markdown("### Bulk and Pseudobulk Matrices")
st.markdown(
    """
Expression matrices are provided in Matrix Market format (`.mtx`) with genes as rows.
Row and column names are in the accompanying `.txt` files:

```python
import scipy.io

matrix = scipy.io.mmread('normalized_data.mtx').tocsr()  # genes x samples
genes = open('genes.txt').read().splitlines()
samples = open('datasets.txt').read().splitlines()
```

```r
library(Matrix)
expression <- readMM("normalized_data.mtx")
rownames(expression) <- readLines("genes.txt")
colnames(expression) <- readLines("datasets.txt")
```
"""
)

st.markdown("### Need Help?")
st.markdown(
    """
If you encounter any issues working with the data:
1. Submit an issue [GitHub repository](https://github.com/Andoniadou-Lab/epitome)
2. Visit the [Scanpy documentation](https://scanpy.readthedocs.io/), [anndataR documentation](https://anndataR.scverse.org/) or [Seurat documentation](https://satijalab.org/seurat/)
3. Contact us (see Contact tab)
"""
)
