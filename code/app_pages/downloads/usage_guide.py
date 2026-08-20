import streamlit as st

from modules.cached_loaders import AVAILABLE_VERSIONS

col1, col2 = st.columns([5, 1])
with col1:
    st.header("Single-Cell Object Usage Guide")
    st.markdown("How to work with downloaded `.h5ad` files in Python and R.")
with col2:
    st.selectbox(
        "Version",
        options=AVAILABLE_VERSIONS,
        key="version_select_download_usage_guide",
        label_visibility="collapsed",
    )

st.subheader("Working with h5ad Files")

# Python/Scanpy Section
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

#plotting UMAP
sc.pl.umap(adata, color='cell_type')
sc.pl.umap(adata, color=['Sox2'])  # Feature plots for specific genes
```

Key components in the h5ad files:
- `.X`: Expression matrix
- `.obs`: Cell metadata (cell types, conditions, etc.)
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
unique(seurat_rna$Author)  # View unique authors

# Dimensionality reduction visualization
DimPlot(seurat_rna, group.by = "cell_type")

# Feature plots
FeaturePlot(seurat_rna, features = c("Sox2", "Pomc"))

# Find markers
markers <- FindAllMarkers(seurat_rna)
```
"""
)

st.markdown("### File Structure")
st.markdown(
    """
Each h5ad file in the epitome contains:
- Raw counts matrix (in adata.layers["counts"])
- Normalized expression matrix (adata.X)
- Standard cell metadata (cell type, age, sex, etc.) (in adata.obs)
- Dimensionality reduction coordinates (PCA, UMAP) (in adata.obsm)
- Cluster annotations (in adata.obs["new_cell_type"])
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
