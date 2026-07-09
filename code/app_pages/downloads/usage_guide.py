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
st.markdown(
    """
### Recommended workflow

1. Download the dataset-level `.h5ad` file from the RNA or ATAC download tabs.
2. Open it in Python with `scanpy.read_h5ad(...)` (or in R via `zellkonverter`).
3. Use metadata columns in `.obs` to subset by sample, condition, lineage, and cell type.
4. Use expression/accessibility matrices in `.X` with `.var_names` for downstream analyses.

### Need help?

If you run into issues using the objects, contact `epitome@kcl.ac.uk`.
"""
)
