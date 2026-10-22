import gzip
import shutil
from dataclasses import dataclass

import scanpy as sc
import tempfile
import zipfile
from pathlib import Path
from epitome_tools.celltyping import get_cell_typing_path
from epitome_tools.doublets import get_doublet_path
from epitome_tools.workflow import (
    cell_type_workflow,
    check_sample_compatibility_normalization,
    doublet_workflow,
)
from .individual_sc import plot_sc_dataset
import os
import streamlit as st
import matplotlib.pyplot as plt
from .utils import create_gene_selector

import pandas as pd
import scipy.io
import anndata as ad
from anndata import AnnData
import numpy as np

# Annotated obs can hold pandas nullable string columns, which anndata only writes
# to the downloadable .h5ad when this opt-in is enabled.
ad.settings.allow_write_nullable_strings = True

#set random seed to 42 for reproducibility
np.random.seed(42)

# Annotation densifies a cells × model-features matrix. Above these sizes that
# step can exhaust the web server, so the site refuses to continue.
MAX_WEB_CELLS = 20_000
MAX_WEB_FEATURES = 40_000
EPITOME_TOOLS_REPO = "https://github.com/Andoniadou-Lab/epitome_tools"

RNA_ASSAYS = ["sc", "sn", "multi_rna"]
ATAC_ASSAYS = ["sn", "multi_atac"]


@dataclass(frozen=True)
class SpeciesSettings:
    species: str
    model_version: str
    gene_examples: str
    typical_features: str


SPECIES_SETTINGS = {
    "mouse": SpeciesSettings(
        species="mouse",
        model_version="v_0.02",
        gene_examples='"Sox2", "Nfic"',
        typical_features="30,000",
    ),
    "human": SpeciesSettings(
        species="human",
        model_version="v_0.02",
        gene_examples='"SOX2", "NFIC"',
        typical_features="36,000",
    ),
}


def available_modalities(species: str) -> list[str]:
    """Modalities with a packaged epitome_tools cell typing model for ``species``."""
    version = SPECIES_SETTINGS[species].model_version
    modalities = []
    for modality in ("rna", "atac"):
        try:
            get_cell_typing_path(version, species, modality)
        except FileNotFoundError:
            continue
        modalities.append(modality)
    return modalities


def has_doublet_model(species: str, modality: str) -> bool:
    try:
        get_doublet_path(SPECIES_SETTINGS[species].model_version, species, modality)
    except FileNotFoundError:
        return False
    return True


def run_epitome_annotation(adata, species: str, modality: str, active_assay: str):
    """Cell typing (with smoothing) plus doublet detection where a model exists."""
    version = SPECIES_SETTINGS[species].model_version
    adata = cell_type_workflow(
        adata,
        active_assay=active_assay,
        modality=modality,
        in_place=True,
        nan_or_zero="zero",
        smoothing=True,
        version=version,
        species=species,
        verbose=False,
    )
    if has_doublet_model(species, modality):
        adata = doublet_workflow(
            adata,
            active_assay=active_assay,
            modality=modality,
            in_place=True,
            nan_or_zero="zero",
            version=version,
            species=species,
            verbose=False,
        )
    return adata


def web_annotation_blockers(n_cells: int, n_features: int, species: str = "mouse") -> list[str]:
    """Warnings that mean the website must not run annotation on this object."""
    blockers: list[str] = []
    if n_cells > MAX_WEB_CELLS:
        blockers.append(
            f"This is a large dataset ({n_cells:,} cells). "
            "Datasets with more than 20,000 cells are not annotated on the website. "
            "Please use the epitome_tools programmatic tool directly: "
            f"{EPITOME_TOOLS_REPO}"
        )
    if n_features > MAX_WEB_FEATURES:
        typical = SPECIES_SETTINGS[species].typical_features
        blockers.append(
            f"This object has {n_features:,} features, and there is likely a problem with it, "
            f"as typical {species} datasets have around {typical} features."
        )
    return blockers
def process_uploaded_file(uploaded_file):
    """
    Process uploaded single-cell data file and return AnnData object
    
    Parameters:
    -----------
    uploaded_file : streamlit.UploadedFile
        The uploaded file object from Streamlit
        
    Returns:
    --------
    anndata.AnnData or None
        Processed AnnData object or None if processing failed
    """
    try:
        file_extension = Path(uploaded_file.name).suffix.lower()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            
            # Save uploaded file to temporary location
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Process based on file type
            if file_extension == '.h5ad':
                # Read AnnData file
                adata = sc.read(temp_path)
                
            elif file_extension == '.h5':
                # Read 10x H5 file
                adata = sc.read_10x_h5(temp_path)
                
            elif file_extension == '.zip':
                # Handle 10x matrix folder (zipped)
                with zipfile.ZipFile(temp_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Look for matrix files in extracted content
                extracted_files = os.listdir(temp_dir)
                matrix_dir = None
                
                # Find directory containing matrix files
                for item in extracted_files:
                    #if any end with gz, then gunzip
                    if item.endswith('.gz'):
                        with gzip.open(os.path.join(temp_dir, item), 'rb') as f_in:
                            with open(os.path.join(temp_dir, item[:-3]), 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                        item = item[:-3]
                    item_path = os.path.join(temp_dir, item)
                    if os.path.isdir(item_path):
                        dir_contents = os.listdir(item_path)
                        if any(f.startswith('matrix') for f in dir_contents):
                            matrix_dir = item_path
                            break
                
                if matrix_dir is None:
                    # Files might be in root of zip
                    matrix_dir = temp_dir
                try:
                    adata = sc.read_10x_mtx(
                        matrix_dir,
                        var_names='gene_symbols',
                        cache=True
                    )
                except:
                    try:
                        matrix_path = os.path.join(matrix_dir, 'matrix.mtx')
                        genes_path = os.path.join(matrix_dir, 'features.tsv')
                        barcodes_path = os.path.join(matrix_dir, 'barcodes.tsv')

                        matrix = scipy.io.mmread(matrix_path).T.tocsr()
                        genes = pd.read_csv(genes_path, sep="\t", header=None)
                        barcodes = pd.read_csv(barcodes_path, header=None)

                    except:
                        matrix_path = os.path.join(matrix_dir, 'matrix.mtx.gz')
                        genes_path = os.path.join(matrix_dir, 'features.tsv.gz')
                        barcodes_path = os.path.join(matrix_dir, 'barcodes.tsv.gz')

                        matrix = scipy.io.mmread(matrix_path).T.tocsr()
                        genes = pd.read_csv(genes_path, sep="\t", header=None)
                        barcodes = pd.read_csv(barcodes_path, header=None)


                    adata = AnnData(matrix)
                    adata.var_names = genes[1]      # gene symbols
                    adata.obs_names = barcodes[0]   # cell barcodes


            else:
                st.error(f"Unsupported file format: {file_extension}")
                return None
                
            return adata
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None


_STATE_KEYS = (
    'uploaded_adata', 'file_processed', 'annotation_params', 'annotation_complete',
    'annotated_adata', 'annotation_blockers', 'current_file_key', 'original_dims',
    'loaded_n_obs', 'loaded_n_vars',
)


def _state_key(species: str, name: str) -> str:
    """Session-state keys are per species so the mouse and human pages never share data."""
    return f"{species}_{name}"


def create_cell_type_annotation_ui(species: str = "mouse"):
    """
    Create the cell type annotation interface with drag-and-drop functionality
    """
    settings = SPECIES_SETTINGS[species]
    modalities = available_modalities(species)
    state = st.session_state
    k = lambda name: _state_key(species, name)  # noqa: E731

    st.header("Automated Cell Type Annotation")

    doublets_available = any(has_doublet_model(species, m) for m in modalities)
    intro = (
        "Upload your single-cell dataset for automated cell type annotation and doublet filtering."
        if doublets_available
        else "Upload your single-cell dataset for automated cell type annotation."
    )
    feature_lines = [
        f'- RNA datasets feature names should be gene symbols (e.g., {settings.gene_examples}). '
        "Cell Ranger output should have ~99% features compatible which is sufficient for the model. "
        "(Even better to try and match feature names with our existing objects in the database.)"
    ]
    format_lines = []
    if "atac" in modalities:
        feature_lines.append(
            "- ATAC datasets feature names from the Consensus Chromatin Landscape (CCL) are required. "
            "For this, fragment counting needs to be repeated using the CCL reference peaks. "
            "This can be performed in Signac (R) or SnapATAC2 (Python)."
        )
        feature_lines.append(
            "- Either RNA or ATAC datasets: the online tool will check for normalization and "
            "log-transformation, and apply these steps if needed. We recommend providing raw counts here. "
            "(For the programmatic tool, we recommend providing normalized and log-transformed data for best results.)"
        )
        format_lines.append("- Either RNA or ATAC: `.h5ad` - AnnData files (scanpy format)")
    else:
        feature_lines.append(
            "- The online tool will check for normalization and log-transformation, and apply these "
            "steps if needed. We recommend providing raw counts here. (For the programmatic tool, "
            "we recommend providing normalized and log-transformed data for best results.)"
        )
        format_lines.append("- RNA: `.h5ad` - AnnData files (scanpy format)")
    format_lines += [
        "- RNA:`.h5` - 10x Genomics HDF5 files",
        "- RNA:`.zip` - Zipped 10x matrix folder (containing matrix.mtx, barcodes.tsv, features.tsv)",
    ]

    st.markdown(
        f"{intro}\n\n"
        "**Data requirements (Input features):**\n\n"
        + "\n".join(feature_lines)
        + "\n\n**Supported formats:**\n\n"
        + "\n".join(format_lines)
        + "\n\n**Troubleshooting:**\n\n"
        "- If you encounter issues with file upload or processing, please ensure your file meets the format requirements above.\n"
        "- For large datasets, consider using the programmatic interface for better performance.\n"
        "- If problems persist, contact epitome@kcl.ac.uk or raise an issue at https://github.com/Andoniadou-Lab/epitome_tools.\n"
    )
    if not modalities:
        st.error(
            f"The installed epitome_tools has no {species} cell typing model "
            f"({settings.model_version}). Please update epitome_tools."
        )
        return

    # Initialize session state variables
    for name, default in (
        ('uploaded_adata', None),
        ('file_processed', False),
        ('annotation_params', {}),
        ('annotation_complete', False),
        ('annotated_adata', None),
        ('annotation_blockers', []),
    ):
        if k(name) not in state:
            state[k(name)] = default
    
    # File upload widget
    st.markdown(
        """
        <style>
        div[data-testid="stFileUploader"] {
            border: 3px dashed #0000ff !important;
            border-radius: 12px !important;
            padding: 1.25rem !important;
            background: rgba(0, 0, 255, 0.03) !important;
        }
        div[data-testid="stFileUploader"] section {
            min-height: 170px !important;
        }
        div[data-testid="stFileUploader"] small {
            font-size: 1rem !important;
            font-weight: 600 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Drag and drop your dataset here or click to browse",
        type=['h5ad', 'h5', 'zip'],
        help="Upload single-cell data in AnnData (.h5ad), 10x H5 (.h5), or zipped 10x matrix folder (.zip) format",
        key=k("file_uploader"),
    )

    #write hint that we do not store data
    st.markdown("""
    **Note:** We do not store your data on the server. All processing is performed temporarily.
                """)
    
    # Process uploaded file and store in session state
    if uploaded_file is not None:
        # Check if this is a new file or same file
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        
        if state.get(k('current_file_key')) != file_key:
            # New file uploaded, reset states
            state[k('current_file_key')] = file_key
            state[k('uploaded_adata')] = None
            state[k('file_processed')] = False
            state[k('annotation_complete')] = False
            state[k('annotated_adata')] = None
            state[k('annotation_blockers')] = []
            state[k('loaded_n_obs')] = None
            state[k('loaded_n_vars')] = None
        
        # Display file information
        st.success(f"File uploaded: {uploaded_file.name}")
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.info(f"File size: {file_size_mb:.1f} MB")
        
        # Process file button
        if not state[k('file_processed')]:
            if st.button("Process Dataset", type="primary", key=k("process_dataset_btn")):
                with st.spinner("Processing dataset..."):
                    # Process the uploaded file
                    adata = process_uploaded_file(uploaded_file)
                    
                    if adata is not None:
                        blockers = web_annotation_blockers(adata.n_obs, adata.n_vars, species)
                        state[k('annotation_blockers')] = blockers
                        state[k('loaded_n_obs')] = adata.n_obs
                        state[k('loaded_n_vars')] = adata.n_vars
                        # Discard the matrix when annotation will not run, so the
                        # later densify step never sees it.
                        state[k('uploaded_adata')] = None if blockers else adata
                        state[k('file_processed')] = True
                        st.rerun()
        
        # Show dataset info and annotation interface if file is processed
        blockers = state.get(k('annotation_blockers')) or []
        if state[k('file_processed')] and blockers:
            n_obs = state.get(k('loaded_n_obs'))
            n_vars = state.get(k('loaded_n_vars'))
            col1, col2 = st.columns(2)
            with col1:
                if n_obs is not None:
                    st.metric("Number of cells", f"{n_obs:,}")
            with col2:
                if n_vars is not None:
                    st.metric("Number of genes", f"{n_vars:,}")
            for message in blockers:
                st.warning(message)
        elif state[k('file_processed')] and state[k('uploaded_adata')] is not None:
            adata = state[k('uploaded_adata')]
            
            # Display basic dataset information
            st.success("Dataset loaded successfully!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Number of cells", f"{adata.n_obs:,}")
            with col2:
                st.metric("Number of genes", f"{adata.n_vars:,}")
            with col3:
                if 'total_counts' in adata.obs.columns:
                    avg_counts = adata.obs['total_counts'].mean()
                    st.metric("Avg. counts/cell", f"{avg_counts:.0f}")
            
            adata.obs['cell_id'] = adata.obs_names
            adata.var['gene_symbol'] = adata.var_names

            # Show data preview
            with st.expander("Dataset Preview"):
                st.write("**Cell metadata (obs):**")
                st.dataframe(adata.obs.head())
                
                st.write("**Gene metadata (var):**")
                st.dataframe(adata.var.head())
            
            # Annotation options
            st.header("Annotation Settings")
            
            col1, col2 = st.columns(2)
            with col1:
                modality = st.selectbox(
                    "Modality",
                    options=modalities,
                    help="Select the data modality (RNA or ATAC experiment)",
                    key=k("modality_select")
                )
            
            with col2:
                # Dynamic assay options based on modality
                assay_options = RNA_ASSAYS if modality == "rna" else ATAC_ASSAYS
                
                active_assay = st.selectbox(
                    "Active Assay",
                    options=assay_options,
                    help="Select the assay type (Single-cell: sc, Single-nucleus: sn, Multiome: multi_rna/multi_atac)",
                    key=k("assay_select")
                )
            
            # Additional options
            with st.expander("Advanced Options"):
                min_counts = st.number_input(
                    "Minimum counts per cell",
                    min_value=500,
                    max_value=5000,
                    value=500,
                    step=50,
                    help="Filter out cells with fewer than this many genes",
                    key=k("min_counts_input")
                )

            # Store current parameters
            current_params = {
                'modality': modality,
                'active_assay': active_assay,
                'min_counts': min_counts
            }
            
            # Run annotation button
            if not state[k('annotation_complete')]:
                if st.button("Run Cell Type Annotation", type="primary", key=k("run_annotation_btn")):
                    run_blockers = web_annotation_blockers(adata.n_obs, adata.n_vars, species)
                    if run_blockers:
                        for message in run_blockers:
                            st.warning(message)
                    else:
                        spinner = (
                            "Running cell type annotation and doublet detection..."
                            if has_doublet_model(species, modality)
                            else "Running cell type annotation..."
                        )
                        with st.spinner(spinner):
                            try:
                                # Create a copy of the data for processing
                                adata_copy = adata.copy()
                            
                                # Store original dimensions
                                original_n_obs = adata_copy.n_obs
                                original_n_vars = adata_copy.n_vars

                                # Basic filtering if requested
                                if min_counts > 0:
                                    sc.pp.filter_cells(adata_copy, min_counts=min_counts)

                                passing, not_normed, not_logged = check_sample_compatibility_normalization(adata_copy,force=False)
                                if not passing:
                                    if not_normed:
                                        sc.pp.normalize_total(adata_copy, target_sum=1e4)
                                        sc.pp.log1p(adata_copy)
                                    elif not_logged:
                                        sc.pp.log1p(adata_copy)

                                check_sample_compatibility_normalization(adata_copy,force=False)

                                # Always check and ensure UMAP exists before visualization
                                if 'X_umap' not in adata_copy.obsm.keys():
                                    st.info("UMAP not found. Computing UMAP for visualization...")
                                    adata_copy.layers["log1p"] = adata_copy.X.copy()
                                    #scale
                                    sc.pp.scale(adata_copy, max_value=10)
                                    if modality == "rna":
                                        sc.pp.highly_variable_genes(adata_copy, min_mean=0.0125, max_mean=3, min_disp=0.5)
                                        sc.tl.pca(adata_copy, svd_solver='full', random_state=42)
                                    elif modality == "atac":
                                        sc.pp.highly_variable_genes(adata_copy, n_top_genes=20000)
                                        sc.tl.pca(adata_copy, svd_solver='arpack', random_state=42)

                                    # Compute UMAP
                                    adata_copy.X = adata_copy.layers["log1p"].copy()
                                    sc.pp.neighbors(adata_copy, n_neighbors=15, n_pcs=42, random_state=42)
                                    sc.tl.umap(adata_copy, random_state=42)
                                    st.success("UMAP computed successfully!")
                                else:
                                    st.success("UMAP embeddings found!")

                                # Run annotation
                                annotated_adata = run_epitome_annotation(
                                    adata_copy, species, modality, active_assay
                                )

                                # Store results in session state
                                state[k('annotated_adata')] = annotated_adata
                                state[k('annotation_params')] = current_params
                                state[k('original_dims')] = (original_n_obs, original_n_vars)
                                state[k('annotation_complete')] = True

                                st.success("Annotation completed!")
                                st.rerun()

                            except Exception as e:
                                st.error(f"Error during annotation: {str(e)}")
                                if st.checkbox("Show detailed error information", key=k("error_details")):
                                    st.exception(e)
            
            # Show results if annotation is complete
            if state[k('annotation_complete')] and state[k('annotated_adata')] is not None:
                show_annotation_results(species)


def _plot_final_dataset(final_adata, species, selected_gene, sort_order, color_map):
    if species == "mouse":
        #in final_data add col new_cell_type which is same as cell_type_final
        final_adata.obs['new_cell_type'] = final_adata.obs['cell_type_final']
        return plot_sc_dataset(final_adata, selected_gene, sort_order, color_map)

    from .pta.individual_sc import plot_pta_sc_dataset

    # The tumour plot prefers an uploaded ``cell_type`` column, so it gets an object
    # that carries only the predicted labels (the expression matrix is shared).
    plot_adata = AnnData(
        X=final_adata.X,
        obs=pd.DataFrame(
            {"cell_type": final_adata.obs["cell_type_final"].astype(str)},
            index=final_adata.obs_names,
        ),
        var=pd.DataFrame(index=final_adata.var_names),
        obsm={"X_umap": final_adata.obsm["X_umap"]},
    )
    return plot_pta_sc_dataset(plot_adata, selected_gene, sort_order, color_map)


def show_annotation_results(species: str = "mouse"):
    """
    Display annotation results, downloads, and visualization
    """
    state = st.session_state
    k = lambda name: _state_key(species, name)  # noqa: E731
    annotated_adata = state[k('annotated_adata')]
    params = state[k('annotation_params')]
    original_n_obs, original_n_vars = state[k('original_dims')]
    has_doublets = 'thresholded_doublet_epitome' in annotated_adata.obs.columns
    
    # Show filtering results
    if original_n_obs != annotated_adata.n_obs or original_n_vars != annotated_adata.n_vars:
        st.info("**Filtering Results:**")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Cells: {original_n_obs:,} → {annotated_adata.n_obs:,}")
        with col2:
            st.write(f"Genes: {original_n_vars:,} → {annotated_adata.n_vars:,}")
    
    # Show UMAP panel with annotation results
    st.header(
        "Annotation & Doublet Detection Results" if has_doublets else "Annotation Results"
    )
    if not has_doublets:
        st.info(
            f"Doublet detection is not yet available for the {species} model, "
            "so only cell types are predicted."
        )
    
    # Create UMAP plots
    try:
        if has_doublets:
            fig, axes = plt.subplots(2, 2, figsize=(15, 15))
            type_axes = axes[0]
        else:
            fig, type_axes = plt.subplots(1, 2, figsize=(15, 7.5))
        
        # Plot 1: Cell types
        sc.pl.umap(annotated_adata, color='cell_type_final', ax=type_axes[0], show=False, frameon=False)
        type_axes[0].set_title('Predicted Cell Types')

        sc.pl.umap(annotated_adata, color='predicted_cell_type_proba', ax=type_axes[1], show=False, frameon=False)
        type_axes[1].set_title('Predicted Cell Type Probabilities')

        if has_doublets:
            # Plot 2: Doublet status
            sc.pl.umap(annotated_adata, color='thresholded_doublet_epitome', ax=axes[1, 0], show=False, frameon=False)
            axes[1, 0].set_title('Doublet Detection')
            
            # Plot 3: Doublet scores
            sc.pl.umap(annotated_adata, color='doublet_score_epitome', ax=axes[1, 1], show=False, frameon=False)
            axes[1, 1].set_title('Doublet Scores')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
    except Exception as plot_error:
        st.error(f"Error creating UMAP plots: {str(plot_error)}")
        if st.checkbox("Show UMAP plot error details", key=k("umap_plot_error_details")):
            st.exception(plot_error)
    
    # Show annotation statistics
    col1, col2 = st.columns(2)
    
    with col1:
        # Cell type distribution
        if 'cell_type_final' in annotated_adata.obs.columns:
            st.write("**Cell Type Distribution:**")
            cell_type_counts = annotated_adata.obs['cell_type_final'].value_counts()
            st.dataframe(cell_type_counts.reset_index())
    
    with col2:
        # Doublet statistics
        if has_doublets:
            doublet_counts = annotated_adata.obs['thresholded_doublet_epitome'].value_counts()
            total_cells = len(annotated_adata.obs)
            doublet_pct = (doublet_counts.get(True, 0) / total_cells) * 100
            
            st.write("**Doublet Detection Summary:**")
            st.metric("Total Cells", f"{total_cells:,}")
            st.metric("Predicted Doublets", f"{doublet_counts.get(True, 0):,}")
            st.metric("Doublet Rate", f"{doublet_pct:.1f}%")
    
    # Download options for annotated data
    st.header("Download Annotated Results")
    
    # Create download buttons
    col1, col2 = st.columns(2)
    
    with col1:
        # Download annotated h5ad
        with tempfile.NamedTemporaryFile(suffix='.h5ad', delete=False) as tmp_file:
            annotated_adata.write_h5ad(tmp_file.name, compression='gzip')
            with open(tmp_file.name, 'rb') as f:
                st.download_button(
                    label="Download Annotated H5AD (This is your single-cell data with our annotations added)",
                    data=f.read(),
                    file_name=f"annotated_{state.get(k('current_file_key'), 'data')}.h5ad",
                    mime="application/octet-stream",
                    key=k("download_h5ad")
                )
    
    with col2:
        # Download cell metadata CSV
        csv_data = annotated_adata.obs.to_csv()
        st.download_button(
            label="Download Cell Metadata (CSV) - this contains all cell annotations and metadata, but no counts data",
            data=csv_data,
            file_name=f"cell_metadata_{state.get(k('current_file_key'), 'data')}.csv",
            mime="text/csv",
            key=k("download_csv")
        )
    
    filter_doublets = False
    if has_doublets:
        # Doublet Filtering Option
        st.header("Optional: Filter Doublets")
        
        filter_doublets = st.checkbox(
            "Remove predicted doublets from dataset",
            value=False,
            help="Remove cells predicted as doublets for downstream analysis",
            key=k("filter_doublets_checkbox")
        )
    
    # Prepare final dataset (with or without doublet filtering)
    if filter_doublets:
        # Filter out doublets
        pre_filter_cells = annotated_adata.n_obs
        final_adata = annotated_adata[~annotated_adata.obs['thresholded_doublet_epitome']].copy()
        post_filter_cells = final_adata.n_obs
        removed_cells = pre_filter_cells - post_filter_cells
        
        st.info(f"**Doublet Filtering Applied:**")
        st.write(f"Cells before filtering: {pre_filter_cells:,}")
        st.write(f"Cells after filtering: {post_filter_cells:,}")
        st.write(f"Doublets removed: {removed_cells:,}")
    else:
        final_adata = annotated_adata.copy()
        if has_doublets:
            st.info("No doublet filtering applied - using full annotated dataset.")
    
    # Visualization Section - Always show this at the end
    st.header("Visualize Final Results")
    
    # Get available genes for selection
    modality = params['modality']
    active_assay = params['active_assay']
    
    available_genes = sorted(final_adata.var_names.tolist())
    
    if available_genes:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_gene = create_gene_selector(
                                        gene_list=available_genes,
                                        key_suffix=f"sc_annotation_{species}")
        
        with col2:
            color_map = st.selectbox(
                "Color Map",
                [
                    "reds",
                    "plasma",
                    "inferno",
                    "magma",
                    "blues",
                    "viridis",
                    "greens",
                    "YlOrRd",#
                ],
                key=k("color_map_select_annotation_final"),
            )
        
        with col3:
            sort_order = st.checkbox("Sort plotted cells by expression", value=False, key=k("sort_annotation_final"))
        
        # Generate plots using imported function
        try:
            gene_fig, cell_type_fig, config = _plot_final_dataset(
                final_adata, species, selected_gene, sort_order, color_map
            )
            
            # Display plots side by side
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(gene_fig, use_container_width=True)
            with col2:
                st.plotly_chart(cell_type_fig, use_container_width=True)
                
            # Add explanation
            with st.container():
                doublet_status = "after doublet filtering" if filter_doublets else "including all cells"
                st.markdown(
                    f"""
                    **Visualization Details:**
                    - **Left plot**: {selected_gene} expression across cells
                    - **Right plot**: Annotated cell types
                    - **Data modality**: {modality.upper()}
                    - **Assay type**: {active_assay}
                    - **Total cells**: {final_adata.n_obs:,} ({doublet_status})
                    """
                )
                
        except Exception as viz_error:
            st.error(f"Error creating visualization: {str(viz_error)}")
            if st.checkbox("Show visualization error details", key=k("viz_error_details")):
                st.exception(viz_error)
    else:
        st.warning("No genes available for visualization.")
    
    # Reset button
    if st.button("Start New Analysis", type="secondary", key=k("reset_analysis")):
        # Clear all session state related to annotation
        for name in _STATE_KEYS:
            state.pop(k(name), None)
        st.rerun()
