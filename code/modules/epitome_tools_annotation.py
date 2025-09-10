import scanpy as sc
import tempfile
import zipfile
from pathlib import Path
from epitome_tools.workflow import celltype_doublet_workflow, check_sample_compatibility_normalization
from .individual_sc import plot_sc_dataset
import plotly.express as px
import os
import streamlit as st
import matplotlib.pyplot as plt
from .utils import create_gene_selector


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
                adata = sc.read_h5ad(temp_path)
                
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
                    item_path = os.path.join(temp_dir, item)
                    if os.path.isdir(item_path):
                        dir_contents = os.listdir(item_path)
                        if any(f.startswith('matrix') for f in dir_contents):
                            matrix_dir = item_path
                            break
                
                if matrix_dir is None:
                    # Files might be in root of zip
                    matrix_dir = temp_dir
                
                adata = sc.read_10x_mtx(
                    matrix_dir,
                    var_names='gene_symbols',
                    cache=True
                )
                
            else:
                st.error(f"Unsupported file format: {file_extension}")
                return None
                
            return adata
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None


def create_cell_type_annotation_ui():
    """
    Create the cell type annotation interface with drag-and-drop functionality
    """
    st.header("Automatic Cell Type Annotation")
    
    st.markdown("""
    Upload your single-cell dataset for automatic cell type annotation and doublet filtering.
    
    **Supported formats:**
    - `.h5ad` - AnnData files (scanpy format)
    - `.h5` - 10x Genomics HDF5 files
    - `.zip` - Zipped 10x matrix folder (containing matrix.mtx, barcodes.tsv, features.tsv)
    """)
    
    # Initialize session state variables
    if 'uploaded_adata' not in st.session_state:
        st.session_state['uploaded_adata'] = None
    if 'file_processed' not in st.session_state:
        st.session_state['file_processed'] = False
    if 'annotation_params' not in st.session_state:
        st.session_state['annotation_params'] = {}
    if 'annotation_complete' not in st.session_state:
        st.session_state['annotation_complete'] = False
    if 'annotated_adata' not in st.session_state:
        st.session_state['annotated_adata'] = None
    
    # File upload widget
    uploaded_file = st.file_uploader(
        "Drag and drop your dataset here or click to browse",
        type=['h5ad', 'h5', 'zip'],
        help="Upload single-cell data in AnnData (.h5ad), 10x H5 (.h5), or zipped 10x matrix folder (.zip) format"
    )

    #write hint that we do not store data
    st.markdown("""
    **Note:** We do not store your data on the server. All processing is performed temporarily.
                """)
    
    # Process uploaded file and store in session state
    if uploaded_file is not None:
        # Check if this is a new file or same file
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        
        if ('current_file_key' not in st.session_state or 
            st.session_state['current_file_key'] != file_key):
            # New file uploaded, reset states
            st.session_state['current_file_key'] = file_key
            st.session_state['uploaded_adata'] = None
            st.session_state['file_processed'] = False
            st.session_state['annotation_complete'] = False
            st.session_state['annotated_adata'] = None
        
        # Display file information
        st.success(f"File uploaded: {uploaded_file.name}")
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.info(f"File size: {file_size_mb:.1f} MB")
        
        # Process file button
        if not st.session_state['file_processed']:
            if st.button("Process Dataset", type="primary", key="process_dataset_btn"):
                with st.spinner("Processing dataset..."):
                    # Process the uploaded file
                    adata = process_uploaded_file(uploaded_file)
                    
                    if adata is not None:
                        st.session_state['uploaded_adata'] = adata
                        st.session_state['file_processed'] = True
                        st.rerun()
        
        # Show dataset info and annotation interface if file is processed
        if st.session_state['file_processed'] and st.session_state['uploaded_adata'] is not None:
            adata = st.session_state['uploaded_adata']
            
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
                    options=["rna", "atac"],
                    help="Select the data modality (RNA or ATAC experiment)",
                    key="modality_select"
                )
            
            with col2:
                # Dynamic assay options based on modality
                if modality == "rna":
                    assay_options = ["sc", "sn", "multi_rna"]
                else:  # atac
                    assay_options = ["sn", "multi_atac"]
                
                active_assay = st.selectbox(
                    "Active Assay",
                    options=assay_options,
                    help="Select the assay type (Single-cell: sc, Single-nucleus: sn, Multiome: multi_rna/multi_atac)",
                    key="assay_select"
                )
            
            # Additional options
            with st.expander("Advanced Options"):
                min_counts = st.number_input(
                    "Minimum counts per cell",
                    min_value=500,
                    max_value=5000,
                    value=800,
                    step=50,
                    help="Filter out cells with fewer than this many genes",
                    key="min_counts_input"
                )

                #choose between nan_or_zero
                nan_or_zero = st.selectbox(
                    options = ["nan","zero"],
                    index = 0,
                    key = "nanzero_select",
                    width=250,
                    help = "Fill missing features for cell typing and doublet detection with 0 values or nan values")

            # Store current parameters
            current_params = {
                'modality': modality,
                'active_assay': active_assay,
                'min_counts': min_counts
            }
            
            # Run annotation button
            if not st.session_state['annotation_complete']:
                if st.button("Run Cell Type Annotation", type="primary", key="run_annotation_btn"):
                    with st.spinner("Running cell type annotation and doublet detection..."):
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

                            # Run annotation
                            annotated_adata = celltype_doublet_workflow(
                                adata_copy,
                                active_assay=active_assay,
                                modality=modality,
                                in_place=True,
                                nan_or_zero = nan_or_zero

                            )

                            # Always check and ensure UMAP exists before visualization
                            if 'X_umap' not in annotated_adata.obsm.keys():
                                st.info("UMAP not found. Computing UMAP for visualization...")
                                if modality == "rna":
                                    sc.pp.highly_variable_genes(annotated_adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
                                    sc.pp.scale(annotated_adata, max_value=10)
                                    sc.tl.pca(annotated_adata, svd_solver='arpack')
                                
                                # Compute UMAP
                                sc.pp.neighbors(annotated_adata)
                                sc.tl.umap(annotated_adata)
                                st.success("UMAP computed successfully!")
                            else:
                                st.success("UMAP embeddings found!")
                            
                            # Store results in session state
                            st.session_state['annotated_adata'] = annotated_adata
                            st.session_state['annotation_params'] = current_params
                            st.session_state['original_dims'] = (original_n_obs, original_n_vars)
                            st.session_state['annotation_complete'] = True
                            
                            st.success("Annotation completed!")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error during annotation: {str(e)}")
                            if st.checkbox("Show detailed error information", key="error_details"):
                                st.exception(e)
            
            # Show results if annotation is complete
            if st.session_state['annotation_complete'] and st.session_state['annotated_adata'] is not None:
                show_annotation_results()


def show_annotation_results():
    """
    Display annotation results, downloads, and visualization
    """
    annotated_adata = st.session_state['annotated_adata']
    params = st.session_state['annotation_params']
    original_n_obs, original_n_vars = st.session_state['original_dims']
    
    # Show filtering results
    if original_n_obs != annotated_adata.n_obs or original_n_vars != annotated_adata.n_vars:
        st.info("**Filtering Results:**")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Cells: {original_n_obs:,} → {annotated_adata.n_obs:,}")
        with col2:
            st.write(f"Genes: {original_n_vars:,} → {annotated_adata.n_vars:,}")
    
    # Show UMAP panel with annotation results
    st.header("Annotation & Doublet Detection Results")
    
    # Create three UMAP plots
    try:
        # Create the three-panel UMAP plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        # Plot 1: Cell types
        sc.pl.umap(annotated_adata, color='cell_type_final', ax=axes[0, 0], show=False, frameon=False)
        axes[0, 0].set_title('Predicted Cell Types')

        sc.pl.umap(annotated_adata, color='predicted_cell_type_proba', ax=axes[0, 1], show=False, frameon=False)
        axes[0, 1].set_title('Predicted Cell Type Probabilities')
        
        # Plot 2: Doublet status
        sc.pl.umap(annotated_adata, color='is_doublet', ax=axes[1, 0], show=False, frameon=False)
        axes[1, 0].set_title('Doublet Detection')
        
        # Plot 3: Doublet scores
        sc.pl.umap(annotated_adata, color='doublet_score', ax=axes[1, 1], show=False, frameon=False)
        axes[1, 1].set_title('Doublet Scores')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
    except Exception as plot_error:
        st.error(f"Error creating UMAP plots: {str(plot_error)}")
        if st.checkbox("Show UMAP plot error details", key="umap_plot_error_details"):
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
        if 'is_doublet' in annotated_adata.obs.columns:
            doublet_counts = annotated_adata.obs['is_doublet'].value_counts()
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
                    file_name=f"annotated_{st.session_state.get('current_file_key', 'data')}.h5ad",
                    mime="application/octet-stream",
                    key="download_h5ad"
                )
    
    with col2:
        # Download cell metadata CSV
        csv_data = annotated_adata.obs.to_csv()
        st.download_button(
            label="Download Cell Metadata (CSV) - this contains all cell annotations and metadata, but no counts data",
            data=csv_data,
            file_name=f"cell_metadata_{st.session_state.get('current_file_key', 'data')}.csv",
            mime="text/csv",
            key="download_csv"
        )
    
    # Doublet Filtering Option
    st.header("Optional: Filter Doublets")
    
    filter_doublets = st.checkbox(
        "Remove predicted doublets from dataset",
        value=False,
        help="Remove cells predicted as doublets for downstream analysis",
        key="filter_doublets_checkbox"
    )
    
    # Prepare final dataset (with or without doublet filtering)
    if filter_doublets and 'is_doublet' in annotated_adata.obs.columns:
        # Filter out doublets
        pre_filter_cells = annotated_adata.n_obs
        final_adata = annotated_adata[~annotated_adata.obs['is_doublet']].copy()
        post_filter_cells = final_adata.n_obs
        removed_cells = pre_filter_cells - post_filter_cells
        
        st.info(f"**Doublet Filtering Applied:**")
        st.write(f"Cells before filtering: {pre_filter_cells:,}")
        st.write(f"Cells after filtering: {post_filter_cells:,}")
        st.write(f"Doublets removed: {removed_cells:,}")
    else:
        final_adata = annotated_adata.copy()
        st.info("No doublet filtering applied - using full annotated dataset.")
    
    # Visualization Section - Always show this at the end
    st.header("Visualize Final Results")
    
    # Get available genes for selection
    modality = params['modality']
    active_assay = params['active_assay']
    
    if modality == "rna":
        available_genes = sorted(final_adata.var_names.tolist())
    else:  # For ATAC, might need different gene list
        available_genes = sorted(final_adata.var_names.tolist())
    
    if available_genes:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            default_gene = (
                st.session_state.get("selected_gene", available_genes[0])
                if st.session_state.get("selected_gene") in available_genes
                else available_genes[0]
            )
            selected_gene = create_gene_selector(
                                        gene_list=available_genes,
                                        key_suffix="sc_annotation")
        
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
                key="color_map_select_annotation_final",
            )
        
        with col3:
            sort_order = st.checkbox("Sort plotted cells by expression", value=False, key="sort_annotation_final")
        
        # Generate plots using imported function
        try:
            # Create plots
            #in final_data add col new_cell_type which is same as cell_type_final
            final_adata.obs['new_cell_type'] = final_adata.obs['cell_type_final']
            gene_fig, cell_type_fig = plot_sc_dataset(
                final_adata, selected_gene, sort_order, color_map
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
            if st.checkbox("Show visualization error details", key="viz_error_details"):
                st.exception(viz_error)
    else:
        st.warning("No genes available for visualization.")
    
    # Reset button
    if st.button("Start New Analysis", type="secondary", key="reset_analysis"):
        # Clear all session state related to annotation
        keys_to_clear = ['uploaded_adata', 'file_processed', 'annotation_params', 
                        'annotation_complete', 'annotated_adata', 'current_file_key', 'original_dims']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()