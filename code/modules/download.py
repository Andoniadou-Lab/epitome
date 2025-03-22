import streamlit as st
import os
import pandas as pd

import streamlit as st
import os
import pandas as pd

def list_available_h5ad_files(base_path, version="v_0.01"):
    """
    List all available h5ad files with metadata for downloading
    
    Parameters:
    -----------
    base_path : str
        Base path to data directory
    version : str
        Version of the dataset
    
    Returns:
    --------
    dict
        Dictionary mapping display names to file paths
    """
    try:
        # Load curation data for metadata
        curation_data = pd.read_parquet(f'{base_path}/data/curation/{version}/cpa.parquet')
        
        # Path to h5ad files
        h5ad_dir = os.path.join(base_path, 'sc_data','datasets', version, 'epitome_h5_files')
        
        # Check if directory exists
        if not os.path.exists(h5ad_dir):
            return {}
        
        # List all .h5ad files
        h5ad_files = [f for f in os.listdir(h5ad_dir) if f.endswith('.h5ad')]
        
        # Create display names using curation data
        downloads = {}
        for h5ad_file in h5ad_files:
            # Extract SRA_ID from filename (remove _processed.h5ad)
            sra_id = h5ad_file.replace('_processed.h5ad', '')
            
            # Find metadata
            dataset_info = curation_data[curation_data['SRA_ID'] == sra_id]
            if not dataset_info.empty:
                author = dataset_info.iloc[0]['Author']
                name = dataset_info.iloc[0]['Name']
                display_name = f"{sra_id} - {name} - {author}"
            else:
                display_name = sra_id
                
            # Full path to file
            file_path = os.path.join(h5ad_dir, h5ad_file)
            downloads[display_name] = file_path
            
        return downloads
    
    except Exception as e:
        st.error(f"Error listing h5ad files: {str(e)}")
        return {}

def create_downloads_ui(base_path, version="v_0.01"):
    """
    Create a downloads UI for h5ad files
    """
    st.header("Download Processed H5AD Files")
    
    # List available files
    downloads = list_available_h5ad_files(base_path, version)
    
    if not downloads:
        st.warning("No h5ad files available for download at this time.")
        return
    
    # Group by author
    author_groups = {}
    for display_name, file_path in downloads.items():
        try:
            # Extract author from display name
            author = display_name.split(" - ")[2]
            if author not in author_groups:
                author_groups[author] = []
            author_groups[author].append((display_name, file_path))
        except:
            # Handle case where display name format is different
            if "Other" not in author_groups:
                author_groups["Other"] = []
            author_groups["Other"].append((display_name, file_path))
    
    # Display files by author
    for author, files in sorted(author_groups.items()):
        with st.expander(f"{author} ({len(files)} datasets)"):
            for display_name, file_path in sorted(files):
                # Create file size string
                try:
                    file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
                    size_str = f"{file_size:.1f} MB"
                except:
                    size_str = "Unknown size"
                
                # Extract SRA_ID
                sra_id = display_name.split(" - ")[0]
                
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"**{display_name}**")
                with col2:
                    # Use st.download_button for the download link
                    with open(file_path, "rb") as f:
                        st.download_button(
                            label=f"Download ({size_str})",
                            data=f,
                            file_name=f"{sra_id}_processed.h5ad",
                            mime="application/octet-stream",
                            key=f"download_{sra_id}",
                            help=f"Download processed h5ad file for {sra_id}"
                        )
                
                # Add divider between entries
                st.markdown("---")

def create_downloads_ui_with_metadata(base_path, version="v_0.01"):
    """
    Create a downloads UI with additional metadata filtering options
    """
    st.header("Download Processed H5AD Files")
    
    try:
        # Load curation data
        curation_data = pd.read_parquet(f'{base_path}/data/curation/{version}/cpa.parquet')
        
        # List available h5ad files
        downloads = list_available_h5ad_files(base_path, version)
        
        if not downloads:
            st.warning("No h5ad files available for download at this time.")
            return
        
        # Filtering options
        st.subheader("Filter Datasets")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Filter by author
            all_authors = sorted(curation_data['Author'].unique())
            selected_authors = st.multiselect(
                'Filter by Author',
                options=all_authors,
                default=None
            )
        
        with col2:
            # Filter by data type
            data_types = sorted(curation_data['sc_sn_atac'].unique())
            selected_types = st.multiselect(
                'Filter by Data Type',
                options=data_types,
                default=None,
                help="sc = single-cell, sn = single-nucleus, atac = chromatin accessibility"
            )
        
        with col3:
            # Filter by cell count
            min_cells = int(curation_data['n_cells'].min())
            max_cells = int(curation_data['n_cells'].max())
            cell_range = st.slider(
                'Filter by Cell Count',
                min_value=min_cells,
                max_value=max_cells,
                value=(min_cells, max_cells)
            )
        
        # Apply filters to curation data
        filtered_data = curation_data.copy()
        
        if selected_authors:
            filtered_data = filtered_data[filtered_data['Author'].isin(selected_authors)]
            
        if selected_types:
            filtered_data = filtered_data[filtered_data['sc_sn_atac'].isin(selected_types)]
            
        filtered_data = filtered_data[
            (filtered_data['n_cells'] >= cell_range[0]) & 
            (filtered_data['n_cells'] <= cell_range[1])
        ]
        
        # Get filtered SRA_IDs
        filtered_sra_ids = set(filtered_data['SRA_ID'].unique())
        
        # Filter downloads based on SRA_IDs
        filtered_downloads = {}
        for display_name, file_path in downloads.items():
            sra_id = display_name.split(" - ")[0]
            if sra_id in filtered_sra_ids:
                filtered_downloads[display_name] = file_path
        
        st.markdown(f"Showing {len(filtered_downloads)} of {len(downloads)} available datasets")
        
        # Group by author
        author_groups = {}
        for display_name, file_path in filtered_downloads.items():
            try:
                # Extract author from display name
                author = display_name.split(" - ")[2]
                if author not in author_groups:
                    author_groups[author] = []
                author_groups[author].append((display_name, file_path))
            except:
                # Handle case where display name format is different
                if "Other" not in author_groups:
                    author_groups["Other"] = []
                author_groups["Other"].append((display_name, file_path))
        
        # Display filtered files grouped by author
        if not author_groups:
            st.warning("No datasets match the selected filters.")
            return
            
        for author, files in sorted(author_groups.items()):
            with st.expander(f"{author} ({len(files)} datasets)", expanded=True if len(author_groups) <= 3 else False):
                for display_name, file_path in sorted(files):
                    # Extract info
                    parts = display_name.split(" - ")
                    sra_id = parts[0]
                    sample_name = parts[1] if len(parts) > 1 else ""
                    
                    # Get detailed metadata
                    dataset_meta = filtered_data[filtered_data['SRA_ID'] == sra_id].iloc[0] if len(filtered_data[filtered_data['SRA_ID'] == sra_id]) > 0 else None
                    
                    # Create columns for layout
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**{display_name}**")
                        if dataset_meta is not None:
                            st.markdown(f"""
                                - **Cells**: {int(dataset_meta['n_cells']):,}
                                - **Type**: {dataset_meta['sc_sn_atac']}
                                - **Sex**: {'Male' if dataset_meta['Comp_sex'] == 1 else 'Female' if dataset_meta['Comp_sex'] == 0 else 'Unknown'}
                                - **Age**: {dataset_meta['Age']}
                                - **Wild-type**: {'Yes' if dataset_meta['Normal'] == 1 else 'No'}
                            """)
                    
                    with col2:
                        # Show file size and provide download button
                        try:
                            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
                            size_str = f"{file_size:.1f} MB"
                        except:
                            size_str = "Unknown size"
                        
                        st.markdown(f"**Size**: {size_str}")
                        st.download_button(
                            label="Download H5AD",
                            data=open(file_path, "rb"),
                            file_name=f"{sra_id}_processed.h5ad",
                            mime="application/octet-stream",
                            help=f"Download processed h5ad file for {sra_id}"
                        )
                    
                    # Add divider
                    st.markdown("---")
    except Exception as e:
        st.error(f"Error loading or displaying downloads: {str(e)}")
        return

def create_bulk_data_downloads_ui(base_path, version="v_0.01"):
    """
    Create UI for downloading bulk data files (matrices, metadata, etc.)
    """
    st.header("Download Bulk Data Files")
    
    # Define available bulk data files
    bulk_data = {
        "Expression Matrices": [
            {
                "name": "Normalized Expression Matrix (.mtx)",
                "path": f"{base_path}/data/expression/{version}/normalized_data.mtx",
                "description": "Log-normalized expression matrix with genes as rows and cells as columns"
            },
            {
                "name": "Gene Information (.parquet)",
                "path": f"{base_path}/data/expression/{version}/genes.parquet",
                "description": "Information about genes in the expression matrix"
            },
            {
                "name": "Metadata (.parquet)",
                "path": f"{base_path}/data/expression/{version}/meta_data.parquet", 
                "description": "Cell metadata including cell type, sample information, and experimental details"
            }
        ],
        "Accessibility Data": [
            {
                "name": "Accessibility Matrix (.mtx)",
                "path": f"{base_path}/data/accessibility/{version}/normalized_data.mtx",
                "description": "Normalized accessibility matrix with peaks as rows and cells as columns"
            },
            {
                "name": "Peak Information (.parquet)",
                "path": f"{base_path}/data/accessibility/{version}/accessibility_features.parquet",
                "description": "Information about genomic peaks in the accessibility matrix"
            },
            {
                "name": "ATAC Metadata (.parquet)",
                "path": f"{base_path}/data/accessibility/{version}/atac_meta_data.parquet",
                "description": "Cell metadata for ATAC-seq samples"
            }
        ],
        "Cell Type Markers": [
            {
                "name": "Cell Type Markers (.parquet)",
                "path": f"{base_path}/data/markers/{version}/cell_typing_markers.parquet",
                "description": "Marker genes for cell type identification"
            },
            {
                "name": "Lineage Markers (.parquet)",
                "path": f"{base_path}/data/markers/{version}/grouping_lineage_markers.parquet",
                "description": "Marker genes for lineage identification and cell grouping"
            }
        ],
        "Curation Data": [
            {
                "name": "Curation Data (.parquet)",
                "path": f"{base_path}/data/curation/{version}/cpa.parquet",
                "description": "Curated metadata for all samples in the atlas"
            }
        ]
    }
    
    # Display each category
    for category, files in bulk_data.items():
        st.subheader(category)
        
        for file_info in files:
            file_path = file_info["path"]
            if os.path.exists(file_path):
                try:
                    file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
                    size_str = f"{file_size:.1f} MB"
                except:
                    size_str = "Unknown size"
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{file_info['name']}**")
                    st.markdown(file_info["description"])
                
                with col2:
                    st.markdown(f"**Size**: {size_str}")
                    with open(file_path, "rb") as f:
                        st.download_button(
                            label="Download",
                            data=f,
                            file_name=os.path.basename(file_path),
                            mime="application/octet-stream",
                            key=f"download_{category}_{os.path.basename(file_path)}",
                            help=f"Download {file_info['name']}"
                        )
            else:
                st.warning(f"{file_info['name']} is not available")
            
            st.markdown("---")