import streamlit as st
import glob
import os
import pandas as pd
from datetime import datetime
from modules.analytics import (
    add_activity,
    get_session_id
)
from modules.versioning import resolve_versioned_path, version_candidates


def _holds_h5ad(directory):
    return os.path.isdir(directory) and any(
        name.endswith(".h5ad") for name in os.listdir(directory)
    )


def resolve_download_file(path_templates, version, pattern=None):
    """Locate a downloadable file for ``version``, else the newest earlier release.

    Each template takes a ``{version}`` placeholder; several can be given because
    releases have shuffled the directory layout. With ``pattern`` a template names a
    directory and the newest matching file inside it wins, which keeps date-stamped
    exports (``wt_adata_merged_post_scvi_2026_08_20.h5ad``) reachable. Returns
    ``(path, resolved_version)``, or ``(None, None)`` if nothing is found.
    """
    if isinstance(path_templates, str):
        path_templates = (path_templates,)

    def first_hit(candidate):
        for template in path_templates:
            location = template.format(version=candidate)
            if pattern is None:
                if os.path.exists(location):
                    return location
            else:
                matches = glob.glob(os.path.join(location, pattern))
                if matches:
                    return max(matches)
        return None

    for candidate in version_candidates(version):
        hit = first_hit(candidate)
        if hit is not None:
            return hit, candidate
    return None, None


def list_available_h5ad_files(base_path, version="v_0.01", rna_atac="rna"):
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
        curation_path, _ = resolve_versioned_path(
            lambda candidate: f"{base_path}/data/curation/{candidate}/cpa.parquet",
            version,
        )
        if curation_path is None:
            return {}
        curation_data = pd.read_parquet(curation_path)

        # Path to h5ad files, falling back to the newest release that has them
        modality_dir = "sc_data" if rna_atac == "rna" else "sc_atac_data"
        h5ad_dir, _ = resolve_versioned_path(
            lambda candidate: os.path.join(
                base_path, modality_dir, "datasets", candidate, "epitome_h5_files"
            ),
            version,
            exists=_holds_h5ad,
        )
        if h5ad_dir is None:
            return {}

        # List all .h5ad files
        h5ad_files = [f for f in os.listdir(h5ad_dir) if f.endswith(".h5ad")]

        # Create display names using curation data
        downloads = {}
        for h5ad_file in h5ad_files:
            # Extract SRA_ID from filename (remove _processed.h5ad)
            sra_id = h5ad_file.replace("_processed", "")
            sra_id = sra_id.replace(".h5ad", "")

            # Find metadata
            if rna_atac == "rna":
                dataset_info = curation_data[curation_data["SRA_ID"] == sra_id]
            elif rna_atac == "atac":
                dataset_info = curation_data[curation_data["GEO"] == sra_id]
            if not dataset_info.empty:
                author = dataset_info.iloc[0]["Author"]
                name = dataset_info.iloc[0]["Name"]
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
                    file_size = os.path.getsize(file_path) / (
                        1024 * 1024
                    )  # Convert to MB
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
                            help=f"Download processed h5ad file for {sra_id}",
                        )

                # Add divider between entries
                st.markdown("---")


def get_binary_file_downloader_html(file_path, file_name):
    """
    Generate an HTML download link for a local file.
    This avoids loading the file into memory until it's clicked.
    """
    import base64
    import os
    
    # Get the file size for display
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
    size_str = f"{file_size:.1f} MB"
    
    # Generate HTML download link based on file path
    download_link = f"""
    <div style="display: flex; flex-direction: column; align-items: center;">
        <div style="margin-bottom: 0.5rem;"><b>Size:</b> {size_str}</div>
        <a href="file_path:{file_path}" 
           download="{file_name}" 
           target="_blank" 
           style="display: inline-block; 
                 padding: 0.5em 1em; 
                 background-color: #4F8BF9; 
                 color: white; 
                 border-radius: 0.3em; 
                 text-decoration: none; 
                 cursor: pointer; 
                 font-weight: 500;">
           Download H5AD
        </a>
    </div>
    """
    return download_link


def register_download_handler():
    """Register custom file download handler in JavaScript via streamlit"""
    import streamlit as st
    
    # Add JavaScript to handle custom downloads
    js_code = """
    <script>
    // Wait for document to fully load
    document.addEventListener('DOMContentLoaded', function() {
        // Watch for elements that appear during updates
        const observer = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                mutation.addedNodes.forEach(function(node) {
                    if (node.nodeType === Node.ELEMENT_NODE) {
                        // Find all links with file_path: prefix
                        const links = node.querySelectorAll('a[href^="file_path:"]');
                        
                        links.forEach(function(link) {
                            link.addEventListener('click', function(e) {
                                e.preventDefault();
                                
                                // Extract the file path
                                const filePath = link.getAttribute('href').replace('file_path:', '');
                                const fileName = link.getAttribute('download');
                                
                                // Create a download request through Streamlit
                                const downloadEvent = new CustomEvent('streamlit:download', {
                                    detail: {
                                        path: filePath,
                                        name: fileName
                                    }
                                });
                                
                                window.dispatchEvent(downloadEvent);
                            });
                        });
                    }
                });
            });
        });
        
        // Watch for DOM changes
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    });
    </script>
    """
    
    # Inject the JavaScript code
    st.markdown(js_code, unsafe_allow_html=True)


def create_downloads_ui_with_metadata_rna(base_path, version="v_0.01"):
    """
    Create a downloads UI with additional metadata filtering options
    """
    st.header("Download Processed H5AD Files")

    try:
        # Load curation data
        curation_path, _ = resolve_download_file(
            f"{base_path}/data/curation/{{version}}/cpa.parquet", version
        )

        # List available h5ad files
        downloads = list_available_h5ad_files(base_path, version, rna_atac="rna")

        if curation_path is None or not downloads:
            st.warning("No h5ad files available for download at this time.")
            return
        curation_data = pd.read_parquet(curation_path)

        # Filtering options
        st.subheader("Filter Datasets")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Filter by author
            all_authors = sorted(curation_data["Author"].unique())
            selected_authors = st.multiselect(
                "Filter by Author", options=all_authors, default=None,
                key = "author_filter"
            )

        with col2:
            # Filter by data type
            data_types = sorted(curation_data["Modality"].unique())
            selected_types = st.multiselect(
                "Filter by Data Type",
                options=data_types,
                default=None,
                help="sc = single-cell, sn = single-nucleus",
                key= "data_type_filter"

            )

        with col3:
            # Filter by cell count
            min_cells = int(curation_data["n_cells"].min())
            max_cells = int(curation_data["n_cells"].max())
            cell_range = st.slider(
                "Filter by Cell Count",
                min_value=min_cells,
                max_value=max_cells,
                value=(min_cells, max_cells),
                key = "cell_count_filter"
            )

        # Apply filters to curation data
        filtered_data = curation_data.copy()

        if selected_authors:
            filtered_data = filtered_data[
                filtered_data["Author"].isin(selected_authors)
            ]

        if selected_types:
            filtered_data = filtered_data[
                filtered_data["Modality"].isin(selected_types)
            ]

        filtered_data = filtered_data[
            (filtered_data["n_cells"] >= cell_range[0])
            & (filtered_data["n_cells"] <= cell_range[1])
        ]

        # Get filtered SRA_IDs
        filtered_sra_ids = set(filtered_data["SRA_ID"].unique())

        # Filter downloads based on SRA_IDs
        filtered_downloads = {}
        for display_name, file_path in downloads.items():
            sra_id = display_name.split(" - ")[0]
            if sra_id in filtered_sra_ids:
                filtered_downloads[display_name] = file_path

        st.markdown(
            f"Showing {len(filtered_downloads)} of {len(downloads)} available datasets"
        )

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

        # Register download handler (only once)
        if "download_handler_registered" not in st.session_state:
            register_download_handler()
            st.session_state.download_handler_registered = True

        for author, files in sorted(author_groups.items()):
            with st.expander(
                f"{author} ({len(files)} datasets)",
                expanded=True if len(author_groups) <= 3 else False,
            ):
                for display_name, file_path in sorted(files):
                    # Extract info
                    parts = display_name.split(" - ")
                    sra_id = parts[0]
                    sample_name = parts[1] if len(parts) > 1 else ""

                    # Get detailed metadata
                    dataset_meta = (
                        filtered_data[filtered_data["SRA_ID"] == sra_id].iloc[0]
                        if len(filtered_data[filtered_data["SRA_ID"] == sra_id]) > 0
                        else None
                    )

                    # Create columns for layout
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.markdown(f"**{display_name}**")
                        if dataset_meta is not None:
                            st.markdown(
                                f"""
                                - **Cells**: {int(dataset_meta['n_cells']):,}
                                - **Type**: {dataset_meta['Modality']}
                                - **Sex**: {'Male' if dataset_meta['Comp_sex'] == 1 else 'Female' if dataset_meta['Comp_sex'] == 0 else 'Unknown'}
                                - **Age**: {dataset_meta['Age']}
                                - **Wild-type**: {'Yes' if dataset_meta['Normal'] == 1 else 'No'}
                            """
                            )

                    with col2:
                        # Alternative approach - use a session state based callback
                        if f"button_{sra_id}" not in st.session_state:
                            st.session_state[f"button_{sra_id}"] = False

                        # Show file size
                        try:
                            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
                            size_str = f"{file_size:.1f} MB"
                            st.markdown(f"**Size**: {size_str}")
                        except:
                            st.markdown("**Size**: Unknown")
                            
                        # Button to trigger download preparation
                        if st.button("Prepare download H5AD", key=f"download_button_{sra_id}"):
                            st.session_state[f"button_{sra_id}"] = True
                        
                        # Only prepare download data if button was clicked
                        if st.session_state[f"button_{sra_id}"]:
                            add_activity(
                            value=sra_id,
                            analysis="Download",
                            user=st.session_state.session_id,
                            time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        )
                            try:
                                with open(file_path, "rb") as f:
                                    st.download_button(
                                        label="Click to download",
                                        data=lambda fp=file_path: open(fp, "rb").read(),
                                        file_name=f"{sra_id}_processed.h5ad",
                                        mime="application/octet-stream",
                                        key=f"actual_download_{sra_id}"
                                    )
                                #st.info(
                                #f"Download only available after pre-print release"
                                #)
                                # Reset the state after download is prepared
                                st.session_state[f"button_{sra_id}"] = False
                            except Exception as e:
                                st.error(f"Error preparing download: {str(e)}")

                    # Add divider
                    st.markdown("---")
    except Exception as e:
        st.error(f"Error loading or displaying downloads: {str(e)}")
        return





def create_downloads_ui_with_metadata_atac(base_path, version="v_0.01"):
    """
    Create a downloads UI with additional metadata filtering options
    """
    st.header("Download Processed H5AD Files")

    try:
        # Load curation data
        curation_path, _ = resolve_download_file(
            f"{base_path}/data/curation/{{version}}/cpa.parquet", version
        )

        # List available h5ad files
        downloads = list_available_h5ad_files(base_path, version,rna_atac="atac")

        if curation_path is None or not downloads:
            st.warning("No h5ad files available for download at this time.")
            return
        curation_data = pd.read_parquet(curation_path)

        # Filtering options
        st.subheader("Filter Datasets")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Filter by author
            all_authors = sorted(curation_data["Author"].unique())
            selected_authors = st.multiselect(
                "Filter by Author", options=all_authors, default=None,
                key = "author_filter_atac"
            )

        with col2:
            # Filter by data type
            data_types = sorted(curation_data["Modality"].unique())
            selected_types = st.multiselect(
                "Filter by Data Type",
                options=data_types,
                default=None,
                help="sc = single-cell, sn = single-nucleus, atac = chromatin accessibility",
                key = "data_type_filter_atac"
            )

        with col3:
            # Filter by cell count
            min_cells = int(curation_data["n_cells"].min())
            max_cells = int(curation_data["n_cells"].max())
            cell_range = st.slider(
                "Filter by Cell Count",
                min_value=min_cells,
                max_value=max_cells,
                value=(min_cells, max_cells),
                key = "cell_count_filter_atac"
            )

        # Apply filters to curation data
        filtered_data = curation_data.copy()

        if selected_authors:
            filtered_data = filtered_data[
                filtered_data["Author"].isin(selected_authors)
            ]

        if selected_types:
            filtered_data = filtered_data[
                filtered_data["Modality"].isin(selected_types)
            ]

        filtered_data = filtered_data[
            (filtered_data["n_cells"] >= cell_range[0])
            & (filtered_data["n_cells"] <= cell_range[1])
        ]

        # Get filtered SRA_IDs
        filtered_sra_ids = set(filtered_data["GEO"].unique())

        # Filter downloads based on SRA_IDs
        filtered_downloads = {}
        for display_name, file_path in downloads.items():
            sra_id = display_name.split(" - ")[0]
            if sra_id in filtered_sra_ids:
                filtered_downloads[display_name] = file_path

        st.markdown(
            f"Showing {len(filtered_downloads)} of {len(downloads)} available datasets"
        )

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

        # Register download handler (only once)
        if "download_handler_registered" not in st.session_state:
            register_download_handler()
            st.session_state.download_handler_registered = True

        for author, files in sorted(author_groups.items()):
            with st.expander(
                f"{author} ({len(files)} datasets)",
                expanded=True if len(author_groups) <= 3 else False,
            ):
                for display_name, file_path in sorted(files):
                    # Extract info
                    parts = display_name.split(" - ")
                    sra_id = parts[0]
                    sample_name = parts[1] if len(parts) > 1 else ""

                    # Get detailed metadata
                    dataset_meta = (
                        filtered_data[filtered_data["SRA_ID"] == sra_id].iloc[0]
                        if len(filtered_data[filtered_data["SRA_ID"] == sra_id]) > 0
                        else None
                    )

                    # Create columns for layout
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.markdown(f"**{display_name}**")
                        if dataset_meta is not None:
                            st.markdown(
                                f"""
                                - **Cells**: {int(dataset_meta['n_cells']):,}
                                - **Type**: {dataset_meta['Modality']}
                                - **Sex**: {'Male' if dataset_meta['Comp_sex'] == 1 else 'Female' if dataset_meta['Comp_sex'] == 0 else 'Unknown'}
                                - **Age**: {dataset_meta['Age']}
                                - **Wild-type**: {'Yes' if dataset_meta['Normal'] == 1 else 'No'}
                            """
                            )

                    with col2:
                        # Alternative approach - use a session state based callback
                        if f"button_{sra_id}" not in st.session_state:
                            st.session_state[f"button_{sra_id}"] = False

                        # Show file size
                        try:
                            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
                            size_str = f"{file_size:.1f} MB"
                            st.markdown(f"**Size**: {size_str}")
                        except:
                            st.markdown("**Size**: Unknown")
                            
                        # Button to trigger download preparation
                        if st.button("Prepare download H5AD", key=f"download_button_{sra_id}_atac"):
                            st.session_state[f"button_{sra_id}"] = True
                        
                        # Only prepare download data if button was clicked
                        if st.session_state[f"button_{sra_id}"]:
                            add_activity(
                            value=sra_id,
                            analysis="Download",
                            user=st.session_state.session_id,
                            time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        )
                            try:
                                with open(file_path, "rb") as f:
                                    st.download_button(
                                        label="Click to download",
                                        data=lambda fp=file_path: open(fp, "rb").read(),
                                        file_name=f"{sra_id}.h5ad",
                                        mime="application/octet-stream",
                                        key=f"actual_download_{sra_id}_atac"
                                    )
                              #  st.info(
                             #   f"Download only available after pre-print release"
                            #)
                                # Reset the state after download is prepared
                                st.session_state[f"button_{sra_id}"] = False
                            except Exception as e:
                                st.error(f"Error preparing download: {str(e)}")

                    # Add divider
                    st.markdown("---")
    except Exception as e:
        st.error(f"Error loading or displaying downloads: {str(e)}")
        return
    



def create_bulk_data_downloads_ui(base_path, version="v_0.02"):
    """
    Create UI for downloading bulk data files (matrices, metadata, etc.)
    """
    st.header("Download Bulk Data Files")

    # Define available bulk data files
    bulk_data = {

        "Single-cell objects" : [
            {
                "name": "Integrated AnnData object (wt cells, RNA) (.h5ad)",
                "path": (
                    f"{base_path}/data/additional_downloads/wt_adata/{{version}}",
                    f"{base_path}/data/additional_downloads/{{version}}/wt_adata",
                    f"{base_path}/data/additional_downloads/{{version}}",
                ),
                "pattern": "wt_adata*.h5ad",
                "description": "Integrated AnnData object containing all wild-type cells (RNA only)",
            },
            {
                "name": "Integrated AnnData object (mut/treated cells, RNA) (.h5ad)",
                "path": (
                    f"{base_path}/data/additional_downloads/mut_adata/{{version}}",
                    f"{base_path}/data/additional_downloads/{{version}}/mut_adata",
                    f"{base_path}/data/additional_downloads/{{version}}",
                ),
                "pattern": "mut_adata*.h5ad",
                "description": "Integrated AnnData object containing all mutant/treated cells (RNA only)",
            },
            {
                "name": "Integrated AnnData object (all cells, ATAC) (.h5ad)",
                "path": (
                    f"{base_path}/data/additional_downloads/atac_adata/{{version}}",
                    f"{base_path}/data/additional_downloads/{{version}}/atac_adata",
                    f"{base_path}/data/additional_downloads/{{version}}",
                ),
                "pattern": "atac_adata*.h5ad",
                "description": "Integrated AnnData object containing all cells (ATAC only)",
            },
        ],

        "Expression Matrices": [
            {
                "name": "Normalized Expression Matrix (.mtx)",
                "path": f"{base_path}/data/expression/{{version}}/normalized_data.mtx",
                "description": "Log-normalized expression matrix with genes as rows and cells as columns",
            },
            {
                "name": "Gene Information (.parquet)",
                "path": f"{base_path}/data/expression/{{version}}/genes.parquet",
                "description": "Information about genes in the expression matrix",
            },
            {
                "name": "Metadata (.parquet)",
                "path": f"{base_path}/data/expression/{{version}}/meta_data.parquet",
                "description": "Cell metadata including cell type, sample information, and experimental details",
            },
        ],
        "Accessibility Data": [
            {
                "name": "Accessibility Matrix (.mtx)",
                "path": f"{base_path}/data/accessibility/{{version}}/normalized_data.mtx",
                "description": "Normalized accessibility matrix with peaks as rows and cells as columns",
            },
            {
                "name": "Peak Information (.parquet)",
                "path": f"{base_path}/data/accessibility/{{version}}/accessibility_features.parquet",
                "description": "Information about genomic peaks in the accessibility matrix",
            },
            {
                "name": "ATAC Metadata (.parquet)",
                "path": f"{base_path}/data/accessibility/{{version}}/atac_meta_data.parquet",
                "description": "Cell metadata for ATAC-seq samples",
            },
        ],
        "Cell Type Markers": [
            {
                "name": "Cell Type Markers (.parquet)",
                "path": f"{base_path}/data/markers/{{version}}/cell_typing_markers.parquet",
                "description": "Marker genes for cell type identification",
            },
            {
                "name": "Lineage Markers (.parquet)",
                "path": f"{base_path}/data/markers/{{version}}/grouping_lineage_markers.parquet",
                "description": "Marker genes for lineage identification and cell grouping",
            },
        ],
        "Curation Data": [
            {
                "name": "Curation Data (.parquet)",
                "path": f"{base_path}/data/curation/{{version}}/cpa.parquet",
                "description": "Curated metadata for all samples in the atlas",
            }
        ],
        "Consensus Chromatin Landscape v2 (parquet)": [
            {
                "name": "Consensus Chromatin Landscape (.parquet)",
                "path": f"{base_path}/data/additional_downloads/consensus_chromatin_landscape/v_0.02/consensus_chromatin_landscape.parquet",
                "description": "Precomputed consensus chromatin landscape for pituitary ATAC datasets (mm10 assembly)",
            }
        ],

        "Consensus Chromatin Landscape v2 (bed)": [
            {
                "name": "Consensus Chromatin Landscape (.bed)",
                "path": f"{base_path}/data/additional_downloads/consensus_chromatin_landscape/v_0.02/consensus_chromatin_landscape.bed",
                "description": "Precomputed consensus chromatin landscape for pituitary ATAC datasets (mm10 assembly)",
            }
        ],
    }

    # Display each category
    for category, files in bulk_data.items():
        st.subheader(category)

        for file_info in files:
            file_path, resolved_version = resolve_download_file(
                file_info["path"], version, file_info.get("pattern")
            )
            if file_path is not None:
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
                    if resolved_version != version:
                        st.caption(f"Fall back to {resolved_version}")
                    
                    # Use button + session state approach to prevent loading all files
                    file_key = f"{category}_{os.path.basename(file_path)}"
                    if f"button_{file_key}" not in st.session_state:
                        st.session_state[f"button_{file_key}"] = False
                    
                    # Button to trigger download preparation
                    if st.button("Prepare download", key=f"download_button_{file_key}"):
                        st.session_state[f"button_{file_key}"] = True
                    
                    # Only prepare download data if button was clicked
                    if st.session_state[f"button_{file_key}"]:
                        get_session_id()
                        add_activity(
                            value=file_info["name"],
                            analysis="Download",
                            user=st.session_state.session_id,
                            time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        )
                        try:
                            with open(file_path, "rb") as f:
                                st.download_button(
                                    label="Click to download",
                                    data=lambda fp=file_path: open(fp, "rb").read(),
                                    file_name=os.path.basename(file_path),
                                    mime="application/octet-stream",
                                    key=f"actual_download_{file_key}"
                                )
                            #st.info(
                            #    f"Download only available after pre-print release"
                            #)
                            # Reset the state after download is prepared
                            st.session_state[f"button_{file_key}"] = False
                        except Exception as e:
                            st.error(f"Error preparing download: {str(e)}")
            else:
                st.warning(
                    f"{file_info['name']} is not available for {version} "
                    "or any earlier version"
                )

            st.markdown("---")