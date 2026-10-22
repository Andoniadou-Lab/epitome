import streamlit as st
import glob
import os
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional, Sequence
from modules.analytics import (
    add_activity,
    get_session_id
)
from modules.versioning import resolve_versioned_path, version_candidates


PREPRINT_NOTICE = "Downloads will be available upon pre-printing"


def _sex_label(value):
    if value == 1:
        return "Male"
    if value == 0:
        return "Female"
    return "Unknown"


def _cells_label(value):
    return f"{int(value):,}" if pd.notna(value) else "Unknown"


def describe_mouse_dataset(meta):
    return f"""
                                - **Cells**: {_cells_label(meta['n_cells'])}
                                - **Type**: {meta['Modality']}
                                - **Sex**: {_sex_label(meta['Comp_sex'])}
                                - **Age**: {meta['Age']}
                                - **Wild-type**: {'Yes' if meta['Normal'] == 1 else 'No'}
                            """


def describe_tumour_dataset(meta):
    age = meta.get("Age_numeric")
    age_str = f"{age:g} years" if pd.notna(age) else "Unknown"
    status = "Healthy" if meta.get("Normal") == 1 else "Tumour"
    lines = [
        f"- **Cells**: {_cells_label(meta.get('n_cells'))}",
        f"- **Type**: {meta.get('Modality', 'Unknown')}",
        f"- **Sex**: {_sex_label(meta.get('Comp_sex'))}",
        f"- **Age**: {age_str}",
        f"- **Status**: {status}",
    ]
    tumour_type = meta.get("Tumor_pta")
    if status == "Tumour" and pd.notna(tumour_type) and str(tumour_type).strip():
        lines.append(f"- **Tumour type**: {str(tumour_type).replace('_', ' ')}")
    return "\n".join(lines)


@dataclass(frozen=True)
class DownloadSite:
    """Where a site keeps its downloadable files and how its download buttons behave.

    Path templates are relative to ``base_path`` and take a ``{version}`` placeholder.
    With ``preprint_only`` every download button serves ``PREPRINT_NOTICE`` as a text
    file instead of the real file.
    """

    curation_template: str
    h5ad_dir_templates: dict = field(default_factory=dict)
    describe_dataset: Callable = describe_mouse_dataset
    available_versions: Optional[Callable[[], Sequence[str]]] = None
    key_prefix: str = ""
    activity_label: str = "Download"
    preprint_only: bool = False

    def versions(self):
        return list(self.available_versions()) if self.available_versions else None


def _pta_versions():
    from modules.pta.config import list_pta_versions

    return list_pta_versions()


MOUSE_SITE = DownloadSite(
    curation_template="data/curation/{version}/cpa.parquet",
    h5ad_dir_templates={
        "rna": "sc_data/datasets/{version}/epitome_h5_files",
        "atac": "sc_atac_data/datasets/{version}/epitome_h5_files",
    },
)

TUMOUR_SITE = DownloadSite(
    curation_template="pta_data/curation/{version}/cpa.parquet",
    h5ad_dir_templates={"rna": "pta_data/sc_data/datasets/{version}/epitome_h5_files"},
    describe_dataset=describe_tumour_dataset,
    available_versions=_pta_versions,
    key_prefix="tumor_",
    activity_label="Tumor Download",
    preprint_only=True,
)


def _holds_h5ad(directory):
    return os.path.isdir(directory) and any(
        name.endswith(".h5ad") for name in os.listdir(directory)
    )


def resolve_download_file(path_templates, version, pattern=None, available=None):
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

    for candidate in version_candidates(version, available):
        hit = first_hit(candidate)
        if hit is not None:
            return hit, candidate
    return None, None


def list_available_h5ad_files(base_path, version="v_0.01", rna_atac="rna", site=MOUSE_SITE):
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
            lambda candidate: os.path.join(
                base_path, site.curation_template.format(version=candidate)
            ),
            version,
            available=site.versions(),
        )
        if curation_path is None:
            return {}
        curation_data = pd.read_parquet(curation_path)

        # Path to h5ad files, falling back to the newest release that has them
        h5ad_template = site.h5ad_dir_templates.get(rna_atac)
        if h5ad_template is None:
            return {}
        h5ad_dir, _ = resolve_versioned_path(
            lambda candidate: os.path.join(
                base_path, h5ad_template.format(version=candidate)
            ),
            version,
            available=site.versions(),
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


def _offer_download(site, file_path, file_name, key):
    """Final download button; with ``site.preprint_only`` it serves the notice text."""
    if site.preprint_only:
        st.download_button(
            label="Click to download",
            data=PREPRINT_NOTICE,
            file_name="download_notice.txt",
            mime="text/plain",
            key=key,
        )
        return
    with open(file_path, "rb"):
        st.download_button(
            label="Click to download",
            data=lambda fp=file_path: open(fp, "rb").read(),
            file_name=file_name,
            mime="application/octet-stream",
            key=key,
        )


def _log_download(site, value):
    get_session_id()
    add_activity(
        value=value,
        analysis=site.activity_label,
        user=st.session_state.session_id,
        time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


_H5AD_MODALITIES = {
    "rna": {
        "id_col": "SRA_ID",
        "key_suffix": "",
        "type_help": "sc = single-cell, sn = single-nucleus",
        "file_name": "{id}_processed.h5ad",
    },
    "atac": {
        "id_col": "GEO",
        "key_suffix": "_atac",
        "type_help": "sc = single-cell, sn = single-nucleus, atac = chromatin accessibility",
        "file_name": "{id}.h5ad",
    },
}


def create_h5ad_downloads_ui(base_path, version, rna_atac="rna", site=MOUSE_SITE):
    """Filterable per-dataset h5ad downloads, grouped by author."""
    spec = _H5AD_MODALITIES[rna_atac]
    id_col = spec["id_col"]
    key = lambda name: f"{site.key_prefix}{name}{spec['key_suffix']}"

    st.header("Download Processed H5AD Files")

    try:
        curation_path, _ = resolve_download_file(
            os.path.join(base_path, site.curation_template),
            version,
            available=site.versions(),
        )
        downloads = list_available_h5ad_files(base_path, version, rna_atac=rna_atac, site=site)

        if curation_path is None or not downloads:
            st.warning("No h5ad files available for download at this time.")
            return
        curation_data = pd.read_parquet(curation_path)

        st.subheader("Filter Datasets")

        col1, col2, col3 = st.columns(3)

        with col1:
            all_authors = sorted(curation_data["Author"].dropna().unique())
            selected_authors = st.multiselect(
                "Filter by Author", options=all_authors, default=None,
                key=key("author_filter"),
            )

        with col2:
            data_types = sorted(curation_data["Modality"].dropna().unique())
            selected_types = st.multiselect(
                "Filter by Data Type",
                options=data_types,
                default=None,
                help=spec["type_help"],
                key=key("data_type_filter"),
            )

        with col3:
            min_cells = int(curation_data["n_cells"].min())
            max_cells = int(curation_data["n_cells"].max())
            cell_range = st.slider(
                "Filter by Cell Count",
                min_value=min_cells,
                max_value=max_cells,
                value=(min_cells, max_cells),
                key=key("cell_count_filter"),
            )

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

        filtered_ids = set(filtered_data[id_col].unique())

        filtered_downloads = {
            display_name: file_path
            for display_name, file_path in downloads.items()
            if display_name.split(" - ")[0] in filtered_ids
        }

        st.markdown(
            f"Showing {len(filtered_downloads)} of {len(downloads)} available datasets"
        )

        author_groups = {}
        for display_name, file_path in filtered_downloads.items():
            parts = display_name.split(" - ")
            author = parts[2] if len(parts) > 2 else "Other"
            author_groups.setdefault(author, []).append((display_name, file_path))

        if not author_groups:
            st.warning("No datasets match the selected filters.")
            return

        if "download_handler_registered" not in st.session_state:
            register_download_handler()
            st.session_state.download_handler_registered = True

        if site.preprint_only:
            st.info(PREPRINT_NOTICE)

        for author, files in sorted(author_groups.items()):
            with st.expander(
                f"{author} ({len(files)} datasets)",
                expanded=True if len(author_groups) <= 3 else False,
            ):
                for display_name, file_path in sorted(files):
                    dataset_id = display_name.split(" - ")[0]
                    matches = filtered_data[filtered_data[id_col] == dataset_id]
                    dataset_meta = matches.iloc[0] if len(matches) > 0 else None

                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.markdown(f"**{display_name}**")
                        if dataset_meta is not None:
                            st.markdown(site.describe_dataset(dataset_meta))

                    with col2:
                        state_key = key(f"button_{dataset_id}")
                        if state_key not in st.session_state:
                            st.session_state[state_key] = False

                        try:
                            file_size = os.path.getsize(file_path) / (1024 * 1024)
                            st.markdown(f"**Size**: {file_size:.1f} MB")
                        except OSError:
                            st.markdown("**Size**: Unknown")

                        if st.button("Prepare download H5AD", key=key(f"download_button_{dataset_id}")):
                            st.session_state[state_key] = True

                        if st.session_state[state_key]:
                            _log_download(site, dataset_id)
                            try:
                                _offer_download(
                                    site,
                                    file_path,
                                    spec["file_name"].format(id=dataset_id),
                                    key=key(f"actual_download_{dataset_id}"),
                                )
                                st.session_state[state_key] = False
                            except Exception as e:
                                st.error(f"Error preparing download: {str(e)}")

                    st.markdown("---")
    except Exception as e:
        st.error(f"Error loading or displaying downloads: {str(e)}")
        return


def create_downloads_ui_with_metadata_rna(base_path, version="v_0.01", site=MOUSE_SITE):
    """Create a downloads UI with additional metadata filtering options"""
    create_h5ad_downloads_ui(base_path, version, rna_atac="rna", site=site)


def create_downloads_ui_with_metadata_atac(base_path, version="v_0.01", site=MOUSE_SITE):
    """Create a downloads UI with additional metadata filtering options"""
    create_h5ad_downloads_ui(base_path, version, rna_atac="atac", site=site)


def mouse_analysis_files(base_path):
    return {

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


def tumour_analysis_files(base_path):
    pta = f"{base_path}/pta_data"
    bulk = f"{pta}/bulk_expression/{{version}}"
    pseudobulk = f"{pta}/pseudobulk/{{version}}"
    abundance = f"{pta}/cell_proportion/{{version}}"
    return {
        "Bulk RNA-seq - Main Cohort (shared genes)": [
            {
                "name": "Normalized Expression Matrix (.mtx)",
                "path": f"{bulk}/normalized_data.mtx",
                "description": "Normalized expression matrix of the integrated bulk cohort (all datasets), genes as rows and samples as columns",
            },
            {
                "name": "Gene Names (.txt)",
                "path": f"{bulk}/genes.txt",
                "description": "Row names of the expression matrix",
            },
            {
                "name": "Sample Names (.txt)",
                "path": f"{bulk}/datasets.txt",
                "description": "Column names of the expression matrix",
            },
            {
                "name": "Sample Metadata (.csv)",
                "path": f"{bulk}/meta_data.csv",
                "description": "Sample metadata including tumour type, lineage, sex and study",
            },
        ],
        "Bulk RNA-seq - Main Cohort (extended gene set)": [
            {
                "name": "Normalized Expression Matrix (.mtx)",
                "path": f"{bulk}/normalized_data_more_genes.mtx",
                "description": "Normalized expression matrix of the raw-aligned datasets with the extended gene set, genes as rows and samples as columns",
            },
            {
                "name": "Gene Names (.txt)",
                "path": f"{bulk}/genes_more_genes.txt",
                "description": "Row names of the extended expression matrix",
            },
            {
                "name": "Sample Names (.txt)",
                "path": f"{bulk}/datasets_more_genes.txt",
                "description": "Column names of the extended expression matrix",
            },
            {
                "name": "Sample Metadata (.csv)",
                "path": f"{bulk}/meta_data_more_genes.csv",
                "description": "Sample metadata for the extended expression matrix",
            },
        ],
        "Bulk RNA-seq - Validation Cohorts": [
            {
                "name": "Zhang et al., 2022 (.csv)",
                "path": f"{bulk}/zhang.csv",
                "description": "Expression table of the Zhang et al., 2022 validation cohort (FPKM)",
            },
            {
                "name": "Jotanovic et al., 2024 (.csv)",
                "path": f"{bulk}/jotanovic.csv",
                "description": "Expression table of the Jotanovic et al., 2024 validation cohort (TPM)",
            },
        ],
        "Bulk RNA-seq Curation": [
            {
                "name": "Bulk Curation Table (.xlsx)",
                "path": f"{pta}/bulk_curation/{{version}}/pituitary_tumor_atlas_bulk_updated_final.xlsx",
                "description": "Curated metadata for all bulk RNA-seq samples in the atlas",
            }
        ],
        "Single-cell Pseudobulk": [
            {
                "name": "Normalized Pseudobulk Matrix (.mtx)",
                "path": f"{pseudobulk}/normalized_data.mtx",
                "description": "Normalized pseudobulk expression per sample and cell type, genes as rows and pseudobulks as columns",
            },
            {
                "name": "Gene Names (.txt)",
                "path": f"{pseudobulk}/genes.txt",
                "description": "Row names of the pseudobulk matrix",
            },
            {
                "name": "Pseudobulk Names (.txt)",
                "path": f"{pseudobulk}/datasets.txt",
                "description": "Column names of the pseudobulk matrix",
            },
        ],
        "Cell Type Abundance": [
            {
                "name": "Cell Type Abundance Matrix (.mtx)",
                "path": f"{abundance}/abundance.mtx",
                "description": "Cell type proportions per single-cell sample",
            },
            {
                "name": "Abundance Row Names (.tsv)",
                "path": f"{abundance}/abundance_rows.tsv",
                "description": "Row names of the abundance matrix",
            },
            {
                "name": "Abundance Column Names (.tsv)",
                "path": f"{abundance}/abundance_cols.tsv",
                "description": "Column names of the abundance matrix",
            },
        ],
        "Curation Data": [
            {
                "name": "Single-cell Curation Data (.parquet)",
                "path": f"{pta}/curation/{{version}}/cpa.parquet",
                "description": "Curated metadata for all single-cell samples in the atlas",
            }
        ],
    }


def create_bulk_data_downloads_ui(
    base_path,
    version="v_0.02",
    site=MOUSE_SITE,
    files_by_category=None,
    header="Download Bulk Data Files",
):
    """
    Create UI for downloading bulk data files (matrices, metadata, etc.)
    """
    st.header(header)

    if files_by_category is None:
        files_by_category = mouse_analysis_files(base_path)
    available = site.versions()

    if site.preprint_only:
        st.info(PREPRINT_NOTICE)

    for category, files in files_by_category.items():
        st.subheader(category)

        for file_info in files:
            file_path, resolved_version = resolve_download_file(
                file_info["path"], version, file_info.get("pattern"), available=available
            )
            if file_path is not None:
                try:
                    file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
                    size_str = f"{file_size:.1f} MB"
                except OSError:
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
                    file_key = f"{site.key_prefix}{category}_{os.path.basename(file_path)}"
                    if f"button_{file_key}" not in st.session_state:
                        st.session_state[f"button_{file_key}"] = False

                    if st.button("Prepare download", key=f"download_button_{file_key}"):
                        st.session_state[f"button_{file_key}"] = True

                    if st.session_state[f"button_{file_key}"]:
                        _log_download(site, file_info["name"])
                        try:
                            _offer_download(
                                site,
                                file_path,
                                os.path.basename(file_path),
                                key=f"actual_download_{file_key}",
                            )
                            st.session_state[f"button_{file_key}"] = False
                        except Exception as e:
                            st.error(f"Error preparing download: {str(e)}")
            else:
                st.warning(
                    f"{file_info['name']} is not available for {version} "
                    "or any earlier version"
                )

            st.markdown("---")
