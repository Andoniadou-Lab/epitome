#importing packages

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import scipy.io
import scipy.sparse
import os
import gc
import polars as pl
import time
import traceback
from datetime import datetime

#importing important epitome modules
from modules.data_loader import (
    load_and_transform_data,
    load_chromvar_data,
    load_isoform_data,
    load_dotplot_data,
    load_accessibility_data,
    load_curation_data,
    load_annotation_data,
    load_motif_data,
    load_enhancer_data,
    load_marker_data,
    load_marker_data_atac,
    load_proportion_data,
    load_atac_proportion_data,
    load_single_cell_dataset,
    load_ligand_receptor_data,
    load_enrichment_results,
    load_motif_genes,
    load_heatmap_data,
    load_sex_dim_data,
    load_gene_curation,
)

from modules.expression import create_expression_plot
from modules.gene_umap_vis import create_gene_umap_plot
from modules.age_correlation import create_age_correlation_plot
from modules.isoforms import create_isoform_plot, filter_isoform_data
from modules.dotplot import (
    create_dotplot,
    filter_dotplot_data,
    create_ligand_receptor_plot,
)
from modules.accessibility import create_accessibility_plot, create_genome_browser_plot
from modules.chromvar import create_chromvar_plot
from modules.utils import (
    filter_data,
    filter_chromvar_data,
    filter_accessibility_data,
    parse_row_info,
    create_color_mapping,
    create_filter_ui,
    create_cell_type_stats_display,
    create_gene_selector,
    create_gene_selector_with_coordinates,
    create_region_selector,
    tab_start_button
)

from modules.proportion_plot import create_proportion_plot

from modules.individual_sc import (
    list_available_datasets,
    plot_sc_dataset,
    get_dataset_info,
)

from modules.display_tables import (
    display_marker_table,
    display_aging_genes_table,
    display_curation_table,
    display_ligand_receptor_table,
    display_enrichment_table,
    display_sex_dimorphism_table,
    display_enhancers_table
)

from modules.gene_gene_corr import (
    create_gene_correlation_plot,
    load_gene_data,
    load_total_counts,
    get_available_genes,
)

from modules.download import (
    create_downloads_ui_with_metadata_rna,
    create_downloads_ui_with_metadata_atac,
    create_bulk_data_downloads_ui
)

from modules.heatmap import process_heatmap_data, analyze_tf_cobinding, plot_heatmap

from modules.analytics import (
    add_activity,
    get_session_id,
)

from modules.epitome_tools_annotation import (
    create_cell_type_annotation_ui
)

from config import Config

BASE_PATH = Config.BASE_PATH
AVAILABLE_VERSIONS = ["v_0.01"]  # List of available versions

logo = f"{BASE_PATH}/data/images/epitome_logo.svg"

# Page Configuration
st.set_page_config(
    page_title="epitome",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS for styling
st.markdown(
    """
    <style>
    .logo-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 0rem 0;
    }
    
    .header-logo {
        width: 200px;  /* Adjust size as needed */
        height: auto;
    }
    
    .footer-logo {
        width: 100px;  /* Smaller size for footer */
        height: auto;
        margin-right: 10px;
    }
    
    .footer-container {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 1rem 0;
        margin-top: 2rem;
        border-top: 1px solid #eee;
    }
    
    .footer-text {
        color: #666;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

import base64

logo = f"{BASE_PATH}/data/images/epitome_logo.svg"
st.markdown(
    f'<div style="margin: 0; padding: 0; text-align: left; margin-top: -2rem; margin-bottom: -4rem;"><img src="data:image/svg+xml;base64,{base64.b64encode(open(logo, "rb").read()).decode()}" width="300" style="margin: 0; padding: 0;"></div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    div[data-testid="stSelectbox"].version-selector {
        position: absolute;
        top: 0;
        right: 0;
        width: 150px !important;
        margin: 10px !important;
        padding: 5px !important;
    }
    div[data-testid="stSelectbox"].version-selector > div[data-baseweb="select"] {
        font-size: 14px !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Add custom CSS for professional styling
st.markdown(
    """
    <style>
    .main {
        padding: 0.5rem 1rem 1rem 1rem;  /* Reduced top padding */
    }
    
    /* Remove top margin from first element after logo */
    .main > div:first-child {
        margin-top: -1rem;
    }
    
    /* Minimize space above tabs */
    .stTabs {
        background-color: #f8f9fa;
        padding: 0.5rem 1rem 1rem 1rem;  /* Reduced top padding */
        border-radius: 5px;
        margin-top: -0.5rem;  /* Pull tabs closer to content above */
    }
    
    /* Reduce space in tab content */
    .stTabs [data-baseweb="tab-list"] {
        margin-bottom: 0.5rem;  /* Reduced from default */
    }
    
    h1 {
        color: #2c3e50;
        margin-bottom: 1rem;  /* Reduced margin */
        margin-top: 0.5rem;   /* Reduced margin */
    }
    h2 {
        color: #34495e;
        margin: 1rem 0;       /* Reduced margin */
    }
    
    .plot-container {
        background-color: white;
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .stats-container {
        padding: 1rem;
        background: #e3f2fd;
        border-radius: 5px;
        margin: 1rem 0;
    }

    .stAlert {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .stSelectbox div[data-baseweb="select"] > div:first-child {
        background-color: #FFFFFF;
        border-color: #2d408d;
    }

    .stMultiSelect div[data-baseweb="select"] > div:first-child {
        background-color: #FFFFFF;
        border-color: #2d408d;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

#defining caching functions

@st.cache_data()
def load_cached_data(version="v_0.01"):
    return load_and_transform_data(version)

@st.cache_data()
def load_cached_chromvar_data(version="v_0.01"):
    return load_chromvar_data(version)

@st.cache_data()
def load_cached_isoform_data(version="v_0.01"):
    return load_isoform_data(version)

@st.cache_data()
def load_cached_dotplot_data(version="v_0.01"):
    return load_dotplot_data(version)

@st.cache_data()
def load_cached_accessibility_data(version="v_0.01"):
    return load_accessibility_data(version)

@st.cache_data()
def load_cached_curation_data(version="v_0.01"):
    return load_curation_data(version)

@st.cache_data()
def load_cached_annotation_data(version="v_0.01"):
    return load_annotation_data(version)

@st.cache_data()
def load_cached_sex_dim_data(version="v_0.01"):
    return load_sex_dim_data(version)

@st.cache_data()
def load_cached_motif_data(version="v_0.01"):
    return load_motif_data(version)

@st.cache_data()
def load_cached_enhancer_data(version="v_0.01"):
    return load_enhancer_data(version)

@st.cache_data()
def load_cached_total_counts(version="v_0.01"):
    base_path = f"{BASE_PATH}/data/large_umap/{version}/adata_export_large_umap"
    return load_total_counts(base_path)

@st.cache_data()
def load_cached_marker_data(version="v_0.01"):
    return load_marker_data(version)

@st.cache_data()
def load_cached_marker_data_atac(version="v_0.01"):
    return load_marker_data_atac(version)

@st.cache_data()
def load_cached_proportion_data(version="v_0.01"):
    return load_proportion_data(version)

@st.cache_data()
def load_cached_ligand_receptor_data(version="v_0.01"):
    return load_ligand_receptor_data(version)

@st.cache_data()
def load_cached_enrichment_data(version="v_0.01"):
    return load_enrichment_results(version)

@st.cache_data()
def load_cached_atac_proportion_data(version="v_0.01"):
    return load_atac_proportion_data(version)

@st.cache_data()
def load_cached_heatmap_data(version="v_0.01"):
    return load_heatmap_data(version)

@st.cache_resource(ttl=600)
def load_cached_single_cell_dataset(dataset, version="v_0.01",rna_atac="rna"):
    return load_single_cell_dataset(dataset, version, rna_atac)

@st.cache_data()
def load_cached_gene_curation(version="v_0.01"):
    return load_gene_curation(version)

#defining function that caches everything for first run after initialization
@st.cache_data()
def load_all_cached_data(version="v_0.01"):
    """
    Load all cached data for the specified version.
    This function is used to load all necessary data at once.
    """
    load_cached_chromvar_data(version=version)
    load_cached_isoform_data(version=version)
    load_cached_dotplot_data(version=version)
    load_cached_accessibility_data(version=version)
    load_cached_curation_data(version=version)
    load_cached_annotation_data(version=version)
    load_cached_sex_dim_data(version=version)
    load_cached_motif_data(version=version)
    load_cached_enhancer_data(version=version)
    load_cached_total_counts(version=version)
    load_cached_marker_data(version=version)
    load_cached_proportion_data(version=version)
    load_cached_ligand_receptor_data(version=version)
    load_cached_enrichment_data(version=version)
    load_cached_atac_proportion_data(version=version)
    load_cached_heatmap_data(version=version)
    load_cached_gene_curation(version=version)

if "current_analysis_tab" not in st.session_state:
    st.session_state["current_analysis_tab"] = None

if "selected_gene" not in st.session_state:
    st.session_state["selected_gene"] = "Sox2"

if "selected_region" not in st.session_state:
    st.session_state["selected_region"] = "chr3:34650405-34652461"

if "cached_all" not in st.session_state:
    st.session_state["cached_all"] = False

epitome_citation = "Kövér, B., Kaufman-Cook, J., Sherwin, O., Vazquez Segoviano, M., Kemkem, Y., Lu, H.-C., & Andoniadou, C. (2025). Electronic Pituitary Omics (epitome) platform. Zenodo. https://doi.org/10.5281/zenodo.17154160"

#main function running the website
def main():
    st.markdown(
        '<p style="margin: 0.1rem 0; font-size: 1rem;">Explore, analyse, and visualise all mouse pituitary datasets. Export raw or processed data, and generate publication-ready figures.</p>',
        unsafe_allow_html=True
    )
    st.markdown('<hr style="margin: 0.1rem 0;">', unsafe_allow_html=True)

    try:

        if not st.session_state["cached_all"]:
            # Load all cached data only once
            print("Loading all cached data...")
            start = time.time()
            load_all_cached_data(version="v_0.01")
            end = time.time()
            print(f"All cached data loaded in {end - start:.2f} seconds")
            st.session_state["cached_all"] = True
            st.rerun()
        else:
            print("All cached data already loaded, skipping...")

    except Exception as e:
        st.error(f"Error loading cached data: {str(e)}")
        traceback.print_exc()

    try:
        print(st.session_state["current_analysis_tab"])
        get_session_id()
        print(st.session_state["session_id"])

        (
            overview_tab,
            rna_tab,
            chromatin_tab,
            multimodal_tab,
            celltyping_tab,
            datasets_tab,
            downloads_tab,
            curation_tab,
            release_tab,
            citation_tab,
            contact_tab,
        ) = st.tabs(
            [
                "Overview",
                "Transcriptome",
                "Chromatin",
                "Multimodal",
                "Automated Cell Typing",
                "Individual Datasets",
                "Downloads",
                "Curation",
                "Release Notes",
                "How to Cite",
                "Contact",
            ]
        )

        with overview_tab:
            

                col1, col2 = st.columns([5, 1])
                with col1:
                    st.header("Overview")
                    st.markdown("A summary of the data available on the platform")
                with col2:
                    selected_version = st.selectbox(
                        "Version",
                        options=AVAILABLE_VERSIONS,
                        key="version_select_overview",
                        label_visibility="collapsed",
                    )

                with st.container():
                    try:

                        curation_data = load_cached_curation_data(version=selected_version)

                        # Calculate statistics
                        rna_samples = len(
                            curation_data[
                                curation_data["Modality"].isin(["sn", "sc", "multi_rna"])
                            ]
                        )
                        atac_samples = len(
                            curation_data[
                                curation_data["Modality"].isin(["atac", "multi_atac"])
                            ]
                        )
                        unique_papers = len(curation_data["Author"].unique())
                        total_cells_rna = int(
                            curation_data[
                                curation_data["Modality"].isin(["sn", "sc", "multi_rna"])
                            ]["n_cells"].sum()
                        )
                        total_cells_atac = int(
                            curation_data[
                                curation_data["Modality"].isin(["atac", "multi_atac"])
                            ]["n_cells"].sum()
                        )
                        total_cells = total_cells_rna + total_cells_atac
                        # in age_numeric turn , to . and
                        curation_data["Age_numeric"] = pd.to_numeric(
                            curation_data["Age_numeric"].astype(str).str.replace(",", "."),
                            errors="coerce",
                        )

                        dev_transcriptome = len(
                            curation_data[
                                (
                                    curation_data["Modality"].isin(
                                        ["sn", "sc", "multi_rna"]
                                    )
                                )
                                & (curation_data["Age_numeric"] < 0)
                            ]
                        )
                        dev_chromatin = len(
                            curation_data[
                                (curation_data["Modality"].isin(["atac", "multi_atac"]))
                                & (curation_data["Age_numeric"] < 0)
                            ]
                        )
                        age_transcriptome = len(
                            curation_data[
                                (
                                    curation_data["Modality"].isin(
                                        ["sn", "sc", "multi_rna"]
                                    )
                                )
                                & (curation_data["Age_numeric"] > 150)
                            ]
                        )
                        age_chromatin = len(
                            curation_data[
                                (curation_data["Modality"].isin(["atac", "multi_atac"]))
                                & (curation_data["Age_numeric"] > 150)
                            ]
                        )

                        # Create four columns
                        col1, col2, col3, col4 = st.columns(4)

                        # Transcriptome samples counter
                        with col1:
                            st.markdown(
                                """
                                <div style="text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px; margin: 10px;">
                                    <h3 style="color: #666; margin-bottom: 5px; font-size: 20px;">Transcriptome Samples</h3>
                                    <div style="font-size: 40px; font-weight: bold; color: #0202ffff;">
                                        {:,}
                                    </div>
                                </div>
                            """.format(
                                    rna_samples
                                ),
                                unsafe_allow_html=True,
                            )

                        # Accessibility samples counter
                        with col2:
                            st.markdown(
                                """
                                <div style="text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px; margin: 10px;">
                                    <h3 style="color: #666; margin-bottom: 5px; font-size: 20px;">Chromatin Samples</h3>
                                    <div style="font-size: 40px; font-weight: bold; color: #0202ffff;">
                                        {:,}
                                    </div>
                                </div>
                            """.format(
                                    atac_samples
                                ),
                                unsafe_allow_html=True,
                            )

                        # Papers counter
                        with col3:
                            st.markdown(
                                """
                                <div style="text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px; margin: 10px;">
                                    <h3 style="color: #666; margin-bottom: 5px; font-size: 20px;">Publications</h3>
                                    <div style="font-size: 40px; font-weight: bold; color: #0202ffff;">
                                        {:,}
                                    </div>
                                </div>
                            """.format(
                                    unique_papers
                                ),
                                unsafe_allow_html=True,
                            )

                        # Cells counter
                        with col4:
                            st.markdown(
                                f"""
                                <div style="text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px; margin: 10px;">
                                    <h3 style="color: #666; margin-bottom: 5px; font-size: 20px;">Total Cells</h3>
                                    <div style="font-size: 40px; font-weight: bold; color: #0202ffff;">
                                        {total_cells:,}
                                    </div>
                                    <div style="font-size: 16px; font-weight: bold;">
                                        <span style="color: #0000ff;">{total_cells_rna:,} RNA</span><br>
                                        <span style="color: #ff2eff;">{total_cells_atac:,} ATAC</span>
                                    </div>
                                </div>
                            """,
                                unsafe_allow_html=True,
                            )

                    except Exception as e:
                        st.error(f"Error loading statistics: {str(e)}")
                        traceback.print_exc() 
                        # Capture the full traceback
                        tb = traceback.format_exc()

                        # Display it in a collapsible section
                        with st.expander("Show full error traceback"):
                            st.code(tb, language='python')

                # In the overview tab
                create_cell_type_stats_display(
                    version=selected_version,
                    display_title="Total Cells by Cell Type",
                    column_count=4,
                    atac_rna="atac+rna",
                )

                with st.container():
                    st.markdown("---")
                    st.subheader("Distribution of samples across ages in the atlas")
                    try:
                        # Create 3 rows with 2 columns each
                        for row in range(3):
                            # Create main columns for layout
                            col1, col2 = st.columns(2)

                            # Row 1: Age distribution histograms
                            if row == 0:
                                with col1:
                                    # Create nested columns to make image smaller
                                    _, img_col, _ = st.columns(
                                        [0.1, 0.8, 0.1]
                                    )  # This creates 20% padding on each side
                                    with img_col:
                                        fig_path = f"{BASE_PATH}/data/figures/{selected_version}/age_distribution_histogram_small.png"
                                        if os.path.exists(fig_path):
                                            st.image(
                                                fig_path,
                                                caption="Histogram showing the number of samples falling into binned ages (days) for transcriptomic samples. The plot has a broken Y-axis to allow visualisation of both low and high abundance ages.",
                                                use_container_width=True,
                                            )
                                        else:
                                            st.warning(
                                                f"Figure not available for version {selected_version}"
                                            )

                                with col2:
                                    _, img_col, _ = st.columns([0.1, 0.8, 0.1])
                                    with img_col:
                                        fig_path = f"{BASE_PATH}/data/figures/{selected_version}/age_distribution_histogram_small_atac.png"
                                        if os.path.exists(fig_path):
                                            st.image(
                                                fig_path,
                                                caption="Histogram showing the number of samples falling into binned ages (days) for chromatin accessibility samples. The plot has a broken Y-axis to allow visualisation of both low and high abundance ages.",
                                                use_container_width=True,
                                            )
                                        else:
                                            st.warning(
                                                f"Figure not available for version {selected_version}"
                                            )

                                st.markdown("---")
                                st.subheader(
                                    "Distribution of metadata categories with relation to each other"
                                )

                            # Row 2: Barplots
                            elif row == 1:
                                with col1:
                                    _, img_col, _ = st.columns([0.1, 0.8, 0.1])
                                    with img_col:
                                        fig_path = f"{BASE_PATH}/data/figures/{selected_version}/barplot1.svg"
                                        if os.path.exists(fig_path):
                                            st.image(
                                                fig_path,
                                                caption="Stacked bar plot of technology metadata, detailing broad assay modality (RNA, ATAC), specific assay modality (single-nucleus, single-cell, multiome), and chemistry versions of respective kits.",
                                                use_container_width=True,
                                            )
                                        else:
                                            st.warning(
                                                f"Figure not available for version {selected_version}"
                                            )

                                with col2:
                                    _, img_col, _ = st.columns([0.1, 0.8, 0.1])
                                    with img_col:
                                        fig_path = f"{BASE_PATH}/data/figures/{selected_version}/barplot2.svg"
                                        if os.path.exists(fig_path):
                                            st.image(
                                                fig_path,
                                                caption="Stacked bar plot of animal metadata, detailing sex, estrous cycle, genetic background, experimental group (control vs perturbed, also showing cases with organoid samples), and whether the sample is sorted or whole pituitary.",
                                                use_container_width=True,
                                            )
                                        else:
                                            st.warning(
                                                f"Figure not available for version {selected_version}"
                                            )

                                st.markdown("---")
                                st.subheader(
                                    "Exponential scaling of single-cell profiling in the pituitary"
                                )

                            # Row 3: Cumulative cell counts
                            else:

                                _, img_col, _ = st.columns(
                                    [0.2, 0.6, 0.2]
                                )  # This creates 20% padding on each side
                                with img_col:
                                    fig_path = f"{BASE_PATH}/data/figures/{selected_version}/cumulative_ncell_over_years_combined.png"
                                    if os.path.exists(fig_path):
                                        st.image(
                                            fig_path, caption="Line plot showing the cumulative number of cells assayed (blue: RNA, pink: ATAC - Chromatin accessibility) over the years since the first publication utlising single-cell profiling on the pituitary until recently. The numbers above each dot represent the number of assayed samples, while the dot sizes are proportional to the number of publications within either modality.", use_container_width=True
                                        )
                                    else:
                                        st.warning(
                                            f"Figure not available for version {selected_version}"
                                        )

                    except Exception as e:
                        st.error(f"Error loading figures: {str(e)}")
                        traceback.print_exc() 
                        # Capture the full traceback
                        tb = traceback.format_exc()

                        # Display it in a collapsible section
                        with st.expander("Show full error traceback"):
                            st.code(tb, language='python')

                # Introduction
                st.markdown("### Current State and Future Directions")
                st.markdown(
                    "Based on our analysis of the current landscape of mouse pituitary research, we've identified several key areas for improvement and future focus."
                )

                # Create sections using columns for better organization
                with st.container():
                    # Age Distribution Section
                    st.subheader("Age Distribution Gaps")
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.markdown("**Embryonic Timepoints**")
                        st.markdown(
                            "- **Transcriptome** Limited samples in embryonic stages ("
                            + str(dev_transcriptome)
                            + " samples)"
                        )
                        st.markdown(
                            "- **Chromatin accessibility:** No samples in embryonic stages ("
                            + str(dev_chromatin)
                            + " samples)"
                        )
                    with col2:
                        st.markdown("**Aging Studies**")
                        st.markdown(
                            "- **Transcriptome:** Few samples from aged mice (>150 days) ("
                            + str(age_transcriptome)
                            + " samples)"
                        )
                        st.markdown(
                            "- **Chromatin accessibility:** No samples from aged mice (>150 days) ("
                            + str(age_chromatin)
                            + " samples)"
                        )
                    st.markdown(
                        "*Note: Chromatin data is particularly limited, with all but one of the samples from ~10-week-old mice.*"
                    )

                # Spatial Data Section
                st.markdown("---")
                with st.container():
                    st.subheader("Assays/Modalities missing from the atlas")
                    st.markdown(
                        "**Spatial Transcriptomics:** Currently no spatially resolved transcriptomics data is available in the literature."
                    )
                    st.markdown(
                        "**Proteomics:** Currently no high-resolution proteomics data is available in the literature."
                    )
                    st.markdown(
                        "**Metabolomics:** Currently no metabolomics data is available in the literature."
                    )
                    st.markdown(
                        "**Metallomics:** Currently limited metallomics data is available in the literature."
                    )
                    st.markdown(
                        "**Methylation:** Currently there is a [single publication](https://www.nature.com/articles/s41467-021-22859-w) with methylation data, and we refer the reader to that publication for more information."
                    )

                # Experimental Design Section
                st.markdown("---")
                with st.container():
                    st.subheader("Experimental Design Recommendations")

                    col1, col2 = st.columns([1, 1])

                    with col1:
                        st.markdown("**Sample Distribution**")
                        st.markdown(
                            "- Avoid further sequencing studies of only wild-type samples aged 50-150 days"
                        )
                        samples_in_this_range = len(
                            curation_data[
                                (curation_data["Age_numeric"] >= 50)
                                & (curation_data["Age_numeric"] <= 150)
                            ]
                        )
                        st.markdown(
                            "- Current count: "
                            + str(samples_in_this_range)
                            + " samples in this age range"
                        )

                    with col2:
                        st.markdown("**Perturbation Studies**")
                        st.markdown("- More mutant/treated samples needed")
                        st.markdown(
                            "- Recommendation: Minimum 2-3 mutant replicates per condition AND at least 2 wild-type control (to account for study specific batch-effects)"
                        )

                # Sexual Dimorphism Section
                st.markdown("---")
                with st.container():
                    st.subheader("Sexual Dimorphism Considerations")
                    st.markdown(
                        """
                    **Key Finding:**
                    - Significant sexual dimorphism observed in most pituitary cell types
                    
                    **Recommendations:**
                    - Design experiments for single sex OR include sufficient samples to account for sex-specific effects
                    - Pooling samples (to reduce costs) from different sexes can be easily demultiplexed using sexually dimorphic genes
                    """
                    )

                # Cell Populations Section
                st.markdown("---")
                with st.container():
                    st.subheader("Updated Markers")
                    st.markdown(
                        """
                    New markers have been identified for known cell populations. We recommend using these highly consistent markers for cell typing (we refer to them as cell typing markers), rather than ad hoc marker genes.
                    Please also refer to our "Automated Cell Typing" section for models that annotate your data reproducibly.
                """
                        )

                # Meta-data description in publications
                st.markdown("---")
                with st.container():
                    st.subheader("Meta-data in Publications")
                    st.markdown(
                        """
                    **Recommendation:**
                    - We have noted several publications with incorrect or incomplete meta-data.
                    
                    **Action:**
                    - Authors should provide complete and accurate meta-data in publications, including single-cell barcoding kits (e.g., 10X V1,V2,V3...), protocols, sex of the mice, and age of the mice.
                    - Authors should deposit their data in a public repository before publication, and make available the raw data - meaning .fastq files, ideally not .bam files.
                    - Authors are encouraged to contact us for corrections or updates to the meta-data.
                    """
                    )

                # Bottom note
                st.markdown("---")
                st.caption(
                    "For detailed methodology and complete findings, please refer to our pre-print publication on bioRxiv (placeholder)."
                )

        with rna_tab:

            # RNA Analysis expander
            with st.container():
                (
                    expression_tab,
                    umap_tab,
                    age_tab,
                    isoform_tab,
                    dotplot_tab,
                    cell_type_tab,
                    gene_gene_relationship_tab,
                    lig_rec_tab
                ) = st.tabs(
                    [
                        "Expression Box Plots",
                        "Expression UMAP",
                        "Age Correlation",
                        "Isoforms",
                        "Dot Plots",
                        "Cell Type Distribution",
                        "Gene-Gene Relationships",
                        "Ligand-Receptor Interactions"
                    ]
                )

                # Expression Distribution tab content
                with expression_tab:

                    st.markdown(
                        "Click the button below to begin transcriptome box plot analysis. This will load the necessary data."
                    )
                    
                    click = tab_start_button(
                        "Expression Distribution",
                        "expr_suffix")

                    if click or (st.session_state["current_analysis_tab"] == "Expression Distribution"):
                        gc.collect()

                        col1, col2 = st.columns([5, 1])
                        with col1:
                            st.header("Expression Box Plots")
                            st.markdown("Generate box plots showing the distribution of gene expression across cell types in the mouse pituitary. Each dot is a pseudobulk sample.")
                        with col2:
                            selected_version = st.selectbox(
                                "Version",
                                options=AVAILABLE_VERSIONS,
                                key="version_select_tab1",
                                label_visibility="collapsed",
                            )

                        with st.spinner("Loading data..."):
                            matrix, genes, meta_data = load_cached_data(
                                version=selected_version
                            )

                        # Sample/Author/Age filtering controls at the top
                        st.subheader("Data Filtering")

                        (
                            filter_type,
                            selected_samples,
                            selected_authors,
                            age_range,
                            only_normal,
                        ) = create_filter_ui(meta_data,sex_analysis=True)
                        
                        
                        # Apply filters to get filtered data
                        filtered_meta = meta_data.copy()

                        # Apply other filters
                        filtered_meta, filtered_matrix = filter_data(
                            meta_data=filtered_meta,
                            age_range=age_range,
                            selected_samples=selected_samples,
                            selected_authors=selected_authors,
                            matrix=matrix,
                            only_normal=only_normal,
                        )

                        #reorder meta and matrix based on alphabetical order of cell types
                        filtered_matrix = filtered_matrix[:, np.argsort(filtered_meta["new_cell_type"])]
                        filtered_meta = filtered_meta.sort_values(by="new_cell_type").reset_index(drop=True)

                        

                        filtered_sra_ids = filtered_meta["SRA_ID"].unique().tolist()

                        
                        create_cell_type_stats_display(
                            version=selected_version,
                            sra_ids=filtered_sra_ids,
                            display_title="Cell Counts in Current Selection",
                            column_count=6,
                            size="small",
                            atac_rna="rna",
                        )

                        # Gene selection with count
                        gene_list = sorted(genes[0].unique())

                        #leave some space out
                        st.markdown("<br>", unsafe_allow_html=True)
                        col1, col2, col3 = st.columns(3)

                        with col1:

                            selected_gene = create_gene_selector(
                                            gene_list=gene_list,
                                key_suffix="gene_select_tab1",
                            )


                            add_activity(value=selected_gene, analysis="Expression Box Plots",
                                    user=st.session_state.session_id,time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                        
                        with col2:
                            # Cell type selection
                            all_cell_types = sorted(filtered_meta["new_cell_type"].unique())
                            cell_type_selection = st.radio(
                                "Cell Type Selection",
                                ["All Cell Types", "Select Specific Cell Types"],
                                key="cell_type_selection_expression",
                                horizontal=True,
                            )

                        with col3:
                            # Additional grouping
                            if filter_type == "Reproduce sex-specific analysis":
                                additional_groups = ["Comp_sex"]
                            else:
                                additional_groups = ["None", "Modality", "Comp_sex"]

                            additional_group = st.selectbox(
                                "Additional Grouping Variable",
                                additional_groups,
                                key="additional_group_select",
                                width=250
                            )

                        selected_cell_types = None
                        if cell_type_selection == "Select Specific Cell Types":
                            selected_cell_types = st.multiselect(
                                "Select Cell Types to Display",
                                options=all_cell_types,
                                default=(
                                    [all_cell_types[0]]
                                    if len(all_cell_types) > 0
                                    else []
                                ),
                                key="selected_cell_types_expression",
                            )

                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        col1, col2 = st.columns(2)
                        # Connect dots toggle
                        with col1:
                            connect_dots = st.checkbox(
                                "Connect Dots",
                                value=False,
                                help="Connect dots with the same SRA_ID (e.g., to visualise if outlier samples across cell types are from the same study)",
                                key="connect_dots_tab1",
                            )

                        with col2:
                            download_as = st.selectbox(
                                        "Download as:",
                                        options=["png", "jpeg", "svg"],
                                        index=0,
                                        key="download_as_expr_boxplot",
                                        width=250
                                    )


                        # Create plot with filtered data and cell type selection
                        fig, config = create_expression_plot(
                            matrix=filtered_matrix,
                            genes=genes,
                            meta_data=filtered_meta,
                            gene_name=selected_gene,
                            additional_group=(
                                None if additional_group == "None" else additional_group
                            ),
                            connect_dots=connect_dots,
                            selected_cell_types=selected_cell_types,
                            download_as=download_as
                        )
                        st.plotly_chart(fig, use_container_width=True, config=config)
                        gc.collect()

                        with st.container():
                            st.markdown(
                                """
                                This box plot shows the distribution of gene expression across different cell types in the mouse pituitary.
                                
                                **X-axis**: Cell types present in the selected samples
                                **Y-axis**: Log10-transformed counts per million* values of the selected gene
                                
                                The plot combines:
                                - Box plot showing the statistical distribution (median, quartiles, and range)
                                - Individual points representing expression in each sample
                                - Optional grouping by additional variables (e.g., sex, data type)
                                - Optional connecting lines between samples from the same source
                                - Cell type filtering to focus on specific cell populations
                                
                                *These values are first normalised using TMM within the Limma-voom workflow.
                            """
                            )

                        # Display the number of samples being shown
                        if filter_type != "No filter":
                            st.info(
                                f"Showing data from {len(filtered_meta)} pseudobulk samples across {len(filtered_meta['new_cell_type'].unique())} cell types"
                            )

                        # Add download button for tab 1
                        gene_idx = genes[genes[0] == selected_gene].index[0]
                        expression_values = (
                            filtered_matrix[gene_idx, :].A1
                            if hasattr(filtered_matrix[gene_idx, :], "A1")
                            else filtered_matrix[gene_idx, :]
                        )
                        download_df = filtered_meta.copy()
                        download_df["Expression"] = expression_values

                        st.download_button(
                                    label="Download Plotting Data",
                                    data=download_df.to_csv(index=False),
                                    file_name=f"{selected_gene}_expression_data.csv",
                                    mime="text/csv",
                                    key="download_button_tab1",
                                    help="Download the current filtered dataset used for plotting",
                                )

                        #add separator line
                        st.markdown("---")
                        # Add marker browser section
                        col1, col2 = st.columns([5, 1])
                        with col1:
                            st.subheader("Marker Gene Browser")
                        with col2:
                            selected_version = st.selectbox(
                                "Version",
                                options=AVAILABLE_VERSIONS,
                                key="version_select_marker_browser",
                                label_visibility="collapsed",
                            )

                        filtered_data = display_marker_table(
                            selected_version, load_cached_marker_data, "expression"
                        )

                        # Add sexually dimorphic genes section
                        st.markdown("---")
                        col1, col2 = st.columns([5, 1])
                        with col1:
                            st.subheader("Sexually Dimorphic Genes")
                        with col2:
                            selected_version = st.selectbox(
                                'Version',
                                options=AVAILABLE_VERSIONS,
                                key='version_select_sex_dim',
                                label_visibility="collapsed"
                            )

                        filtered_sex_dim_data = display_sex_dimorphism_table(sex_dim_data=load_cached_sex_dim_data(), key_prefix="sex_dimorphism")
                
                with umap_tab:

                    #massive red text this tab is currently being fixed
                    st.markdown("<span style='color:red; font-size:20px'>Note: This tab is currently being updated to fix some issues. It will be back online soon.</span>", unsafe_allow_html=True)

                    st.markdown(
                        "Click the button below to start viewing UMAP visualisations. This will load the necessary data."
                    )
                    #click = tab_start_button(
                    #    "UMAP Visualisation",
                    #    "begin_umap_analysis")

                    if click or (st.session_state["current_analysis_tab"] == "UMAP Visualisation"):
                        gc.collect()
                    

                        col1, col2 = st.columns([5, 1])
                        with col1:
                            st.header("UMAP visualisation")
                            st.markdown("Generate UMAP plots showing gene expression across the entire atlas. Each dot is a cell.")
                        with col2:
                            selected_version = st.selectbox(
                                "Version",
                                options=AVAILABLE_VERSIONS,
                                key="version_select_umap",
                                label_visibility="collapsed",
                            )

                        try:
                            # Get available genes
                            base_path = f"{BASE_PATH}/data/large_umap/{selected_version}/adata_export_large_umap"
                            available_genes = get_available_genes(base_path)

                            # Load metadata
                            obs_data = pd.read_parquet(f"{base_path}/obs.parquet")

                            # Sample filtering UI
                            st.subheader("Data Filtering")
                            valid_sra_ids = obs_data["SRA_ID"].unique().tolist()
                            curation = load_cached_curation_data(
                                version=selected_version
                            )
                            # remove rows where age_numeric is not a number
                            filtered_meta = curation[
                                curation["SRA_ID"].isin(valid_sra_ids)
                            ].copy()
                            (
                                filter_type,
                                selected_samples,
                                selected_authors,
                                age_range,
                                only_normal,
                            ) = create_filter_ui(filtered_meta, key_suffix="umap")

                            if filter_type == "Sample":
                                filtered_meta = filtered_meta[
                                    filtered_meta["Name"].isin(selected_samples)
                                ]
                            elif filter_type == "Author":
                                filtered_meta = filtered_meta[
                                    filtered_meta["Author"].isin(selected_authors)
                                ]
                            elif filter_type == "Age" and age_range:
                                #convert to float
                                #change , to .
                                filtered_meta["Age_numeric"] = pd.to_numeric(
                                    filtered_meta["Age_numeric"].str.replace(",", "."),
                                    errors="coerce",
                                )
                                filtered_meta["Age_numeric"] = filtered_meta["Age_numeric"].astype(float)
                                age_mask = (
                                    filtered_meta["Age_numeric"].notna()
                                    & (filtered_meta["Age_numeric"] >= age_range[0])
                                    & (filtered_meta["Age_numeric"] <= age_range[1])
                                )
                                filtered_meta = filtered_meta[age_mask]

                            if only_normal:
                                filtered_meta = filtered_meta[
                                    filtered_meta["Normal"] == 1
                                ]

                            filtered_sra_ids = filtered_meta["SRA_ID"].unique().tolist()

                            # Cell type selection
                            st.subheader("Cell Type Selection")
                            all_cell_types = sorted(obs_data["assignments"].unique()) #change to new_cell_type
                            selected_cell_types = st.multiselect(
                                "Select Cell Types",
                                options=all_cell_types,
                                default=all_cell_types,
                                key="selected_cell_types_umap",
                            )

                            selected_cell_types_display = [ct.split("_")[0] for ct in selected_cell_types]

                            create_cell_type_stats_display(
                                version=selected_version,
                                sra_ids=filtered_sra_ids,
                                display_title="Cell Counts in Current Selection",
                                column_count=6,
                                size="small",
                                cell_types=(
                                    selected_cell_types_display 
                                ),
                                atac_rna="rna",
                            )

                            # Gene selection
                            col1, col2, col3, col4 = st.columns(4)

                            with col1:
                                selected_gene = create_gene_selector(
                                        gene_list=available_genes,
                                    key_suffix="umap_gene_select",
                                )

                            with col2:
                                color_map = st.selectbox(
                                    "Color Map",
                                    [
                                        "blues",
                                        "reds",
                                        "plasma",
                                        "inferno",
                                        "magma",
                                        "viridis",
                                        "greens",
                                        "YlOrRd",
                                    ],
                                    key="color_map_select_datasets1")
                            with col3:
                                sort_order = st.checkbox("Sort plotted cells by expression", value=False, key="sort3")
                            
                            metadata_cols = ['assignments', 'Comp_sex', '10X version', 'Modality', 
                                            'pct_counts_mt', 'pct_counts_ribo', 'pct_counts_malat', 'Normal']
                            
                            with col4:
                                metadata_col = st.selectbox(
                                    "Color second plot by",
                                    options=metadata_cols,
                                    index=0,
                                    key="color_by_select",
                                )

                            download_as = st.selectbox(
                                        "Download as:",
                                        options=["png", "jpeg", "svg"],
                                        index=0,
                                        key="download_as_umap",
                                        width=250
                                    )

                            add_activity(value=selected_gene, analysis="UMAP Plot",
                                        user=st.session_state.session_id,time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                            
                            total_counts = load_cached_total_counts(version=selected_version)

                            gene_fig, cell_type_fig, config = create_gene_umap_plot(
                                selected_gene,
                                base_path,
                                obs_data,
                                total_counts.values,
                                selected_samples=filtered_sra_ids,
                                selected_cell_types=selected_cell_types,
                                color_map=color_map,
                                sort_order=sort_order,
                                metadata_col=metadata_col,
                                download_as=download_as
                            )
                            gc.collect()

                            # Display plots side by side
                            col1, col2 = st.columns(2)
                            with col1:
                                st.plotly_chart(gene_fig, use_container_width=True, config=config)
                            with col2:
                                st.plotly_chart(cell_type_fig, use_container_width=True, config=config)
                            gc.collect()
                            # Add explanation in a container
                            with st.container():
                                st.markdown(
                                    """
                                    This plot shows the expression of a selected gene across different cell types in the mouse pituitary.
                                    Datasets were integrated using scVI, and the latent space was used for identifying nearest neighbours and generating a UMAP plot.
                                    
                                    **X-axis**: Arbitrary UMAP axis 1
                                    **Y-axis**: Arbitrary UMAP axis 2
                                    
                                    The visualization includes:
                                    - Scatter plot of gene expression values
                                    - Hover information showing sample details
                                    - Dynamic point opacity based on total number of points
                                            
                                    Note: For any statistically robust visualisation use the box plots or dot plots. The UMAP coordinates are arbitrary and do not necessarily represent or relate to anything biological. For concerns on the use of UMAPs, read: doi.org/10.1371/journal.pcbi.1011288
                                """
                                )

                            
                        except Exception as e:
                            st.error(f"Error creating plots: {str(e)}")
                            traceback.print_exc() 
                            # Capture the full traceback
                            tb = traceback.format_exc()

                            # Display it in a collapsible section
                            with st.expander("Show full error traceback"):
                                st.code(tb, language='python')

                # Age Correlation tab content
                with age_tab:

                    st.markdown(
                        "Click the button below to begin age-dependent gene expression analysis. This will load the necessary data."
                    )

                    click = tab_start_button(
                        "Age Correlation",
                        "begin_age_analysis")

                    if click or (st.session_state["current_analysis_tab"] == "Age Correlation"):
                        gc.collect()
                        col1, col2 = st.columns([5, 1])
                        with col1:
                            st.header("Age Correlation Analysis")
                            st.markdown("Analyze how gene expression changes in mouse pituitary cell types across different ages. Each dot is a pseudobulk sample.")
                        with col2:
                            selected_version = st.selectbox(
                                "Version",
                                options=AVAILABLE_VERSIONS,
                                key="version_select_tab2",
                                label_visibility="collapsed",
                            )

                        if "matrix" not in st.session_state:
                            with st.spinner("Loading expression data..."):
                                matrix, genes, meta_data = load_cached_data(
                                    version=selected_version
                                )

                        st.subheader("Data Filtering")

                        (
                            filter_type,
                            selected_samples,
                            selected_authors,
                            age_range,
                            only_normal
                            
                        ) = create_filter_ui(meta_data, age_analysis=True, key_suffix="age_corr")

                        if only_normal:
                            total_sra_ids = len(set(meta_data["SRA_ID"].unique()))
                            normal_sra_ids = len(
                                set(
                                    meta_data[meta_data["Normal"] == 1][
                                        "SRA_ID"
                                    ].unique()
                                )
                            )
                            st.info(
                                f"Samples remaining after wild-type filter: {normal_sra_ids} ({total_sra_ids} without filter)"
                            )

                        # Apply filters to get filtered data
                        filtered_meta = meta_data
                        filtered_matrix = matrix

                        if filter_type == "Sample":
                            filtered_meta = filtered_meta[
                                filtered_meta["Name"].isin(selected_samples)
                            ]
                        elif filter_type == "Author":
                            filtered_meta = filtered_meta[
                                filtered_meta["Author"].isin(selected_authors)
                            ]
                        elif filter_type == "Age" and age_range:
                            age_mask = (
                                filtered_meta["Age_numeric"].notna()
                                & (filtered_meta["Age_numeric"] >= age_range[0])
                                & (filtered_meta["Age_numeric"] <= age_range[1])
                            )
                            filtered_meta = filtered_meta[age_mask]

                        elif filter_type == "Reproduce age-dependent analysis":
                            age_mask = (
                                filtered_meta["Age_numeric"].notna()
                                & (filtered_meta["Age_numeric"] >= 0)
                                & (filtered_meta["Age_numeric"] <= 2000)
                            )
                            filtered_meta = filtered_meta[age_mask]

                        if only_normal:
                            filtered_meta = filtered_meta[filtered_meta["Normal"] == 1]

                        # Gene selection for tab 2 - with Il6 default
                        gene_list = sorted(genes[0].unique())


                        col1, col2, col3, col4 = st.columns(4)

                        with col1:

                            selected_gene = create_gene_selector(
                                            gene_list=gene_list,
                                key_suffix="gene_select_tab2"
                            )

                        with col2:
                            #split at _
                            filtered_meta["new_cell_type"] = [
                                ct.split("_")[0] for ct in filtered_meta["new_cell_type"]
                            ]
                            cell_types = sorted(filtered_meta["new_cell_type"].unique())
                            selected_cell_type = st.selectbox(
                                "Select Cell Type",
                                cell_types,
                                index=(
                                    cell_types.index("Stem")
                                    if "Stem" in cell_types
                                    else 0
                                ),
                                width=250
                            )

                        with col3:

                            # Data type filter
                            data_type_options = [
                                "All Data Types",
                                "Single Cell Only (sc)",
                                "Single Nucleus Only (sn)",
                                "Multi-modal RNA Only",
                            ]
                            if filter_type == "Reproduce age-dependent analysis":
                                data_type_options = ["Single Cell Only (sc)"]

                            selected_data_type = st.selectbox(
                                "Data Type Filter:",
                                options=data_type_options,
                                index=0,
                                help="Filter to show only specific data types",
                                width=250
                            )

                            # Convert UI selection to parameter for the plot function
                            data_type_filter = None
                            if selected_data_type == "Single Cell Only (sc)":
                                data_type_filter = "sc"
                            elif selected_data_type == "Single Nucleus Only (sn)":
                                data_type_filter = "sn"
                            elif selected_data_type == "Multi-modal RNA Only":
                                data_type_filter = "multi_rna"

                        with col4:
                        
                            # Color by option (existing code)
                            color_by = st.selectbox(
                                "Color points by:",
                                ["None", "Comp_sex", "Modality"],
                                key="color_select_age_correlation",
                                help="Choose a variable to color the points by",
                                width=250
                            )

                        # Add toggle options
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            use_log_age = st.checkbox(
                                "Use log10 scale for age", value=True
                            )
                        with col2:
                            show_trendline = st.checkbox(
                                "Show trendline",
                                value=True,
                                help="Display linear regression trendline",
                            )
                        with col3:
                            remove_zeros = st.checkbox(
                                "Remove zero values",
                                value=False,
                                help="Remove cells with expression values < 0.01. Some highly contaminating (ambient RNA) transcripts might have been overcorrected in some datasets.",
                            )

                        with col4:
                            download_as = st.selectbox(
                                        "Download as:",
                                        options=["png", "jpeg", "svg"],
                                        index=0,
                                        key="download_as_umap",
                                        width=250
                                    )

                        if data_type_filter == "sc":
                            filtered_meta = filtered_meta[
                                filtered_meta["Modality"].isin(["sc"])
                            ]
                        elif data_type_filter == "sn":
                            filtered_meta = filtered_meta[
                                filtered_meta["Modality"].isin(["sn"])
                            ]
                        elif data_type_filter == "multi_rna":
                            filtered_meta = filtered_meta[
                                filtered_meta["Modality"].isin(["multi_rna"])
                            ]

                        filtered_sra_ids = filtered_meta["SRA_ID"].unique().tolist()

                        # Update matrix to match filtered metadata
                        filtered_matrix = matrix[:, filtered_meta.index]

                        create_cell_type_stats_display(
                            version=selected_version,
                            sra_ids=filtered_sra_ids,
                            display_title="Cell Counts in Current Selection",
                            cell_types=[selected_cell_type],
                            size="small",
                            column_count=1,
                            atac_rna="rna",
                        )

                        add_activity(value=selected_gene, analysis="Age Correlation",
                                    user=st.session_state.session_id,time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                        
                        # Create the plot with log and color options
                        fig, config, r_squared, p_value, aging_genes_df = (
                            create_age_correlation_plot(
                                matrix=filtered_matrix,
                                genes=genes,
                                meta_data=filtered_meta,
                                gene_name=selected_gene,
                                cell_type=selected_cell_type,
                                use_log_age=use_log_age,
                                remove_zeros=remove_zeros,
                                color_by=None if color_by == "None" else color_by,
                                show_trendline=show_trendline,
                                data_type_filter=data_type_filter,
                                download_as=download_as
                            )
                        )

                        st.plotly_chart(fig, use_container_width=True, config=config)
                        gc.collect()

                        with st.container():
                            st.markdown(
                                """
                                This plot visualizes the correlation between gene expression and age for a specific cell type.
                                
                                **X-axis**: Age in days (can be log10-transformed)
                                **Y-axis**: Log10-transformed counts per million* values of the selected gene
                                
                                The plot includes:
                                - Scatter points representing individual samples
                                - Trend line showing the linear correlation
                                - Optional coloring by sex or data type
                                - R-squared value and p-value statistics - Note these do not exactly match the statistical results from Limma-voom used in the publication
                                        
                                *These values are first normalised using TMM within the Limma-voom workflow.
                            """
                            )

                        # Correlation Statistics
                        st.subheader("Correlation Statistics (based on selected/displayed data)")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("R-squared", f"{r_squared:.3f}")
                        with col2:
                            st.metric("P-value", f"{p_value:.3e}")

                        # Create download data for tab 2
                        gene_idx = genes[genes[0] == selected_gene].index[0]
                        expression_values = (
                            matrix[gene_idx, :].A1
                            if hasattr(matrix[gene_idx, :], "A1")
                            else matrix[gene_idx, :]
                        )
                        cell_type_mask = (
                            meta_data["new_cell_type"] == selected_cell_type
                        )

                        download_df = meta_data[cell_type_mask].copy()
                        download_df["Expression"] = expression_values[cell_type_mask]
                        download_df["R_squared"] = r_squared
                        download_df["P_value"] = p_value

                        st.download_button(
                            label="Download Age Correlation Data",
                            data=download_df.to_csv(index=False),
                            file_name=f"{selected_gene}_{selected_cell_type}_age_correlation.csv",
                            mime="text/csv",
                            key="download_button_tab2",
                            help="Download the current age correlation dataset",
                        )

                        # Add Aging Genes Table
                        st.subheader("Aging Genes Reference Table")
                        filtered_df = display_aging_genes_table(aging_genes_df, "aging")

                with isoform_tab:

                    st.markdown(
                        "Click the button below to begin isoform analysis. This will load the necessary data."
                    )

                    click = tab_start_button(
                        "Isoform Analysis",
                        "begin_isoform_analysis")

                    if click or (st.session_state["current_analysis_tab"] == "Isoform Analysis"):
                        gc.collect()
                        col1, col2 = st.columns([5, 1])
                        with col1:
                            st.header("Transcript-Level Expression")
                            st.markdown("Explore transcript-level expression of genes across cell types in the mouse pituitary. Each dot is a pseudobulk sample.")
                        with col2:
                            selected_version = st.selectbox(
                                "Version",
                                options=AVAILABLE_VERSIONS,
                                key="version_select_tab3",
                                label_visibility="collapsed",
                            )

                        with st.spinner("Loading isoform data..."):
                            isoform_matrix, isoform_features, isoform_samples = (
                                load_cached_isoform_data(version=selected_version)
                            )
                            curation_data = load_cached_curation_data(
                                version=selected_version
                            )

                            # Filter curation data to only include SRA_IDs present in isoform data
                            valid_sra_ids = set(isoform_samples["SRA_ID"].unique())
                            filtered_curation = curation_data[
                                curation_data["SRA_ID"].isin(valid_sra_ids)
                            ].copy()
                            # in filtered curation, turn Age_numeric to float

                            # filtered_curation['Age_numeric'] = filtered_curation['Age_numeric'].astype(float)

                            # Create filtering UI
                            st.subheader("Data Filtering")
                            (
                                filter_type,
                                selected_samples,
                                selected_authors,
                                age_range,
                                only_normal,
                            ) = create_filter_ui(
                                filtered_curation, key_suffix="isoform"
                            )

                            # Apply filters to get valid SRA_IDs
                            valid_sra_ids = set(filtered_curation["SRA_ID"].unique())

                            if filter_type == "Sample":
                                valid_sra_ids &= set(
                                    filtered_curation[
                                        filtered_curation["Name"].isin(selected_samples)
                                    ]["SRA_ID"].unique()
                                )
                            elif filter_type == "Author":
                                valid_sra_ids &= set(
                                    filtered_curation[
                                        filtered_curation["Author"].isin(
                                            selected_authors
                                        )
                                    ]["SRA_ID"].unique()
                                )
                            elif filter_type == "Age" and age_range:
                                # Convert age values to float before comparison
                                filtered_curation["Age_numeric"] = filtered_curation[
                                    "Age_numeric"
                                ].apply(
                                    lambda x: (
                                        float(str(x).replace(",", ".")) if pd.notnull(x) else np.nan
                                    )
                                )
                                age_mask = (
                                    filtered_curation["Age_numeric"].notna()
                                    & (
                                        filtered_curation["Age_numeric"]
                                        >= float(age_range[0])
                                    )
                                    & (
                                        filtered_curation["Age_numeric"]
                                        <= float(age_range[1])
                                    )
                                )
                                valid_sra_ids &= set(
                                    filtered_curation[age_mask]["SRA_ID"].unique()
                                )

                            if only_normal:
                                valid_sra_ids &= set(
                                    filtered_curation[filtered_curation["Normal"] == 1][
                                        "SRA_ID"
                                    ].unique()
                                )

                            # Filter isoform samples based on valid SRA_IDs
                            sample_mask = isoform_samples["SRA_ID"].isin(valid_sra_ids)
                            
                            
                            if not isinstance(sample_mask, np.ndarray):
                                # Convert from pandas Series to numpy array
                                sample_mask = sample_mask.to_numpy() if hasattr(sample_mask, 'to_numpy') else np.array(sample_mask)

                            # Ensure sample_mask is a boolean array
                            if sample_mask.dtype != bool:
                                sample_mask = sample_mask.astype(bool)

                            # Now use the numpy array (boolean mask) for indexing the sparse matrix
                            filtered_matrix = isoform_matrix[:, sample_mask]

                            filtered_matrix = isoform_matrix[:, sample_mask]
                            filtered_samples = isoform_samples[sample_mask].copy()

                            # Display filtered data info
                            if filter_type != "No filter" or only_normal:
                                st.info(
                                    f"Showing data from {len(filtered_samples)} pseudobulk samples"
                                )

                            # Gene selection for isoform plot
                            gene_list = sorted(isoform_features["gene_name"].unique())

                            # Cell type filter
                            all_cell_types = sorted(
                                filtered_samples["cell_type"].unique()
                            )

                            
                            cell_type_option = st.radio(
                                "Cell Type Selection",
                                ["All Cell Types", "Select Specific Cell Types"],
                                key="cell_type_radio",
                            )

                            selected_cell_types = None
                            if cell_type_option == "Select Specific Cell Types":
                                selected_cell_types = st.multiselect(
                                    "Select Cell Types to Display",
                                    #remove _.* from cell types for display
                                    [ct.split("_")[0] for ct in all_cell_types],
                                    default=[ct.split("_")[0] for ct in all_cell_types][0],
                                    key="cell_type_select",
                                )

                            filtered_sra_ids = (
                                filtered_samples["SRA_ID"].unique().tolist()
                            )

                            
                            #
                            create_cell_type_stats_display(
                                version=selected_version,
                                sra_ids=filtered_sra_ids,
                                display_title="Cell Counts in Current Selection",
                                cell_types=(
                                    #split at " "
                                    [ct.split(" ")[0] for ct in selected_cell_types]
                                    if cell_type_option == "Select Specific Cell Types"
                                    else None
                                ),
                                column_count=6,
                                size="small",
                                atac_rna="rna",
                            )

                            col1, col2 = st.columns(2)
                            with col1:
                                selected_gene = create_gene_selector(
                                        gene_list=gene_list, key_suffix="gene_select_tab3"
                                )
                            with col2:
                                download_as = st.selectbox(
                                        "Download as:",
                                        options=["png", "jpeg", "svg"],
                                        index=0,
                                        key="download_as_umap",
                                        width=250
                                    )

                            # Create and display the plot
                            if selected_gene:

                                add_activity(value=selected_gene, analysis="Isoform Plot",
                                    user=st.session_state.session_id,time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                                #split at _
                                
                                fig, config, error_message = create_isoform_plot(
                                    filtered_matrix,
                                    isoform_features,
                                    filtered_samples,
                                    filtered_curation,
                                    selected_gene,
                                    (
                                        selected_cell_types
                                        if cell_type_option
                                        == "Select Specific Cell Types"
                                        else None
                                    ),
                                    download_as=download_as
                                )

                                if error_message:
                                    st.error(error_message)
                                else:
                                    st.plotly_chart(
                                        fig, use_container_width=True, config=config
                                    )
                                    gc.collect()
                                    with st.container():
                                        st.markdown(
                                            """
                                            This plot displays transcript-level expression data for a selected gene across cell types.
                                            
                                            **X-axis**: Transcript IDs grouped by cell type
                                            **Y-axis**: Log10-transformed counts per million values of each transcript
                                            
                                            The plot shows:
                                            - Box plots for each transcript's expression distribution
                                            - Individual points representing expression in each sample
                                            - Grouping by cell type and transcript ID
                                            - Hover information including sample metadata
                                        """
                                        )

                                # Display transcript information
                                transcript_count = len(
                                    isoform_features[
                                        isoform_features["gene_name"] == selected_gene
                                    ]
                                )
                                st.info(
                                    f"Number of transcripts for {selected_gene}: {transcript_count}"
                                )

                                # Display transcript table with Ensembl links
                                try:
                                    # Get transcripts for the selected gene
                                    gene_transcripts = isoform_features[
                                        isoform_features["gene_name"] == selected_gene
                                    ]

                                    # Load the filtered transcript list if it exists
                                    filtered_transcripts = []
                                    filtered_transcripts_path = f"{BASE_PATH}/data/isoforms/v_0.01/filtered_transcripts_list.csv"
                                    if os.path.exists(filtered_transcripts_path):
                                        try:
                                            filtered_df = pd.read_csv(
                                                filtered_transcripts_path
                                            )
                                            # Remove version numbers from transcripts (e.g., ENSMUST00000070532.8 -> ENSMUST00000070532)
                                            filtered_transcripts = [
                                                t.split(".")[0]
                                                for t in filtered_df.iloc[:, 1]
                                            ]
                                        except Exception as e:
                                            st.warning(
                                                f"Error loading filtered transcripts: {e}"
                                            )
                                            traceback.print_exc() 
                                            # Capture the full traceback
                                            tb = traceback.format_exc()

                                            # Display it in a collapsible section
                                            with st.expander("Show full error traceback"):
                                                st.code(tb, language='python')

                                    # Create table data
                                    st.markdown("### Transcript Details")
                                    st.markdown(
                                        "The table below shows all transcripts for this gene with links to Ensembl."
                                    )

                                    # Build the table
                                    table_md = [
                                        "| Transcript ID | Ensembl Link |",
                                        "|-------------|--------------|",
                                    ]

                                    has_filtered = False

                                    for _, transcript in gene_transcripts.iterrows():
                                        transcript_id = transcript["transcript_id"]
                                        # Get base transcript ID without version
                                        base_transcript_id = (
                                            transcript_id.split(".")[0]
                                            if "." in transcript_id
                                            else transcript_id
                                        )
                                        # Check if this is a filtered transcript
                                        is_filtered = (
                                            base_transcript_id in filtered_transcripts
                                        )
                                        if is_filtered:
                                            has_filtered = True

                                        # Create Ensembl link
                                        ensembl_link = f"https://www.ensembl.org/Mus_musculus/Transcript/Summary?t={transcript_id}"

                                        # Mark filtered transcripts with a star
                                        star = "⭐ " if is_filtered else ""
                                        # Create a row with the transcript ID and link
                                        row = f"| {star}{transcript_id} | [View in Ensembl]({ensembl_link}) |"
                                        table_md.append(row)

                                    # Display table using markdown
                                    st.markdown("\n".join(table_md))

                                    # Add legend for the star
                                    if has_filtered:
                                        st.markdown(
                                            "⭐ Isoform with uniquely mapping reads in 90% of datasets - likely to be more reliably quantified"
                                        )
                                except Exception as e:
                                    st.error(f"Error displaying transcript table: {e}")
                                    if st.checkbox("Show detailed error"):
                                        st.exception(e)
                                        traceback.print_exc() 
                                        # Capture the full traceback
                                        tb = traceback.format_exc()

                                        # Display it in a collapsible section
                                        with st.expander("Show full error traceback"):
                                            st.code(tb, language='python')

                                # Create download data for tab 3
                                gene_transcripts = isoform_features[
                                    isoform_features["gene_name"] == selected_gene
                                ]
                                plot_data = []

                                for idx, transcript in gene_transcripts.iterrows():
                                    expression = isoform_matrix[idx].toarray().flatten()
                                    for sample_idx, expr_val in enumerate(expression):
                                        cell_type = isoform_samples.iloc[sample_idx][
                                            "cell_type"
                                        ]
                                        if (
                                            cell_type != "Erythrocytes"
                                        ):  # Exclude Erythrocytes
                                            if (
                                                selected_cell_types is None
                                                or cell_type in selected_cell_types
                                            ):
                                                plot_data.append(
                                                    {
                                                        "Gene": selected_gene,
                                                        "Transcript": transcript[
                                                            "transcript_id"
                                                        ],
                                                        "Cell_Type": cell_type,
                                                        "Expression": expr_val,
                                                    }
                                                )

                                if plot_data:
                                    download_df = pd.DataFrame(plot_data)
                                    st.download_button(
                                        label="Download Transcript Data",
                                        data=download_df.to_csv(index=False),
                                        file_name=f"{selected_gene}_transcript_data.csv",
                                        mime="text/csv",
                                        key="download_button_tab3",
                                        help="Download the current transcript expression dataset",
                                    )

                with dotplot_tab:

                    st.markdown(
                        "Click the button below to begin transcriptome dot plot analysis. This will load the necessary data."
                    )

                    click = tab_start_button(
                        "Dot plot Analysis",
                        "begin_dotplot_analysis")

                    if click or (st.session_state["current_analysis_tab"] == "Dot plot Analysis"):
                        gc.collect()
                        col1, col2 = st.columns([5, 1])
                        with col1:
                            st.header("Gene Expression Dot Plot")
                            st.markdown("Compare gene expression patterns across cell types in the mouse pituitary. Each dot provides summary statistics across all samples.")
                        with col2:
                            selected_version = st.selectbox(
                                "Version",
                                options=AVAILABLE_VERSIONS,
                                key="version_select_tab4",
                                label_visibility="collapsed",
                            )

                        with st.spinner("Loading data..."):
                            # Load the data
                            (
                                proportion_matrix,
                                genes1,
                                rows1,
                                expression_matrix,
                                genes2,
                                rows2,
                            ) = load_cached_dotplot_data(version=selected_version)
                            curation_data = load_cached_curation_data(
                                version=selected_version
                            )

                            # Parse row information to get SRA_IDs
                            row_info = parse_row_info(
                                rows1
                            )  # Assuming rows1 and rows2 have the same SRA_IDs
                            valid_sra_ids = set(row_info["SRA_ID"].unique())

                            # Filter curation data to only include relevant SRA_IDs
                            filtered_curation = curation_data[
                                curation_data["SRA_ID"].isin(valid_sra_ids)
                            ].copy()

                            # Create filtering UI
                            st.subheader("Data Filtering")
                            (
                                filter_type,
                                selected_samples,
                                selected_authors,
                                age_range,
                                only_normal,
                            ) = create_filter_ui(
                                filtered_curation, key_suffix="dotplot"
                            )

                            # Apply filters
                            (
                                filtered_curation,
                                filtered_prop_matrix,
                                filtered_expr_matrix,
                                filtered_rows1,
                                filtered_rows2,
                            ) = filter_dotplot_data(
                                proportion_matrix,
                                expression_matrix,
                                rows1,
                                rows2,
                                filtered_curation,
                                selected_samples if filter_type == "Sample" else None,
                                selected_authors if filter_type == "Author" else None,
                                age_range if filter_type == "Age" else None,
                                only_normal,
                            )

                            # Display filtered data info
                            if filter_type != "No filter" or only_normal:
                                st.info(
                                    f"Showing data from {len(filtered_rows1)} pseudobulk samples"
                                )

                        try:
                            # Gene selection for dot plot
                            available_genes = sorted(set(genes1.iloc[:, 0].unique()))
                            default_genes = (
                                [st.session_state["selected_gene"]]
                                if st.session_state["selected_gene"] in available_genes
                                else [available_genes[0]]
                            )
                            
                            selected_genes = st.multiselect(
                                f"Select Genes for Dot Plot ({len(available_genes)} genes)",
                                available_genes,
                                default=default_genes,
                                max_selections=60,
                                help="Choose genes to display in the dot plot (maximum 60)",
                            )

                            if selected_genes:
                                # Add cell type selection
                                all_cell_types = sorted(
                                    set(
                                        [
                                            (
                                                cell_type.split("_")[1]
                                                if "_" in cell_type
                                                else cell_type
                                            )
                                            for cell_type in filtered_rows1[
                                                filtered_rows1.columns[0]
                                            ]
                                        ]
                                    )
                                )

                                cell_type_selection = st.radio(
                                    "Cell Type Selection",
                                    ["All Cell Types", "Select Specific Cell Types"],
                                    key="cell_type_selection_dotplot",
                                )

                                selected_cell_types = None
                                if cell_type_selection == "Select Specific Cell Types":
                                    selected_cell_types = st.multiselect(
                                        "Select Cell Types",
                                        options=all_cell_types,
                                        default=[all_cell_types[0]],
                                        key="cell_type_multiselect_dotplot",
                                    )

                                create_cell_type_stats_display(
                                    version=selected_version,
                                    sra_ids=(
                                        filtered_curation[
                                            filtered_curation["Name"].isin(
                                                selected_samples
                                            )
                                        ]["SRA_ID"]
                                        .unique()
                                        .tolist()
                                        if selected_samples
                                        else filtered_curation["SRA_ID"]
                                        .unique()
                                        .tolist()
                                    ),
                                    display_title="Cell Counts in Current Selection",
                                    cell_types=(
                                        selected_cell_types
                                        if cell_type_selection
                                        == "Select Specific Cell Types"
                                        else None
                                    ),
                                    column_count=6,
                                    size="small",
                                    atac_rna="rna",
                                )
                                #insert a break
                                st.markdown("<br>", unsafe_allow_html=True)
                                col1, col2 = st.columns(2)
                                with col1:
                                    chosen_color_scheme = st.selectbox(
                                        "Select Color Scheme",
                                        options=["Blue","Red","Viridis","Cividis"],
                                        index=0,
                                        key="color_scheme_dotplot",
                                        width=250
                                    )
                                with col2:

                                    download_as = st.selectbox(
                                        "Download as:",
                                        options=["png", "jpeg", "svg"],
                                        index=0,
                                        key="download_as_dotplot",
                                        width=250
                                    )
                                    
                                add_activity(value=selected_genes, analysis="Dot Plot",
                                    user=st.session_state.session_id,time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                                
                                # Update the create_dotplot call to include cell type filtering
                                fig, config = create_dotplot(
                                    filtered_prop_matrix,
                                    filtered_expr_matrix,
                                    genes1,
                                    genes2,
                                    filtered_rows1,
                                    filtered_rows2,
                                    selected_genes,
                                    selected_cell_types=(
                                        selected_cell_types
                                        if cell_type_selection
                                        == "Select Specific Cell Types"
                                        else None
                                    ),
                                    color_scheme=chosen_color_scheme,
                                    download_as=download_as
                                )
                                gc.collect()

                                st.plotly_chart(
                                    fig, use_container_width=True, config=config
                                )

                                with st.container():
                                    st.markdown(
                                        """
                                        This plot provides a comprehensive view of gene expression patterns across cell types.
                                        
                                        **X-axis**: Selected genes
                                        **Y-axis**: Cell types
                                        
                                        The visualization combines two metrics:
                                        - Dot size: Proportion of cells expressing the gene. Mean across datasets.
                                        - Dot color: Average expression level in expressing cells. Mean log2(counts_per_10k + 1) across datasets.
                                        
                                        This allows simultaneous visualization of expression prevalence and intensity.
                                    """
                                    )
                                st.info(
                                    f"Showing data from {len(filtered_rows1)} pseudobulk samples"
                                )

                                # Create download data
                                plot_data = []
                                proportion_matrix_array = (
                                    filtered_prop_matrix.toarray()
                                    if hasattr(filtered_prop_matrix, "toarray")
                                    else np.array(filtered_prop_matrix)
                                )
                                expression_matrix_array = (
                                    filtered_expr_matrix.toarray()
                                    if hasattr(filtered_expr_matrix, "toarray")
                                    else np.array(filtered_expr_matrix)
                                )

                                genes_list1 = [
                                    str(gene)
                                    for gene in genes1[genes1.columns[0]].tolist()
                                ]
                                genes_list2 = [
                                    str(gene)
                                    for gene in genes2[genes2.columns[0]].tolist()
                                ]

                                for gene in selected_genes:
                                    if gene in genes_list1 and gene in genes_list2:
                                        gene_idx1 = genes_list1.index(gene)
                                        gene_idx2 = genes_list2.index(gene)

                                        for i, row in enumerate(
                                            filtered_rows1[filtered_rows1.columns[0]]
                                        ):
                                            cell_type = (
                                                row.split("_")[1] if "_" in row else row
                                            )
                                            sra_id = (
                                                row.split("_")[0] if "_" in row else ""
                                            )

                                            plot_data.append(
                                                {
                                                    "Gene": gene,
                                                    "Cell_Type": cell_type,
                                                    "Dataset": sra_id,
                                                    "Proportion": proportion_matrix_array[
                                                        i, gene_idx1
                                                    ],
                                                    "Expression": expression_matrix_array[
                                                        i, gene_idx2
                                                    ],
                                                }
                                            )

                                if plot_data:
                                    download_df = pd.DataFrame(plot_data)
                                    st.download_button(
                                        label="Download Dot Plot Data",
                                        data=download_df.to_csv(index=False),
                                        file_name="dotplot_data.csv",
                                        mime="text/csv",
                                        key="download_button_tab4",
                                        help="Download the current dot plot dataset",
                                    )
                            else:
                                st.warning(
                                    "Please select at least one gene to display the dot plot."
                                )

                        except Exception as e:
                            st.error(f"Error processing dot plot data: {e}")
                            traceback.print_exc() 
                            # Capture the full traceback
                            tb = traceback.format_exc()

                            # Display it in a collapsible section
                            with st.expander("Show full error traceback"):
                                st.code(tb, language='python')

                        # Add marker browser section
                        col1, col2 = st.columns([5, 1])
                        with col1:
                            st.subheader("Marker Gene Browser")
                        with col2:
                            selected_version = st.selectbox(
                                "Version",
                                options=AVAILABLE_VERSIONS,
                                key="version_select_markers_dotplot",
                                label_visibility="collapsed",
                            )

                        filtered_data = display_marker_table(
                            selected_version, load_cached_marker_data, "dotplot"
                        )

                with cell_type_tab:
                    gc.collect()
                    st.markdown(
                        "Click the button below to begin cell type proportion analysis. This will load the necessary data."
                    )

                    click = tab_start_button(
                        "Proportion Analysis",
                        "begin_proportion_analysis")

                    if click or (st.session_state["current_analysis_tab"] == "Proportion Analysis"):
                        gc.collect()
                        col1, col2 = st.columns([5, 1])
                        with col1:
                            st.header("Cell Type Distribution")
                            st.markdown("Visualize the distribution of cell type abundance across samples in the mouse pituitary.")
                        with col2:
                            selected_version = st.selectbox(
                                "Version",
                                options=AVAILABLE_VERSIONS,
                                key="version_select_proportion",
                                label_visibility="collapsed",
                            )

                        with st.spinner("Loading proportion data..."):
                            proportion_matrix, proportion_rows, proportion_cols = (
                                load_cached_proportion_data(version=selected_version)
                            )

                        # Filter controls
                        st.subheader("Data Filtering")
                        filter_type = st.radio(
                            "Filter data by:",
                            ["No filter", "Sample", "Author"],
                            key="filter_type_proportion",
                        )
                        meta_data = load_cached_curation_data(version=selected_version)
                        selected_samples = all_samples = sorted(
                            meta_data["Name"].unique()
                        )
                        selected_authors = all_authors = sorted(
                            meta_data["Author"].unique()
                        )

                        if filter_type == "Sample":
                            selected_samples = st.multiselect(
                                "Select Samples",
                                all_samples,
                                default=[all_samples[0]],
                                help="Choose which samples to include in the analysis",
                                key="samples_multiselect_proportion",
                            )
                        elif filter_type == "Author":
                            selected_authors = st.multiselect(
                                "Select Authors",
                                all_authors,
                                default=[all_authors[0]],
                                help="Choose which authors' data to include",
                                key="authors_multiselect_proportion",
                            )

                        # Filter toggles in columns
                        col1, col2 = st.columns(2)
                        with col1:
                            only_normal = st.checkbox(
                                "Show only control samples",
                                value=False,
                                help="Samples that are wild-type, untreated etc. (In curation, Normal == 1)",
                                key="only_normal_proportion",
                            )
                            only_whole = st.checkbox(
                                "Show only whole-pituitary samples",
                                value=False,
                                help="Samples not sorted to enrich for a given sub-population (In curation, Sorted == 0)",
                                key="only_whole_proportion",
                            )
                            show_mean = st.checkbox(
                                "Show mean proportions",
                                value=False,
                                help="Show average cell type proportions across selected samples",
                                key="show_mean_proportion",
                            )
                        with col2:
                            group_by_sex = st.checkbox(
                                "Group by Sex",
                                value=False,
                                help="Create separate plots for male and female samples",
                                key="group_by_sex_proportion",
                            )
                            order_by_age = st.checkbox(
                                "Order by Age",
                                value=False,
                                help="Order samples by age",
                                key="order_by_age_proportion",
                            )

                            download_as = st.selectbox(
                                        "Download as:",
                                        options=["png", "jpeg", "svg"],
                                        index=0,
                                        key="download_as_atac_proportion",
                                        width=250
                                    )

                            # Show log age option only when ordering by age
                            use_log_age = False
                            # if order_by_age:
                            # use_log_age = st.checkbox('Use log10(Age)', value=False,
                            #                        help="Use log10 scale for age and group similar ages",
                            #                        key='use_log_age_proportion')

                        filtered_meta = meta_data.copy()
                        if only_normal:
                            filtered_meta = filtered_meta[filtered_meta["Normal"] == 1]
                        if only_whole:
                            filtered_meta = filtered_meta[filtered_meta["Sorted"] == 0]
                        if filter_type == "Author":
                            filtered_meta = filtered_meta[
                                filtered_meta["Author"].isin(selected_authors)
                            ]
                        if filter_type == "Sample":
                            filtered_meta = filtered_meta[
                                filtered_meta["Name"].isin(selected_samples)
                            ]

                        create_cell_type_stats_display(
                            version=selected_version,
                            # make it selected samples if empty then use  all samples
                            sra_ids=filtered_meta["SRA_ID"].unique().tolist(),
                            display_title="Cell Counts in Current Selection",
                            column_count=6,
                            size="small",
                            atac_rna="rna",
                        )

                        add_activity(value="NA", analysis="Cell Type Proportions",
                                    user=st.session_state.session_id,time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

                        # Create plot
                        fig_male, fig_female, config, error_message = (
                            create_proportion_plot(
                                matrix=proportion_matrix,
                                rows=proportion_rows,
                                columns=proportion_cols,
                                meta_data=meta_data,
                                selected_samples=(
                                    selected_samples
                                    if filter_type == "Sample"
                                    else None
                                ),
                                selected_authors=(
                                    selected_authors
                                    if filter_type == "Author"
                                    else None
                                ),
                                only_normal=only_normal,
                                only_whole=only_whole,
                                group_by_sex=group_by_sex,
                                order_by_age=order_by_age,
                                show_mean=show_mean,
                                use_log_age=use_log_age,
                                download_as=download_as
                            )
                        )

                        if error_message:
                            st.warning(error_message)
                        elif group_by_sex:
                            if fig_male is not None:
                                st.plotly_chart(
                                    fig_male, use_container_width=True, config=config
                                )
                            if fig_female is not None:
                                st.plotly_chart(
                                    fig_female, use_container_width=True, config=config
                                )
                        else:
                            if (
                                fig_male is not None
                            ):  # Using fig_male as the main figure
                                st.plotly_chart(
                                    fig_male, use_container_width=True, config=config
                                )

                        gc.collect()

                        with st.container():
                            st.markdown(
                                """
                                This plot shows the relative proportions of different cell types across samples.
                                
                                **X-axis**: Samples (can be ordered by age)
                                **Y-axis**: Percentage of each cell type
                                
                                Features:
                                - Stacked bar chart showing relative proportions
                                - Optional grouping by sex
                                - Option to show mean proportions
                                - Age-based ordering and log-transformation
                                - Smooth visualization option for age-based trends
                            """
                            )

                        # Add download button for data
                        if hasattr(proportion_matrix, "toarray"):
                            prop_data = proportion_matrix.toarray()
                        else:
                            prop_data = proportion_matrix

                        prop_df = pd.DataFrame(
                            prop_data,
                            index=proportion_rows.iloc[:, 0],
                            columns=proportion_cols.iloc[:, 0],
                        )

                        st.download_button(
                            label="Download Proportion Data",
                            data=prop_df.to_csv(index=True),
                            file_name="cell_type_proportions.csv",
                            mime="text/csv",
                            help="Download the cell type proportion data",
                            key="download_button_proportion",
                        )

                with gene_gene_relationship_tab:

                    st.markdown(
                        "Click the button below to begin gene-gene correlation analysis. This will load the necessary data."
                    )

                    click = tab_start_button(
                        "gene_gene_corr_analysis",
                        "begin_gene_corr_analysis")

                    if click or (st.session_state["current_analysis_tab"] == "gene_gene_corr_analysis"):
                        gc.collect()

                        col1, col2 = st.columns([5, 1])
                        with col1:
                            st.header("Gene-Gene Relationships")
                            st.markdown("Explore correlations between the expression levels of two genes across cell types in the mouse pituitary. Each point represents a cell.")
                        with col2:
                            selected_version = st.selectbox(
                                "Version",
                                options=AVAILABLE_VERSIONS,
                                key="version_select_gene_corr",
                                label_visibility="collapsed",
                            )

                        try:
                            # Get available genes
                            base_path = f"{BASE_PATH}/data/gene_gene_corr/{selected_version}/adata_export"
                            available_genes = get_available_genes(base_path)

                            # Load metadata
                            obs_data = pd.read_parquet(f"{base_path}/obs.parquet")

                            # Gene selection
                            col1, col2 = st.columns(2)
                            with col1:
                                gene1 = st.selectbox(
                                    "Select First Gene",
                                    options=available_genes,
                                    index=(
                                        available_genes.index("Sox2")
                                        if "Sox2" in available_genes
                                        else 0
                                    ),
                                    key="gene1_select",
                                    width=250
                                )
                            with col2:
                                gene2 = st.selectbox(
                                    "Select Second Gene",
                                    options=available_genes,
                                    index=(
                                        available_genes.index("Sox9")
                                        if "Sox9" in available_genes
                                        else 0
                                    ),
                                    key="gene2_select",
                                    width=250
                                )

                            # Sample filtering UI
                            st.subheader("Data Filtering")
                            valid_sra_ids = obs_data["SRA_ID"].unique().tolist()
                            curation = load_cached_curation_data(
                                version=selected_version
                            )
                            # remove rows where age_numeric is not a number
                            filtered_meta = curation[
                                curation["SRA_ID"].isin(valid_sra_ids)
                            ].copy()
                            (
                                filter_type,
                                selected_samples,
                                selected_authors,
                                age_range,
                                only_normal,
                            ) = create_filter_ui(filtered_meta, key_suffix="gene_corr")

                            if filter_type == "Sample":
                                filtered_meta = filtered_meta[
                                    filtered_meta["Name"].isin(selected_samples)
                                ]
                            elif filter_type == "Author":
                                filtered_meta = filtered_meta[
                                    filtered_meta["Author"].isin(selected_authors)
                                ]
                            elif filter_type == "Age" and age_range:
                                age_mask = (
                                    filtered_meta["Age_numeric"].notna()
                                    & (filtered_meta["Age_numeric"] >= age_range[0])
                                    & (filtered_meta["Age_numeric"] <= age_range[1])
                                )
                                filtered_meta = filtered_meta[age_mask]

                            if only_normal:
                                filtered_meta = filtered_meta[
                                    filtered_meta["Normal"] == 1
                                ]

                            filtered_sra_ids = filtered_meta["SRA_ID"].unique().tolist()

                            # Cell type selection
                            st.subheader("Cell Type Selection")
                            all_cell_types = sorted(obs_data["new_cell_type"].unique())
                            selected_cell_types = st.multiselect(
                                "Select Cell Types",
                                options=all_cell_types,
                                default=(
                                    ["Stem_cells"]
                                    if "Stem_cells" in all_cell_types
                                    else None
                                ),
                                key="selected_cell_types_gene_gene",
                            )

                            # Color by cell type toggle
                            color_by_celltype = st.checkbox(
                                "Color by Cell Type",
                                value=True,
                                help="Color points by cell type or show all points in a single color",
                            )

                            create_cell_type_stats_display(
                                version=selected_version,
                                sra_ids=filtered_sra_ids,
                                display_title="Cell Counts in Current Selection",
                                column_count=6,
                                size="small",
                                cell_types=(
                                    "all"
                                    if selected_cell_types is None
                                    else selected_cell_types
                                ),
                                atac_rna="rna",
                            )
                            
                            add_activity(value=[gene1, gene2], analysis="Gene-Gene Correlation",
                                    user=st.session_state.session_id,time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                            

                            # Create plot
                            fig, config, stats, error = create_gene_correlation_plot(
                                gene1,
                                gene2,
                                base_path,
                                obs_data,
                                selected_samples=filtered_sra_ids,
                                color_by_celltype=color_by_celltype,
                                selected_cell_types=selected_cell_types,
                            )
                            gc.collect()

                            if error:
                                st.error(f"Error creating plot: {error}")
                            elif fig:
                                st.plotly_chart(
                                    fig, use_container_width=True, config=config
                                )

                                # Display overall statistics
                                st.subheader("Overall Spearman Correlation (r)")
                                st.metric("Correlation", f"{stats['correlation']:.3f}")
                                # Pct coexpression
                                st.subheader("Percentage of Coexpression")
                                st.metric(
                                    "Percentage", f"{stats['pct_coexpression']:.2f}%"
                                )

                                # Create download data
                                gene1_data = load_gene_data(gene1, base_path)
                                gene2_data = load_gene_data(gene2, base_path)
                                download_df = pd.DataFrame(
                                    {
                                        gene1: gene1_data[gene1_data.columns[0]],
                                        gene2: gene2_data[gene2_data.columns[0]],
                                        "Cell_Type": obs_data["new_cell_type"],
                                        "SRA_ID": obs_data["SRA_ID"],
                                    }
                                )

                                if valid_sra_ids is not None:
                                    download_df = download_df[
                                        download_df["SRA_ID"].isin(valid_sra_ids)
                                    ]

                                if selected_cell_types is not None:
                                    download_df = download_df[
                                        download_df["Cell_Type"].isin(
                                            selected_cell_types
                                        )
                                    ]

                                st.download_button(
                                    label="Download Correlation Data",
                                    data=download_df.to_csv(index=False),
                                    file_name=f"correlation_{gene1}_{gene2}.csv",
                                    mime="text/csv",
                                    key="download_button_gene_corr",
                                    help="Download the current correlation dataset",
                                )

                                with st.container():
                                    st.markdown(
                                        """
                                        This plot shows the correlation between single-cell expression levels of two selected genes.
                                        
                                        **X-axis**: Expression level of first selected gene
                                        **Y-axis**: Expression level of second selected gene
                                        
                                        The visualization includes:
                                        - Scatter plot of gene expression values
                                        - Optional coloring by cell type
                                        - Correlation line (dashed red)
                                        - Hover information showing sample details
                                        - Dynamic point opacity based on total number of points
                                                
                                        Note: This feature was requested by a user. The analysis may not be very informative, and any gene-gene relationship should not be assumed to be a result of regulation or the lack thereof. Gene counts here are simply log1p(counts_per_10k) values and batch effects are not accounted for in any way.
                                    """
                                    )

                        except Exception as e:
                            st.error(f"Error in gene correlation analysis: {str(e)}")
                            if st.checkbox("Show detailed error"):
                                st.exception(e)
                                traceback.print_exc() 
                                # Capture the full traceback
                                tb = traceback.format_exc()

                                # Display it in a collapsible section
                                with st.expander("Show full error traceback"):
                                    st.code(tb, language='python')

                with lig_rec_tab:

                    st.markdown(
                        "Click the button below to begin ligand-receptor analysis. This will load the necessary data."
                    )

                    click = tab_start_button(
                        "ligrec",
                        "begin_ligrec_analysis")

                    if click or (st.session_state["current_analysis_tab"] == "ligrec"):
                        gc.collect()
                        col1, col2 = st.columns([5, 1])

                        with col1:
                            st.header("Ligand-Receptor Interactions")
                            st.markdown("Explore ligand-receptor interactions between cell types in the mouse pituitary. Each dot represents a ligand-receptor pair between a source and target cell type.")
                        with col2:
                            selected_version = st.selectbox(
                                "Version",
                                options=AVAILABLE_VERSIONS,
                                key="version_select_lr",
                                label_visibility="collapsed",
                            )

                        with st.spinner("Loading ligand-receptor data..."):
                            liana_df = load_cached_ligand_receptor_data(
                                version=selected_version
                            )

                            all_genes = (
                                liana_df["ligand_complex"].values.tolist()
                                + liana_df["receptor_complex"].values.tolist()
                            )
                            all_genes = sorted(set(all_genes))

                            # Cell type selection
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                source_types = sorted(liana_df["source"].unique())
                                selected_source = st.multiselect(
                                    "Select Source Cell Types",
                                    options=source_types,
                                    default=[
                                        "Somatotrophs",
                                        "Lactotrophs",
                                        "Thyrotrophs",
                                        "Corticotrophs",
                                        "Melanotrophs",
                                        "Stem_cells",
                                        "Gonadotrophs",
                                    ],
                                    key="source_select_lr",
                                )

                            with col2:
                                target_types = sorted(liana_df["target"].unique())
                                selected_target = st.multiselect(
                                    "Select Target Cell Types",
                                    options=target_types,
                                    default=[
                                        "Somatotrophs",
                                        "Lactotrophs",
                                        "Thyrotrophs",
                                        "Corticotrophs",
                                        "Melanotrophs",
                                        "Stem_cells",
                                        "Gonadotrophs",
                                    ],
                                    key="target_select_lr",
                                )

                            with col3:
                                col3_1, col3_2, col3_3 = st.columns(3)
                                with col3_1:
                                    top_n = st.number_input(
                                        "Number of top interactions",
                                        min_value=5,
                                        max_value=40,
                                        value=30,
                                        step=5,
                                        key="top_n_lr",
                                        help="Shows top N interactions (maximum 40)",
                                    )
                                with col3_2:
                                    sort_by = st.selectbox(
                                        "Sort by",
                                        options=["magnitude", "specificity"],
                                        key="sort_by_lr",
                                        help="Choose whether to sort by magnitude or specificity rank",
                                        width=250
                                    )
                                with col3_3:
                                    x_axis_order = st.selectbox(
                                        "X-axis Order",
                                        options=["sender", "target"],
                                        key="x_axis_order_lr",
                                        help="Order x-axis by sender or target cell type",
                                        width=250
                                    )

                            # Add include/exclude interaction filters
                            st.subheader("Filter Specific Interactions")
                            col1, col2 = st.columns(2)

                            with col1:
                                include_interactions = st.multiselect(
                                    "Include Only These Interactions",
                                    options=all_genes,
                                    default=None,
                                    key="include_ligands_lr",
                                    help="Only show interactions with these interactions (leave empty to include all)",
                                )

                            with col2:

                                default_exclude = [
                                    "Gh",
                                    "Prl",
                                    "Tshb",
                                    "Pomc",
                                    "Fshb",
                                    "Lhb",
                                ]
                                valid_defaults = [
                                    gene
                                    for gene in default_exclude
                                    if gene in all_genes
                                ]

                                exclude_interactions = st.multiselect(
                                    "Exclude These Interactions",
                                    options=all_genes,
                                    default=valid_defaults,
                                    key="exclude_receptors_lr",
                                    help="Remove interactions with these interactions",
                                )

                            # Filter the dataframe based on include/exclude selections
                            filtered_df = liana_df.copy()

                            # Apply include filter if any ligands are selected
                            if include_interactions:
                                filtered_df = filtered_df[
                                    (
                                        filtered_df["ligand_complex"].isin(
                                            include_interactions
                                        )
                                    )
                                    | (
                                        filtered_df["receptor_complex"].isin(
                                            include_interactions
                                        )
                                    )
                                ]

                            # Apply exclude filter
                            if exclude_interactions:
                                filtered_df = filtered_df[
                                    (
                                        ~filtered_df["receptor_complex"].isin(
                                            exclude_interactions
                                        )
                                    )
                                    & (
                                        ~filtered_df["ligand_complex"].isin(
                                            exclude_interactions
                                        )
                                    )
                                ]
                                
                            add_activity(value=[selected_source, selected_target],
                                    analysis="Ligand-Receptor Interactions",
                                    user=st.session_state.session_id,time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

                            chosen_color_scheme = st.selectbox(
                                    "Select Color Scheme",
                                    options=["Blue","Red","Viridis","Cividis"],
                                    index=0,
                                    key="color_scheme_lr",
                                    width=250
                                )
                            
                            # Create and display the plot
                            fig, config, plot_df = create_ligand_receptor_plot(
                                filtered_df,
                                selected_source=(
                                    selected_source if selected_source else None
                                ),
                                selected_target=(
                                    selected_target if selected_target else None
                                ),
                                top_n=top_n,
                                sort_by=sort_by,
                                order_by=x_axis_order,  # Pass the x-axis ordering option
                                color_scheme=chosen_color_scheme
                            )
                            gc.collect()

                            st.plotly_chart(
                                fig, use_container_width=True, config=config
                            )

                            with st.container():
                                st.markdown(
                                    """
                                    This plot shows significant ligand-receptor interactions between cell types.
                                    
                                    **X-axis**: Cell type pairs (Source → Target)
                                    **Y-axis**: Ligand-Receptor pairs
                                    
                                    The visualization shows:
                                    - Dot size*: Specificity of the interaction (-log10 transformed)
                                    - Dot color*: Magnitude of the interaction (-log10 transformed)
                                    - Hover information includes exact specificity and magnitude values
                                    
                                    Use the controls above to:
                                    - Select specific source and target cell types
                                    - Adjust the number of top interactions shown
                                    - Choose whether to sort by magnitude or specificity
                                    - Download the plot as an SVG file
                                    
                                    * Magnitude and specificity values are derived from robust rank aggregation. Briefly these values represent the minimum probability of having observed a given interaction x times at a given rank by chance. For more insight please consult the LIANA+ (https://doi.org/10.1038/s41556-024-01469-w) and RRA (doi: 10.1093/bioinformatics/btr709) papers.
                                """
                                )

                            # Table section
                            st.markdown("---")
                            st.subheader("Detailed Interaction Table")
                            # Add toggle for table view
                            show_all_interactions = st.checkbox(
                                "Show all interactions in table (not just plotted/filtered ones)",
                                value=False,
                                help="Toggle between showing only the filtered interactions or all interactions in the dataset",
                            )

                            st.markdown(
                                """
                                Explore and filter ligand-receptor interactions. Use the column headers to sort and filter the data.
                                The table shows:
                                - Complete interaction pairs
                                - Source and target cell types
                                - Specificity and magnitude ranks
                            """
                            )

                            # Use either all interactions or just the filtered ones based on toggle
                            table_data = (
                                liana_df if show_all_interactions else filtered_df
                            )

                            filtered_data = display_ligand_receptor_table(
                                table_data, key_prefix="lr_interactions"
                            )

                            # Add informational text based on the toggle selection
                            if show_all_interactions:
                                st.info(
                                    f"Showing all {len(liana_df)} interactions in the dataset. The plot above shows only the top {top_n} filtered interactions."
                                )
                            else:
                                st.info(
                                    f"Showing {len(filtered_df)} interactions that match the current filter criteria and appear in the plot above."
                                )

        with chromatin_tab:

            with st.container():
                accessibility_tab, motif_tab, cell_type_atac_tab = st.tabs(
                    [
                        "Accessibility Distribution (Motifs/Enhancers)",
                        "Motif Enrichment (ChromVAR)",
                        "Cell Type Distribution",
                    ]
                )

                with accessibility_tab:
                    st.markdown(
                        "Click the button below to begin chromatin accessibility analysis. This will load the necessary data."
                    )

                    click = tab_start_button(
                        "chromatin_analysis",
                        "begin_chromatin_analysis")

                    if click or (st.session_state["current_analysis_tab"] == "chromatin_analysis"):
                        gc.collect()
                        col1, col2 = st.columns([5, 1])
                        with col1:
                            st.header("Accessibility Distribution")
                            st.markdown("Visualize the distribution of chromatin accessibility across cell types in the mouse pituitary. Each dot is a pseudobulk sample.")
                        with col2:
                            selected_version = st.selectbox(
                                "Version",
                                options=AVAILABLE_VERSIONS,
                                key="version_select_tab5",
                                label_visibility="collapsed",
                            )

                        with st.spinner("Loading accessibility data..."):
                            (
                                accessibility_matrix,
                                accessibility_meta,
                                features,
                                columns,
                            ) = load_cached_accessibility_data(version=selected_version)

                        try:

                            # Sample/Author filtering controls
                            st.subheader("Data Filtering")

                            (
                                filter_type,
                                selected_samples,
                                selected_authors,
                                age_range,
                                only_normal,
                            ) = create_filter_ui(
                                accessibility_meta, key_suffix="accessibility"
                            )

                            # Apply filters
                            filtered_meta = accessibility_meta.copy()

                            if filter_type == "Age" and "age_range" in locals():
                                age_mask = (
                                    filtered_meta["Age_numeric"] >= age_range[0]
                                ) & (filtered_meta["Age_numeric"] <= age_range[1])
                                filtered_meta = filtered_meta[age_mask]

                            filtered_meta, filtered_matrix = filter_accessibility_data(
                                meta_data=filtered_meta,
                                selected_samples=selected_samples,
                                selected_authors=selected_authors,
                                matrix=accessibility_matrix,
                                only_normal=only_normal,
                            )

                            # Display cell type statistics
                            filtered_geo_ids = filtered_meta["GEO"].unique().tolist()

                            create_cell_type_stats_display(
                                version=selected_version,
                                sra_ids=filtered_geo_ids,
                                display_title="Cell Counts in Current Selection",
                                column_count=6,
                                size="small",
                                atac_rna="atac",
                            )

                            # Feature selection
                            selected_feature = st.selectbox(
                                f"Select Feature ({len(features)} features)",
                                features,
                                key="feature_select_tab5",
                                width=250
                            )

                            # Additional grouping
                            additional_group = st.selectbox(
                                "Additional Grouping Variable",
                                ["None", "Modality", "Comp_sex"],
                                key="additional_group_select_tab5",
                                width=250
                            )

                            # Connect dots toggle
                            connect_dots = st.checkbox(
                                "Connect Dots",
                                value=False,
                                help="Connect dots with the same SRA_ID (e.g., to visualise if outlier samples across cell types are from the same study)",
                                key="connect_dots_tab5",
                            )

                            # Create plot
                            if selected_feature:
                                add_activity(value=selected_feature, analysis="Accessibility Distribution",
                                    user=st.session_state.session_id,time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                                
                                fig, config = create_accessibility_plot(
                                    matrix=filtered_matrix,
                                    features=features,
                                    meta_data=filtered_meta,
                                    feature_name=selected_feature,
                                    additional_group=(
                                        None
                                        if additional_group == "None"
                                        else additional_group
                                    ),
                                    connect_dots=connect_dots,
                                )
                                st.plotly_chart(
                                    fig, use_container_width=True, config=config
                                )
                                gc.collect()
                                with st.container():
                                    st.markdown(
                                        """
                                        This plot shows the distribution of chromatin accessibility fragments counts.
                                        
                                        **X-axis**: Cell types present in the selected samples
                                        **Y-axis**: Log10 fragment counts per million* for the selected genomic region
                                        
                                        The plot combines:
                                        - Box plot showing statistical distribution
                                        - Individual points representing accessibility in each sample
                                        - Optional grouping by additional variables
                                        - Optional connecting lines between samples from the same source
                                        
                                        * Fragment counts were first TMMwsp normalised in the Limma-voom workflow.
                                    """
                                    )

                                # Display filtered data info
                                if filter_type != "No filter":
                                    sample_count = filtered_meta.shape[0]
                                    cell_type_count = filtered_meta[
                                        "cell_type"
                                    ].nunique()
                                    st.info(
                                        f"Showing data from {sample_count} pseudobulk samples across {cell_type_count} cell types"
                                    )

                                # Add download button
                                feature_idx = features.index(selected_feature)
                                if scipy.sparse.issparse(filtered_matrix):
                                    accessibility_values = (
                                        filtered_matrix[feature_idx].toarray().flatten()
                                    )
                                else:
                                    accessibility_values = filtered_matrix[feature_idx]

                                download_df = filtered_meta.copy()
                                download_df["Accessibility"] = accessibility_values

                                st.download_button(
                                    label="Download Plotting Data",
                                    data=download_df.to_csv(index=False),
                                    file_name=f"{selected_feature}_accessibility_data.csv",
                                    mime="text/csv",
                                    key="download_button_tab5",
                                    help="Download the current filtered dataset used for plotting",
                                )
                                # Add gene-based region selection
                                st.subheader("Region Selection")
                                selection_method = st.radio(
                                    "Select region by:",
                                    ["Gene", "Coordinates"],
                                    key="region_selection_method",
                                )

                                if selection_method == "Gene":
                                    # Gene selection dropdown with default value
                                    
                                    gene_list = load_motif_genes(version=selected_version).tolist()
                                    annotation_df = load_cached_annotation_data(version=selected_version)
                                    
                                    selected_gene, selected_region = create_gene_selector_with_coordinates(
                                        gene_list=gene_list,
                                        key_suffix="browser",
                                        annotation_df=annotation_df,
                                        selected_version=selected_version,
                                        flank_fraction=0.2,
                                        label="Select Gene"
                                    )

                                    gene_data_pl = annotation_df.filter(pl.col("gene_name") == selected_gene)

                                    # Convert to Pandas for further processing
                                    gene_data = gene_data_pl.to_pandas()

                                    if not gene_data.empty:
                                        gene_chr = gene_data["seqnames"].iloc[0]
                                        gene_start = int(gene_data["start"].min())
                                        gene_end = int(gene_data["end"].max())

                                        # Calculate extended region (20% extension)
                                        gene_length = gene_end - gene_start
                                        extension = int(gene_length * 0.2)

                                        extended_start = max(0, gene_start - extension)
                                        extended_end = gene_end + extension

                                        selected_region = f"{gene_chr}:{extended_start}-{extended_end}"
                                        

                                        # Display gene coordinates
                                        st.info(
                                            f"Gene coordinates: {gene_chr}:{gene_start:,}-{gene_end:,}"
                                        )
                                    else:
                                        st.warning(
                                            f"No coordinate data found for gene {selected_gene}"
                                        )
                                        selected_region = f"chr3:34650405-34652461"

                                else:

                                    annotation_df = load_cached_annotation_data(
                                        version=selected_version
                                    )

                                    selected_region = create_region_selector(
                                        key_suffix="browser",
                                        label="Select Region"
                                    )
                                    

                                # Add genome browser section
                                st.subheader("Genome Browser View")

                                # Create color mapping for cell types
                                cell_types = sorted(filtered_meta["cell_type"].unique())
                                color_map = create_color_mapping(cell_types)

                                # Add cell type filtering for genome browser
                                st.subheader("Cell Type Selection")
                                all_cell_types = sorted(
                                    filtered_meta["cell_type"].unique()
                                )
                                selected_cell_types_browser = st.multiselect(
                                    "Select Cell Types to Display",
                                    options=all_cell_types,
                                    default=all_cell_types,
                                    help="Choose which cell types to display in the genome browser",
                                )

                                # Filter metadata and matrix for selected cell types
                                cell_type_mask = filtered_meta["cell_type"].isin(selected_cell_types_browser)
                                browser_meta = filtered_meta[cell_type_mask].copy()
                                browser_matrix = filtered_matrix[:, cell_type_mask.values]

                                # Add motif selection
                                st.subheader("Motif Selection")
                                # Load motifs from ChromVAR data
                                (
                                    chromvar_matrix,
                                    chromvar_meta,
                                    chromvar_features,
                                    chromvar_columns,
                                ) = load_cached_chromvar_data(version=selected_version)

                                motif_data = load_cached_motif_data(
                                    version=selected_version
                                )

                                selected_motifs = st.multiselect(
                                    "Select Motifs to Display (max 10)",
                                    options=sorted(motif_data.select("motif").unique().to_series().to_list()),
                                    max_selections=10,
                                    help="Choose up to 10 motifs to display in the genome browser",
                                )

                                st.info(
                                    "Only those motifs are shown that fall in a given consensus peak."
                                )
                                if selected_motifs:
                                    add_activity(value=[selected_region, selected_motifs],
                                    analysis="Genome Browser",
                                    user=st.session_state.session_id,time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                                else:
                                    add_activity(value=selected_region,
                                    analysis="Genome Browser",
                                    user=st.session_state.session_id,time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                                
                                show_enhancers = st.checkbox(
                                    "Show enhancers",
                                    value=False,
                                    help="Display enhancer regions in the genome browser",
                                    key="show_enhancers",
                                )

                                enhancer_df = load_cached_enhancer_data(
                                    version=selected_version
                                )

                                # Create genome browser plot with selected motifs and filtered cell types
                                filtered_enhancers, browser_fig, browser_config, error_message = (
                                    create_genome_browser_plot(
                                        matrix=browser_matrix,
                                        features=features,
                                        meta_data=browser_meta,
                                        selected_region=selected_region,
                                        selected_version=selected_version,
                                        annotation_df=annotation_df,
                                        show_enhancers=show_enhancers,
                                        enhancer_df = enhancer_df,

                                        motif_df=motif_data,
                                        color_map={
                                            ct: color_map[ct]
                                            for ct in selected_cell_types_browser
                                        },
                                        selected_motifs=(
                                            selected_motifs if selected_motifs else None
                                        ),
                                    )
                                )

                                if error_message:
                                    st.warning(error_message)
                                elif browser_fig:
                                    st.plotly_chart(
                                        browser_fig,
                                        use_container_width=True,
                                        config=browser_config,
                                    )
                                    gc.collect()

                                    with st.container():
                                        st.markdown(
                                            """
                                            This genome browser view shows accessibility profiles and genomic features.
                                            
                                            **X-axis**: Genomic coordinates (base pairs)
                                            **Y-axis**: Multiple tracks showing:
                                            - Fragment counts per million (log10)
                                            - Gene annotations
                                            - Selected motif positions
                                            
                                            Features:
                                            - Cell type-specific accessibility profiles
                                            - Gene structure visualization
                                            - Motif position markers
                                            - Interactive zooming and panning
                                        """
                                        )

                                    # Add download button for browser data
                                    browser_data = []
                                    for feature in features:
                                        try:
                                            feat_chr, feat_range = feature.split(":")
                                            feat_start, feat_end = map(
                                                int, feat_range.split("-")
                                            )

                                            if (
                                                feat_chr == selected_chr
                                                and feat_start <= end_pos
                                                and feat_end >= start_pos
                                            ):

                                                feature_idx = features.index(feature)
                                                for cell_type in cell_types:
                                                    cell_indices = filtered_meta[
                                                        filtered_meta["cell_type"]
                                                        == cell_type
                                                    ].index
                                                    if hasattr(
                                                        filtered_matrix, "toarray"
                                                    ):
                                                        signal = (
                                                            filtered_matrix[
                                                                feature_idx,
                                                                cell_indices,
                                                            ]
                                                            .toarray()
                                                            .mean()
                                                        )
                                                    else:
                                                        signal = filtered_matrix[
                                                            feature_idx, cell_indices
                                                        ].mean()

                                                    browser_data.append(
                                                        {
                                                            "Feature": feature,
                                                            "Cell_Type": cell_type,
                                                            "Mean_Signal": float(
                                                                signal
                                                            ),
                                                        }
                                                    )
                                        except:
                                            continue

                                    if browser_data:
                                        browser_df = pd.DataFrame(browser_data)
                                        st.download_button(
                                            label="Download Browser Data",
                                            data=browser_df.to_csv(index=False),
                                            file_name=f"genome_browser_{selected_chr}_{start_pos}_{end_pos}.csv",
                                            mime="text/csv",
                                            key="download_button_browser",
                                            help="Download the data shown in the genome browser plot",
                                        )
                                    
                                    # Download enhancer data if shown
                                    if show_enhancers and enhancer_df is not None:
                                        
                                        st.markdown("---")
                                        col1, col2 = st.columns([5, 1])
                                        with col1:
                                            st.subheader("Enhancer Regions")
                                        with col2:
                                            selected_version = st.selectbox(
                                                'Version',
                                                options=AVAILABLE_VERSIONS,
                                                key='version_select_sex_dim',
                                                label_visibility="collapsed"
                                            )

                                        filtered_enhancer_data = display_enhancers_table(enhancers_data=filtered_enhancers, key_prefix="enhancers")

                                #add separator line
                                st.markdown("---")
                                # Add marker browser section
                                col1, col2 = st.columns([5, 1])
                                with col1:
                                    st.subheader("Marker Peaks Browser")
                                with col2:
                                    selected_version = st.selectbox(
                                        "Version",
                                        options=AVAILABLE_VERSIONS,
                                        key="version_select_marker_browser",
                                        label_visibility="collapsed",
                                    )

                                filtered_data = display_marker_table(
                                    selected_version, load_cached_marker_data_atac, "accessibility"
                                )

                        except Exception as e:
                            st.error(
                                f"Error in accessibility data processing: {str(e)}"
                            )
                            traceback.print_exc() 
                            # Capture the full traceback
                            tb = traceback.format_exc()

                            # Display it in a collapsible section
                            with st.expander("Show full error traceback"):
                                st.code(tb, language='python')
                            st.error("Please check your data paths and file formats.")

                with motif_tab:
                    st.markdown(
                        "Click the button below to begin TF enrichment analysis. This will load the necessary data."
                    )

                    click = tab_start_button(
                        "chromvar",
                        "begin_chromvar_analysis")

                    if click or (st.session_state["current_analysis_tab"] == "chromvar"):
                        gc.collect()
                        col1, col2 = st.columns([5, 1])
                        with col1:
                            st.header("Motif Enrichment (ChromVAR)")
                            st.markdown("Explore transcription factor motif enrichment across cell types in the mouse pituitary using ChromVAR. Each dot represents a pseudobulk sample.")
                        with col2:
                            selected_version = st.selectbox(
                                "Version",
                                options=AVAILABLE_VERSIONS,
                                key="version_select_tab6",
                                label_visibility="collapsed",
                            )

                        with st.spinner("Loading ChromVAR data..."):
                            chromvar_matrix, chromvar_meta, features, columns = (
                                load_cached_chromvar_data(version=selected_version)
                            )

                        try:
                            # Load chromvar data
                            chromvar_matrix, chromvar_meta, features, columns = (
                                load_cached_chromvar_data(version=selected_version)
                            )

                            # Sample/Author filtering controls
                            st.subheader("Data Filtering")

                            (
                                filter_type,
                                selected_samples,
                                selected_authors,
                                age_range,
                                only_normal,
                            ) = create_filter_ui(chromvar_meta, key_suffix="chromvar")

                            # Apply filters
                            filtered_meta = (
                                chromvar_meta.copy()
                            )  # Make a copy to avoid modifying original

                            if filter_type == "Age" and "age_range" in locals():
                                age_mask = (
                                    filtered_meta["Age_numeric"] >= age_range[0]
                                ) & (filtered_meta["Age_numeric"] <= age_range[1])
                                filtered_meta = filtered_meta[age_mask]

                            try:
                                filtered_meta, filtered_matrix = filter_chromvar_data(
                                    meta_data=filtered_meta,
                                    selected_samples=selected_samples,
                                    selected_authors=selected_authors,
                                    matrix=chromvar_matrix,
                                    only_normal=only_normal,
                                )

                                filtered_geo_ids = (
                                    filtered_meta["GEO"].unique().tolist()
                                )
                                
                                create_cell_type_stats_display(
                                    version=selected_version,
                                    sra_ids=filtered_geo_ids,
                                    display_title="Cell Counts in Current Selection",
                                    column_count=6,
                                    size="small",
                                    atac_rna="atac",
                                )

                                # Motif selection
                                default_motif = (
                                    "MA0143.4_SOX2"
                                    if "MA0143.4_SOX2" in features
                                    else features[0]
                                )
                                selected_motif = st.selectbox(
                                    f"Select Motif ({len(features)} motifs)",
                                    features,
                                    index=features.index(default_motif),
                                    key="motif_select_tab6",
                                    width=250
                                )

                                # Additional grouping
                                additional_group = st.selectbox(
                                    "Additional Grouping Variable",
                                    ["None", "Modality", "Comp_sex"],
                                    key="additional_group_select_tab6",
                                    width=250
                                )

                                # Connect dots toggle
                                connect_dots = st.checkbox(
                                    "Connect Dots",
                                    value=False,
                                    help="Connect dots with the same SRA_ID (e.g., to visualise if outlier samples across cell types are from the same study)",
                                    key="connect_dots_tab6",
                                )

                                # Create plot
                                if selected_motif:

                                    add_activity(value=selected_motif, analysis="ChromVAR",
                                    user=st.session_state.session_id,time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


                                    fig, config = create_chromvar_plot(
                                        matrix=filtered_matrix,
                                        features=features,
                                        meta_data=filtered_meta,
                                        motif_name=selected_motif,
                                        additional_group=(
                                            None
                                            if additional_group == "None"
                                            else additional_group
                                        ),
                                        connect_dots=connect_dots,
                                    )
                                    st.plotly_chart(
                                        fig, use_container_width=True, config=config
                                    )
                                    gc.collect()
                                    with st.container():
                                        st.markdown(
                                            """
                                            This plot shows the enrichment of transcription factor motifs across cell types.
                                            
                                            **X-axis**: Cell types present in the selected samples
                                            **Y-axis**: ChromVAR deviation score* (motif enrichment)
                                            
                                            The plot combines:
                                            - Box plot showing the distribution of enrichment scores
                                            - Individual points representing each sample
                                            - Optional grouping by additional variables
                                            - Optional connecting lines between samples from the same source
                                                    
                                            *Briefly these are Z-scores of the fragments falling into peaks with a given motif compared to a null distribution of random peaks. For more insight please consult the ChromVAR (https://doi.org/10.1038/nmeth.4401) paper.
                                        """
                                        )

                                    # Display filtered data info using DataFrame methods
                                    if filter_type != "No filter":
                                        sample_count = filtered_meta.shape[0]
                                        cell_type_count = filtered_meta[
                                            "cell_type"
                                        ].nunique()
                                        st.write(
                                            f"Showing data from {sample_count} pseudobulk samples across {cell_type_count} cell types"
                                        )

                                    # Add download button
                                    motif_idx = features.index(selected_motif)
                                    if scipy.sparse.issparse(filtered_matrix):
                                        enrichment_values = (
                                            filtered_matrix[motif_idx]
                                            .toarray()
                                            .flatten()
                                        )
                                    else:
                                        enrichment_values = filtered_matrix[motif_idx]

                                    download_df = filtered_meta.copy()
                                    download_df["Enrichment"] = enrichment_values

                                    st.download_button(
                                        label="Download Plotting Data",
                                        data=download_df.to_csv(index=False),
                                        file_name=f"{selected_motif}_enrichment_data.csv",
                                        mime="text/csv",
                                        key="download_button_tab6",
                                        help="Download the current filtered dataset used for plotting",
                                    )

                            except Exception as e:
                                st.error(
                                    f"Error in data filtering or plotting: {str(e)}"
                                )
                                traceback.print_exc() 
                                # Capture the full traceback
                                tb = traceback.format_exc()

                                # Display it in a collapsible section
                                with st.expander("Show full error traceback"):
                                    st.code(tb, language='python')

                            # Add enrichment results section
                            st.markdown("---")
                            st.subheader("Motif Enrichment Analysis")

                            # Load enrichment results
                            enrichment_df = load_cached_enrichment_data(
                                version=selected_version
                            )

                            # Display the interactive table
                            filtered_data = display_enrichment_table(
                                enrichment_df, key_prefix="motif_enrichment"
                            )

                            # Add explanation
                            with st.container():
                                st.markdown(
                                    """
                                    Note:
                                    These results are related but separate from ChromVAR, as they have been derived from the same data and the same motifs.
                                    The difference is that these are simple enrichment results on differentially accessible peaks.
                                    This has the advantage over ChromVAR that we can define biologically meaningful comparisons in differential accessibility
                                    between lineages. The downside is that it performs enrichment by considering peaks as equal, not taking into account their 
                                    magnitude of fragment counts.
                                    
                                            
                                    This table shows the results of motif enrichment analysis across different cell type comparisons.
                                    Each row represents a motif that is significantly enriched in one group compared to another.
                                    
                                    Key columns:
                                    - **Motif**: The enriched transcription factor binding motif
                                    - **Adjusted p-value**: Statistical significance after multiple testing correction
                                    - **Log2 Fold Change**: Magnitude and direction of enrichment
                                    - **Grouping**: The cell type comparison group
                                    - **Direction**: Whether the motif is enriched (UP) or depleted (DOWN) in the first group
                                    
                                    Use the filters above to focus on specific groupings or directions of interest.
                                """
                                )

                        except Exception as e:
                            st.error(f"Error loading ChromVAR data: {str(e)}")
                            st.error("Please check your data paths and file formats.")
                            traceback.print_exc() 
                            # Capture the full traceback
                            tb = traceback.format_exc()

                            # Display it in a collapsible section
                            with st.expander("Show full error traceback"):
                                st.code(tb, language='python')

                with cell_type_atac_tab:
                    gc.collect()
                    st.markdown(
                        "Click the button below to begin cell type distribution analysis. This will load the necessary data."
                    )

                    click = tab_start_button(
                        "distribution_atac",
                        "begin_cell_type_atac_analysis")

                    if click or (st.session_state["current_analysis_tab"] == "distribution_atac"):
                        gc.collect()
                        col1, col2 = st.columns([5, 1])
                        with col1:
                            st.header("ATAC Cell Type Distribution")
                            st.markdown("Visualize the distribution of cell type abundance across samples in the mouse pituitary.")
                        with col2:
                            selected_version = st.selectbox(
                                "Version",
                                options=AVAILABLE_VERSIONS,
                                key="version_select_atac_proportion",
                                label_visibility="collapsed",
                            )

                        with st.spinner("Loading ATAC proportion data..."):
                            proportion_matrix, proportion_rows, proportion_cols = (
                                load_cached_atac_proportion_data(
                                    version=selected_version
                                )
                            )

                            if proportion_matrix is None:
                                st.error(
                                    "Failed to load ATAC proportion data. Please check the data files."
                                )
                            else:
                                # Filter controls
                                st.subheader("Data Filtering")
                                filter_type = st.radio(
                                    "Filter data by:",
                                    ["No filter", "Sample", "Author"],
                                    key="filter_type_atac_proportion",
                                )
                                meta_data = load_cached_curation_data(
                                    version=selected_version
                                )

                                # Map GEO to SRA_ID for matching
                                geo_to_sra = dict(
                                    zip(meta_data["GEO"], meta_data["SRA_ID"])
                                )
                                sra_to_meta = {}
                                for _, row in meta_data.iterrows():
                                    if pd.notna(row["GEO"]):
                                        sra_to_meta[row["GEO"]] = row

                                # in meta_data keep only where Modality is "atac" or "multi_atac"
                                meta_data = meta_data[
                                    meta_data["Modality"].isin(["atac", "multi_atac"])
                                ]

                                # Get unique sample names and authors
                                all_samples = sorted(meta_data["Name"].unique())
                                all_authors = sorted(meta_data["Author"].unique())
                                selected_samples = all_samples
                                selected_authors = all_authors

                                if filter_type == "Sample":
                                    selected_samples = st.multiselect(
                                        "Select Samples",
                                        all_samples,
                                        default=[all_samples[0]],
                                        help="Choose which samples to include in the analysis",
                                        key="samples_multiselect_atac_proportion",
                                    )
                                elif filter_type == "Author":
                                    selected_authors = st.multiselect(
                                        "Select Authors",
                                        all_authors,
                                        default=[all_authors[0]],
                                        help="Choose which authors' data to include",
                                        key="authors_multiselect_atac_proportion",
                                    )

                                # Filter toggles in columns
                                col1, col2 = st.columns(2)
                                with col1:
                                    only_normal = st.checkbox(
                                        "Show only control samples",
                                        value=False,
                                        help="Samples that are wild-type, untreated etc. (In curation, Normal == 1)",
                                        key="only_normal_atac_proportion",
                                    )
                                    only_whole = st.checkbox(
                                        "Show only whole-pituitary samples",
                                        value=False,
                                        help="Samples not sorted to enrich for a given sub-population (In curation, Sorted == 0)",
                                        key="only_whole_atac_proportion",
                                    )
                                    show_mean = st.checkbox(
                                        "Show mean proportions",
                                        value=False,
                                        help="Show average cell type proportions across selected samples",
                                        key="show_mean_atac_proportion",
                                    )
                                with col2:
                                    group_by_sex = st.checkbox(
                                        "Group by Sex",
                                        value=False,
                                        help="Create separate plots for male and female samples",
                                        key="group_by_sex_atac_proportion",
                                    )
                                    order_by_age = st.checkbox(
                                        "Order by Age",
                                        value=False,
                                        help="Order samples by age",
                                        key="order_by_age_atac_proportion",
                                    )

                                    download_as = st.selectbox(
                                        "Download as:",
                                        options=["png", "jpeg", "svg"],
                                        index=0,
                                        key="download_as_atac_proportion",
                                        width=250
                                    )

                                    # Show log age option only when ordering by age
                                    use_log_age = False
                                    # if order_by_age:
                                    #    use_log_age = st.checkbox('Use log10(Age)', value=False,
                                    #                            help="Use log10 scale for age and group similar ages",
                                    #                            key='use_log_age_atac_proportion')

                                # Apply filters to meta_data
                                filtered_meta = meta_data.copy()

                                if filter_type == "Sample":
                                    filtered_meta = filtered_meta[
                                        filtered_meta["Name"].isin(selected_samples)
                                    ]
                                elif filter_type == "Author":
                                    filtered_meta = filtered_meta[
                                        filtered_meta["Author"].isin(selected_authors)
                                    ]
                                if only_normal:
                                    filtered_meta = filtered_meta[
                                        filtered_meta["Normal"] == 1
                                    ]

                                if only_whole:
                                    filtered_meta = filtered_meta[
                                        filtered_meta["Sorted"] == 0
                                    ]

                                # Get valid GEO IDs from filtered metadata
                                valid_geo_ids = list(
                                    set(filtered_meta["GEO"].dropna().unique())
                                )

                                atac_meta = filtered_meta.copy()
                                # keep only the rows that have a valid GEO ID
                                atac_meta = atac_meta[
                                    atac_meta["GEO"].isin(valid_geo_ids)
                                ]

                                # Display statistics about cell types if we have SRA_IDs
                                # ADJUST THIS WHOLE function FOR ATAC SAMPLES!!
                                create_cell_type_stats_display(
                                    version=selected_version,
                                    sra_ids=atac_meta["GEO"].tolist(),
                                    display_title="Cell Counts in Current Selection",
                                    column_count=6,
                                    size="small",
                                    atac_rna="atac",
                                )

                                # Create plot
                                add_activity(value="NA",
                                    analysis="Cell Type Proportions ATAC",
                                    user=st.session_state.session_id,time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

                                fig_male, fig_female, config, error_message = (
                                    
                                    create_proportion_plot(
                                        matrix=proportion_matrix,
                                        rows=proportion_rows,
                                        columns=proportion_cols,
                                        meta_data=meta_data,
                                        selected_samples=(
                                            selected_samples
                                            if filter_type == "Sample"
                                            else None
                                        ),
                                        selected_authors=(
                                            selected_authors
                                            if filter_type == "Author"
                                            else None
                                        ),
                                        only_normal=only_normal,
                                        only_whole=only_whole,
                                        group_by_sex=group_by_sex,
                                        order_by_age=order_by_age,
                                        show_mean=show_mean,
                                        use_log_age=use_log_age,
                                        atac_rna="atac",
                                        download_as=download_as
                                    )
                                )

                                if error_message:
                                    st.warning(error_message)
                                elif group_by_sex:
                                    if fig_male is not None:
                                        st.plotly_chart(
                                            fig_male,
                                            use_container_width=True,
                                            config=config,
                                        )
                                    if fig_female is not None:
                                        st.plotly_chart(
                                            fig_female,
                                            use_container_width=True,
                                            config=config,
                                        )
                                else:
                                    if (
                                        fig_male is not None
                                    ):  # Using fig_male as the main figure
                                        st.plotly_chart(
                                            fig_male,
                                            use_container_width=True,
                                            config=config,
                                        )

                                gc.collect()

                                with st.container():
                                    st.markdown(
                                        """
                                        This plot shows the relative proportions of different cell types across ATAC-seq samples.
                                        
                                        **X-axis**: Samples (can be ordered by age)
                                        **Y-axis**: Percentage of each cell type
                                        
                                        Features:
                                        - Stacked bar chart showing relative proportions
                                        - Optional grouping by sex
                                        - Option to show mean proportions
                                        - Age-based ordering and log-transformation
                                        - Smooth visualization option for age-based trends
                                    """
                                    )

                                # Add download button for data
                                if hasattr(proportion_matrix, "toarray"):
                                    prop_data = proportion_matrix.toarray()
                                else:
                                    prop_data = proportion_matrix

                                prop_df = pd.DataFrame(
                                    prop_data,
                                    index=proportion_rows.iloc[:, 0],
                                    columns=proportion_cols.iloc[:, 0],
                                )

                                st.download_button(
                                    label="Download Proportion Data",
                                    data=prop_df.to_csv(index=True),
                                    file_name="cell_type_proportions.csv",
                                    mime="text/csv",
                                    help="Download the cell type proportion data",
                                    key="download_button_proportion",
                                )

        with multimodal_tab:
            gc.collect()
            with st.container():
                multimodal_heatmap_tab = st.tabs(
                    ["Multimodal heatmap of TFs"]
                )

                with multimodal_heatmap_tab[0]:

                    st.markdown(
                        "Click the button below to begin analysis of TFs in lineage decisions. This will load the necessary data."
                    )

                    click = tab_start_button(
                        "multimodal_heatmap",
                        "begin_multi_heatmap_analysis")

                    if click or (st.session_state["current_analysis_tab"] == "multimodal_heatmap"):
                        gc.collect()
                        col1, col2 = st.columns([5, 1])

                        with col1:
                            st.header("Multimodal Heatmap of TFs")
                            st.markdown("Explore transcription factors involved in lineage decisions in the mouse pituitary by integrating gene expression and chromatin accessibility data. The heatmap displays scaled expression and accessibility values for key TFs across different cell types.")
                        with col2:
                            selected_version = st.selectbox(
                                "Version",
                                options=AVAILABLE_VERSIONS,
                                key="version_select_multi_heatmap",
                                label_visibility="collapsed",
                            )

                        with st.spinner("Loading multimodal heatmap data..."):
                            try:
                                # Load heatmap data
                                (
                                    motif_analysis_summary,
                                    coefs,
                                    rna_res,
                                    atac_res,
                                    mat,
                                    features,
                                    columns,
                                ) = load_cached_heatmap_data(version=selected_version)

                                # Define the groupings with more informative descriptions
                                groupings = {
                                    "grouping_1_up": {
                                        "name": "Stem Cells vs Hormone-Producing Cells (UP in Stem Cells)",
                                        "group_a": ["Stem_cells"],
                                        "group_b": [
                                            "Melanotrophs",
                                            "Corticotrophs",
                                            "Somatotrophs",
                                            "Lactotrophs",
                                            "Thyrotrophs",
                                            "Gonadotrophs",
                                        ],
                                    },
                                    "grouping_1_down": {
                                        "name": "Stem Cells vs Hormone-Producing Cells (DOWN in Stem Cells)",
                                        "group_a": ["Stem_cells"],
                                        "group_b": [
                                            "Melanotrophs",
                                            "Corticotrophs",
                                            "Somatotrophs",
                                            "Lactotrophs",
                                            "Thyrotrophs",
                                            "Gonadotrophs",
                                        ],
                                    },
                                    "grouping_2_up": {
                                        "name": "Gonadotrophs vs Others (UP in Gonadotrophs)",
                                        "group_a": ["Gonadotrophs"],
                                        "group_b": [
                                            "Stem_cells",
                                            "Melanotrophs",
                                            "Corticotrophs",
                                            "Somatotrophs",
                                            "Lactotrophs",
                                            "Thyrotrophs",
                                        ],
                                    },
                                    "grouping_2_down": {
                                        "name": "Gonadotrophs vs Others (DOWN in Gonadotrophs)",
                                        "group_a": ["Gonadotrophs"],
                                        "group_b": [
                                            "Stem_cells",
                                            "Melanotrophs",
                                            "Corticotrophs",
                                            "Somatotrophs",
                                            "Lactotrophs",
                                            "Thyrotrophs",
                                        ],
                                    },
                                    "grouping_3_up": {
                                        "name": "TPIT-lineage vs Others (UP in TPIT-lineage)",
                                        "group_a": ["Melanotrophs", "Corticotrophs"],
                                        "group_b": [
                                            "Stem_cells",
                                            "Gonadotrophs",
                                            "Somatotrophs",
                                            "Lactotrophs",
                                            "Thyrotrophs",
                                        ],
                                    },
                                    "grouping_3_down": {
                                        "name": "TPIT-lineage vs Others (DOWN in TPIT-lineage)",
                                        "group_a": ["Melanotrophs", "Corticotrophs"],
                                        "group_b": [
                                            "Stem_cells",
                                            "Gonadotrophs",
                                            "Somatotrophs",
                                            "Lactotrophs",
                                            "Thyrotrophs",
                                        ],
                                    },
                                    "grouping_4_up": {
                                        "name": "Melanotrophs vs Corticotrophs (UP in Melanotrophs)",
                                        "group_a": ["Melanotrophs"],
                                        "group_b": ["Corticotrophs"],
                                    },
                                    "grouping_4_down": {
                                        "name": "Melanotrophs vs Corticotrophs (DOWN in Melanotrophs)",
                                        "group_a": ["Melanotrophs"],
                                        "group_b": ["Corticotrophs"],
                                    },
                                    "grouping_5_up": {
                                        "name": "PIT1-lineage vs Others (UP in PIT1-lineage)",
                                        "group_a": [
                                            "Somatotrophs",
                                            "Lactotrophs",
                                            "Thyrotrophs",
                                        ],
                                        "group_b": [
                                            "Stem_cells",
                                            "Gonadotrophs",
                                            "Melanotrophs",
                                            "Corticotrophs",
                                        ],
                                    },
                                    "grouping_5_down": {
                                        "name": "PIT1-lineage vs Others (DOWN in PIT1-lineage)",
                                        "group_a": [
                                            "Somatotrophs",
                                            "Lactotrophs",
                                            "Thyrotrophs",
                                        ],
                                        "group_b": [
                                            "Stem_cells",
                                            "Gonadotrophs",
                                            "Melanotrophs",
                                            "Corticotrophs",
                                        ],
                                    },
                                    "grouping_6_up": {
                                        "name": "Lactotrophs vs Other PIT1-lineage (UP in Lactotrophs)",
                                        "group_a": ["Lactotrophs"],
                                        "group_b": ["Somatotrophs", "Thyrotrophs"],
                                    },
                                    "grouping_6_down": {
                                        "name": "Lactotrophs vs Other PIT1-lineage (DOWN in Lactotrophs)",
                                        "group_a": ["Lactotrophs"],
                                        "group_b": ["Somatotrophs", "Thyrotrophs"],
                                    },
                                    "grouping_7_up": {
                                        "name": "Somatotrophs vs Other PIT1-lineage (UP in Somatotrophs)",
                                        "group_a": ["Somatotrophs"],
                                        "group_b": ["Lactotrophs", "Thyrotrophs"],
                                    },
                                    "grouping_7_down": {
                                        "name": "Somatotrophs vs Other PIT1-lineage (DOWN in Somatotrophs)",
                                        "group_a": ["Somatotrophs"],
                                        "group_b": ["Lactotrophs", "Thyrotrophs"],
                                    },
                                    "grouping_8_up": {
                                        "name": "Thyrotrophs vs Other PIT1-lineage (UP in Thyrotrophs)",
                                        "group_a": ["Thyrotrophs"],
                                        "group_b": ["Lactotrophs", "Somatotrophs"],
                                    },
                                    "grouping_8_down": {
                                        "name": "Thyrotrophs vs Other PIT1-lineage (DOWN in Thyrotrophs)",
                                        "group_a": ["Thyrotrophs"],
                                        "group_b": ["Lactotrophs", "Somatotrophs"],
                                    },
                                }

                                # Get analyses available in the data
                                available_analyses = motif_analysis_summary[
                                    "analysis"
                                ].unique()

                                # Filter groupings to only show those available in the data
                                available_groupings = {
                                    k: v
                                    for k, v in groupings.items()
                                    if k in available_analyses
                                }

                                # Default selection
                                default_grouping = next(iter(available_groupings))

                                # Create a dictionary mapping display names to codes
                                grouping_display_to_code = {
                                    v["name"]: k for k, v in available_groupings.items()
                                }

                                # Analysis selection (grouping and direction)
                                st.subheader("Select Cell Type Comparison")

                                # Display a more informative selection menu
                                selected_grouping_name = st.selectbox(
                                    "Select Grouping and Direction",
                                    options=list(grouping_display_to_code.keys()),
                                    index=0,
                                    key="selected_grouping_name",
                                    help="Choose the cell type comparison and direction for analysis",
                                    width=250
                                )

                                # Get the actual analysis code
                                selected_analysis = grouping_display_to_code[
                                    selected_grouping_name
                                ]

                                # Show the grouping information to the user
                                group_info = groupings[selected_analysis]

                                # Display the grouping details in a nicely formatted way
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("#### Group A (Test Group)")
                                    st.markdown(
                                        ", ".join(
                                            [
                                                f"**{cell}**"
                                                for cell in group_info["group_a"]
                                            ]
                                        )
                                    )
                                with col2:
                                    st.markdown("#### Group B (Reference Group)")
                                    st.markdown(
                                        ", ".join(
                                            [
                                                f"**{cell}**"
                                                for cell in group_info["group_b"]
                                            ]
                                        )
                                    )

                                # Selection method
                                st.subheader("TF Selection Method")

                                selection_method = st.radio(
                                    "How would you like to select transcription factors?",
                                    [
                                        "Use top RNA and ATAC hits",
                                        "Select specific transcription factors",
                                        "Show all multimodal hits",
                                    ],
                                    key="tf_selection_method",
                                )

                                chosen_names = None
                                top_x_rna = 10
                                top_x_atac = 10
                                multimodal_selection = False

                                if selection_method == "Use top RNA and ATAC hits":
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        top_x_rna = st.slider(
                                            "Number of top RNA hits",
                                            min_value=5,
                                            max_value=30,
                                            value=10,
                                            step=1,
                                            key="top_x_rna_slider",
                                        )
                                    with col2:
                                        top_x_atac = st.slider(
                                            "Number of top ATAC hits",
                                            min_value=5,
                                            max_value=30,
                                            value=10,
                                            step=1,
                                            key="top_x_atac_slider",
                                        )
                                elif (
                                    selection_method
                                    == "Select specific transcription factors"
                                ):
                                    # Filter for the selected analysis
                                    filtered_results = motif_analysis_summary[
                                        motif_analysis_summary["analysis"]
                                        == selected_analysis
                                    ]

                                    # Get unique TF names
                                    available_tfs = sorted(
                                        filtered_results["gene"].unique()
                                    )

                                    # TF selection multiselect
                                    chosen_names = st.multiselect(
                                        "Select Transcription Factors",
                                        options=available_tfs,
                                        default=(
                                            available_tfs[:5]
                                            if len(available_tfs) >= 5
                                            else available_tfs
                                        ),
                                        key="selected_tfs",
                                        help="Select specific transcription factors to include in the heatmap",
                                    )
                                elif selection_method == "Show all multimodal hits":
                                    multimodal_selection = True

                                else:
                                    st.warning(
                                        "Please select a valid method for transcription factor selection."
                                    )

                                    if not chosen_names:
                                        st.warning(
                                            "No multimodal hits found for this analysis. Please select another comparison or method."
                                        )

                                # Visualization options
                                st.subheader("Visualization Options")

                                # Create columns for layout
                                col1, col2 = st.columns(2)
                                with col1:
                                    color_by_pval = st.checkbox(
                                        "Color by P-value",
                                        value=False,
                                        help="If checked, color the heatmap by -log10(p-value) instead of fold change",
                                    )

                                    use_motif_name = st.checkbox(
                                        "Use Motif IDs",
                                        value=False,
                                        help="If checked, use motif IDs instead of gene names in the heatmap",
                                    )

                                with col2:
                                    log10_pval_cap = st.number_input(
                                        "-log10(p-value) Cap",
                                        min_value=3,
                                        max_value=200,
                                        value=10,
                                        step=1,
                                        help="Maximum -log10(p-value) to display. Higher values will be capped at this value. This helps to visualize results more clearly.",
                                    )

                                    fc_cap = st.number_input(
                                        "Log2 Fold Change Cap",
                                        min_value=1.0,
                                        max_value=50.0,
                                        value=2.0,
                                        step=0.5,
                                        help="Maximum log2 fold change to display. Higher values will be capped at this value. This helps to visualize results more clearly.",
                                    )

                                # Add filtering thresholds
                                st.subheader("Filtering Thresholds")
                                col1, col2, col3 = st.columns(3)

                                with col1:
                                    AveExpr_threshold = st.number_input(
                                        "Min. Avg Expression",
                                        min_value=-2.0,
                                        max_value=10.0,
                                        value=0.0,
                                        step=0.5,
                                        help="Minimum average expression value to include a TF",
                                    )

                                with col2:
                                    mean_log2fc_threshold = st.number_input(
                                        "Min. Log2 Fold Change",
                                        min_value=0.0,
                                        max_value=5.0,
                                        value=0.0,
                                        step=0.5,
                                        help="Minimum log2 fold change to include a TF",
                                    )

                                with col3:
                                    fold_enrichment_threshold = st.number_input(
                                        "Min. Motif Enrichment",
                                        min_value=0.0,
                                        max_value=5.0,
                                        value=0.0,
                                        step=0.5,
                                        help="Minimum chromatin fold enrichment to include a TF",
                                    )

                                # Process button
                                process_button = st.button(
                                    "Process and Generate Heatmap",
                                    key="process_heatmap_button",
                                )

                                if process_button:
                                    with st.spinner("Processing heatmap data..."):
                                        try:

                                        
                                            add_activity(value = selected_analysis,
                                                analysis="Multimodal Heatmap",
                                                user=st.session_state.session_id,
                                                time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                                            # Process the heatmap data
                                            sub_matrix, motifs, plot_results, all_results = (
                                                process_heatmap_data(
                                                    motif_analysis_summary,
                                                    coefs,
                                                    rna_res,
                                                    atac_res,
                                                    mat,
                                                    features,
                                                    columns,
                                                    chosen_names=chosen_names,
                                                    top_x_rna=top_x_rna,
                                                    top_x_atac=top_x_atac,
                                                    grouping=selected_analysis,
                                                    multimodal=multimodal_selection,
                                                    AveExpr_threshold=AveExpr_threshold,
                                                    mean_log2fc_threshold=mean_log2fc_threshold,
                                                    fold_enrichment_threshold=fold_enrichment_threshold
                                                )
                                            )

                                            # Analyze TF co-binding
                                            results_df, fold_change_matrix = (
                                                analyze_tf_cobinding(
                                                    sub_matrix,
                                                    motifs,
                                                    return_matrix=True,
                                                )
                                            )
                                            

                                            
                                            # Create heatmap plot with additional options
                                            heatmap_fig = plot_heatmap(
                                                results_df,
                                                motifs,
                                                plot_results,
                                                sig_threshold=0.05,
                                                #fold_change_matrix=fold_change_matrix,
                                                use_motif_name=use_motif_name,
                                                log10_pval_cap=log10_pval_cap,
                                                fc_cap_log2=fc_cap,
                                            )

                                            # Display the plot
                                            st.pyplot(heatmap_fig)

                                            # Add download button for the plot
                                            from io import BytesIO

                                            buf = BytesIO()
                                            heatmap_fig.savefig(
                                                buf, format="svg", bbox_inches="tight"
                                            )
                                            buf.seek(0)

                                            st.download_button(
                                                label="Download Heatmap Plot",
                                                data=buf,
                                                file_name=f"tf_heatmap_{selected_analysis}.svg",
                                                mime="image/svg+xml",
                                                key="download_heatmap_plot",
                                            )

                                            # Add explanation
                                            with st.container():
                                                st.markdown(
                                                    """
                                                    ### TF Co-binding Heatmap Explanation
                                                    
                                                    This heatmap visualizes the co-binding patterns between transcription factors (TFs) based on their binding to shared genomic regions.
                                                    
                                                    **What the colors represent:**
                                                    - Fill: Dependent on the selected metric (fold change or -log10(p-value))
                                                            - Fold change: Change of co-binding compared to what's expected by chance
                                                            - log10(p-value): Significance of co-binding relationship as calculated by Fisher's exact test
                                                    - Blocks: Highlight significant co-binding relationships, often TF families.
                                                    - Capping: Values above the threshold are capped to improve visibility.

                                                    - The diagonal represents self-binding and is set to maximum value. To reveal other patterns these maximum values are capped.
                                                    - Row annotations show the hit category, which can be RNA, ATAC, or multimodal.
                                                    - Column annotations show the expression level of each TF when available. These help prioritise within TF families.
                                                                                                        
                                                """
                                                )

                                            # Detailed results table
                                            st.subheader("Co-binding Statistics")

                                            # Sort and filter the results

                                            # only include where sig is not nan
                                            filtered_results = results_df[
                                                ~results_df["adjusted_p_value"].isna()
                                            ]
                                            # also remove diagonal entries, so where TF1 != TF2
                                            filtered_results = filtered_results[
                                                filtered_results["TF1"]
                                                != filtered_results["TF2"]
                                            ]
                                            filtered_results = (
                                                filtered_results.sort_values(
                                                    "fold_change", ascending=False
                                                )
                                            )
                                            display_filtered_results = filtered_results[
                                                [
                                                    "TF1",
                                                    "TF2",
                                                    "fold_change",
                                                    "adjusted_p_value",
                                                    "observed_rate",
                                                    "expected_rate",
                                                ]
                                            ]
                                            # rename TF1 and TF2 to Motif1 and Motif2
                                            display_filtered_results = (
                                                display_filtered_results.rename(
                                                    columns={
                                                        "TF1": "Motif1",
                                                        "TF2": "Motif2",
                                                    }
                                                )
                                            )
                                            # using the motif and gene column in plot_results, get the gene name for the motif and place them in TF1 and TF2
                                            tf1_vals = []
                                            tf2_vals = []
                                            for (
                                                index,
                                                row,
                                            ) in display_filtered_results.iterrows():
                                                tf1_vals.append(
                                                    plot_results[
                                                        plot_results["motif"]
                                                        == row["Motif1"]
                                                    ]["gene"].values[0]
                                                )
                                                tf2_vals.append(
                                                    plot_results[
                                                        plot_results["motif"]
                                                        == row["Motif2"]
                                                    ]["gene"].values[0]
                                                )
                                            display_filtered_results["TF1"] = tf1_vals
                                            display_filtered_results["TF2"] = tf2_vals
                                            # make TF1 and TF2 first two cols
                                            display_filtered_results = (
                                                display_filtered_results[
                                                    [
                                                        "TF1",
                                                        "Motif1",
                                                        "TF2",
                                                        "Motif2",
                                                        "observed_rate",
                                                        "expected_rate",
                                                        "fold_change",
                                                        "adjusted_p_value",
                                                    ]
                                                ]
                                            )

                                            # adjusted p-value is in scientific notation
                                            # make smallest adjusted p-value 1e-300
                                            display_filtered_results[
                                                "adjusted_p_value"
                                            ] = display_filtered_results[
                                                "adjusted_p_value"
                                            ].apply(
                                                lambda x: 1e-300 if x == 0 else x
                                            )

                                            display_filtered_results[
                                                "-log10 adjusted_p_value"
                                            ] = display_filtered_results[
                                                "adjusted_p_value"
                                            ].apply(
                                                lambda x: -np.log10(x)
                                            )
                                            # sort by this
                                            display_filtered_results = (
                                                display_filtered_results.sort_values(
                                                    "-log10 adjusted_p_value",
                                                    ascending=False,
                                                )
                                            )
                                            display_filtered_results = (
                                                display_filtered_results.drop(
                                                    columns=["adjusted_p_value"]
                                                )
                                            )
                                            # Display the top results
                                            st.dataframe(
                                                display_filtered_results,
                                                hide_index=True,
                                            )

                                            # Add download button for results
                                            st.download_button(
                                                label="Download Co-binding Statistics",
                                                data=filtered_results.to_csv(
                                                    index=False
                                                ),
                                                file_name=f"tf_cobinding_stats_{selected_analysis}.csv",
                                                mime="text/csv",
                                                key="download_cobinding_stats",
                                            )

                                            # Display TF information
                                            st.subheader("Selected TF Information")
                                            # Make note
                                            st.markdown(
                                                "Note: Means are log2 counts per million (CPM) derived from the Limma-voom workflow."
                                            )

                                            # Get information for the selected TFs
                                            tf_info = all_results.copy()

                                            # Select and rename columns for clarity
                                            display_columns = {
                                                "gene": "Gene",
                                                "motif": "Motif ID",
                                                "hit_type": "Evidence Type",
                                                "pvalue_x": "ATAC P-value",
                                                "pvalue_y": "RNA P-value",
                                                "fold.enrichment": "ATAC Fold Enrichment",
                                                "log2fc": "RNA Log2 Fold Change",
                                                "means_group1": "Mean Group 1",
                                                "means_group2": "Mean Group 2",
                                                "motif_exists": "Motif Present in Database",
                                            }

                                            # Only keep columns that exist in the DataFrame
                                            valid_columns = [
                                                col
                                                for col in display_columns.keys()
                                                if col in tf_info.columns
                                            ]

                                            # Create the display DataFrame
                                            display_df = tf_info[valid_columns].copy()
                                            display_df.columns = [
                                                display_columns[col]
                                                for col in valid_columns
                                            ]

                                            #change ATAC P-value to ATAC -log10 P-value
                                            display_df["ATAC -log10 P-value"] = -np.log10(
                                                display_df["ATAC P-value"]+1e-300
                                            )
                                            display_df["RNA -log10 P-value"] = -np.log10(
                                                display_df["RNA P-value"]+1e-300
                                            )
                                            #round to 2
                                            display_df["ATAC -log10 P-value"] = display_df[
                                                "ATAC -log10 P-value"
                                            ].round(2)
                                            display_df["RNA -log10 P-value"] = display_df[
                                                "RNA -log10 P-value"
                                            ].round(2)
                                            
                                            #remove
                                            display_df.drop(columns=["ATAC P-value", "RNA P-value"], inplace=True)

                                            # Display the DataFrame
                                            st.dataframe(
                                                display_df.sort_values(
                                                    "Evidence Type", ascending=False
                                                ),
                                                hide_index=True,
                                            )

                                            # Add download button for TF info
                                            st.download_button(
                                                label="Download TF Information",
                                                data=display_df.to_csv(index=False),
                                                file_name=f"tf_info_{selected_analysis}.csv",
                                                mime="text/csv",
                                                key="download_tf_info",
                                            )

                                        except Exception as e:
                                            st.error(
                                                f"Error processing heatmap data: {str(e)}"
                                            )
                                            if st.checkbox(
                                                "Show detailed error information"
                                            ):
                                                st.exception(e)
                                                traceback.print_exc() 
                                            # Capture the full traceback
                                            tb = traceback.format_exc()

                                            # Display it in a collapsible section
                                            with st.expander("Show full error traceback"):
                                                st.code(tb, language='python')

                            except Exception as e:
                                st.error(f"Error loading heatmap data: {str(e)}")
                                st.error(
                                    "Please check that all required data files are available."
                                )
                                if st.checkbox("Show detailed error information"):
                                    st.exception(e)
                                    traceback.print_exc() 
                                    # Capture the full traceback
                                    tb = traceback.format_exc()

                                    # Display it in a collapsible section
                                    with st.expander("Show full error traceback"):
                                        st.code(tb, language='python')
        
        with celltyping_tab:
            col1, col2 = st.columns([5, 1])
            with col1:
                st.header("Cell Type Annotation")
                st.markdown("Annotate cell types and detect doublets in your own single-cell datasets using our bespoke pituitary-specific models.")
            with col2:
                selected_version = st.selectbox(
                    "Version",
                    options=AVAILABLE_VERSIONS,
                    key="version_select_tab8",
                    label_visibility="collapsed",
                )


            click = tab_start_button(
                "cell_type_analysis",
                "begin_cell_type_analysis")

            if click or (st.session_state["current_analysis_tab"] == "cell_type_analysis"):
                gc.collect()
                create_cell_type_annotation_ui()


        with datasets_tab:
            
            with st.container():
                (
                    sc_rna_tab,
                    sc_atac_tab

                ) = st.tabs(
                    [
                        "RNA datasets",
                        "ATAC datasets"
                    ]
                )

                # Expression Distribution tab content
                with sc_rna_tab:

                    st.markdown(
                        "Click the button below to visualise individual datasets. This will load the necessary data."
                    )

                    click = tab_start_button(
                            "sc_analysis",
                            "begin_sc_analysis")

                    if click or (st.session_state["current_analysis_tab"] == "sc_analysis"):
                        gc.collect()
                        col1, col2 = st.columns([5, 1])
                        with col1:
                            st.header("Individual Datasets")
                            st.markdown("Visualize gene expression patterns and cell type distributions in published single-cell RNA-seq datasets from the mouse pituitary.")
                        with col2:
                            selected_version = st.selectbox(
                                "Version",
                                options=AVAILABLE_VERSIONS,
                                key="version_select_datasets_rna",
                                label_visibility="collapsed",
                            )

                        available_datasets = list_available_datasets(
                            BASE_PATH,
                            os.path.join(BASE_PATH, "sc_data", "datasets"),
                            selected_version,
                        )

                        default_dataset = (
                            "Ruf-Zamojski et al. (2021) - FrozPit-MM2 - SRX8489835"
                        )
                        sorted_dataset_names = sorted(available_datasets.keys())

                        if default_dataset in available_datasets:
                            default_index = sorted_dataset_names.index(default_dataset)
                        else:
                            default_index = 0

                        selected_display_name = st.selectbox(
                            "Select a dataset",
                            options=sorted_dataset_names,
                            index=default_index,
                            key="dataset_select_datasets_rna",
                            width=500
                        )

                        if selected_display_name:
                            # Get the SRA_ID from the display name
                            selected_dataset = available_datasets[selected_display_name]

                            #if selected dataset is embryonic: SRX22219345, SRX22219346 or SRX26708357
                            if selected_dataset in ["SRX22219345", "SRX22219346", "SRX26708357"]:
                                st.warning("Embryonic datasets may have different cell type annotations and characteristics compared to postnatal datasets. Please interpret the results with caution. We recommend downloading the data from epitome and annotating on your own.")

                            # Load dataset using just the SRA_ID
                            with st.spinner("Loading dataset..."):
                                adata = load_cached_single_cell_dataset(
                                    selected_dataset, selected_version,rna_atac="rna"
                                )
                                available_genes = adata.var_names.tolist()

                            if adata is not None:
                                # Display dataset info
                                dataset_info = get_dataset_info(adata)
                                st.write("Dataset Information")

                                st.metric("Total Cells", dataset_info["Total Cells"])
                                st.metric("Total Genes", dataset_info["Total Genes"])
                                create_cell_type_stats_display(
                                    version=selected_version,
                                    # make it selected samples if empty then use  all samples
                                    sra_ids=[selected_dataset.split(" ")[0]],
                                    display_title="Cell Counts in Current Selection",
                                    column_count=6,
                                    size="small",
                                )

                                # Gene selection and plot options
                                col1, col2, col3, col4 = st.columns(4)

                                with col1:
                                    selected_gene = create_gene_selector(
                                        gene_list=available_genes,
                                        key_suffix="gene_select_datasets1")
                                    
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
                                            "YlOrRd",
                                        ],
                                        key="color_map_select_datasets2",
                                        width=250
                                    )
                                with col3:
                                    sort_order = st.checkbox("Sort plotted cells by expression", value=False, key="sort1")
                                
                                with col4:
                                    download_as = st.selectbox(
                                        "Download as:",
                                        options=["png", "jpeg", "svg"],
                                        index=0,
                                        key="download_as_sc",
                                        width=250
                                    )

                                try:
                                    

                                    # Create plots
                                    gene_fig, cell_type_fig, config = plot_sc_dataset(
                                        adata, selected_gene, sort_order, color_map,
                                        download_as=download_as
                                    )

                                    add_activity(value = [selected_dataset, selected_gene],
                                                analysis="Individual Dataset RNA",
                                                user=st.session_state.session_id,
                                                time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                                    

                                    # Display plots side by side
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.plotly_chart(gene_fig, use_container_width=True, config=config)
                                    with col2:
                                        st.plotly_chart(cell_type_fig, use_container_width=True, config=config)
                                    gc.collect()
                                    # Add explanation in a container
                                    with st.container():
                                        st.markdown(
                                            """
                                            These UMAP plots show the single-cell data structure for individual datasets.
                                            
                                            Left plot:
                                                    
                                            **X-axis**: Arbitrary UMAP dimension 1
                                                    
                                            **Y-axis**: Arbitrary UMAP dimension 2
                                                    
                                            Color represents expression level of selected gene
                                            
                                            
                                            Right plot:
                                                    
                                            **X-axis**: Arbitrary UMAP dimension 1
                                                    
                                            **Y-axis**: Arbitrary UMAP dimension 2
                                                    
                                            Color represents cell type annotations
                                            
                                            Note: UMAP dimensions are arbitrary and do not have specific units. These plots are 
                                            used here due to community demand, we recommend statistical interpretations over simple visualizations.
                                            For concerns on the use of UMAPs, read: doi.org/10.1371/journal.pcbi.1011288

                                            Gene expression values are log1p(counts_per_10k) transformed.
                                            Cell types were assigned automatically, as described in our manuscript.
                                                    
                                            Features:
                                            - Interactive visualization of cellular heterogeneity
                                            - Gene expression patterns across cell populations
                                            - Cell type distribution in low-dimensional space
                                            - Optional sorting by expression intensity
                                        """
                                        )

                                    # Add QC PDF display section
                                    st.markdown("---")
                                    st.subheader("Quality Control Report")

                                    # Construct path to QC PDF
                                    qc_pdf_path = f"{BASE_PATH}/sc_data/qc/{selected_version}/summary_pdfs/summary_{selected_dataset}.pdf"

                                    if os.path.exists(qc_pdf_path):
                                        try:
                                            # Read and display PDF
                                            with open(qc_pdf_path, "rb") as f:
                                                pdf_bytes = f.read()

                                            # Add download button
                                            st.download_button(
                                                label="Download QC Report",
                                                data=pdf_bytes,
                                                file_name=f"qc_report_{selected_dataset}.pdf",
                                                mime="application/pdf",
                                                key="download_button_qc",
                                            )

                                        except Exception as e:
                                            st.error(f"Error displaying QC report: {str(e)}")
                                            traceback.print_exc() 
                                            # Capture the full traceback
                                            tb = traceback.format_exc()

                                            # Display it in a collapsible section
                                            with st.expander("Show full error traceback"):
                                                st.code(tb, language='python')
                                    else:
                                        st.warning("No QC report available for this dataset")

                                except Exception as e:
                                    st.error(f"Error creating plots: {str(e)}")
                                    traceback.print_exc() 
                                    # Capture the full traceback
                                    tb = traceback.format_exc()

                                    # Display it in a collapsible section
                                    with st.expander("Show full error traceback"):
                                        st.code(tb, language='python')

                with sc_atac_tab:

                    st.markdown(
                        "Click the button below to visualise individual datasets. This will load the necessary data."
                    )

                    click = tab_start_button(
                            "sc_atac_analysis",
                            "begin_sc_atac_analysis")

                    if click or (st.session_state["current_analysis_tab"] == "sc_atac_analysis"):
                        gc.collect()
                        col1, col2 = st.columns([5, 1])
                        with col1:
                            st.header("Individual Datasets")
                            st.markdown("Visualize gene expression patterns and cell type distributions in published single-nucleus ATAC-seq datasets from the mouse pituitary.")
                        with col2:
                            selected_version = st.selectbox(
                                "Version",
                                options=AVAILABLE_VERSIONS,
                                key="version_select_datasets_atac",
                                label_visibility="collapsed",
                            )

                        available_datasets = list_available_datasets(
                            BASE_PATH,
                            os.path.join(BASE_PATH, "sc_atac_data", "datasets"),
                            selected_version,
                        )

                        
                        default_dataset = (
                            "Ruf-Zamojski et al. (2021) - Male_Pit_4 - GSM4594390"
                        )
                        sorted_dataset_names = sorted(available_datasets.keys())

                        if default_dataset in available_datasets:
                            default_index = sorted_dataset_names.index(default_dataset)
                        else:
                            default_index = 0

                        selected_display_name = st.selectbox(
                            "Select a dataset",
                            options=sorted_dataset_names,
                            index=default_index,
                            key="dataset_select_datasets_atac",
                            width=500
                        )

                        if selected_display_name:
                            # Get the SRA_ID from the display name
                            selected_dataset = available_datasets[selected_display_name]

                            # Load dataset using just the SRA_ID
                            with st.spinner("Loading dataset..."):
                                adata = load_cached_single_cell_dataset(
                                    selected_dataset, selected_version,rna_atac="atac"
                                )
                                adata.obs["new_cell_type"] = adata.obs["cell_type"].copy()
                                available_genes = adata.var_names.tolist()

                            if adata is not None:
                                # Display dataset info
                                dataset_info = get_dataset_info(adata)
                                st.write("Dataset Information")

                                st.metric("Total Cells", dataset_info["Total Cells"])
                                st.metric("Total Peaks", dataset_info["Total Genes"])
                                create_cell_type_stats_display(
                                    version=selected_version,
                                    # make it selected samples if empty then use  all samples
                                    sra_ids=[selected_dataset.split(" ")[0]],
                                    display_title="Cell Counts in Current Selection",
                                    column_count=6,
                                    size="small",
                                )

                                # Gene selection and plot options
                                col1, col2, col3 = st.columns([2, 1, 1])

                                with col1:
                                    
                                    selected_gene = create_gene_selector(
                                        gene_list=available_genes,
                                        key_suffix="gene_select_datasets2")
                                    

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
                                            "YlOrRd",
                                        ],
                                        key="color_map_select_datasets3",
                                        width=250
                                    )
                                with col3:
                                    sort_order = st.checkbox("Sort plotted cells by expression", value=False, key="sort2")

                                try:
                                    # Create plots
                                    gene_fig, cell_type_fig, config = plot_sc_dataset(
                                        adata, selected_gene, sort_order, color_map
                                    )
                                    
                                    add_activity(value = [selected_dataset, selected_gene],
                                        analysis="Individual Dataset ATAC",
                                        user=st.session_state.session_id,
                                        time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                                    

                                    # Display plots side by side
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.plotly_chart(gene_fig, use_container_width=True, config=config)
                                    with col2:
                                        st.plotly_chart(cell_type_fig, use_container_width=True, config=config)
                                    gc.collect()
                                    # Add explanation in a container
                                    with st.container():
                                        st.markdown(
                                            """
                                            These UMAP plots show the single-cell data structure for individual datasets.
                                            
                                            Left plot:
                                                    
                                            **X-axis**: Arbitrary UMAP dimension 1
                                                    
                                            **Y-axis**: Arbitrary UMAP dimension 2
                                                    
                                            Color represents expression level of selected gene
                                            
                                            
                                            Right plot:
                                                    
                                            **X-axis**: Arbitrary UMAP dimension 1
                                                    
                                            **Y-axis**: Arbitrary UMAP dimension 2
                                                    
                                            Color represents cell type annotations
                                            
                                            Note: UMAP dimensions are arbitrary and do not have specific units. These plots are 
                                            used here due to community demand, we recommend statistical interpretations over simple visualizations.
                                            For concerns on the use of UMAPs, read: doi.org/10.1371/journal.pcbi.1011288

                                            Gene expression values are log1p(counts_per_10k) transformed.
                                            Cell types were assigned automatically, as described in our manuscript.
                                                    
                                            Features:
                                            - Interactive visualization of cellular heterogeneity
                                            - Gene expression patterns across cell populations
                                            - Cell type distribution in low-dimensional space
                                            - Optional sorting by expression intensity
                                        """
                                        )

                                    # Add QC PDF display section
                                    st.markdown("---")
                                    st.subheader("Quality Control Report")

                                    # Construct path to QC PDF
                                    qc_pdf_path = f"{BASE_PATH}/sc_atac_data/qc/{selected_version}/{selected_dataset}.pdf"

                                    if os.path.exists(qc_pdf_path):
                                        try:
                                            # Read and display PDF
                                            with open(qc_pdf_path, "rb") as f:
                                                pdf_bytes = f.read()

                                            # Add download button
                                            st.download_button(
                                                label="Download QC Report",
                                                data=pdf_bytes,
                                                file_name=f"qc_report_{selected_dataset}.pdf",
                                                mime="application/pdf",
                                                key="download_button_qc",
                                            )

                                        except Exception as e:
                                            st.error(f"Error displaying QC report: {str(e)}")
                                            traceback.print_exc() 
                                            # Capture the full traceback
                                            tb = traceback.format_exc()

                                            # Display it in a collapsible section
                                            with st.expander("Show full error traceback"):
                                                st.code(tb, language='python')
                                    else:
                                        st.warning("No QC report available for this dataset")

                                except Exception as e:
                                    st.error(f"Error creating plots: {str(e)}")
                                    traceback.print_exc() 
                                    # Capture the full traceback
                                    tb = traceback.format_exc()

                                    # Display it in a collapsible section
                                    with st.expander("Show full error traceback"):
                                        st.code(tb, language='python')

        with downloads_tab:
            
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.header("Downloads")
                    st.markdown("Download processed single-cell RNA-seq and ATAC-seq datasets, along with intermediate analysis files. Each dataset is provided in h5ad format, compatible with Python (Scanpy) and R (Seurat).")
                with col2:
                    selected_version = st.selectbox(
                        "Version",
                        options=AVAILABLE_VERSIONS,
                        key="version_select_downloads",
                        label_visibility="collapsed",
                    )

                # Create tabs for different download sections
                h5ad_tab_rna,h5ad_tab_atac, bulk_tab, usage_tab = st.tabs(
                    [
                        "Dataset Files (h5ad) - RNA",
                        "Dataset Files (h5ad) - ATAC",
                        "Analysis Data Files",
                        "Single-Cell Object Usage Guide",
                    ]
                )

                with h5ad_tab_rna:
                    create_downloads_ui_with_metadata_rna(BASE_PATH, version=selected_version)

                with h5ad_tab_atac:
                    create_downloads_ui_with_metadata_atac(BASE_PATH, version=selected_version)

                with bulk_tab:
                    create_bulk_data_downloads_ui(BASE_PATH, version=selected_version)

                with usage_tab:
                    # Keep your existing usage guide here
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
                    The most efficient way to work with h5ad files in R is using `reticulate` with `Seurat`:

                    ```r
                    # Install required packages if needed
                    install.packages("reticulate")
                    install.packages("Seurat")

                    # Load libraries
                    library(reticulate)
                    library(Seurat)

                    # Import Python modules through reticulate
                    anndata <- import("anndata")
                    py_index <- import("pandas")$Index

                    # Read the h5ad file
                    adata <- anndata$read_h5ad("your_dataset.h5ad")

                    # Convert to Seurat object
                    # Note: Seurat expects count matrix in column-major order (genes x cells)
                    counts <- t(adata$X)  # Transpose to get genes x cells
                    meta_data <- as.data.frame(adata$obs)  # Convert cell metadata to data frame
                    gene_names <- py_to_r(adata$var_names$to_list())  # Get gene names

                    # Create Seurat object
                    seurat_object <- CreateSeuratObject(
                        counts = counts,
                        meta.data = meta_data
                    )

                    # Set gene names
                    rownames(seurat_object) <- gene_names

                    # Optional: Transfer cell type annotations
                    seurat_object$cell_type <- seurat_object$new_cell_type

                    # Normalize data (if needed)
                    seurat_object <- NormalizeData(
                        seurat_object,
                        normalization.method = "LogNormalize",
                        scale.factor = 10000
                    )

                    # Basic operations
                    dim(seurat_object)  # View dimensions
                    head(seurat_object@meta.data)  # View metadata
                    unique(seurat_object$Author)  # View unique authors
                    ```

                    Important notes:
                    - This approach uses Python's anndata through reticulate, avoiding any conversion steps
                    - The count matrix is automatically transposed to match Seurat's expected format
                    - All metadata from the h5ad file is preserved in the Seurat object
                    - You can directly work with the object using standard Seurat functions after conversion
                    """
                    )

                    st.markdown(
                        """
                    For additional operations, such as visualization and analysis:

                    ```r
                    # Dimensionality reduction visualization
                    DimPlot(seurat_object, group.by = "cell_type")

                    # Feature plots
                    FeaturePlot(seurat_object, features = c("Sox2", "Pomc"))

                    # Find markers
                    markers <- FindAllMarkers(seurat_object)
                    ```
                    """
                    )

                    st.markdown("### File Structure")
                    st.markdown(
                        """
                    Each h5ad file in the epitome contains:
                    - Raw counts matrix (in adata.raw.X)
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
                    2. Visit the [Scanpy documentation](https://scanpy.readthedocs.io/) or [Seurat documentation](https://satijalab.org/seurat/)
                    3. Contact us (see Contact tab)
                    """
                    )

        with curation_tab:
            col1, col2 = st.columns([5, 1])
            with col1:
                st.header("Data Curation Information")
                st.markdown("Browse detailed metadata for all samples included in the epitome.")
            with col2:
                selected_version = st.selectbox(
                    "Version",
                    options=AVAILABLE_VERSIONS,
                    key="version_select_tab9",
                    label_visibility="collapsed",
                )

            try:
                # Load curation data
                curation_data = load_cached_curation_data(version=selected_version)
                # do not expose the following columns passed_qc, species, pseudoaligned, filtering_junk, median_cellassign_prob, passed_qc_tcc
                curation_data = curation_data.drop(
                    columns=[
                        "passed_qc",
                        "species",
                        "pseudoaligned",
                        "filtering_junk",
                        "median_cellassign_prob",
                        "passed_qc_tcc",
                    ]
                )
                
                filtered_data = display_curation_table(
                    curation_data, key_prefix="curation"
                )
            except FileNotFoundError:
                st.error("Curation data file not found. Please check the file path.")
            except Exception as e:
                st.error(f"Error loading curation data: {str(e)}")
                traceback.print_exc() 
                # Capture the full traceback
                tb = traceback.format_exc()

                # Display it in a collapsible section
                with st.expander("Show full error traceback"):
                    st.code(tb, language='python')

        with release_tab:
            st.header("Release Notes")
            st.markdown("Details of features and datasets included in each version of the epitome.")
            st.info(
            "v_0.01: First release of the epitome, including all mouse pituitary datasets published before June, 2025.\n\n"
            "Transcriptome analysis:\n"
            "- Expression Box Plots and UMAPs: Visualize gene expression across cell types with filtering options\n"
            "- Age Correlation: Analyze expression-age relationships with statistical metrics\n"
            "- Isoforms: Explore transcript-level expression with ensembl annotations\n"
            "- Dot Plots: Compare expression patterns showing magnitude and prevalence\n"
            "- Cell Type Distribution: Examine proportions with sex and age grouping\n"
            "- Gene-Gene Relationships: Analyze correlations with cell type specificity\n"
            "- Ligand-Receptor Interactions: Identify communication pathways\n"
            "- Sexually Dimorphic Genes and Marker Browser: Access comprehensive tables\n\n"
            "Chromatin analysis:\n"
            "- Accessibility Distribution: Visualize chromatin accessibility patterns\n"
            "- Interactive Genome Browser: View genomic regions with gene annotations and motifs\n"
            "- Motif Enrichment (ChromVAR): Analyze TF binding with enrichment scores\n"
            "- Cell Type Distribution: View ATAC-seq based population proportions\n\n"
            "Multimodal analysis:\n"
            "- TF Heatmaps: Visualize TF co-binding with RNA and ATAC evidence\n"
            "- Lineage-specific Factors: Explore TFs driving cell fate decisions\n\n"
            "Individual datasets:\n"
            "- Interactive RNA and ATAC UMAPs: Explore single datasets with QC reports\n\n"
            "Automated cell type and doublet annotation:\n"
            "- Access to our Cell Type Model in the browser\n"
            "- Access to our Doublet Model in the browser\n"
            "- Downloadable results and visualisation of user uploaded data, without any coding\n\n"
            "Data access:\n"
            "- Downloadable Cell Type Model: Use our model in your own analysis\n"
            "- Downloadable H5AD Files: Access pre-processed data with metadata\n"
            "- Analysis Data Files: Download matrices and processed data\n"
            "- Comprehensive Curation: Browse detailed metadata\n"
            "- Usage Guides: Instructions for Python (Scanpy) and R (Seurat)\n"
            "\nFor more information, see Methods in our pre-print on bioRxiv (placeholder)."
            "\nThe codebase for this release is found on [GitHub](https://github.com/Andoniadou-Lab/epitome)"
        )
            
        with citation_tab:
            st.header("How to Cite")
            st.markdown("Guide on citing the epitome and original datasets.")

            

            st.subheader("Cite Us")

            st.markdown("##### Citing the Consensus Pituitary Atlas")
            st.markdown(
                """
                When referring to results or methods from the atlas, please cite our preprint:
                
                [Preprint citation placeholder]
            """
            )

            st.markdown("##### Citing the Epitome")
            st.markdown(
                f"""
                When using the website to access data, generate hypotheses, or create figures, please cite:

                {epitome_citation}
            """
            )

            st.markdown("---")

            st.markdown("##### Examples")
            st.markdown(
                f"""
                Scenario 1: You have used a result from our Consensus Pituitary Atlas publication, but not the epitome.
                
                "Gal is more abundant in female mouse pituitaries compared to male ones [1]."

                1. [Preprint citation placeholder]

                Scenario 2: You are retrieving a uniformly pre-processed dataset.

                "To evaluate whether our gene of interest, Bean1, is affected by Prop1, we retrieved a Prop1 knockout dataset [1] from the electronic pituitary omics platform [2].
                
                1. Cite source paper of dataset 
                2. {epitome_citation}

                Scenario 3: You have used the Epitome to access the atlas, and then created a figure.

                "Using the electronic pituitary omics platform [1] which collates all existing single-cell transcriptomic data on the pituitary [2], we found that our gene of interest, Bean1, is mostly present in gonadotrophs ."

                1. {epitome_citation}
                2. [Preprint citation placeholder]

                
            """
            )
            st.markdown("---")

            st.subheader("Cite Others")
            st.markdown(
                """
                Please also cite the relevant original publications, when you use a single or few datasets:
                
                1.  Ruf-Zamojski F, Zhang Z, Zamojski M, Smith GR, Mendelev N, Liu H, et al. Single nucleus multi-omics regulatory landscape of the murine pituitary. Nat Commun. 2021 May 11;12(1):2677.
                2.  Cheung LYM, George AS, McGee SR, Daly AZ, Brinkmeier ML, Ellsworth BS, et al. Single-Cell RNA Sequencing Reveals Novel Markers of Male Pituitary Stem Cells and Hormone-Producing Cell Types. Endocrinology. 2018 Dec 1;159(12):3910–24.
                3.	Mayran A, Sochodolsky K, Khetchoumian K, Harris J, Gauthier Y, Bemmo A, et al. Pioneer and nonpioneer factor cooperation drives lineage specific chromatin opening. Nat Commun. 2019 Aug 23;10(1):3807.
                4.	Chen Q, Leshkowitz D, Blechman J, Levkowitz G. Single-Cell Molecular and Cellular Architecture of the Mouse Neurohypophysis. eNeuro. 2020;7(1):ENEURO.0345-19.2019.
                5.	Ho Y, Hu P, Peel MT, Chen S, Camara PG, Epstein DJ, et al. Single-cell transcriptomic analysis of adult mouse pituitary reveals sexual dimorphism and physiologic demand-induced cellular plasticity. Protein Cell. 2020 Aug;11(8):565–83.
                6.	Lopez JP, Brivio E, Santambrogio A, De Donno C, Kos A, Peters M, et al. Single-cell molecular profiling of all three components of the HPA axis reveals adrenal ABCB1 as a regulator of stress adaptation. Sci Adv. 2021 Jan;7(5):eabe4497.
                7.	Ruggiero-Ruff RE, Le BH, Villa PA, Lainez NM, Athul SW, Das P, et al. Single-Cell Transcriptomics Identifies Pituitary Gland Changes in Diet-Induced Obesity in Male Mice. Endocrinology. 2024 Mar 1;165(3):bqad196.
                8.	Vennekens A, Laporte E, Hermans F, Cox B, Modave E, Janiszewski A, et al. Interleukin-6 is an activator of pituitary stem cells upon local damage, a competence quenched in the aging gland. Proc Natl Acad Sci U S A. 2021 Jun 22;118(25):e2100052118.
                9.	Laporte E, Hermans F, De Vriendt S, Vennekens A, Lambrechts D, Nys C, et al. Decoding the activated stem cell phenotype of the neonatally maturing pituitary. eLife. 2022 Jun 14;11:e75742.
                10.	Li Y, Wang J, Wang R, Chang Y, Wang X. Gut bacteria induce IgA expression in pituitary hormone-secreting cells during aging. iScience. 2023 Oct 20;26(10):107747.
                11.	Miles TK, Odle AK, Byrum SD, Lagasse A, Haney A, Ortega VG, et al. Anterior Pituitary Transcriptomics Following a High-Fat Diet: Impact of Oxidative Stress on Cell Metabolism. Endocrinology. 2023 Dec 23;165(2):bqad191.
                12.	Bohaczuk SC, Thackray VG, Shen J, Skowronska-Krawczyk D, Mellon PL. FSHB  Transcription is Regulated by a Novel 5′ Distal Enhancer With a Fertility-Associated Single Nucleotide Polymorphism. Endocrinology. 2021 Jan 1;162(1):bqaa181.
                13.	Schang G, Ongaro L, Brûlé E, Zhou X, Wang Y, Boehm U, et al. Transcription factor GATA2 may potentiate follicle-stimulating hormone production in mice via induction of the BMP antagonist gremlin in gonadotrope cells. J Biol Chem. 2022 Jul 1;298(7):102072.
                14.	Lin YF, Schang G, Buddle ERS, Schultz H, Willis TL, Ruf-Zamojski F, et al. Steroidogenic Factor 1 Regulates Transcription of the Inhibin B Coreceptor in Pituitary Gonadotrope Cells. Endocrinology. 2022 Aug 12;163(11):bqac131.
                15.	Rizzoti K, Chakravarty P, Sheridan D, Lovell-Badge R. SOX9-positive pituitary stem cells differ according to their position in the gland and maintenance of their progeny depends on context. Sci Adv. 9(40):eadf6911.
                16.	Cheung LYM, Menage L, Rizzoti K, Hamilton G, Dumontet T, Basham K, et al. Novel Candidate Regulators and Developmental Trajectory of Pituitary Thyrotropes. Endocrinology. 2023 May 15;164(6):bqad076.
                17.	Allensworth-James M, Banik J, Odle A, Hardy L, Lagasse A, Moreira ARS, et al. Control of the Anterior Pituitary Cell Lineage Regulator POU1F1 by the Stem Cell Determinant Musashi. Endocrinology. 2021 Mar 1;162(3):bqaa245.
                18.	Moncho-Amor V, Chakravarty P, Galichet C, Matheu A, Lovell-Badge R, Rizzoti K. SOX2 is required independently in both stem and differentiated cells for pituitary tumorigenesis in p27-null mice. Proc Natl Acad Sci U S A. 2021 Feb 16;118(7):e2017115118.
                19.	Bastedo WE, Scott RW, Arostegui M, Underhill TM. Single-cell analysis of mesenchymal cells in permeable neural vasculature reveals novel diverse subpopulations of fibroblasts. Fluids Barriers CNS. 2024 Apr 5;21(1):31.
                20.	Cheung LYM, Camper SA. PROP1-Dependent Retinoic Acid Signaling Regulates Developmental Pituitary Morphogenesis and Hormone Expression. Endocrinology. 2020 Jan 8;161(2):bqaa002.
                21.	Zhang Z, Ruf-Zamojski F, Zamojski M, Bernard DJ, Chen X, Troyanskaya OG, et al. Peak-agnostic high-resolution cis-regulatory circuitry mapping using single cell multiome data. Nucleic Acids Res. 2024 Jan 25;52(2):572–82.
                22.	Masser BE, Brinkmeier ML, Lin Y, Liu Q, Miyazaki A, Nayeem J, et al. Gene Misexpression in a Smoc2+ve/Sox2-Low Population in Juvenile Prop1-Mutant Pituitary Gland. J Endocr Soc. 2024 Oct 1;8(10):bvae146.
                23.	Sheridan D, Chakravarty P, Golan G, Shiakola Y, Galichet C, Mollard P, et al. Gonadotrophs have a dual origin, with most derived from pituitary stem cells during minipuberty [Internet]. bioRxiv; 2024 [cited 2024 Sep 13]. p. 2024.09.09.610834. Available from: https://www.biorxiv.org/content/10.1101/2024.09.09.610834v2
                24.	Qian Q, Li M, Zhang Z, Davis SW, Rahmouni K, Norris AW, et al. Obesity disrupts the pituitary-hepatic UPR communication leading to NAFLD progression. Cell Metab. 2024 Jul 2;36(7):1550-1565.e9.
                25.	Martinez-Mayer J, Brinkmeier ML, O’Connell SP, Ukagwu A, Marti MA, Miras M, et al. Knockout mice with pituitary malformations help identify human cases of hypopituitarism. Genome Med. 2024 May 31;16:75.
                26.	Kim Y bin, Lee S, Kim NY, Lee EJ, Oh JH, Oh CM, et al. 12404 Aging-Associated Decline in the Pituitary Gland Using Single-Cell Transcriptomes. J Endocr Soc. 2024 Oct 1;8(Supplement_1):bvae163.1127.
                27.	Khetchoumian K, Sochodolsky K, Lafont C, Gouhier A, Bemmo A, Kherdjemil Y, et al. Paracrine FGF1 signaling directs pituitary architecture and size. Proc Natl Acad Sci. 2024 Oct;121(40):e2410269121.
                28.	Wang Y, Thistlethwaite W, Tadych A, Ruf-Zamojski F, Bernard DJ, Cappuccio A, et al. Automated single-cell omics end-to-end framework with data-driven batch inference. Cell Syst. 2024 Oct 16;15(10):982-990.e5.
                29.	Huang Y, Wang Q, Zhou W, Jiang Y, He K, Huang W, et al. Prenatal p25-activated Cdk5 induces pituitary tumorigenesis through MCM2 phosphorylation-mediated cell proliferation. Neoplasia N Y N. 2024 Oct 3;57:101054.
                30.	Vriendt SD, Laporte E, Abaylı B, Hoekx J, Hermans F, Lambrechts D, et al. Single-cell transcriptome atlas of male mouse pituitary across postnatal life highlighting its stem cell landscape. iScience [Internet]. 2025 Feb 21 [cited 2025 Mar 13];28(2). Available from: https://www.cell.com/iscience/abstract/S2589-0042(24)02935-3
                31.	Brinkmeier ML, Cheung LYM, O’Connell SP, Gutierrez DK, Rhoads EC, Camper SA, et al. Nucleoredoxin regulates WNT signaling during pituitary stem cell differentiation. Hum Mol Genet. 2025 Mar 5;ddaf032.
                32.	Ongaro L, Zhou X, Wang Y, Schultz H, Zhou Z, Buddle ERS, et al. Muscle-derived myostatin is a major endocrine driver of follicle-stimulating hormone synthesis. Science. 2025 Jan 17;387(6731):329–36.
                33.	Miles TK, Odle AK, Byrum SD, Lagasse AN, Haney AC, Ortega VG, et al. Ablation of Leptin Receptor Signaling Alters Somatotrope Transcriptome Maturation in Female Mice. Endocrinology. 2025 Feb 18;bqaf036.
                34. Rebboah E, Weber R, Abdollahzadeh E, Swarna N, Sullivan DK, Trout D, et al. Systematic cell-type resolved transcriptomes of 8 tissues in 8 lab and wild-derived mouse strains captures global and local expression variation. bioRxiv. 2025 Jan 1;2025.04.21.649844.            
                        """
            )

        with contact_tab:
            st.header("Contact Us")
            st.markdown("Get in touch for data submission, collaboration, corrections, or career opportunities.")
            st.subheader("Submit Your Data")
            st.markdown(
                """
                We welcome submissions of new mouse pituitary datasets. To submit your data:
                1. Ensure your raw data is deposited in a public repository (SRA, ENA, GEO, ArrayExpress, etc.)
                2. Fill out our [data submission form](https://forms.office.com/Pages/ResponsePage.aspx?id=FM9wg_MWFky4PHJAcWVDVtCPt0Xedb9ClGRxkEBa4fZUM1o5T01KTkVLQUFKWkFNTU5FVkRBRVoxVy4u&embed=true)
                3. Email us at **epitome at kcl dot ac dot uk** with:
                - Publication details
                - Repository accession numbers
                - Any additional metadata (Genotype, Sex, Age, etc. - see existing curation)
            """
            )

            st.subheader("Are you sitting on unpublished data?")
            st.markdown(
                """
                If you have data that did not make it into a paper, but you would like to share it with the community (and assign a DOI and get cited!), we can help.
                Reach out to us at 
                **epitome at kcl dot ac dot uk**
                with a brief description of your data and we can help you get it into the atlas.
                """
            )

            st.subheader("Reach Out for Collaboration")
            st.markdown(
                """
                We're interested in collaborating on:
                - Including our atlas analysis results in your publication
                - Combining our data with yours to increase statistical power
                - Developing new methods that work on an atlas-scale
                - Adding new modalities (methylation, proteomics, spatial data etc.)
                Contact us at **epitome at kcl dot ac dot uk** with a brief proposal.
            """
            )

            st.subheader("Work with Us")
            st.markdown(
                """
                There is plenty of more work to be done on the systematic analysis of pituitary gland datasets.
                If you are interested in joining our team as a student, please reach out to
                **cynthia dot andoniadou at kcl dot ac dot uk**
                with your CV and a brief statement of interest.
            """
            )

            st.subheader("Submit a Correction")
            st.markdown(
                """
                We are humans too. Did we get something wrong? Did we miss your dataset?
                Please let us know by supplying the relevant information through [this form]("https://forms.office.com/Pages/ResponsePage.aspx?id=FM9wg_MWFky4PHJAcWVDVtCPt0Xedb9ClGRxkEBa4fZUNjlDOURVSTRYMUxHSkpIUDE5OVNUTk1SVS4u&embed=true") or email.
                
                - Data corrections
                - Metadata updates
                - Website functionality issues
                
                Email us at: **epitome at kcl dot ac dot uk** with detailed information about the correction needed.
            """
            )

        st.markdown("---")

        # Footer
        st.markdown(
            "The <i>e<span style='color:#0000ff;'>pit</span>ome</i> is maintained by the <strong>Andoniadou Lab</strong> at <strong>King's College London</strong>. <a href='https://bsky.app/profile/pituitarylab.bsky.social'>Bluesky</a>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "Lead curator: Bence Kövér [Bluesky](https://bsky.app/profile/bencekover.bsky.social) (Email: epitome at kcl dot ac dot uk)"
        )
        st.markdown("[GitHub repository](https://github.com/Andoniadou-Lab/epitome)")
        st.markdown("[Preprint placeholder]")
        st.markdown(f"{epitome_citation}")
        st.image(logo, width=50)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check your data paths and file formats.")
        traceback.print_exc() 
        # Capture the full traceback
        tb = traceback.format_exc()

        # Display it in a collapsible section
        with st.expander("Show full error traceback"):
            st.code(tb, language='python')


if __name__ == "__main__":
    main()