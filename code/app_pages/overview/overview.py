import os
import traceback

import pandas as pd
import streamlit as st

from config import Config
from modules.cached_loaders import AVAILABLE_VERSIONS, load_cached_curation_data
from modules.citations import print_citation
from modules.utils import create_cell_type_stats_display

BASE_PATH = Config.BASE_PATH

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
    st.subheader("Sex Differences")
    st.markdown(
        """
    **Key Finding:**
    - Significant sex differences observed in most pituitary cell types

    **Recommendations:**
    - Design experiments for single sex OR include sufficient samples to account for sex-specific effects
    - Pooling samples (to reduce costs) from different sexes can be easily demultiplexed using sex-biased genes
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
    f"For detailed methodology and complete findings, please refer to our publication in Cell Reports {print_citation}."
)
