import os
import traceback
from datetime import datetime

import streamlit as st

from config import Config
from modules.analytics import add_activity
from modules.cached_loaders import AVAILABLE_VERSIONS, load_cached_single_cell_dataset
from modules.individual_sc import get_dataset_info, list_available_datasets, plot_sc_dataset
from modules.ui.plot_settings import download_format_select, plot_settings_panel
from modules.ui.plot_summary import plot_summary_caption
from modules.utils import create_cell_type_stats_display, create_gene_selector

BASE_PATH = Config.BASE_PATH

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

        with plot_settings_panel("Plot settings"):
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
                )
            with col3:
                sort_order = st.checkbox("Sort plotted cells by expression", value=False, key="sort1")

            with col4:
                download_as = download_format_select(
                    "download_as_sc", formats=("png", "jpeg", "svg")
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
            plot_summary_caption(
                f"{dataset_info['Total Cells']:,} cells",
                f"{len(dataset_info['Cell Types'])} cell types",
                selected_dataset.split(" ")[0],
                f"gene: {selected_gene}",
            )
            #gc.collect()
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
                        on_click="ignore",
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
