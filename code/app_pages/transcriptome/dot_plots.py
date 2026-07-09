import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl
import streamlit as st

from config import Config
from modules.analytics import add_activity
from modules.cached_loaders import (
    AVAILABLE_VERSIONS,
    load_cached_curation_data,
    load_cached_data,
    load_cached_dotplot_data,
    load_cached_gene_curation,
    load_cached_isoform_data,
    load_cached_ligand_receptor_data,
    load_cached_marker_data,
    load_cached_proportion_data,
    load_cached_sex_dim_data,
)
from modules.age_correlation import create_age_correlation_plot
from modules.display_tables import (
    display_aging_genes_table,
    display_ligand_receptor_table,
    display_marker_table,
    display_sex_dimorphism_table,
)
from modules.dotplot import (
    create_dotplot,
    create_ligand_receptor_plot,
    filter_dotplot_data,
)
from modules.expression import create_expression_plot
from modules.gene_gene_corr import (
    create_gene_correlation_plot,
    get_available_genes,
    load_gene_data,
)
from modules.gene_umap_vis import create_gene_umap_plot
from modules.isoforms import create_isoform_plot, filter_isoform_data
from modules.proportion_plot import create_proportion_plot
from modules.ui.plot_settings import download_format_select, plot_settings_panel
from modules.ui.plot_summary import plot_summary_caption
from modules.utils import (
    create_cell_type_stats_display,
    create_filter_ui,
    create_gene_selector,
    filter_data,
    parse_row_info,
    to_array,
)

BASE_PATH = Config.BASE_PATH

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

    row_info = parse_row_info(rows1)
    valid_sra_ids = set(row_info["SRA_ID"].unique())
    filtered_curation = curation_data[
        curation_data["SRA_ID"].isin(valid_sra_ids)
    ].copy()

with plot_settings_panel("Plot settings"):
    (
        filter_type,
        selected_samples,
        selected_authors,
        age_range,
        only_normal,
        modality,
    ) = create_filter_ui(
        filtered_curation, key_suffix="dotplot"
    )

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

    col1, col2 = st.columns(2)
    with col1:
        chosen_color_scheme = st.selectbox(
            "Select Color Scheme",
            options=["Blue", "Red", "Viridis", "Cividis"],
            index=0,
            key="color_scheme_dotplot",
            width=250,
        )
    with col2:
        download_as = download_format_select(
            "download_as_dotplot", formats=("png", "jpeg", "svg")
        )

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

try:
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

        group_by_extras = st.multiselect(
            "Group by (in addition to Cell type)",
            options=["Comp_sex", "Modality"],
            default=[],
            key="group_by_extras_dotplot",
            help=(
                "Optionally stack metadata fields onto cell type to produce "
                "composite groups (e.g. Somatotrophs_Male_sc)."
            ),
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
        st.markdown("<br>", unsafe_allow_html=True)

        add_activity(
            value=selected_genes,
            analysis="Dot Plot",
            user=st.session_state.session_id,
            time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

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
            download_as=download_as,
            meta_data=filtered_curation,
            group_by_extras=group_by_extras,
        )
        #gc.collect()

        st.plotly_chart(
            fig, use_container_width=True, config=config
        )
        n_cell_types = (
            len(selected_cell_types)
            if selected_cell_types
            else len(all_cell_types)
        )
        plot_summary_caption(
            f"{len(filtered_rows1)} pseudobulk samples",
            f"{len(selected_genes)} genes",
            f"{n_cell_types} cell types",
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
                on_click="ignore",
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
