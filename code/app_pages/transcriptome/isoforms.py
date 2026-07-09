import os
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
    to_array,
)

BASE_PATH = Config.BASE_PATH

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

with plot_settings_panel("Plot settings"):
    (
        filter_type,
        selected_samples,
        selected_authors,
        age_range,
        only_normal,
        modality,
    ) = create_filter_ui(
        filtered_curation, key_suffix="isoform"
    )

    gene_list = sorted(isoform_features["gene_name"].unique())
    all_cell_types = sorted(
        isoform_samples["cell_type"].unique()
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
            [ct.split("_")[0] for ct in all_cell_types],
            default=[ct.split("_")[0] for ct in all_cell_types][0],
            key="cell_type_select",
        )

    col1, col2 = st.columns(2)
    with col1:
        selected_gene = create_gene_selector(
            gene_list=gene_list, key_suffix="gene_select_tab3"
        )
    with col2:
        download_as = download_format_select(
            "download_as_isoform", formats=("png", "jpeg", "svg")
        )

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

elif filter_type == "Modality" and modality:
    valid_sra_ids &= set(
        filtered_curation[
            filtered_curation["Modality"].isin(modality)
        ]["SRA_ID"].unique()
    )

if only_normal:
    valid_sra_ids &= set(
        filtered_curation[filtered_curation["Normal"] == 1][
            "SRA_ID"
        ].unique()
    )

sample_mask = isoform_samples["SRA_ID"].isin(valid_sra_ids)

if not isinstance(sample_mask, np.ndarray):
    sample_mask = (
        sample_mask.to_numpy()
        if hasattr(sample_mask, "to_numpy")
        else np.array(sample_mask)
    )

if sample_mask.dtype != bool:
    sample_mask = sample_mask.astype(bool)

filtered_matrix = isoform_matrix[:, sample_mask]
filtered_samples = isoform_samples[sample_mask].copy()

if filter_type != "No filter" or only_normal:
    st.info(
        f"Showing data from {len(filtered_samples)} pseudobulk samples"
    )

filtered_sra_ids = (
    filtered_samples["SRA_ID"].unique().tolist()
)

create_cell_type_stats_display(
    version=selected_version,
    sra_ids=filtered_sra_ids,
    display_title="Cell Counts in Current Selection",
    cell_types=(
        [ct.split(" ")[0] for ct in selected_cell_types]
        if cell_type_option == "Select Specific Cell Types"
        else None
    ),
    column_count=6,
    size="small",
    atac_rna="rna",
)

if selected_gene:
    add_activity(
        value=selected_gene,
        analysis="Isoform Plot",
        user=st.session_state.session_id,
        time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

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
        download_as=download_as,
    )

    if error_message:
        st.error(error_message)
    else:
        st.plotly_chart(
            fig, use_container_width=True, config=config
        )
        transcript_count = len(
            isoform_features[
                isoform_features["gene_name"] == selected_gene
            ]
        )
        n_cell_types = (
            len(selected_cell_types)
            if cell_type_option == "Select Specific Cell Types" and selected_cell_types
            else filtered_curation["new_cell_type"].nunique()
        )
        plot_summary_caption(
            selected_gene,
            f"{len(filtered_samples)} pseudobulk samples",
            f"{transcript_count} transcripts",
            f"{n_cell_types} cell types",
        )
        with st.container():
            st.markdown(
                """
                This plot displays transcript-level expression data for a selected gene across cell types.

                **X-axis**: Transcript IDs grouped by cell type
                **Y-axis**: Log10-transformed counts per million values of each transcript

                The box plot also enables:
                - Grouping by cell type and transcript ID
                - Hovering reveals information including sample metadata
            """
            )

    try:
        gene_transcripts = isoform_features[
            isoform_features["gene_name"] == selected_gene
        ]

        filtered_transcripts = []
        filtered_transcripts_path = f"{BASE_PATH}/data/isoforms/v_0.01/filtered_transcripts_list.csv"
        if os.path.exists(filtered_transcripts_path):
            try:
                filtered_df = pd.read_csv(
                    filtered_transcripts_path
                )
                filtered_transcripts = [
                    t.split(".")[0]
                    for t in filtered_df.iloc[:, 1]
                ]
            except Exception as e:
                st.warning(
                    f"Error loading filtered transcripts: {e}"
                )
                traceback.print_exc()
                tb = traceback.format_exc()
                with st.expander("Show full error traceback"):
                    st.code(tb, language="python")

        st.markdown("### Transcript Details")
        st.markdown(
            "The table below shows all transcripts for this gene with links to Ensembl."
        )

        table_md = [
            "| Transcript ID | Ensembl Link |",
            "|-------------|--------------|",
        ]

        has_filtered = False

        for _, transcript in gene_transcripts.iterrows():
            transcript_id = transcript["transcript_id"]
            base_transcript_id = (
                transcript_id.split(".")[0]
                if "." in transcript_id
                else transcript_id
            )
            is_filtered = (
                base_transcript_id in filtered_transcripts
            )
            if is_filtered:
                has_filtered = True

            ensembl_link = f"https://www.ensembl.org/Mus_musculus/Transcript/Summary?t={transcript_id}"

            star = "⭐ " if is_filtered else ""
            row = f"| {star}{transcript_id} | [View in Ensembl]({ensembl_link}) |"
            table_md.append(row)

        st.markdown("\n".join(table_md))

        if has_filtered:
            st.markdown(
                "⭐ Isoform with uniquely mapping reads in 90% of datasets - likely to be more reliably quantified"
            )

        st.info(
            "The isoform analysis detailed in this tab has not been included in any of our current manuscripts. We, however, thought it might be a valuable community resource for hypothesis generation. Neither the methods nor the results have been peer-reviewed, so treat this information tentatively."
        )
    except Exception as e:
        st.error(f"Error displaying transcript table: {e}")
        if st.checkbox("Show detailed error"):
            st.exception(e)
            traceback.print_exc()
            tb = traceback.format_exc()
            with st.expander("Show full error traceback"):
                st.code(tb, language="python")

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
            if cell_type != "Erythrocytes":
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
            on_click="ignore",
            label="Download Transcript Data",
            data=download_df.to_csv(index=False),
            file_name=f"{selected_gene}_transcript_data.csv",
            mime="text/csv",
            key="download_button_tab3",
            help="Download the current transcript expression dataset",
        )
