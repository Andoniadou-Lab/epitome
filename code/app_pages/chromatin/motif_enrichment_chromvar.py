import traceback
from datetime import datetime

import streamlit as st

from modules.analytics import add_activity
from modules.cached_loaders import (
    AVAILABLE_VERSIONS,
    load_cached_chromvar_data,
    load_cached_enrichment_data,
)
from modules.chromvar import create_chromvar_plot
from modules.display_tables import display_enrichment_table
from modules.ui.plot_settings import plot_settings_panel
from modules.ui.plot_summary import plot_summary_caption
from modules.utils import (
    create_cell_type_stats_display,
    create_filter_ui,
    create_gene_selector,
    filter_chromvar_data,
    to_array,
)

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

    try:
        with plot_settings_panel("Plot settings"):
            (
                filter_type,
                selected_samples,
                selected_authors,
                age_range,
                only_normal,
                modality
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
                selected_samples = filtered_meta["Name"].values.tolist()

            if filter_type == "Modality" and "modality" in locals():
                filtered_meta = filtered_meta[
                    filtered_meta["Modality"].isin(modality)
                ]
                selected_samples = filtered_meta["Name"].values.tolist()

            filtered_meta, filtered_matrix = filter_chromvar_data(
                meta_data=chromvar_meta,
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

            default_motif = (
                "MA0143.4_SOX2"
                if "MA0143.4_SOX2" in features
                else features[0]
            )
            col1, col2 = st.columns(2)
            with col1:
                selected_motif = st.selectbox(
                    f"Select Motif ({len(features)} motifs)",
                    features,
                    index=features.index(default_motif),
                    key="motif_select_tab6",
                )
                additional_group = st.selectbox(
                    "Additional Grouping Variable",
                    ["None", "Modality", "Comp_sex"],
                    key="additional_group_select_tab6",
                )
            with col2:
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
            plot_summary_caption(
                selected_motif,
                f"{filtered_meta.shape[0]} pseudobulk samples",
                f"{filtered_meta['cell_type'].nunique()} cell types",
            )
            #gc.collect()
            with st.container():
                st.markdown(
                    f"""
                    This plot shows the enrichment of transcription factor motifs across cell types.

                    **X-axis**: Cell types present in the selected samples
                    **Y-axis**: ChromVAR deviation score* (motif enrichment)

                    **Box:** centre line = **median**; top and bottom edges = **75th** and **25th** percentiles

                    **Whiskers:** extend to the **5th** and **95th** percentiles

                    **Points:** each dot is one sample; hover a dot for sample-level detail

                    The box plot also enables:
                    - Optional grouping by additional variables
                    - Optional connecting lines between samples from the same source

                    *Briefly these are Z-scores of the fragments falling into peaks with a given motif compared to a null distribution of random peaks. For more insight please consult the ChromVAR (https://doi.org/10.1038/nmeth.4401) paper.
                """
                )

            # Add download button
            motif_idx = features.index(selected_motif)
            enrichment_values = to_array(filtered_matrix[motif_idx, :])

            download_df = filtered_meta.copy()
            download_df["Enrichment"] = enrichment_values

            st.download_button(
                on_click="ignore",
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
