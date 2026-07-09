import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from modules.analytics import add_activity
from modules.cached_loaders import (
    AVAILABLE_VERSIONS,
    load_cached_enrichment_data,
    load_cached_heatmap_data,
    load_cached_motif_data,
)
from modules.data_loader import load_motif_genes
from modules.display_tables import display_enrichment_table
from modules.heatmap import analyze_tf_cobinding, plot_heatmap, process_heatmap_data
from modules.ui.plot_settings import plot_settings_panel
from modules.ui.plot_summary import plot_summary_caption
from modules.utils import create_gene_selector

col1, col2 = st.columns([5, 1])

with col1:
    st.header("Multimodal Heatmap of TFs")
    st.markdown("Explore transcription factors involved in lineage decisions in the mouse pituitary. The heatmap displays TF motif co-occurrence with differential expression and accessibility between selected cell type groupings.")
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

        with plot_settings_panel("Plot settings"):
            # Analysis selection (grouping and direction)
            selected_grouping_name = st.selectbox(
                "Select Grouping and Direction",
                options=list(grouping_display_to_code.keys()),
                index=0,
                key="selected_grouping_name",
                help="Choose the cell type comparison and direction for analysis",
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

            st.markdown("#### TF Selection Method")

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

            st.markdown("#### Visualization Options")

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

            st.markdown("#### Filtering Thresholds")
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
                    plot_summary_caption(
                        f"{len(motifs)} transcription factors",
                        selected_grouping_name,
                    )

                    # Add download button for the plot
                    from io import BytesIO

                    buf = BytesIO()
                    heatmap_fig.savefig(
                        buf, format="svg", bbox_inches="tight"
                    )
                    buf.seek(0)

                    st.download_button(
                        on_click="ignore",
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
                        on_click="ignore",
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
                        on_click="ignore",
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
