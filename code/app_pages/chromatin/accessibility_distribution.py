import traceback
from datetime import datetime

import pandas as pd
import polars as pl
import streamlit as st

from config import Config
from modules.accessibility import create_accessibility_plot, create_genome_browser_plot
from modules.analytics import add_activity
from modules.cached_loaders import (
    AVAILABLE_VERSIONS,
    load_cached_accessibility_data,
    load_cached_annotation_data,
    load_cached_chromvar_data,
    load_cached_enhancer_data,
    load_cached_marker_data_atac,
    load_cached_motif_data,
    preprocess_features_cached,
)
from modules.data_loader import load_motif_genes
from modules.display_tables import display_enhancers_table, display_marker_table
from modules.ui.plot_settings import plot_settings_panel
from modules.ui.plot_summary import plot_summary_caption
from modules.utils import (
    create_cell_type_stats_display,
    create_color_mapping,
    create_filter_ui,
    create_gene_selector_with_coordinates,
    create_region_selector,
    filter_accessibility_data,
    to_array,
)

BASE_PATH = Config.BASE_PATH

@st.fragment
def render_genome_browser(browser_matrix, features, browser_meta, selected_region, 
                          selected_version, annotation_df, enhancer_df, motif_data,
                          color_map, selected_cell_types_browser):
    """Genome browser as a fragment — button clicks only rerun this section."""
    
    selected_motifs = st.multiselect(
        "Select Motifs to Display (max 10)",
        options=sorted(motif_data.select("motif").unique().to_series().to_list()),
        max_selections=10,
        help="Choose up to 10 motifs to display in the genome browser",
    )

    st.info("Only those motifs are shown that fall in a given consensus peak.")
    
    if selected_motifs:
        add_activity(value=[selected_region, selected_motifs],
            analysis="Genome Browser",
            user=st.session_state.session_id,
            time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    else:
        add_activity(value=selected_region,
            analysis="Genome Browser",
            user=st.session_state.session_id,
            time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    show_enhancers = st.checkbox(
        "Show enhancers",
        value=False,
        help="Display enhancer regions in the genome browser",
        key="show_enhancers",
    )

    filtered_enhancers, browser_fig, browser_config, error_message = (
        create_genome_browser_plot(
            matrix=browser_matrix,
            features=features,
            feature_df=preprocess_features_cached(features),
            meta_data=browser_meta,
            selected_region=selected_region,
            selected_version=selected_version,
            annotation_df=annotation_df,
            show_enhancers=show_enhancers,
            enhancer_df=enhancer_df,
            motif_df=motif_data,
            color_map={ct: color_map[ct] for ct in selected_cell_types_browser},
            selected_motifs=selected_motifs if selected_motifs else None,
        )
    )

    if error_message:
        st.warning(error_message)
    elif browser_fig:
        st.plotly_chart(browser_fig, use_container_width=True, config=browser_config)
        plot_summary_caption(
            selected_region,
            f"{len(browser_meta)} pseudobulk samples",
            f"{len(selected_cell_types_browser)} cell types",
        )

    return filtered_enhancers,show_enhancers

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

    with plot_settings_panel("Plot settings"):
        (
            filter_type,
            selected_samples,
            selected_authors,
            age_range,
            only_normal,
            modality
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
            selected_samples = filtered_meta["Name"].values.tolist()

        if filter_type == "Modality" and "modality" in locals():
            filtered_meta = filtered_meta[
                filtered_meta["Modality"].isin(modality)
            ]
            selected_samples = filtered_meta["Name"].values.tolist()

        filtered_meta, filtered_matrix = filter_accessibility_data(
            meta_data=accessibility_meta,
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
        )

        # Additional grouping
        additional_group = st.selectbox(
            "Additional Grouping Variable",
            ["None", "Modality", "Comp_sex"],
            key="additional_group_select_tab5",
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
        plot_summary_caption(
            selected_feature,
            f"{filtered_meta.shape[0]} pseudobulk samples",
            f"{filtered_meta['cell_type'].nunique()} cell types",
        )
        #gc.collect()
        with st.container():
            st.markdown(
                f"""
                This plot shows the distribution of chromatin accessibility fragments counts.

                **X-axis**: Cell types present in the selected samples
                **Y-axis**: Log10 fragment counts per million* for the selected genomic region

                **Box:** centre line = **median**; top and bottom edges = **75th** and **25th** percentiles

                **Whiskers:** extend to the **5th** and **95th** percentiles

                **Points:** each dot is one sample; hover a dot for sample-level detail

                The box plot also enables:
                - Optional grouping by additional variables
                - Optional connecting lines between samples from the same source

                * Fragment counts were first TMMwsp normalised in the Limma-voom workflow.
            """
            )

        # Add download button
        feature_idx = features.index(selected_feature)
        accessibility_values = to_array(filtered_matrix[feature_idx, :])

        download_df = filtered_meta.copy()
        download_df["Accessibility"] = accessibility_values

        st.download_button(
            on_click="ignore",
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
        st.subheader("Cell Type Selection")
        all_cell_types = sorted(filtered_meta["cell_type"].unique())
        selected_cell_types_browser = st.multiselect(
            "Select Cell Types to Display",
            options=all_cell_types,
            default=all_cell_types,
            help="Choose which cell types to display in the genome browser",
        )
        cell_type_mask = filtered_meta["cell_type"].isin(selected_cell_types_browser)
        browser_meta = filtered_meta[cell_type_mask].copy()
        browser_matrix = filtered_matrix[:, cell_type_mask.values]

        (
            chromvar_matrix, chromvar_meta,
            chromvar_features, chromvar_columns,
        ) = load_cached_chromvar_data(version=selected_version)

        motif_data = load_cached_motif_data(version=selected_version)
        enhancer_df = load_cached_enhancer_data(version=selected_version)

        st.subheader("Motif Selection")

        # Call as fragment — navigation buttons only rerun this
        filtered_enhancers,show_enhancers = render_genome_browser(
            browser_matrix, features, browser_meta,
            selected_region, selected_version,
            annotation_df, enhancer_df, motif_data,
            color_map, selected_cell_types_browser,
        )


        #gc.collect()

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
                        signal = float(to_array(filtered_matrix[feature_idx, cell_indices]).mean())

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
                on_click="ignore",
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
        selected_version_markers = st.selectbox(
            "Version",
            options=AVAILABLE_VERSIONS,
            key="version_select_marker_browser_atac",
            label_visibility="collapsed",
        )

    filtered_data = display_marker_table(
        selected_version_markers, load_cached_marker_data_atac, "accessibility"
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
