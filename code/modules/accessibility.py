import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import polars as pl
from .utils import create_color_mapping
import streamlit as st

def create_accessibility_plot(
    matrix, features, meta_data, feature_name, additional_group=None, connect_dots=False
):
    """
    Create a box plot for accessibility data with strip plot overlay, supporting additional grouping and connected dots
    """
    # Convert matrix to array if needed
    if hasattr(matrix, "toarray"):
        matrix = matrix.tocsr()

    # Get feature values
    feature_idx = features.index(feature_name)
    feature_values = matrix[feature_idx].toarray().flatten()

    # Create plot dataframe
    plot_df = meta_data.copy()
    plot_df["Accessibility"] = feature_values

    # Create consistent color mapping
    color_map = create_color_mapping(plot_df["cell_type"].unique())

    if additional_group:
        # Sort data by cell type and additional group
        plot_df = plot_df.sort_values(["cell_type", additional_group])

        # Create a categorical x-axis that groups by cell type first
        cell_types = sorted(plot_df["cell_type"].unique())
        secondary_groups = sorted(plot_df[additional_group].unique())

        # Create a position mapping for proper spacing
        position_map = {}
        current_pos = 0
        for cell_type in cell_types:
            for group in secondary_groups:
                position_map[(cell_type, str(group))] = current_pos
                current_pos += 1
            current_pos += 1  # Add extra space between cell types

        # Create x positions for plotting
        plot_df["x_position"] = plot_df.apply(
            lambda row: position_map[(row["cell_type"], str(row[additional_group]))],
            axis=1,
        )

        if additional_group == "Comp_sex":
            # Define special colors for female (0) and male (1)
            sex_color_map = {
                "Female": "#FFA500",  # female
                "Male": "#63B3ED",     # male
            }
            # Create box plot with proper spacing
            fig = px.box(
                plot_df,
                x="x_position",
                y="Accessibility",
                color=additional_group,
                points=False,
                hover_data=["GEO"],
                title=f"{feature_name} Accessibility by Cell Type and {additional_group}",
                color_discrete_map=sex_color_map,
            )

            # Create matching strip plot
            strip_fig = px.strip(
                plot_df,
                x="x_position",
                y="Accessibility",
                color=additional_group,
                hover_data=["GEO"],
                color_discrete_map=sex_color_map,
            )

        else:
            # Create box plot with proper spacing
            fig = px.box(
                plot_df,
                x="x_position",
                y="Accessibility",
                color=additional_group,
                points=False,
                hover_data=["GEO"],
                title=f"{feature_name} Accessibility by Cell Type and {additional_group}",
            )

            # Create matching strip plot
            strip_fig = px.strip(
                plot_df,
                x="x_position",
                y="Accessibility",
                color=additional_group,
                hover_data=["GEO"],
            )

        # Update x-axis to show cell type labels centered for each group
        tick_positions = []
        tick_labels = []
        for cell_type in cell_types:
            # Get positions for this cell type
            cell_type_positions = [
                pos for (ct, _), pos in position_map.items() if ct == cell_type
            ]
            tick_positions.append(sum(cell_type_positions) / len(cell_type_positions))
            tick_labels.append(cell_type)

        fig.update_xaxes(ticktext=tick_labels, tickvals=tick_positions, tickangle=45)
    else:
        fig = px.box(
            plot_df,
            x="cell_type",
            y="Accessibility",
            color="cell_type",
            points=False,
            color_discrete_map=color_map,
            hover_data=["GEO"],
            title=f"{feature_name} Accessibility by Cell Type",
        )

        # Add strip plot
        strip_fig = px.strip(
            plot_df,
            x="cell_type",
            y="Accessibility",
            color="cell_type",
            color_discrete_map=color_map,
            hover_data=["GEO"],
        )

    # Update strip plot traces to match the box plot
    for trace in strip_fig.data:
        trace.update(marker=dict(opacity=0.4, size=6), showlegend=False)
        fig.add_trace(trace)

    # Optionally connect dots with the same SRA_ID
    if connect_dots:
        for sra_id in plot_df["SRA_ID"].unique():
            sra_df = plot_df[plot_df["SRA_ID"] == sra_id]
            # Use x_position if it exists (for grouped data), otherwise use cell type
            x_values = (
                sra_df["x_position"]
                if "x_position" in sra_df.columns
                else sra_df["cell_type"]
            )
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=sra_df["Accessibility"],
                    mode="lines+markers",
                    line=dict(color="gray", width=1),
                    marker=dict(size=6, opacity=0.6),
                    name=sra_id,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    fig.update_layout(
        xaxis_title="Cell Type",
        yaxis_title=f"{feature_name} Acc. Score",
        showlegend=True,
        xaxis={"tickangle": 45, "tickfont": {"size": 20}, "title_font": {"size": 20}},
        yaxis={"title_font": {"size": 20}, "tickfont": {"size": 20}},
        height=600,
        width=None,
    )

    config = {
        "toImageButtonOptions": {
            "format": "svg",
            "filename": f"{feature_name}_accessibility",
            "height": 800,
            "width": 1600,
            "scale": 4,
        }
    }

    return fig, config

def create_genome_browser_plot(
    matrix,
    features,
    meta_data,
    selected_region,
    selected_version,
    show_enhancers=False,
    enhancer_df=None,
    annotation_df=None,
    motif_df = None,
    color_map=None,
    selected_motifs=None,
):
    """
    Create a genome browser-style plot showing accessibility peaks, gene annotations, and selected motifs.

    Parameters:
        matrix: accessibility matrix
        features: feature list
        meta_data: metadata DataFrame
        selected_region: string of format "chrX:start-end"
        selected_motifs: list of selected motif names (up to 10)
        color_map: optional color mapping for cell types
    """
    import plotly.graph_objects as go
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import re
    import polars as pl

    # Define color palette for motifs
    motif_color_palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    # Initialize session state for genome region if it doesn't exist
    if 'base_genome_region' not in st.session_state:
        st.session_state['base_genome_region'] = selected_region
    
    # If the selected_region (gene region) has changed, reset the view
    if st.session_state['base_genome_region'] != selected_region:
        st.session_state['base_genome_region'] = selected_region
        st.session_state['genome_region'] = selected_region
    
    # Initialize genome_region if it doesn't exist
    if 'genome_region' not in st.session_state:
        st.session_state['genome_region'] = selected_region
    
    # Use the session state value
    current_region = st.session_state['genome_region']
    
    # Parse current region
    chr_name, region_range = current_region.split(":")
    region_start, region_end = map(int, region_range.split("-"))
    region_width = region_end - region_start

    # Add navigation controls (zoom in/out, move left/right)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("⬅️ Move Left", key="move_left"):
            # Move 20% to the left
            shift_amount = int(region_width * 0.3)
            new_start = max(0, region_start - shift_amount)
            new_end = new_start + region_width
            st.session_state['genome_region'] = f"{chr_name}:{new_start}-{new_end}"
            st.rerun()
    
    with col2:
        if st.button("➡️ Move Right", key="move_right"):
            # Move 20% to the right
            shift_amount = int(region_width * 0.3)
            new_start = region_start + shift_amount
            new_end = new_start + region_width
            st.session_state['genome_region'] = f"{chr_name}:{new_start}-{new_end}"
            st.rerun()
    
    with col3:
        if st.button("🔍 Zoom In", key="zoom_in"):
            # Zoom in 10% (make region 10% smaller)
            zoom_amount = int(region_width * 0.3)
            center = (region_start + region_end) // 2
            new_start = center - (region_width // 2) + zoom_amount
            new_end = center + (region_width // 2) - zoom_amount
            st.session_state['genome_region'] = f"{chr_name}:{new_start}-{new_end}"
            st.rerun()
    
    with col4:
        if st.button("🔍 Zoom Out", key="zoom_out"):
            # Zoom out 10% (make region 10% larger)
            zoom_amount = int(region_width * 0.3)
            center = (region_start + region_end) // 2
            new_start = max(0, center - (region_width // 2) - zoom_amount)
            new_end = center + (region_width // 2) + zoom_amount
            st.session_state['genome_region'] = f"{chr_name}:{new_start}-{new_end}"
            st.rerun()
    
    # Reset button to return to original region
    if st.button("🔄 Reset View", key="reset_view"):
        st.session_state['genome_region'] = selected_region
        st.rerun()
    
    # Show current region information
    st.info(f"Current region: {current_region} ({region_width:,} bp width)")

    # Define all track heights and spacing constants
    scale_bar_width = region_width * 0.02
    track_spacing = 1.0
    accessibility_track_height = 0.8
    gene_track_height = 0.3
    motif_height = 0.2  # Height for motif markers
    motif_spacing = 0.4  # Spacing between motif tracks
    enhancer_height = 0.2  # Height for enhancer markers
    enhancer_spacing = 0.4  # Spacing between enhancer tracks

    def hex_to_rgba(hex_color, alpha=0.3):
        """Convert hex color to rgba with alpha."""
        rgb_match = re.match(r"rgb\((\d+),\s*(\d+),\s*(\d+)\)", hex_color)
        if rgb_match:
            r, g, b = map(int, rgb_match.groups())
            return f"rgba({r}, {g}, {b}, {alpha})"

        hex_color = hex_color.lstrip("#")
        try:
            rgb = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
            return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})"
        except ValueError:
            return f"rgba(100, 100, 100, {alpha})"

    # Create the plot
    fig = go.Figure()

    # Step 1: First filter for genes that overlap with the viewing window
    # This gives us the gene names we need to get complete information for
    region_annotations_partial = annotation_df.filter(
        (pl.col("seqnames") == chr_name) & 
        (pl.col("end") >= region_start) & 
        (pl.col("start") <= region_end)
    )
    
    # Get list of unique gene names that are visible in the window
    visible_genes = region_annotations_partial.select("gene_name").unique().to_series().to_list()
    
    # Step 2: Get complete gene information for all visible genes
    # This will include all exons, even those outside the viewing window
    region_annotations_pl = annotation_df.filter(
        (pl.col("seqnames") == chr_name) & 
        (pl.col("gene_name").is_in(visible_genes))
    )

    # Convert to pandas for the rest of the function
    region_annotations = region_annotations_pl.to_pandas()

    # Get unique genes and their complete coordinates
    genes_data = []
    for gene in visible_genes:
        gene_annot = region_annotations[region_annotations["gene_name"] == gene]
        gene_start = gene_annot["start"].min()
        gene_end = gene_annot["end"].max()
        genes_data.append(
            {
                "gene_name": gene,
                "start": gene_start,
                "end": gene_end,
                "median_pos": (gene_start + gene_end) / 2,
                # Check if gene extends beyond visible region
                "extends_left": gene_start < region_start,
                "extends_right": gene_end > region_end
            }
        )

    # Sort genes by start position
    genes_data = sorted(genes_data, key=lambda x: x["start"])

    # Assign tracks optimally
    tracks = []  # List of tracks, each track is a list of genes
    gene_tracks = {}  # Dictionary mapping gene names to track numbers

    for gene_data in genes_data:
        placed = False
        for track_idx, track in enumerate(tracks):
            can_place = True
            for existing_gene in track:
                if not (
                    gene_data["end"]
                    < existing_gene["start"] - (region_end - region_start) * 0.05
                    or gene_data["start"]
                    > existing_gene["end"] + (region_end - region_start) * 0.05
                ):
                    can_place = False
                    break

            if can_place:
                track.append(gene_data)
                gene_tracks[gene_data["gene_name"]] = track_idx
                placed = True
                break

        if not placed:
            if len(tracks) < 10:  # Limit to 10 tracks
                tracks.append([gene_data])
                gene_tracks[gene_data["gene_name"]] = len(tracks) - 1
            else:
                tracks[-1].append(gene_data)
                gene_tracks[gene_data["gene_name"]] = len(tracks) - 1

    # Create gene colors
    gene_colors = {}
    color_palette = (
        px.colors.qualitative.Set1
        + px.colors.qualitative.Dark2
        + px.colors.qualitative.Bold
        + [
            "#1f77b4",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
            "#ff7f0e",
        ]
    )

    for i, gene_data in enumerate(genes_data):
        gene_colors[gene_data["gene_name"]] = color_palette[i % len(color_palette)]
    print(gene_colors)
    
    # Create default color map if none provided
    if color_map is None:
        cell_types = sorted(meta_data["cell_type"].unique())
        colors = px.colors.qualitative.Set3
        color_map = dict(zip(cell_types, colors[: len(cell_types)]))

    # Filter features within the selected region
    feature_data = []
    for i, feature in enumerate(features):
        try:
            feat_chr, feat_range = feature.split(":")
            feat_start, feat_end = map(int, feat_range.split("-"))

            if (
                feat_chr == chr_name
                and feat_start <= region_end
                and feat_end >= region_start
            ):
                feature_data.append(
                    {"index": i, "chr": feat_chr, "start": feat_start, "end": feat_end}
                )
        except:
            continue

    feature_df = pd.DataFrame(feature_data)
    
    # MODIFIED: Handle case when no features are found
    if len(feature_df) == 0:
        # Add info message about no fragments
        st.info("No accessibility fragments found in this region.")
        # Create empty tracks_data but still with cell types
        tracks_data = []
        max_signal = 1.0  # Default max signal
    else:
        # Calculate tracks data (original logic)
        cell_types = sorted(meta_data["cell_type"].unique(), reverse=True)
        tracks_data = []

        for cell_type in cell_types:
            # Get the boolean mask for this cell type
            cell_mask = meta_data["cell_type"] == cell_type

            # Only proceed if we have any cells of this type
            if cell_mask.any():
                for _, feature in feature_df.iterrows():
                    try:
                        # Extract the relevant slice of the matrix using the mask
                        if hasattr(matrix, "toarray"):
                            # For sparse matrices
                            feature_slice = matrix[feature["index"], :]
                            # Apply the mask to get only cells of this type
                            cell_type_values = feature_slice[:, cell_mask.values]
                            signal = (
                                cell_type_values.toarray().mean()
                                if cell_type_values.size > 0
                                else 0
                            )
                        else:
                            # For dense matrices
                            cell_type_values = matrix[feature["index"], cell_mask.values]
                            signal = (
                                cell_type_values.mean() if cell_type_values.size > 0 else 0
                            )

                        tracks_data.append(
                            {
                                "cell_type": cell_type,
                                "start": feature["start"],
                                "end": feature["end"],
                                "signal": float(signal),
                                "color": color_map.get(cell_type, "#000000"),
                            }
                        )
                    except Exception as e:
                        print(
                            f"Error processing feature {feature['index']} for cell type {cell_type}: {str(e)}"
                        )
                        # Continue with next feature instead of failing
                        continue

        # Calculate max signal for scaling
        max_signal = max([track["signal"] for track in tracks_data]) if tracks_data else 1.0
        if max_signal < 0.01:
            max_signal = 0.01

    tracks_df = pd.DataFrame(tracks_data)
    
    # Get cell types for track creation (even if no features)
    cell_types = sorted(meta_data["cell_type"].unique(), reverse=True)

    # Add gene tracks and labels
    for track_idx, track in enumerate(tracks):
        track_y = len(cell_types) + track_idx + 1

        for gene_data in track:
            gene_name = gene_data["gene_name"]
            gene_exons = region_annotations[
                region_annotations["gene_name"] == gene_name
            ]

            # Add gene name label - positioned within the visible window
            gene_strand = gene_exons['strand'].iloc[0] if 'strand' in gene_exons.columns else '+'
            arrow_symbol = "→" if gene_strand == '+' else "←" 
            
            # Calculate label position (ensure it's within the visible window)
            label_pos = gene_data['median_pos']
            if label_pos < region_start:
                label_pos = region_start + (region_width * 0.05)
            elif label_pos > region_end:
                label_pos = region_end - (region_width * 0.05)
                
            # Add gene name label with direction arrow
            fig.add_annotation(
                x=label_pos,
                y=track_y + gene_track_height * 1.5,
                text=f"{gene_name} {arrow_symbol}",
                showarrow=False,
                yanchor='bottom',
                font=dict(size=12, color=gene_colors[gene_name]),
                align='center'
            )

            # Add connecting line - handle genes that extend beyond the visible region
            # Use the full gene extent, but clip it to the visible window for display
            gene_start_raw = gene_data["start"]
            gene_end_raw = gene_data["end"]
            
            # Determine the visible portion of the gene
            visible_start = max(gene_start_raw, region_start)
            visible_end = min(gene_end_raw, region_end)
            
            # Create connecting line for the main gene body
            rgba_color = hex_to_rgba(gene_colors[gene_name])
            
            # For genes that extend beyond the viewing window, use special styling
            if gene_data["extends_left"] or gene_data["extends_right"]:
                # Main line
                fig.add_shape(
                    type="line",
                    x0=visible_start,
                    x1=visible_end,
                    y0=track_y + gene_track_height / 2,
                    y1=track_y + gene_track_height / 2,
                    line=dict(color=rgba_color, width=2),
                    layer="below",
                )
                
                # Add extension indicators
                if gene_data["extends_left"]:
                    # Add dashed line segment or arrow to indicate extension left
                    fig.add_shape(
                        type="line",
                        x0=region_start,
                        x1=region_start + (region_width * 0.01),
                        y0=track_y + gene_track_height / 2,
                        y1=track_y + gene_track_height / 2,
                        line=dict(color=gene_colors[gene_name], width=3, dash="dash"),
                        layer="below",
                    )
                
                if gene_data["extends_right"]:
                    # Add dashed line segment or arrow to indicate extension right
                    fig.add_shape(
                        type="line",
                        x0=region_end - (region_width * 0.01),
                        x1=region_end,
                        y0=track_y + gene_track_height / 2,
                        y1=track_y + gene_track_height / 2,
                        line=dict(color=gene_colors[gene_name], width=3, dash="dash"),
                        layer="below",
                    )
            else:
                # For genes fully contained in the window, draw a normal line
                fig.add_shape(
                    type="line",
                    x0=gene_start_raw,
                    x1=gene_end_raw,
                    y0=track_y + gene_track_height / 2,
                    y1=track_y + gene_track_height / 2,
                    line=dict(color=rgba_color, width=2),
                    layer="below",
                )

            # Add exon boxes - only show those within the visible window
            for _, exon in gene_exons.iterrows():
                # Skip exons completely outside the viewing window
                if exon["end"] < region_start or exon["start"] > region_end:
                    continue
                    
                # For visible exons, only show the visible portion
                visible_exon_start = max(exon["start"], region_start)
                visible_exon_end = min(exon["end"], region_end)
                
                fig.add_shape(
                    type="rect",
                    x0=visible_exon_start,
                    x1=visible_exon_end,
                    y0=track_y,
                    y1=track_y + gene_track_height,
                    fillcolor=gene_colors[gene_name],
                    opacity=0.6,
                    layer="above",
                    line_width=1,
                )

    # Add accessibility tracks
    for i, cell_type in enumerate(cell_types):
        track_y = i * track_spacing

        # Add cell type label
        fig.add_annotation(
            x=region_start - scale_bar_width * 3,
            y=i + 0.4,
            text=cell_type,
            showarrow=False,
            xanchor="right",
            xref="x",
            yref="y",
            font=dict(color=color_map.get(cell_type, "#000000"), size=14),
            align="right",
        )

        # Add scale bar
        fig.add_trace(
            go.Scatter(
                x=[
                    region_start - scale_bar_width * 3,
                    region_start - scale_bar_width * 3,
                ],
                y=[track_y, track_y + accessibility_track_height],
                mode="lines",
                line=dict(color="black", width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Add scale bar labels
        fig.add_annotation(
            x=region_start - scale_bar_width * 3,
            y=track_y,
            text="0",
            showarrow=False,
            xanchor="right",
            font=dict(size=10),
        )
        fig.add_annotation(
            x=region_start - scale_bar_width * 3,
            y=track_y + accessibility_track_height,
            text=f"{max_signal:.2f}",
            showarrow=False,
            xanchor="right",
            font=dict(size=10),
        )

        # Add baseline track
        fig.add_trace(
            go.Scatter(
                x=[region_start, region_end],
                y=[track_y, track_y],
                mode="lines",
                line=dict(color="#e9ecef", width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Add peaks (only if we have track data)
        if not tracks_df.empty:
            track_data = tracks_df[tracks_df["cell_type"] == cell_type]
            for _, peak in track_data.iterrows():
                normalized_height = (
                    peak["signal"] / max_signal
                ) * accessibility_track_height
                peak_middle = (peak["start"] + peak["end"]) / 2

                fig.add_trace(
                    go.Scatter(
                        x=[peak["start"], peak_middle, peak["end"]],
                        y=[track_y, track_y + normalized_height, track_y],
                        mode="none",
                        fill="toself",
                        fillcolor=peak["color"],
                        line=dict(width=0),
                        opacity=0.6,
                        showlegend=False,
                        hovertemplate=(
                            f"Cell Type: {cell_type}<br>"
                            + "Position: %{x:,}<br>"
                            + "Signal: %{customdata:.3f}<br>"
                            + "<extra></extra>"
                        ),
                        customdata=[peak["signal"]] * 3,
                    )
                )

    # Add motif tracks if selected
    if selected_motifs:
        # Filter the Polars DataFrame for motifs within the visible window
        filtered_motif_pl_df = motif_df.filter(
            (pl.col("seqnames") == chr_name)
            & (pl.col("end") >= region_start)
            & (pl.col("start") <= region_end)
            & pl.col("motif").is_in(selected_motifs)
        )

        # Convert the filtered Polars DataFrame back to pandas
        motif_df_pandas = filtered_motif_pl_df.to_pandas()

        # Calculate minimum motif width (1% of total region width)
        min_motif_width = max(region_width * 0.01, 10)  # At least 10bp wide

        motif_base_y = len(cell_types) + len(tracks) + 1
        motif_colors = {
            motif: motif_color_palette[i % len(motif_color_palette)]
            for i, motif in enumerate(selected_motifs)
        }

        for i, motif in enumerate(selected_motifs):
            motif_y = motif_base_y + (i * motif_spacing)
            motif_data = motif_df_pandas[motif_df_pandas["motif"] == motif]

            # Add motif label
            fig.add_annotation(
                x=region_start - scale_bar_width * 2,
                y=motif_y + motif_height / 2,
                text=motif,
                showarrow=False,
                xanchor="right",
                font=dict(size=10, color=motif_colors[motif]),
                align="right",
            )

            # Add motif markers
            for _, site in motif_data.iterrows():
                if site["start"] <= region_end and site["end"] >= region_start:
                    motif_start = max(site["start"], region_start)
                    motif_end = min(site["end"], region_end)
                    motif_width = motif_end - motif_start

                    # Apply minimum width if needed
                    if motif_width < min_motif_width:
                        center = (motif_start + motif_end) / 2
                        motif_start = center - (min_motif_width / 2)
                        motif_end = center + (min_motif_width / 2)

                    # Create a rectangle using go.Scatter with fixed text for hover
                    fig.add_trace(
                        go.Scatter(
                            x=[
                                motif_start,
                                motif_end,
                                motif_end,
                                motif_start,
                                motif_start,
                            ],
                            y=[
                                motif_y,
                                motif_y,
                                motif_y + motif_height,
                                motif_y + motif_height,
                                motif_y,
                            ],
                            mode="none",
                            fill="toself",
                            fillcolor=motif_colors[motif],
                            opacity=0.8,
                            line=dict(width=1, color=motif_colors[motif]),
                            text=f"Motif: {motif}<br>Position: {site['start']:,}-{site['end']:,}<br>Length: {site['end'] - site['start']:,} bp",
                            hoverinfo="text",
                            showlegend=False,
                        )
                    )
                    
    if show_enhancers and enhancer_df is not None:
        # Filter enhancers within the visible window
        enhancer_data_filtered = enhancer_df.filter(
        (pl.col("seqnames") == chr_name) & 
        (pl.col("end") >= region_start) & 
        (pl.col("start") <= region_end) &
        (pl.col("gene").is_in(gene_colors.keys()) if enhancer_df["gene"].dtype == "str" else pl.lit(True))
    )
        # Convert the filtered Polars DataFrame back to pandas
        enhancer_df_pandas = enhancer_data_filtered.to_pandas()
        enhancer_genes = enhancer_df_pandas["gene"].unique()

        # Calculate minimum motif width (1% of total region width)
        min_enhancer_width = max(region_width * 0.01, 10)  # At least 10bp wide

        enhancer_base_y = len(cell_types) + len(tracks) + 2 + (
            len(selected_motifs) * motif_spacing if selected_motifs else 0
        ) + 0.5

        for i, gene in enumerate(enhancer_genes):

            try:
                color_to_use = gene_colors[gene]
            except:
                color_to_use = "#000000"

            enhancer_y = enhancer_base_y + (i * enhancer_spacing)
            enhancer_data = enhancer_df_pandas[enhancer_df_pandas["gene"] == gene]
            
            #keep rows with unique start
            enhancer_data = enhancer_data.drop_duplicates(subset=["start", "end"])

            # Add motif label
            fig.add_annotation(
                x=region_start - scale_bar_width * 2,
                y=enhancer_y + enhancer_height / 2,
                text=f"Enhancer for {gene}",
                showarrow=False,
                xanchor="right",
                font=dict(size=10, color= color_to_use),
                align="right",
            )
            
            # Add motif markers
            for _, site in enhancer_data.iterrows():
                if site["start"] <= region_end and site["end"] >= region_start:
                    enhancer_start = max(site["start"], region_start)
                    enhancer_end = min(site["end"], region_end)
                    enhancer_width = enhancer_end - enhancer_start
                    opacity = np.abs(site["cor"])
                    #red if negative correlation
                    line_color = "red" if site["cor"] < 0 else "green"

                    # Apply minimum width if needed
                    if enhancer_width < min_enhancer_width:
                        center = (enhancer_start + enhancer_end) / 2
                        enhancer_start = center - (min_enhancer_width / 2)
                        enhancer_end = center + (min_enhancer_width / 2)

                    # Create a rectangle using go.Scatter with fixed text for hover
                    fig.add_trace(
                    go.Scatter(
                        x=[
                            enhancer_start,
                            enhancer_end,
                            enhancer_end,
                            enhancer_start,
                            enhancer_start,
                        ],
                        y=[
                            enhancer_y,
                            enhancer_y,
                            enhancer_y + enhancer_height,
                            enhancer_y + enhancer_height,
                            enhancer_y,
                        ],
                        mode="lines",  # Use "lines" to draw the outline
                        fill="toself",  # Still fills the box
                        fillcolor=color_to_use,
                        line=dict(color=line_color, width=2),
                        opacity=opacity,
                        text=(
                            f"Enhancer for gene: {gene}<br>"
                            f"Correlation: {site['cor']:.2f}<br>"
                            f"Position: {site['start']:,}-{site['end']:,}<br>"
                            f"Length: {site['end'] - site['start']:,} bp"
                        ),
                        hoverinfo="text",
                        showlegend=False,
                    )
                )

    # Update layout
    motif_height_addition = len(selected_motifs) * 40 if selected_motifs else 0
    enhancer_height_addition = (
        len(enhancer_genes) * 40 if show_enhancers and enhancer_df is not None else 0
    )

    plot_height = (
        100 + (len(cell_types) * 80) + (len(tracks) * 60) + motif_height_addition + enhancer_height_addition
    )
    y_range_max = (
        (len(cell_types) + len(tracks)) * track_spacing
        + (len(selected_motifs) * motif_spacing if selected_motifs else 0)
        + (len(enhancer_genes) * enhancer_spacing if show_enhancers and enhancer_df is not None else 0)
        + 3.5
    )

    fig.update_layout(
        title=f"Accessibility Profile - {current_region}",
        xaxis=dict(
            title="Genomic Position",
            showgrid=False,
            zeroline=False,
            range=[region_start - scale_bar_width * 3, region_end],
            tickformat=",d",
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-0.5, y_range_max],
        ),
        height=plot_height,
        width=1200,
        showlegend=False,
        plot_bgcolor="white",
        margin=dict(l=200, r=50, t=50, b=50),
    )

    # Configure download options
    config = {
        "toImageButtonOptions": {
            "format": "svg",
            "filename": f"genome_browser_{chr_name}_{region_start}_{region_end}",
            "height": plot_height,
            "width": 1200,
            "scale": 2,
        }
    }
    
    #if ehancer_df_pandas exists
    if show_enhancers and enhancer_df is not None:
        return enhancer_df_pandas, fig, config, None
    else:
        return None, fig, config, None