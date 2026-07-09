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
from modules.ui.plot_settings import plot_settings_panel
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

with plot_settings_panel("Plot settings"):
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
                width=250,
            )
        with col3_3:
            x_axis_order = st.selectbox(
                "X-axis Order",
                options=["sender", "target"],
                key="x_axis_order_lr",
                help="Order x-axis by sender or target cell type",
                width=250,
            )

    st.markdown("**Filter Specific Interactions**")
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

    chosen_color_scheme = st.selectbox(
        "Select Color Scheme",
        options=["Blue", "Red", "Viridis", "Cividis"],
        index=0,
        key="color_scheme_lr",
        width=250,
    )

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

add_activity(
    value=[selected_source, selected_target],
    analysis="Ligand-Receptor Interactions",
    user=st.session_state.session_id,
    time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
)

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
    order_by=x_axis_order,
    color_scheme=chosen_color_scheme,
)

st.plotly_chart(
    fig, use_container_width=True, config=config
)
n_pairs = plot_df[["source", "target"]].drop_duplicates().shape[0] if not plot_df.empty else 0
plot_summary_caption(
    f"{len(plot_df)} interactions plotted",
    f"{n_pairs} cell-type pairs",
    f"top {top_n} by {sort_by}",
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

st.markdown("---")
st.subheader("Detailed Interaction Table")
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

table_data = (
    liana_df if show_all_interactions else filtered_df
)

filtered_data = display_ligand_receptor_table(
    table_data, key_prefix="lr_interactions"
)

if show_all_interactions:
    st.info(
        f"Showing all {len(liana_df)} interactions in the dataset. The plot above shows only the top {top_n} filtered interactions."
    )
else:
    st.info(
        f"Showing {len(filtered_df)} interactions that match the current filter criteria and appear in the plot above."
    )
