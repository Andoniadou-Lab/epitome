import traceback
from datetime import datetime

import streamlit as st

from modules.analytics import add_activity
from modules.other.page_layout import other_page_header
from modules.other.phylogeny import (
    CELL_TYPE_LABELS,
    DEFAULT_GENES,
    available_cell_types,
    change_table,
    load_cell_type_tree,
    load_dge_table,
    orthogroup_genes,
    phylogeny_dir,
    plot_gene_tree_plotly,
    tip_reason_table,
)
from modules.ui.plot_settings import download_format_select, plot_settings_panel
from modules.ui.plot_summary import plot_summary_caption
from modules.utils import create_gene_selector

selected_version = other_page_header(
    "Phylogeny",
    "Ancestral reconstruction of pituitary cell-type markers across species. "
    "A gene is a marker (red) or not (grey) at each assayed tip; Fitch parsimony "
    "fills in the ancestors. Species never assayed for that cell type are omitted.",
    "version_select_other_phylogeny",
)

directory, _resolved = phylogeny_dir(selected_version)
if directory is None:
    st.warning(
        f"No phylogeny tables found for version {selected_version} "
        "or any earlier version. Expected CSVs under other_data/phylogeny/."
    )
    st.stop()

cell_types = available_cell_types(directory)
if not cell_types:
    st.warning("No cell-type DGE tables found in this version.")
    st.stop()

with plot_settings_panel("Plot settings"):
    col1, col2, col3 = st.columns(3)
    with col1:
        cell_type = st.selectbox(
            "Cell type",
            options=cell_types,
            format_func=lambda ct: CELL_TYPE_LABELS.get(ct, ct),
            key="other_phylo_cell_type",
        )
    with col3:
        download_as = download_format_select(
            "other_phylo_download", formats=("png", "jpeg", "svg")
        )

    dge = load_dge_table(str(directory), cell_type)
    genes = orthogroup_genes(dge)
    suggested = DEFAULT_GENES.get(cell_type)
    if suggested in genes and st.session_state.get("selected_gene") not in genes:
        st.session_state["selected_gene"] = suggested
    with col2:
        selected_gene = create_gene_selector(
            gene_list=genes,
            key_suffix="other_phylo_gene",
            label=f"Orthogroup ({len(genes):,} human symbols)",
        )

try:
    tree = load_cell_type_tree(str(directory), cell_type)
    fig, config, reconstructed = plot_gene_tree_plotly(
        tree,
        selected_gene,
        title=f"{CELL_TYPE_LABELS.get(cell_type, cell_type)}: {selected_gene}",
        download_as=download_as,
    )
    add_activity(
        value=[cell_type, selected_gene],
        analysis="Other Phylogeny",
        user=st.session_state.session_id,
        time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    st.plotly_chart(fig, use_container_width=True, config=config)
    plot_summary_caption(
        CELL_TYPE_LABELS.get(cell_type, cell_type),
        selected_gene,
        f"{len(reconstructed)} species",
        f"parsimony cost {reconstructed.cost}",
        version=selected_version,
        loader_key="other_phylogeny",
    )

    reasons = tip_reason_table(reconstructed, selected_gene)
    changes = change_table(reconstructed)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Tips**")
        st.dataframe(reasons, hide_index=True, use_container_width=True)
    with col_b:
        st.markdown("**Gains and losses**")
        if changes.empty:
            st.caption("No reconstructed gain or loss on this tree.")
        else:
            st.dataframe(changes, hide_index=True, use_container_width=True)

    st.markdown(
        """
        Tips are always true or false for an assayed species. **Marker** means
        a significant hit (even if the background table missed the orthogroup).
        **Not a marker** means the gene was tested and not significant.
        **Orthogroup missing** is also false in Fitch; the dark-grey label is
        display only and is not fed back into the parsimony. Internal nodes
        follow the reconstructed state. The root defaults to absent when the
        reconstruction is ambiguous.
        """
    )
except Exception as exc:
    st.error(f"Error creating the phylogeny: {exc}")
    with st.expander("Show full traceback"):
        st.code(traceback.format_exc(), language="python")
