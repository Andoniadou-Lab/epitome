import traceback
from datetime import datetime

import streamlit as st

from modules.analytics import add_activity
from modules.display_tables import display_volcano_results_table
from modules.pta.data_loader import load_volcano_manifest, load_volcano_results
from modules.pta.page_layout import pta_page_header
from modules.pta.volcano import create_volcano_plot
from modules.ui.plot_settings import download_format_select, plot_settings_panel
from modules.ui.plot_summary import plot_summary_caption

selected_version = pta_page_header(
    "Volcano Plots",
    "Visualise bulk tumour differential-expression results. "
    "All genes are shown; significance thresholds affect colouring only.",
    "version_select_tumor_volcano",
)

try:
    comparisons = load_volcano_manifest(version=selected_version)
except FileNotFoundError as exc:
    st.error(
        "Volcano manifest not found. Expected "
        f"`pta_data/epitome_volcanos/{selected_version}/volcanos.json`."
    )
    st.code(str(exc))
    st.stop()

if not comparisons:
    st.warning("No comparisons defined for this version.")
    st.stop()

labels = {c["id"]: c["name"] for c in comparisons}
selected_id = st.selectbox(
    "Comparison",
    options=list(labels.keys()),
    format_func=lambda cid: labels[cid],
    key="tumor_volcano_comparison",
)
entry = next(c for c in comparisons if c["id"] == selected_id)

st.markdown(
    f"**{entry['group_a']}** vs **{entry['group_b']}**  \n"
    f"{entry['explanation']}"
)

try:
    results = load_volcano_results(selected_version, selected_id)
    gene_options = sorted(results["gene"].dropna().astype(str).unique())

    with plot_settings_panel("Plot settings"):
        col1, col2 = st.columns(2)
        with col1:
            pval_threshold = st.number_input(
                "adj.P.Val threshold (visual)",
                min_value=1e-10,
                max_value=1.0,
                value=0.05,
                format="%.4f",
                key="tumor_volcano_pval",
            )
            logfc_threshold = st.number_input(
                "|logFC| threshold (visual)",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.1,
                key="tumor_volcano_logfc",
            )
            label_top_n = st.slider(
                "Label top N genes (by adj.P.Val)",
                min_value=0,
                max_value=30,
                value=10,
                key="tumor_volcano_labels",
            )
            highlight_genes = st.multiselect(
                "Highlight genes",
                options=gene_options,
                default=[],
                max_selections=40,
                key="tumor_volcano_highlight_genes",
            )
        with col2:
            highlight_tf = st.checkbox("Highlight transcription factors", key="tumor_volcano_tf")
            highlight_ligand = st.checkbox("Highlight ligands", key="tumor_volcano_ligand")
            highlight_receptor = st.checkbox("Highlight receptors", key="tumor_volcano_receptor")
            highlight_metabolism = st.checkbox("Highlight metabolism genes", key="tumor_volcano_metabolism")
            highlight_clinical_target = st.checkbox(
                "Highlight clinical targets", key="tumor_volcano_clinical"
            )
            download_as = download_format_select("tumor_volcano_download", formats=("png", "svg"))

    fig, config = create_volcano_plot(
        results,
        title=entry["name"],
        pval_threshold=pval_threshold,
        logfc_threshold=logfc_threshold,
        highlight_tf=highlight_tf,
        highlight_ligand=highlight_ligand,
        highlight_receptor=highlight_receptor,
        highlight_metabolism=highlight_metabolism,
        highlight_clinical_target=highlight_clinical_target,
        highlight_genes=highlight_genes or None,
        label_top_n=label_top_n,
        download_as=download_as,
    )
    add_activity(
        value=[selected_id, pval_threshold, logfc_threshold],
        analysis="Tumor Volcano Plot",
        user=st.session_state.session_id,
        time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    st.plotly_chart(fig, use_container_width=True, config=config)
    plot_summary_caption(
        f"{len(results)} genes",
        entry["name"],
        "dashed lines show visual thresholds",
    )

    st.markdown("---")
    st.subheader("Results table")
    display_volcano_results_table(results, key_prefix="tumor_volcano")

except Exception as exc:
    st.error(f"An error occurred: {exc}")
    with st.expander("Show full traceback"):
        st.code(traceback.format_exc(), language="python")
