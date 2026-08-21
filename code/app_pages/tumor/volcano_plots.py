import traceback
from datetime import datetime

import streamlit as st

from modules.analytics import add_activity
from modules.display_tables import display_volcano_results_table
from modules.pta.data_loader import (
    flatten_volcano_comparisons,
    flatten_volcano_markers,
    load_volcano_manifest,
    load_volcano_results,
)
from modules.pta.page_layout import pta_page_header
from modules.pta.volcano import create_volcano_plot
from modules.ui.plot_settings import download_format_select, plot_settings_panel
from modules.ui.plot_summary import plot_summary_caption

selected_version = pta_page_header(
    "Volcano Plots",
    "Visualise bulk tumour differential-expression results. "
    "Pairwise plots show all genes; one-group marker plots show the significant "
    "marker genes only. Significance thresholds affect colouring only.",
    "version_select_tumor_volcano",
)

try:
    families = load_volcano_manifest(version=selected_version)
except FileNotFoundError as exc:
    st.error(
        "Volcano manifest not found. Expected "
        f"`pta_data/epitome_volcanos/{selected_version}/volcanos.json`."
    )
    st.code(str(exc))
    st.stop()

if not families or not (
    flatten_volcano_comparisons(families) or flatten_volcano_markers(families)
):
    st.warning("No comparisons defined for this version.")
    st.stop()

family_ids = [family["id"] for family in families]
family_labels = {family["id"]: family["name"] for family in families}
selected_family_id = st.selectbox(
    "Comparison family",
    options=family_ids,
    format_func=lambda fid: family_labels[fid],
    key="tumor_volcano_family",
)
family = next(item for item in families if item["id"] == selected_family_id)
if family.get("description"):
    st.caption(family["description"])

has_markers = bool(family.get("markers"))
has_contrasts = bool(family.get("comparisons"))
plot_kind_options = []
if has_contrasts:
    plot_kind_options.append("Pairwise comparison")
if has_markers:
    plot_kind_options.append("One-group markers")
plot_kind = st.radio(
    "Plot type",
    options=plot_kind_options,
    horizontal=True,
    key=f"tumor_volcano_plot_kind_{selected_family_id}",
)

entries = (
    family.get("markers") or []
    if plot_kind == "One-group markers"
    else family.get("comparisons") or []
)
if not entries:
    st.warning("No plots of this type in this family.")
    st.stop()

comp_labels = {c["id"]: c["name"] for c in entries}
selected_id = st.selectbox(
    "Markers" if plot_kind == "One-group markers" else "Comparison",
    options=list(comp_labels.keys()),
    format_func=lambda cid: comp_labels[cid],
    key=f"tumor_volcano_comparison_{selected_family_id}_{plot_kind}",
)
entry = next(c for c in entries if c["id"] == selected_id)

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
        value=[selected_family_id, selected_id, pval_threshold, logfc_threshold],
        analysis="Tumor Volcano Plot",
        user=st.session_state.session_id,
        time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    st.plotly_chart(fig, use_container_width=True, config=config)
    n_shared = int((results["source"] == "shared").sum()) if "source" in results.columns else None
    n_aligned = (
        int((results["source"] == "just_aligned").sum()) if "source" in results.columns else None
    )
    extra = None
    if plot_kind == "One-group markers":
        extra = "marker genes only (not the full transcriptome)"
    elif n_shared is not None and n_aligned is not None:
        extra = f"{n_shared:,} shared-universe genes + {n_aligned:,} just-aligned-only"
    plot_summary_caption(
        f"{len(results)} genes",
        entry["name"],
        extra,
        "dashed lines show visual thresholds",
        version=selected_version,
        loader_keys=("pta_volcano_manifest", "pta_volcano_results"),
    )

    st.markdown("---")
    st.subheader("Results table")
    display_volcano_results_table(
        results,
        key_prefix="tumor_volcano",
        version=selected_version,
        loader_keys=("pta_volcano_manifest", "pta_volcano_results"),
    )

except Exception as exc:
    st.error(f"An error occurred: {exc}")
    with st.expander("Show full traceback"):
        st.code(traceback.format_exc(), language="python")
