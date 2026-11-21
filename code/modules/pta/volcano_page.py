"""Volcano page body shared by the bulk and single-cell volcano tabs."""

import traceback
from datetime import datetime

import streamlit as st

from modules.analytics import add_activity
from modules.display_tables import display_volcano_results_table
from modules.pta.config import PtaConfig
from modules.pta.data_loader import (
    flatten_volcano_comparisons,
    flatten_volcano_markers,
    load_volcano_manifest,
    load_volcano_results,
    volcano_loader_key,
)
from modules.pta.page_layout import pta_page_header
from modules.pta.volcano import (
    create_volcano_plot,
    volcano_data,
    volcano_publication_figure,
    volcano_title,
)
from modules.publication_export import plot_downloads
from modules.ui.plot_settings import download_format_select, plot_settings_panel
from modules.ui.plot_summary import plot_summary_caption


def render_volcano_page(
    *,
    title: str,
    description: str,
    collection: str,
    key_prefix: str,
    version_key: str,
    analysis_name: str,
) -> None:
    selected_version = pta_page_header(title, description, version_key)
    loader_keys = (
        volcano_loader_key("pta_volcano_manifest", collection),
        volcano_loader_key("pta_volcano_results", collection),
    )

    try:
        families = load_volcano_manifest(version=selected_version, collection=collection)
    except FileNotFoundError as exc:
        folder = PtaConfig.VOLCANO_COLLECTION_DIRS[collection]
        st.error(
            "Volcano manifest not found. Expected "
            f"`pta_data/{folder}/{selected_version}/volcanos.json`."
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
        key=f"{key_prefix}_family",
    )
    family = next(item for item in families if item["id"] == selected_family_id)
    if family.get("description"):
        st.caption(family["description"])

    plot_kind_options = []
    if family.get("comparisons"):
        plot_kind_options.append("Pairwise comparison")
    if family.get("markers"):
        plot_kind_options.append("One-group markers")
    plot_kind = st.radio(
        "Plot type",
        options=plot_kind_options,
        horizontal=True,
        key=f"{key_prefix}_plot_kind_{selected_family_id}",
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
        key=f"{key_prefix}_comparison_{selected_family_id}_{plot_kind}",
    )
    entry = next(c for c in entries if c["id"] == selected_id)

    st.markdown(
        f"**{entry['group_a']}** vs **{entry['group_b']}**  \n"
        f"{entry['explanation']}"
    )

    try:
        results = load_volcano_results(selected_version, selected_id, collection=collection)
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
                    key=f"{key_prefix}_pval",
                )
                logfc_threshold = st.number_input(
                    "|logFC| threshold (visual)",
                    min_value=0.0,
                    max_value=10.0,
                    value=1.0,
                    step=0.1,
                    key=f"{key_prefix}_logfc",
                )
                label_top_n = st.slider(
                    "Label top N genes (by adj.P.Val)",
                    min_value=0,
                    max_value=30,
                    value=10,
                    key=f"{key_prefix}_labels",
                )
                highlight_genes = st.multiselect(
                    "Highlight genes",
                    options=gene_options,
                    default=[],
                    max_selections=40,
                    key=f"{key_prefix}_highlight_genes",
                )
            with col2:
                highlight_tf = st.checkbox("Highlight transcription factors", key=f"{key_prefix}_tf")
                highlight_ligand = st.checkbox("Highlight ligands", key=f"{key_prefix}_ligand")
                highlight_receptor = st.checkbox("Highlight receptors", key=f"{key_prefix}_receptor")
                highlight_metabolism = st.checkbox(
                    "Highlight metabolism genes", key=f"{key_prefix}_metabolism"
                )
                highlight_clinical_target = st.checkbox(
                    "Highlight clinical targets", key=f"{key_prefix}_clinical"
                )
                download_as = download_format_select(f"{key_prefix}_download", formats=("png", "svg"))

        fig, config = create_volcano_plot(
            results,
            title=volcano_title(entry["group_a"], entry["group_b"]),
            group_a=entry["group_a"],
            group_b=entry["group_b"],
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
            analysis=analysis_name,
            user=st.session_state.session_id,
            time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        st.plotly_chart(fig, width="stretch", config=config)
        export_fig, export_w, export_h = volcano_publication_figure(fig)
        plot_downloads(
            export_fig, export_w, export_h,
            filename=f"volcano_{selected_id}",
            download_as=download_as,
            data=volcano_data(fig),
            key=f"{key_prefix}_{selected_id}",
        )
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
            loader_keys=loader_keys,
        )

        st.markdown("---")
        st.subheader("Results table")
        display_volcano_results_table(
            results,
            key_prefix=key_prefix,
            version=selected_version,
            loader_keys=loader_keys,
        )

    except Exception as exc:
        st.error(f"An error occurred: {exc}")
        with st.expander("Show full traceback"):
            st.code(traceback.format_exc(), language="python")
