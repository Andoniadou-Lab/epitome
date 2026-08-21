import traceback

import pandas as pd
import streamlit as st

from config import Config
from modules.pta.data_loader import load_pta_bulk_curation, load_pta_scrna_curation
from modules.pta.page_layout import pta_page_header
from modules.pta.stats import pta_rna_stats_path
from modules.utils import create_cell_type_stats_display

BASE_PATH = Config.BASE_PATH
ACCENT = "#cc0000"

BULK_CENSUS_GROUPS = {
    "Lineage": "Lineage_pta",
    "Subtype": "Subtype_pta",
    "Cell type": "Cell_type_pta",
}
BULK_CENSUS_CROSSTABS = {
    "None": None,
    "Invasion": ("Invasion_pta",),
    "Ki-67": ("KI67_pta", "Ki67_pta"),
    "GNAS genotype": ("GNAS_geno_pta",),
    "USP8 genotype": ("USP8_geno_pta",),
}


def _metric_card(title: str, value: str) -> None:
    st.markdown(
        f"""
        <div style="text-align:center;padding:20px;background:#f8f9fa;border-radius:10px;">
            <h3 style="color:#666;font-size:20px;">{title}</h3>
            <div style="font-size:40px;font-weight:bold;color:{ACCENT};">{value}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def _label_series(series: pd.Series) -> pd.Series:
    labelled = series.astype("string").fillna("Unknown").str.strip()
    return labelled.replace({"": "Unknown", "<NA>": "Unknown", "nan": "Unknown", "None": "Unknown"})


def _resolve_column(df: pd.DataFrame, *candidates: str) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _bulk_value_counts(df: pd.DataFrame, column: str) -> pd.DataFrame:
    counts = _label_series(df[column]).value_counts()
    return counts.rename_axis(column).reset_index(name="Samples")


def _bulk_crosstab(df: pd.DataFrame, row_col: str, col_col: str) -> pd.DataFrame:
    table = pd.crosstab(
        _label_series(df[row_col]),
        _label_series(df[col_col]),
        margins=True,
        margins_name="Total",
    )
    body = table.drop(index="Total", errors="ignore")
    if "Total" in table.columns and not body.empty:
        order = body["Total"].sort_values(ascending=False).index.tolist() + ["Total"]
        table = table.loc[order]
    return table


selected_version = pta_page_header(
    "Overview",
    "Summary of human pituitary tumour atlas data — single-cell RNA-seq, bulk RNA-seq, and pseudobulk profiles.",
    "version_select_tumor_overview",
)

try:
    curation = load_pta_scrna_curation(version=selected_version)
    bulk = load_pta_bulk_curation(version=selected_version)
    rna_samples = len(curation[curation["Modality"].isin(["sn", "sc", "multi_rna"])])
    unique_papers = curation["Author"].nunique()
    total_cells = int(pd.to_numeric(curation["n_cells"], errors="coerce").fillna(0).sum())
    bulk_samples = len(bulk)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        _metric_card("scRNA-seq Samples", f"{rna_samples:,}")
    with col2:
        _metric_card("Bulk RNA-seq Samples", f"{bulk_samples:,}")
    with col3:
        _metric_card("Publications", f"{unique_papers:,}")
    with col4:
        _metric_card("Total Cells (scRNA)", f"{total_cells:,}")

    st.markdown("---")
    st.subheader("Modality breakdown")
    modality_counts = (
        curation["Modality"].value_counts().rename_axis("Modality").reset_index(name="Samples")
    )
    st.dataframe(modality_counts, use_container_width=True, hide_index=True)

    if "Tumor_pta" in curation.columns:
        st.subheader("Tumour type breakdown")
        tumor_counts = (
            curation["Tumor_pta"].value_counts().head(15).rename_axis("Tumour type").reset_index(name="Samples")
        )
        st.dataframe(tumor_counts, use_container_width=True, hide_index=True)

    stats_df = pd.read_parquet(pta_rna_stats_path(selected_version))
    cell_types_no_other = [
        c for c in stats_df.columns if c != "dataset" and c.strip().lower() != "other"
    ]
    create_cell_type_stats_display(
        version=selected_version,
        display_title="Total Cells by Cell Type (scRNA)",
        column_count=4,
        atac_rna="rna",
        cell_types=cell_types_no_other,
        rna_stats_path=pta_rna_stats_path(selected_version),
    )

    st.markdown("---")
    st.subheader("Bulk RNA-seq census")
    st.caption(
        "Sample counts by tumour classification, optionally cross-tabulated against invasion, "
        "Ki-67, GNAS genotype, or USP8 genotype."
    )
    available_groups = {
        label: col for label, col in BULK_CENSUS_GROUPS.items() if col in bulk.columns
    }
    if not available_groups:
        st.info("Bulk metadata does not include Lineage, Subtype, or Cell type columns.")
    else:
        group_col, cross_col = st.columns(2)
        with group_col:
            group_label = st.selectbox(
                "Count samples by",
                options=list(available_groups),
                key="tumor_overview_bulk_group",
            )
        with cross_col:
            available_crosstabs = ["None"]
            resolved_crosstabs: dict[str, str] = {}
            for label, candidates in BULK_CENSUS_CROSSTABS.items():
                if candidates is None:
                    continue
                resolved = _resolve_column(bulk, *candidates)
                if resolved is not None:
                    available_crosstabs.append(label)
                    resolved_crosstabs[label] = resolved
            cross_label = st.selectbox(
                "Cross-tab against",
                options=available_crosstabs,
                key="tumor_overview_bulk_crosstab",
            )

        group_column = available_groups[group_label]
        counts = _bulk_value_counts(bulk, group_column)
        st.dataframe(counts, use_container_width=True, hide_index=True)

        if cross_label != "None":
            cross_column = resolved_crosstabs[cross_label]
            st.markdown(f"**{group_label} × {cross_label}**")
            crosstab = _bulk_crosstab(bulk, group_column, cross_column)
            st.dataframe(crosstab, use_container_width=True)

    st.markdown("### Data included in this release")
    st.markdown(
        f"- **Single-cell RNA-seq**: {rna_samples:,} tumour samples across "
        f"{unique_papers:,} studies, with dot plots, cell-type abundance, and individual dataset UMAPs.\n"
        f"- **Bulk RNA-seq**: {bulk_samples:,} tumour samples with expression boxplots and heatmaps.\n"
        "- **Pseudobulk**: inferred cell-cluster profiles from integrated scRNA-seq."
    )

except FileNotFoundError as exc:
    st.error(f"Tumour atlas overview data not found for `{selected_version}`.")
    st.code(str(exc))
except Exception as exc:
    st.error(f"An error occurred: {exc}")
    with st.expander("Show full traceback"):
        st.code(traceback.format_exc(), language="python")
