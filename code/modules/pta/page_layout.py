"""Shared layout helpers for tumour atlas pages."""

import pandas as pd
import streamlit as st

from modules.pta.cell_type_labels import sort_pta_categories
from modules.pta.config import PtaConfig, list_pta_versions


def pta_page_header(title: str, subtitle: str, version_key: str) -> str:
    col1, col2 = st.columns([5, 1])
    with col1:
        st.header(title)
        st.markdown(subtitle)
    with col2:
        versions = list_pta_versions()
        if not versions:
            st.caption("No PTA data")
            return "v_0.04"
        return st.selectbox(
            "Version",
            options=versions,
            key=version_key,
            label_visibility="collapsed",
        )


def pta_bulk_cohort_toggle(key: str) -> str:
    """Three-way cohort selector for bulk boxplot / heatmap pages."""
    options = list(PtaConfig.BULK_COHORTS)
    return st.radio(
        "Cohort",
        options=options,
        format_func=lambda cohort: PtaConfig.BULK_COHORTS[cohort]["label"],
        horizontal=True,
        key=key,
        help=(
            "Main cohort is the integrated bulk RNA-seq collection (~1,700 samples). "
            "Zhang et al., 2022 and Jotanovic et al., 2024 are independent TPM validation cohorts."
        ),
    )


def bulk_expression_unit(cohort: str) -> str:
    return PtaConfig.BULK_COHORTS.get(cohort, PtaConfig.BULK_COHORTS["main_cohort"])["unit"]


def bulk_matrix_caption(cohort: str, matrix_id: str) -> str:
    spec = PtaConfig.BULK_COHORTS.get(cohort, PtaConfig.BULK_COHORTS["main_cohort"])
    if cohort == "main_cohort":
        if matrix_id == "shared":
            return "shared-gene matrix (all datasets)"
        return "just-aligned matrix (raw-aligned datasets only)"
    return f"{spec['label']} validation cohort ({spec['unit']})"


def grouping_category_multiselect(
    meta: pd.DataFrame,
    columns,
    *,
    key_prefix: str,
    merge_mixed: bool = True,
    key_suffix: str = "",
) -> dict[str, list]:
    """Author-style multiselects for X-axis grouping levels (all selected by default)."""
    cols = []
    for col in columns:
        if not col or col in {"None", "(none)"}:
            continue
        if col in meta.columns and col not in cols:
            cols.append(col)
    selected: dict[str, list] = {}
    if not cols:
        return selected
    boxes = st.columns(len(cols))
    suffix = f"_{key_suffix}" if key_suffix else ""
    for box, col in zip(boxes, cols):
        options = sort_pta_categories(
            meta[col].unique(), col, merge_mixed=merge_mixed
        )
        with box:
            selected[col] = st.multiselect(
                f"{col} ({len(options)})",
                options=options,
                default=options,
                key=f"{key_prefix}_{col}_{int(bool(merge_mixed))}{suffix}",
                help="Choose which categories to show on this grouping. All are included by default.",
            )
    return selected
