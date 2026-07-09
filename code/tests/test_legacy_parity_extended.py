"""Extended content-vs-content parity checks against epitome_legacy.py."""

from __future__ import annotations

import ast
import re

import pytest

from legacy_parity_support import (
    CODE_DIR,
    DOWNLOAD_PAGES,
    DOWNLOADS_PARENT,
    LEGACY_LINES,
    LEGACY_TEXT,
    PAGES,
    _BUILTIN_CALLABLES,
    _dedent,
    bare_function_calls,
    count_calls,
    extract_display_strings,
    extract_kw_strings,
    extract_merged_string_literals,
    extract_positional_strings,
    extract_regex_matches,
    imported_names_from_source,
    legacy_source_for,
    local_function_defs,
    normalize_code,
    page_body,
    page_source,
)

# Legacy top-level tab labels (epitome_legacy.py ~596–609).
LEGACY_MAIN_NAV = [
    "Overview",
    "Transcriptome",
    "Chromatin",
    "Multimodal",
    "Automated Cell Typing",
    "Individual Datasets",
    "Downloads",
    "Curation",
    "Release Notes",
    "How to Cite",
    "Contact",
]

LEGACY_TRANSCRIPTOME_TABS = [
    "Expression Box Plots",
    "Expression UMAP",
    "Age Correlation",
    "Isoforms",
    "Dot Plots",
    "Cell Type Distribution",
    "Gene-Gene Relationships",
    "Ligand-Receptor Interactions",
]

LEGACY_CHROMATIN_TABS = [
    "Accessibility Distribution (Motifs/Enhancers)",
    "Motif Enrichment (ChromVAR)",
    "Cell Type Distribution",
]

LEGACY_DOWNLOAD_TABS = [
    "Dataset Files (h5ad) - RNA",
    "Dataset Files (h5ad) - ATAC",
    "Analysis Data Files",
    "Single-Cell Object Usage Guide",
]

LEGACY_DATASET_TABS = ["RNA datasets", "ATAC datasets"]

# Phrases that must not appear in migrated pages (dev / mistaken edits).
FORBIDDEN_PAGE_PHRASES = [
    "page_header(",
    "BOX_AND_WHISKER",
    "The plot combines",
    "Click the button below to begin transcriptome box plot analysis",
    "Click the button below to visualise individual datasets",
    "tab_start_button",
    "current_analysis_tab",
]

BOX_PLOT_HELP_MARKERS_EXPRESSION = [
    "**Box:** centre line = **median**",
    "**Whiskers:** extend to the **5th** and **95th** percentiles",
    "**Points:** each dot is one sample",
    "Log10-transformed counts per million",
]

BOX_PLOT_HELP_MARKERS_CHROMATIN = [
    "**Box:** centre line = **median**",
    "**Whiskers:** extend to the **5th** and **95th** percentiles",
    "**Points:** each dot is one sample",
]

CONTACT_URLS = [
    "https://forms.office.com/Pages/ResponsePage.aspx?id=FM9wg_MWFky4PHJAcWVDVtCPt0Xedb9ClGRxkEBa4fZUM1o5T01KTkVLQUFKWkFNTU5FVkRBRVoxVy4u&embed=true",
    "https://forms.office.com/Pages/ResponsePage.aspx?id=FM9wg_MWFky4PHJAcWVDVtCPt0Xedb9ClGRxkEBa4fZUNjlDOURVSTRYMUxHSkpIUDE5OVNUTk1SVS4u&embed=true",
]


def _paired_sources():
    for relpath, start, end, indent, skip_fn in PAGES:
        yield relpath, legacy_source_for(relpath), page_source(relpath, start, end, indent, skip_fn)
    for relpath, start, end, indent in DOWNLOAD_PAGES:
        legacy = "\n".join(DOWNLOADS_PARENT + _dedent(LEGACY_LINES[start - 1 : end], indent))
        yield relpath, legacy, page_body(CODE_DIR / relpath)


def test_no_lazy_load_remnants_in_app_pages():
    failures: list[str] = []
    for path in (CODE_DIR / "app_pages").rglob("*.py"):
        if "tumor" in path.parts:
            continue
        text = path.read_text()
        for phrase in FORBIDDEN_PAGE_PHRASES:
            if phrase in text:
                failures.append(f"{path.relative_to(CODE_DIR)}: contains {phrase!r}")
    assert not failures, "\n".join(failures)


def test_no_footer_rendered_in_page_files():
    for path in (CODE_DIR / "app_pages").rglob("*.py"):
        text = path.read_text()
        assert "render_footer(" not in text, f"{path.name} should not call render_footer directly"
        assert "render_page_footer(" not in text, f"{path.name} should not call render_page_footer directly"


def test_main_navigation_labels_match_legacy_tabs():
    shell = (CODE_DIR / "modules/site_shell.py").read_text()
    for label in LEGACY_MAIN_NAV:
        assert label in shell, f"Missing nav section/page label: {label}"


def test_transcriptome_page_titles_match_legacy_subtabs():
    shell = (CODE_DIR / "modules/site_shell.py").read_text()
    for title in LEGACY_TRANSCRIPTOME_TABS:
        assert f'title="{title}"' in shell


def test_chromatin_and_download_titles_match_legacy():
    shell = (CODE_DIR / "modules/site_shell.py").read_text()
    for title in LEGACY_CHROMATIN_TABS + LEGACY_DOWNLOAD_TABS + LEGACY_DATASET_TABS:
        assert f'title="{title}"' in shell


def test_widget_keys_match_legacy_per_page():
    failures: list[str] = []
    for relpath, legacy, page in _paired_sources():
        legacy_keys = extract_kw_strings(legacy, "selectbox", "key") | extract_kw_strings(
            legacy, "multiselect", "key"
        )
        page_keys = extract_kw_strings(page, "selectbox", "key") | extract_kw_strings(
            page, "multiselect", "key"
        )
        if legacy_keys != page_keys:
            failures.append(
                f"{relpath}: keys legacy-only={sorted(legacy_keys - page_keys)} "
                f"page-only={sorted(page_keys - legacy_keys)}"
            )
    assert not failures, "\n".join(failures)


def test_version_select_keys_match_legacy():
    failures: list[str] = []
    for relpath, legacy, page in _paired_sources():
        legacy_keys = extract_regex_matches(legacy, r'key="(version_select[^"]+)"')
        page_keys = extract_regex_matches(page, r'key="(version_select[^"]+)"')
        if legacy_keys != page_keys:
            failures.append(f"{relpath}: {legacy_keys} vs {page_keys}")
    assert not failures, "\n".join(failures)


def test_st_headers_match_legacy_per_page():
    failures: list[str] = []
    for relpath, legacy, page in _paired_sources():
        legacy_h = extract_positional_strings(legacy, "header")
        page_h = extract_positional_strings(page, "header")
        if legacy_h != page_h:
            failures.append(f"{relpath}: headers differ")
    assert not failures, "\n".join(failures)


def test_expander_titles_match_legacy():
    failures: list[str] = []
    for relpath, legacy, page in _paired_sources():
        legacy_e = extract_positional_strings(legacy, "expander")
        page_e = extract_positional_strings(page, "expander")
        if legacy_e != page_e:
            failures.append(f"{relpath}: expanders {legacy_e} vs {page_e}")
    assert not failures, "\n".join(failures)


def test_spinner_messages_match_legacy():
    failures: list[str] = []
    for relpath, legacy, page in _paired_sources():
        legacy_s = extract_positional_strings(legacy, "spinner", min_len=5)
        page_s = extract_positional_strings(page, "spinner", min_len=5)
        if legacy_s != page_s:
            failures.append(f"{relpath}: spinners differ")
    assert not failures, "\n".join(failures)


def test_error_message_strings_match_legacy():
    failures: list[str] = []
    for relpath, legacy, page in _paired_sources():
        legacy_e = extract_positional_strings(legacy, "error", min_len=15)
        page_e = extract_positional_strings(page, "error", min_len=15)
        # Ignore f-string-only errors (dynamic); compare static literals.
        legacy_static = {s for s in legacy_e if not s.startswith("Error loading") or "{" not in s}
        page_static = {s for s in page_e if not s.startswith("Error loading") or "{" not in s}
        if legacy_static != page_static:
            failures.append(f"{relpath}: static errors differ")
    assert not failures, "\n".join(failures)


def test_add_activity_analysis_names_match_legacy():
    failures: list[str] = []
    for relpath, legacy, page in _paired_sources():
        legacy_a = extract_regex_matches(legacy, r'analysis="([^"]+)"')
        page_a = extract_regex_matches(page, r'analysis="([^"]+)"')
        if legacy_a != page_a:
            failures.append(
                f"{relpath}: analysis names legacy-only={legacy_a - page_a} page-only={page_a - legacy_a}"
            )
    assert not failures, "\n".join(failures)


def test_plot_helper_call_counts_match_legacy():
    plot_fns = [
        "create_expression_plot",
        "create_gene_umap_plot",
        "create_age_correlation_plot",
        "create_isoform_plot",
        "create_dotplot",
        "create_ligand_receptor_plot",
        "create_proportion_plot",
        "create_gene_correlation_plot",
        "create_accessibility_plot",
        "create_genome_browser_plot",
        "create_chromvar_plot",
        "plot_heatmap",
        "plot_sc_dataset",
        "create_cell_type_annotation_ui",
    ]
    failures: list[str] = []
    for relpath, legacy, page in _paired_sources():
        for fn in plot_fns:
            lc = count_calls(legacy, fn)
            pc = count_calls(page, fn)
            if lc != pc:
                failures.append(f"{relpath}: {fn} legacy={lc} page={pc}")
    assert not failures, "\n".join(failures)


def test_cached_loader_calls_match_legacy_per_page():
    failures: list[str] = []
    for relpath, legacy, page in _paired_sources():
        legacy_l = set(extract_regex_matches(legacy, r"\b(load_cached_\w+)\s*\("))
        page_l = set(extract_regex_matches(page, r"\b(load_cached_\w+)\s*\("))
        if legacy_l != page_l:
            failures.append(f"{relpath}: loaders {legacy_l} vs {page_l}")
    assert not failures, "\n".join(failures)


def test_try_except_structure_count_match_legacy():
    failures: list[str] = []
    for relpath, legacy, page in _paired_sources():
        legacy_try = legacy.count("try:")
        page_try = page.count("try:")
        if legacy_try != page_try:
            failures.append(f"{relpath}: try blocks legacy={legacy_try} page={page_try}")
    assert not failures, "\n".join(failures)


def test_boxplot_help_text_present_on_legacy_pages():
    specs = [
        (
            "app_pages/transcriptome/expression_box_plots.py",
            1072,
            1313,
            24,
            None,
            BOX_PLOT_HELP_MARKERS_EXPRESSION,
        ),
        (
            "app_pages/chromatin/accessibility_distribution.py",
            3320,
            3701,
            24,
            "render_genome_browser",
            BOX_PLOT_HELP_MARKERS_CHROMATIN,
        ),
        (
            "app_pages/chromatin/motif_enrichment_chromvar.py",
            3714,
            3953,
            24,
            None,
            BOX_PLOT_HELP_MARKERS_CHROMATIN,
        ),
    ]
    for relpath, start, end, indent, skip_fn, markers in specs:
        if skip_fn:
            src = (CODE_DIR / relpath).read_text()
        else:
            src = page_source(relpath, start, end, indent, skip_fn)
        for marker in markers:
            assert marker in src, f"{relpath} missing box-plot help marker: {marker}"


def test_overview_counter_labels_match_legacy():
    legacy = legacy_source_for("app_pages/overview/overview.py")
    page = page_source("app_pages/overview/overview.py", 615, 1030, 16, None)
    labels = [
        "Transcriptome Samples",
        "Chromatin Samples",
        "Publications",
        "Total Cells",
    ]
    for label in labels:
        assert label in legacy
        assert label in page


def test_contact_form_urls_match_legacy():
    legacy = legacy_source_for("app_pages/contact/contact.py")
    page = page_source("app_pages/contact/contact.py", 5778, 5837, 12, None)
    for url in CONTACT_URLS:
        assert url in legacy
        assert url in page


def test_citation_page_bibliography_count_matches_legacy():
    legacy = legacy_source_for("app_pages/citation/how_to_cite.py")
    page = page_source("app_pages/citation/how_to_cite.py", 5677, 5775, 12, None)
    legacy_refs = len(re.findall(r"^\s*\d+\.", legacy, re.MULTILINE))
    page_refs = len(re.findall(r"^\s*\d+\.", page, re.MULTILINE))
    assert legacy_refs == page_refs
    assert page_refs >= 39


def test_release_notes_mentions_cell_reports_and_biorxiv():
    page = page_source("app_pages/release_notes/release_notes.py", 5627, 5674, 12, None)
    assert "publication in Cell Reports" in page
    assert "pre-print on bioRxiv" in page
    assert "v_0.02:" in page
    assert "v_0.01:" in page


def test_cell_typing_page_is_legacy_minimal():
    body = normalize_code(page_body(CODE_DIR / "app_pages/cell_typing/automated_cell_typing.py"))
    assert body.strip() == "create_cell_type_annotation_ui()"
    legacy_body = normalize_code("create_cell_type_annotation_ui()")
    assert body == legacy_body


def test_genome_browser_fragment_strings_in_accessibility_page():
    fragment = "\n".join(LEGACY_LINES[478:534])
    page = (CODE_DIR / "app_pages/chromatin/accessibility_distribution.py").read_text()
    for snippet in [
        "Select Motifs to Display (max 10)",
        "Only those motifs are shown that fall in a given consensus peak.",
        "Show enhancers",
    ]:
        assert snippet in fragment
        assert snippet in page


def test_merged_string_literals_match_legacy_per_page():
    """Merged literals (incl. split help text and multi-part st.info) match legacy tab bodies."""
    failures: list[str] = []
    for relpath, legacy, page in _paired_sources():
        legacy_strings = extract_merged_string_literals(legacy)
        page_strings = extract_merged_string_literals(page)
        only_page = page_strings - legacy_strings
        only_legacy = legacy_strings - page_strings
        if only_page or only_legacy:
            failures.append(
                f"{relpath}: page-only={len(only_page)} legacy-only={len(only_legacy)}"
            )
    assert not failures, "\n".join(failures)


def test_every_bare_call_in_page_body_is_imported():
    """AST check: each bare function call in a page file must be imported or defined locally."""
    failures: list[str] = []
    for path in sorted((CODE_DIR / "app_pages").rglob("*.py")):
        if "tumor" in path.parts:
            continue
        full = path.read_text()
        imported = imported_names_from_source(full)
        local = local_function_defs(full)
        for fn in bare_function_calls(full):
            if fn in _BUILTIN_CALLABLES:
                continue
            if fn not in imported and fn not in local:
                failures.append(f"{path.relative_to(CODE_DIR)}: {fn}")
    assert not failures, "Missing imports for bare calls:\n" + "\n".join(failures)


def test_page_bare_calls_match_legacy_tab_bodies():
    """Bare function calls in each page body must match its legacy tab body exactly."""
    failures: list[str] = []
    for relpath, legacy, page in _paired_sources():
        legacy_calls = bare_function_calls(legacy)
        page_calls = bare_function_calls(page)
        if legacy_calls != page_calls:
            failures.append(
                f"{relpath}: legacy-only={sorted(legacy_calls - page_calls)} "
                f"page-only={sorted(page_calls - legacy_calls)}"
            )
    assert not failures, "\n".join(failures)


def test_legacy_module_symbols_used_in_pages_are_imported():
    """Any symbol imported from modules.* in legacy and called in a page must be imported there."""
    legacy_module_symbols: set[str] = set()
    for node in ast.walk(ast.parse(LEGACY_TEXT)):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("modules"):
            for alias in node.names:
                legacy_module_symbols.add(alias.name)

    failures: list[str] = []
    for path in sorted((CODE_DIR / "app_pages").rglob("*.py")):
        if "tumor" in path.parts:
            continue
        full = path.read_text()
        imported = imported_names_from_source(full)
        local = local_function_defs(full)
        for fn in bare_function_calls(full):
            if fn in legacy_module_symbols and fn not in imported and fn not in local:
                failures.append(f"{path.relative_to(CODE_DIR)}: {fn}")
    assert not failures, "Legacy module symbols not imported:\n" + "\n".join(failures)


def test_display_tables_functions_used_match_legacy():
    display_fns = [
        "display_marker_table",
        "display_aging_genes_table",
        "display_curation_table",
        "display_ligand_receptor_table",
        "display_enrichment_table",
        "display_sex_dimorphism_table",
        "display_enhancers_table",
    ]
    legacy_total = {fn: 0 for fn in display_fns}
    page_total = {fn: 0 for fn in display_fns}
    for relpath, legacy, page in _paired_sources():
        for fn in display_fns:
            legacy_total[fn] += count_calls(legacy, fn)
            page_total[fn] += count_calls(page, fn)
    assert page_total == legacy_total


def test_session_state_defaults_match_legacy():
    legacy_defaults = {
        "selected_gene": "Sox2",
        "selected_region": "chr3:34650405-34652461",
    }
    shell = (CODE_DIR / "modules/site_shell.py").read_text()
    epitome = (CODE_DIR / "epitome.py").read_text()
    for key, val in legacy_defaults.items():
        assert f'"{key}"' in shell or f'"{key}"' in epitome
        assert val in shell or val in LEGACY_TEXT


def _page_src_for(relpath: str) -> str:
    for start_spec in PAGES:
        if start_spec[0] == relpath:
            return page_source(*start_spec)
    return page_body(CODE_DIR / relpath)


@pytest.mark.parametrize(
    "relpath",
    [p[0] for p in PAGES] + [p[0] for p in DOWNLOAD_PAGES],
)
def test_page_line_count_within_legacy_bounds(relpath: str):
    """Page bodies should not be dramatically shorter/longer than legacy (sanity guard)."""
    legacy = legacy_source_for(relpath)
    page = _page_src_for(relpath)
    legacy_lines = len(normalize_code(legacy).splitlines())
    page_lines = len(normalize_code(page).splitlines())
    ratio = page_lines / max(legacy_lines, 1)
    assert 0.85 <= ratio <= 1.15, f"{relpath}: line ratio {ratio:.2f} ({page_lines}/{legacy_lines})"
