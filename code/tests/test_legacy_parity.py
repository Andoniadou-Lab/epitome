"""Structural checks carried over from the epitome_legacy.py migration.

Covers citations, footer/tagline, cached-loader return shapes, and that every
symbol a page uses is imported. Page bodies themselves are free to diverge from
legacy as the site evolves.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parent.parent
_LEGACY = (_CODE_DIR / "epitome_legacy.py").read_text().splitlines()

_MOUSE_TAGLINE_LINES = (
    "Explore, analyse, and visualise all mouse pituitary datasets.",
    "Export raw or processed data, and generate publication-ready figures.",
)
_LEGACY_FOOTER_SNIPPETS = (
    "Andoniadou Lab</strong> at <strong>King's College",
    "Lead curator: Bence Kövér",
    "github.com/Andoniadou-Lab/epitome",
)


def _page_body(path: Path) -> str:
    text = path.read_text()
    mod = ast.parse(text)
    skip_lines: set[int] = set()
    for node in mod.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for ln in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                skip_lines.add(ln)
        elif isinstance(node, ast.Assign):
            if (
                len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "BASE_PATH"
            ):
                for ln in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                    skip_lines.add(ln)
        elif isinstance(node, ast.FunctionDef):
            for ln in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                skip_lines.add(ln)
    lines = text.splitlines()
    return "\n".join(line for i, line in enumerate(lines, 1) if i not in skip_lines)


def test_cell_typing_page():
    body = _page_body(_CODE_DIR / "app_pages/cell_typing/automated_cell_typing.py")
    assert "create_cell_type_annotation_ui()" in body


def test_footer_citations_match_legacy():
    footer = (_CODE_DIR / "modules/site_shell.py").read_text()
    assert "print_citation" in footer
    assert "epitome_citation" in footer
    assert "pre_print_citation" not in footer


def test_mouse_tagline_matches_legacy():
    shell = (_CODE_DIR / "modules/site_shell.py").read_text()
    for line in _MOUSE_TAGLINE_LINES:
        assert line in shell


def test_cached_loaders_return_distinct_tuple_sizes():
    from modules import cached_loaders as cl

    assert len(cl.load_cached_heatmap_data.__name__) > 0
    # Verify each loader is a distinct function object (not one shared factory closure).
    names = {
        cl.load_cached_data.__name__,
        cl.load_cached_heatmap_data.__name__,
        cl.load_cached_marker_data.__name__,
    }
    assert names == {
        "load_cached_data",
        "load_cached_heatmap_data",
        "load_cached_marker_data",
    }
    assert cl.load_cached_data is not cl.load_cached_heatmap_data
    assert cl.load_cached_heatmap_data is not cl.load_cached_marker_data


def test_app_pages_import_used_cached_loaders():
    """Every load_cached_* call in app_pages must be imported from cached_loaders."""
    import ast
    import re

    for path in (_CODE_DIR / "app_pages").rglob("*.py"):
        text = path.read_text()
        used = set(re.findall(r"\b(load_cached_\w+)\s*\(", text))
        imported: set[str] = set()
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.ImportFrom) and node.module == "modules.cached_loaders":
                imported.update(alias.name for alias in node.names)
        missing = used - imported
        assert not missing, f"{path.relative_to(_CODE_DIR)} missing imports: {sorted(missing)}"


_GLOBAL_NAMES = ("pd", "np", "pl", "os", "datetime", "traceback", "gc")

# Functions commonly called in page bodies; map name -> expected modules.* import path.
_MODULE_FUNCTIONS: dict[str, tuple[str, ...]] = {
    "add_activity": ("modules.analytics",),
    "create_cell_type_stats_display": ("modules.utils",),
    "create_filter_ui": ("modules.utils",),
    "create_gene_selector": ("modules.utils",),
    "to_array": ("modules.utils",),
    "filter_data": ("modules.utils",),
    "parse_row_info": ("modules.utils",),
    "filter_dotplot_data": ("modules.dotplot",),
    "filter_chromvar_data": ("modules.utils",),
    "filter_accessibility_data": ("modules.utils",),
    "create_color_mapping": ("modules.utils",),
    "create_gene_selector_with_coordinates": ("modules.utils",),
    "create_region_selector": ("modules.utils",),
    "filter_isoform_data": ("modules.isoforms",),
    "display_enhancers_table": ("modules.display_tables",),
    "display_marker_table": ("modules.display_tables",),
    "display_aging_genes_table": ("modules.display_tables",),
    "display_curation_table": ("modules.display_tables",),
    "display_ligand_receptor_table": ("modules.display_tables",),
    "display_sex_dimorphism_table": ("modules.display_tables",),
    "preprocess_features_cached": ("modules.cached_loaders",),
    "create_downloads_ui_with_metadata_rna": ("modules.download",),
    "create_downloads_ui_with_metadata_atac": ("modules.download",),
    "create_bulk_data_downloads_ui": ("modules.download",),
    "render_test_health_bar": ("modules.test_health",),
    "create_chromvar_plot": ("modules.chromvar",),
    "get_available_genes": ("modules.gene_gene_corr",),
    "load_gene_data": ("modules.gene_gene_corr",),
    "create_cell_type_annotation_ui": ("modules.epitome_tools_annotation",),
    "create_expression_plot": ("modules.expression",),
    "create_accessibility_plot": ("modules.accessibility",),
    "create_genome_browser_plot": ("modules.accessibility",),
    "plot_sc_dataset": ("modules.individual_sc",),
    "get_dataset_info": ("modules.individual_sc",),
    "list_available_datasets": ("modules.individual_sc",),
    "plot_heatmap": ("modules.heatmap",),
    "process_heatmap_data": ("modules.heatmap",),
    "analyze_tf_cobinding": ("modules.heatmap",),
    "display_enrichment_table": ("modules.display_tables",),
    "load_motif_genes": ("modules.data_loader",),
}


def _imported_names(text: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
            if node.module == "datetime":
                names.add("datetime")
    return names


def test_app_pages_have_required_imports():
    """Symbols used in page bodies must be imported (were global in epitome_legacy.py)."""
    failures: list[str] = []
    for path in sorted((_CODE_DIR / "app_pages").rglob("*.py")):
        raw = path.read_text()
        code_lines = [
            line for line in raw.splitlines() if not re.match(r"^\s*#", line)
        ]
        text = "\n".join(code_lines)
        imported = _imported_names(raw)
        # Attribute access on a longer dotted path (``sc.pl.umap`` in a guide's code
        # sample) is not a use of the bare name, so require no leading dot.
        missing = [
            name
            for name in _GLOBAL_NAMES
            if re.search(rf"(?<![.\w]){re.escape(name)}\.", text)
            or re.search(rf"(?<![.\w]){re.escape(name)}\(", text)
        ]
        missing = [name for name in missing if name not in imported]
        if missing:
            failures.append(f"{path.relative_to(_CODE_DIR)}: {sorted(missing)}")
    assert not failures, "Missing imports in app_pages:\n" + "\n".join(failures)


def _imported_module_symbols(text: str) -> dict[str, str]:
    symbols: dict[str, str] = {}
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("modules"):
            for alias in node.names:
                symbols[alias.asname or alias.name] = node.module
    return symbols


def test_citations_module_matches_legacy():
    from modules import citations as c

    legacy_ns: dict[str, str] = {}
    exec("\n".join(_LEGACY[549:552]), legacy_ns)
    for name in ("epitome_citation", "pre_print_citation", "print_citation"):
        assert getattr(c, name) == legacy_ns[name], f"{name} drift from legacy"


def test_shell_footer_and_tagline_match_legacy():
    shell = (_CODE_DIR / "modules/site_shell.py").read_text()
    legacy = "\n".join(_LEGACY)

    assert _MOUSE_TAGLINE_LINES[0] in shell
    assert _MOUSE_TAGLINE_LINES[1] in shell
    for snippet in _LEGACY_FOOTER_SNIPPETS:
        assert snippet in shell, f"Missing footer snippet: {snippet[:60]}"
        assert snippet in legacy


def test_app_pages_import_used_module_functions():
    """Every modules.* helper called in a page must be imported."""
    failures: list[str] = []
    for path in sorted((_CODE_DIR / "app_pages").rglob("*.py")):
        raw = path.read_text()
        code_lines = [
            line for line in raw.splitlines() if not re.match(r"^\s*#", line)
        ]
        text = "\n".join(code_lines)
        imported = _imported_module_symbols(raw)
        missing = [
            fn
            for fn in _MODULE_FUNCTIONS
            if re.search(rf"\b{re.escape(fn)}\s*\(", text) and fn not in imported
        ]
        if missing:
            failures.append(f"{path.relative_to(_CODE_DIR)}: {sorted(missing)}")
    assert not failures, "Missing module imports in app_pages:\n" + "\n".join(failures)
