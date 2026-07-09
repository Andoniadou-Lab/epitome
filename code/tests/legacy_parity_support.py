"""Shared helpers for legacy ↔ multipage parity tests."""

from __future__ import annotations

import ast
import builtins
import re
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
LEGACY_LINES = (CODE_DIR / "epitome_legacy.py").read_text().splitlines()
LEGACY_TEXT = "\n".join(LEGACY_LINES)

PAGES: list[tuple[str, int, int, int, str | None]] = [
    ("app_pages/overview/overview.py", 615, 1030, 16, None),
    ("app_pages/transcriptome/expression_box_plots.py", 1072, 1313, 24, None),
    ("app_pages/transcriptome/expression_umap.py", 1331, 1541, 24, None),
    ("app_pages/transcriptome/age_correlation.py", 1556, 1846, 24, None),
    ("app_pages/transcriptome/isoforms.py", 1860, 2229, 24, None),
    ("app_pages/transcriptome/dot_plots.py", 2243, 2564, 24, None),
    ("app_pages/transcriptome/cell_type_distribution.py", 2578, 2789, 24, None),
    ("app_pages/transcriptome/gene_gene_relationships.py", 2804, 3030, 24, None),
    ("app_pages/transcriptome/ligand_receptor_interactions.py", 3044, 3296, 24, None),
    ("app_pages/chromatin/accessibility_distribution.py", 3320, 3701, 24, "render_genome_browser"),
    ("app_pages/chromatin/motif_enrichment_chromvar.py", 3714, 3953, 24, None),
    ("app_pages/chromatin/cell_type_distribution_atac.py", 3967, 4229, 24, None),
    ("app_pages/multimodal/heatmap_tfs.py", 4250, 4958, 24, None),
    ("app_pages/datasets/rna_datasets.py", 4992, 5198, 24, None),
    ("app_pages/datasets/atac_datasets.py", 5212, 5406, 24, None),
    ("app_pages/curation/curation.py", 5584, 5624, 12, None),
    ("app_pages/release_notes/release_notes.py", 5627, 5674, 12, None),
    ("app_pages/citation/how_to_cite.py", 5677, 5775, 12, None),
    ("app_pages/contact/contact.py", 5778, 5837, 12, None),
]

DOWNLOAD_PAGES: list[tuple[str, int, int, int]] = [
    ("app_pages/downloads/h5ad_rna.py", 5433, 5433, 20),
    ("app_pages/downloads/h5ad_atac.py", 5436, 5436, 20),
    ("app_pages/downloads/analysis_data_files.py", 5439, 5439, 20),
    ("app_pages/downloads/usage_guide.py", 5443, 5581, 20),
]


def _dedent(lines: list[str], n: int) -> list[str]:
    out: list[str] = []
    for line in lines:
        if line.strip() == "":
            out.append("")
        elif line.startswith(" " * n):
            out.append(line[n:])
        else:
            out.append(line.lstrip())
    return out


DOWNLOADS_PARENT = _dedent(LEGACY_LINES[5409:5420], 16)


def normalize_code(code: str) -> str:
    lines: list[str] = []
    for line in code.splitlines():
        s = line.rstrip()
        if s.strip().startswith("#"):
            continue
        s = re.sub(r"Config\.BASE_PATH", "BASE_PATH", s)
        lines.append(s)
    return "\n".join(lines).strip()


def page_body(path: Path) -> str:
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


def expected_legacy(start: int, end: int, indent: int) -> str:
    return "\n".join(_dedent(LEGACY_LINES[start - 1 : end], indent))


def page_source(relpath: str, start: int, end: int, indent: int, skip_fn: str | None) -> str:
    path = CODE_DIR / relpath
    if skip_fn:
        raw = path.read_text()
        idx = raw.find("col1, col2 = st.columns([5, 1])")
        return raw[idx:]
    return page_body(path)


def iter_mapped_page_sources() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for relpath, start, end, indent, skip_fn in PAGES:
        out.append((relpath, page_source(relpath, start, end, indent, skip_fn)))
    for relpath, start, end, indent in DOWNLOAD_PAGES:
        legacy_src = "\n".join(DOWNLOADS_PARENT + _dedent(LEGACY_LINES[start - 1 : end], indent))
        out.append((relpath, page_body(CODE_DIR / relpath)))
        # legacy side stored separately when needed
    return out


def legacy_source_for(relpath: str) -> str:
    for path, start, end, indent, skip_fn in PAGES:
        if path == relpath:
            return expected_legacy(start, end, indent)
    for path, start, end, indent in DOWNLOAD_PAGES:
        if path == relpath:
            return "\n".join(DOWNLOADS_PARENT + _dedent(LEGACY_LINES[start - 1 : end], indent))
    raise KeyError(relpath)


def _string_parts(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.JoinedStr):
        parts = [
            v.value
            for v in node.values
            if isinstance(v, ast.Constant) and isinstance(v.value, str)
        ]
        return ["".join(parts)] if parts else []
    return []


def extract_st_calls(source: str, attr: str) -> list[ast.Call]:
    calls: list[ast.Call] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "st"
            and node.func.attr == attr
        ):
            calls.append(node)
    return calls


def extract_kw_strings(source: str, attr: str, kw: str) -> set[str]:
    out: set[str] = set()
    for call in extract_st_calls(source, attr):
        for keyword in call.keywords:
            if keyword.arg == kw:
                for part in _string_parts(keyword.value):
                    out.add(part)
    return out


def extract_positional_strings(source: str, attr: str, min_len: int = 1) -> set[str]:
    out: set[str] = set()
    for call in extract_st_calls(source, attr):
        if call.args:
            for part in _string_parts(call.args[0]):
                if len(part.strip()) >= min_len:
                    out.add(part.strip())
    return out


def extract_display_strings(source: str, min_len: int = 10) -> set[str]:
    display_attrs = {
        "header",
        "subheader",
        "markdown",
        "write",
        "info",
        "warning",
        "error",
        "success",
        "caption",
        "title",
        "text",
    }
    out: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        if not (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "st"
            and node.func.attr in display_attrs
        ):
            continue
        for arg in node.args:
            for part in _string_parts(arg):
                normalized = re.sub(r"\s+", " ", part.strip())
                if len(normalized) >= min_len:
                    out.add(normalized)
    return out


def extract_regex_matches(source: str, pattern: str) -> set[str]:
    return set(re.findall(pattern, source))


def extract_merged_string_literals(source: str, min_len: int = 40) -> set[str]:
    """String literals and merged adjacent literals (help text, multi-arg st.info)."""

    def flatten(node: ast.AST) -> list[str]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return [node.value]
        if isinstance(node, (ast.Tuple, ast.List)):
            parts: list[str] = []
            for elt in node.elts:
                parts.extend(flatten(elt))
            return parts
        return []

    merged: set[str] = set()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            parts: list[str] = []
            for arg in node.args:
                parts.extend(flatten(arg))
            if parts:
                text = "".join(parts).strip()
                if len(text) >= min_len:
                    merged.add(re.sub(r"\s+", " ", text))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            for kw in node.keywords:
                if kw.arg in ("help", "label", "placeholder"):
                    parts = flatten(kw.value)
                    if parts:
                        text = "".join(parts).strip()
                        if len(text) >= min_len:
                            merged.add(re.sub(r"\s+", " ", text))
    return merged


def count_calls(source: str, name: str) -> int:
    return len(re.findall(rf"\b{re.escape(name)}\s*\(", source))


def imported_names_from_source(source: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def local_function_defs(source: str) -> set[str]:
    return {
        node.name
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
    }


def bare_function_calls(source: str) -> set[str]:
    return {
        node.func.id
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


_BUILTIN_CALLABLES = set(dir(builtins))
