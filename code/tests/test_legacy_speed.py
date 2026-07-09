"""Speed and warmup parity vs epitome_legacy.py cached loaders."""

from __future__ import annotations

import ast
import re
import time
from pathlib import Path

import pytest

from legacy_parity_support import CODE_DIR, LEGACY_LINES

# Per-loader cold-call ceilings (seconds) on a typical dev machine with local data.
LOADER_CEILINGS = {
    "load_curation_data": 2.0,
    "load_and_transform_data": 8.0,
    "load_dotplot_data": 8.0,
    "load_heatmap_data": 8.0,
    "load_chromvar_data": 8.0,
    "load_accessibility_data": 10.0,
    "load_marker_data": 3.0,
    "load_proportion_data": 3.0,
}


def _legacy_cached_loader_names() -> list[str]:
    names: list[str] = []
    for line in LEGACY_LINES[314:477]:
        m = re.match(r"def (load_cached_\w+)", line)
        if m:
            names.append(m.group(1))
    return names


def _new_cached_loader_names() -> list[str]:
    src = (CODE_DIR / "modules" / "cached_loaders.py").read_text()
    return re.findall(r"def (load_cached_\w+)", src)


def _legacy_warmup_calls() -> list[str]:
    body = "\n".join(LEGACY_LINES[455:476])
    return re.findall(r"(load_cached_\w+)\(", body)


def _new_warmup_calls() -> list[str]:
    body = (CODE_DIR / "modules" / "cached_loaders.py").read_text()
    m = re.search(
        r"def load_all_cached_data\(.*?\n(?P<body>(?:    .*\n)+)",
        body,
    )
    assert m, "load_all_cached_data body not found"
    return re.findall(r"(load_cached_\w+)\(", m.group("body"))


def _loader_to_underlying(cached_name: str) -> str:
    mapping = {
        "load_cached_data": "load_and_transform_data",
        "load_cached_enrichment_data": "load_enrichment_results",
        "load_cached_single_cell_dataset": "load_single_cell_dataset",
    }
    if cached_name in mapping:
        return mapping[cached_name]
    return cached_name.replace("load_cached_", "load_")


@pytest.fixture(scope="module")
def loader_timings() -> dict[str, float]:
    from modules import data_loader as dl

    timings: dict[str, float] = {}
    for cached, ceiling in LOADER_CEILINGS.items():
        underlying = _loader_to_underlying(cached)
        fn = getattr(dl, underlying)
        t0 = time.perf_counter()
        fn("v_0.02")
        timings[underlying] = time.perf_counter() - t0
        assert timings[underlying] <= ceiling, (
            f"{underlying} took {timings[underlying]:.2f}s (ceiling {ceiling}s)"
        )
    return timings


def test_legacy_and_new_cached_loader_sets_match():
    legacy = set(_legacy_cached_loader_names())
    new = set(_new_cached_loader_names())
    assert legacy == new, f"legacy-only={legacy - new} new-only={new - legacy}"


def test_new_warmup_is_superset_of_legacy():
    legacy = set(_legacy_warmup_calls())
    new = set(_new_warmup_calls())
    assert legacy <= new, f"Missing from new warmup: {legacy - new}"
    # Document intentional additions (expression + ATAC markers warmed on first visit).
    assert "load_cached_data" in new - legacy
    assert "load_cached_marker_data_atac" in new - legacy


def test_warmup_underlying_sequence_within_budget(loader_timings):
    """Sum of legacy warmup loaders (direct, no Streamlit cache) stays bounded."""
    from config import Config
    from modules import data_loader as dl

    base = Config.BASE_PATH
    isoform_npz = base / "data/isoforms/v_0.02/isoforms_matrix.mtx.npz"
    if not isoform_npz.is_file():
        pytest.skip("Isoform data not present for full warmup timing")

    t0 = time.perf_counter()
    for cached in _legacy_warmup_calls():
        fn = getattr(dl, _loader_to_underlying(cached))
        fn("v_0.02")
    legacy_total = time.perf_counter() - t0

    t0 = time.perf_counter()
    for cached in _new_warmup_calls():
        fn = getattr(dl, _loader_to_underlying(cached))
        fn("v_0.02")
    new_total = time.perf_counter() - t0

    assert new_total <= legacy_total * 1.35 + 3.0, (
        f"New warmup {new_total:.1f}s vs legacy {legacy_total:.1f}s"
    )


def test_core_loaders_report_timings(loader_timings, capsys):
    """Print loader timings for manual comparison (legacy uses same underlying code)."""
    for name, elapsed in sorted(loader_timings.items(), key=lambda x: x[1], reverse=True):
        print(f"{name}: {elapsed:.2f}s")
    assert loader_timings["load_and_transform_data"] > 0


def test_page_script_ast_parse_is_fast():
    """Multipage scripts should parse quickly (negligible vs data load)."""
    from legacy_parity_support import PAGES

    t0 = time.perf_counter()
    for relpath, *_ in PAGES:
        ast.parse((CODE_DIR / relpath).read_text())
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, f"Parsing {len(PAGES)} page files took {elapsed:.2f}s"


def test_cached_loader_decorators_match_legacy():
    legacy_src = "\n".join(LEGACY_LINES[314:452])
    new_src = (CODE_DIR / "modules" / "cached_loaders.py").read_text()

    for name in _legacy_cached_loader_names():
        legacy_chunk = legacy_src.split(f"def {name}")[1].split("\ndef ")[0]
        new_chunk = new_src.split(f"def {name}")[1].split("\ndef ")[0]
        legacy_deco = "cache_resource" if "@st.cache_resource" in legacy_chunk.split("def")[0] else "cache_data"
        new_deco = "cache_resource" if "@st.cache_resource" in new_chunk.split("def")[0] else "cache_data"
        assert legacy_deco == new_deco, f"{name}: {legacy_deco} vs {new_deco}"


def test_legacy_vs_new_loader_bodies_equivalent():
    """Loader try/except bodies call the same underlying load_* functions."""
    legacy_src = "\n".join(LEGACY_LINES[314:452])
    new_src = (CODE_DIR / "modules" / "cached_loaders.py").read_text()
    for name in _legacy_cached_loader_names():
        legacy_chunk = legacy_src.split(f"def {name}")[1].split("\ndef ")[0]
        new_chunk = new_src.split(f"def {name}")[1].split("\ndef ")[0]
        legacy_calls = set(re.findall(r"load_[a-z_]+\(", legacy_chunk))
        new_calls = set(re.findall(r"load_[a-z_]+\(", new_chunk))
        assert legacy_calls == new_calls, f"{name}: {legacy_calls} vs {new_calls}"
