"""Smoke tests for human pituitary tumour atlas (PTA) modules."""

from __future__ import annotations

from pathlib import Path

import pytest

from config import Config

CODE_DIR = Path(__file__).resolve().parent.parent
TUMOR_PAGES = list((CODE_DIR / "app_pages" / "tumor").glob("*.py"))
V = "v_0.04"
PTA = Config.BASE_PATH / "pta_data"


def test_pta_data_v004_present():
    required = [
        PTA / "curation" / V / "cpa.parquet",
        PTA / "dotplot" / V / "matrix1.mtx",
        PTA / "dotplot" / V / "matrix2.mtx",
        PTA / "cell_proportion" / V / "abundance.mtx",
        PTA / "overview" / V / "rna_cell_type_counts.parquet",
        PTA / "bulk_curation" / V / "pituitary_tumor_atlas_bulk_updated_final.xlsx",
        PTA / "bulk_expression" / V / "concatted_matrix_shared.csv",
        PTA / "sc_data" / "datasets" / V / "epitome_h5_files" / "HRS1408776.h5ad",
        PTA / "epitome_volcanos" / V / "volcanos.json",
        PTA / "epitome_volcanos" / V / "dream_NR5A1_vs_POU1F1.csv",
    ]
    missing = [str(p.relative_to(PTA)) for p in required if not p.is_file()]
    pseudobulk_h5ad = list((PTA / "pseudobulk" / V).glob("*.h5ad"))
    if not pseudobulk_h5ad:
        missing.append(f"pseudobulk/{V}/*.h5ad")
    assert not missing, f"Missing PTA data files: {missing}"


def test_list_pta_versions():
    from modules.pta.config import list_pta_versions

    versions = list_pta_versions()
    assert versions == ["v_0.04"]


def test_tumor_pages_import_cleanly():
    from legacy_parity_support import _BUILTIN_CALLABLES, imported_names_from_source, local_function_defs, bare_function_calls

    failures = []
    for path in TUMOR_PAGES:
        full = path.read_text()
        imported = imported_names_from_source(full)
        local = local_function_defs(full)
        for fn in bare_function_calls(full):
            if fn in _BUILTIN_CALLABLES:
                continue
            if fn not in imported and fn not in local:
                failures.append(f"{path.name}: {fn}")
    assert not failures, failures


def test_pseudobulk_path_resolves():
    from modules.pta.config import PtaConfig

    path = PtaConfig.pseudobulk_path(V)
    assert path.is_file(), path


@pytest.mark.parametrize("module_path", [
    "modules.pta.config",
    "modules.pta.data_loader",
    "modules.pta.boxplot",
    "modules.pta.heatmap",
    "modules.pta.individual_sc",
    "modules.pta.stats",
    "modules.pta.gene_annotation",
    "modules.pta.volcano",
    "modules.ui.plot_settings",
])
def test_pta_modules_import(module_path: str):
    __import__(module_path)


def test_pta_scrna_curation_loads():
    from modules.pta.data_loader import load_pta_scrna_curation

    df = load_pta_scrna_curation(V)
    assert len(df) == 118
    assert "SRA_ID" in df.columns


def test_volcano_manifest_and_paths():
    from modules.pta.config import PtaConfig
    from modules.pta.data_loader import load_volcano_manifest

    manifest = load_volcano_manifest(V)
    assert len(manifest) == 3
    for entry in manifest:
        assert "id" in entry and "file" in entry
        path = PtaConfig.volcano_dir(V) / entry["file"]
        assert path.is_file(), f"Missing volcano CSV: {path}"


def test_volcano_plot_renders():
    from modules.pta.data_loader import load_volcano_results
    from modules.pta.volcano import create_volcano_plot

    df = load_volcano_results(V, "NR5A1_vs_POU1F1")
    fig, config = create_volcano_plot(df, title="test", highlight_genes=["GH1"])
    assert len(fig.data) >= 1
    assert config["toImageButtonOptions"]["width"] == 800
    assert config["toImageButtonOptions"]["height"] == 800
