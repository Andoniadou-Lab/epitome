"""Tests for shared data-version fallback helpers."""

from modules.versioning import (
    MOUSE_AVAILABLE_VERSIONS,
    format_version_label,
    get_resolved_version,
    record_resolved_version,
    version_candidates,
    version_sort_key,
)


def test_mouse_available_versions_include_v003():
    assert MOUSE_AVAILABLE_VERSIONS[0] == "v_0.03"
    assert "v_0.02" in MOUSE_AVAILABLE_VERSIONS
    assert "v_0.01" in MOUSE_AVAILABLE_VERSIONS


def test_version_candidates_prefer_requested_then_lower():
    assert version_candidates("v_0.03") == ["v_0.03", "v_0.02", "v_0.01"]
    assert version_candidates("v_0.02") == ["v_0.02", "v_0.01"]
    assert version_candidates("v_0.01") == ["v_0.01"]


def test_version_sort_key_orders_semver_like():
    assert version_sort_key("v_0.03") > version_sort_key("v_0.02")
    assert version_sort_key("v_0.02") > version_sort_key("v_0.01")


def test_format_version_label_fallback_text():
    assert format_version_label("v_0.03", "v_0.03") == "v_0.03"
    assert format_version_label("v_0.03", "v_0.02") == "Fall back to v_0.02"


def test_record_and_get_resolved_version():
    record_resolved_version("expression", "v_0.03", "v_0.02")
    assert get_resolved_version("v_0.03", "expression") == "v_0.02"
    record_resolved_version("dotplot", "v_0.03", "v_0.03")
    assert get_resolved_version("v_0.03", "dotplot") == "v_0.03"


def test_get_resolved_version_ignores_undeclared_loaders():
    # A fallback in an unrelated loader must not mislabel this plot.
    record_resolved_version("accessibility", "v_0.03", "v_0.01")
    record_resolved_version("proportion", "v_0.03", "v_0.03")
    assert get_resolved_version("v_0.03", "proportion") == "v_0.03"
    assert get_resolved_version("v_0.03") == "v_0.03"


def test_get_resolved_version_multiple_loaders_reports_oldest():
    record_resolved_version("isoforms", "v_0.03", "v_0.03")
    record_resolved_version("curation", "v_0.03", "v_0.02")
    resolved = get_resolved_version("v_0.03", loader_keys=("isoforms", "curation"))
    assert resolved == "v_0.02"
    assert format_version_label("v_0.03", resolved) == "Fall back to v_0.02"
