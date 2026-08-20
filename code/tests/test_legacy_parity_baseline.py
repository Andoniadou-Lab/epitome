"""Legacy parity is always evaluated at v_0.02 (the version legacy shipped).

Newer releases add datasets, release-note blocks and references on top; those
additions must not be read as drift from ``epitome_legacy.py``.
"""

from legacy_parity_support import (
    LEGACY_PARITY_VERSION,
    is_newer_than_parity_baseline,
    reference_keys,
    strip_newer_release_blocks,
)


def test_parity_baseline_is_v_0_02():
    assert LEGACY_PARITY_VERSION == "v_0.02"


def test_only_versions_above_baseline_count_as_newer():
    assert is_newer_than_parity_baseline("v_0.03")
    assert is_newer_than_parity_baseline("v_0.04")
    assert not is_newer_than_parity_baseline("v_0.02")
    assert not is_newer_than_parity_baseline("v_0.01")


def test_strip_newer_release_blocks_keeps_baseline_and_older():
    source = '\n'.join(
        [
            'import streamlit as st',
            'st.info(',
            '"v_0.03: Third release\\n"',
            ')',
            'st.info(',
            '"v_0.02: Second release\\n"',
            ')',
            'st.info(',
            '"v_0.01: First release\\n"',
            ')',
        ]
    )
    stripped = strip_newer_release_blocks(source)
    assert "v_0.03" not in stripped
    assert "v_0.02: Second release" in stripped
    assert "v_0.01: First release" in stripped


def test_reference_keys_survive_renumbering_and_author_expansion():
    legacy = "    39.\tSochodolsky, K. (2026). BDNF engages pituitary stem cells."
    page = (
        "    41. Sochodolsky, K., Khetchoumian, K., Balsalobre, A., and Drouin, J. "
        "(2026). BDNF engages pituitary stem cells."
    )
    assert reference_keys(legacy) == reference_keys(page) == {"sochodolsky:2026"}


def test_reference_keys_detect_a_dropped_reference():
    legacy = "1. Smith, A. (2020). A paper.\n2. Jones, B. (2021). Another paper."
    page = "1. Smith, A. (2020). A paper."
    assert reference_keys(legacy) - reference_keys(page) == {"jones:2021"}
