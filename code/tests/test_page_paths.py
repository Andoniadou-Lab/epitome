"""Every ``st.Page`` path registered for navigation must exist on disk."""

from __future__ import annotations

import re
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parent.parent
_PAGE_SOURCES = (
    _CODE_DIR / "epitome.py",
    _CODE_DIR / "modules" / "site_shell.py",
)


def test_navigation_page_files_exist():
    paths: list[str] = []
    for source in _PAGE_SOURCES:
        if source.is_file():
            text = source.read_text()
            paths.extend(re.findall(r'st\.Page\("([^"]+)"', text))
            paths.extend(re.findall(r'page_with_footer\("([^"]+)"\)', text))
    assert paths, "No st.Page entries found in epitome.py / site_shell.py"
    missing = [p for p in paths if not (_CODE_DIR / p).is_file()]
    assert not missing, f"Missing page files: {missing}"
