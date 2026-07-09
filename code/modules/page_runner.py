"""Wrap app page scripts so the site footer renders after page content."""

from __future__ import annotations

import runpy
from collections.abc import Callable
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parent.parent


def page_with_footer(script_relpath: str) -> Callable[[], None]:
    """Build a ``st.Page`` callable that runs *script_relpath* then the footer."""
    script_path = _CODE_ROOT / script_relpath

    def _run() -> None:
        runpy.run_path(str(script_path), run_name="__epitome_page__")
        from modules.site_shell import render_page_footer

        render_page_footer()

    _run.__name__ = script_path.stem.replace("-", "_")
    _run.__doc__ = f"Run {script_relpath} with shared footer."
    return _run
