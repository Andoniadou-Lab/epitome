"""Shared data-version helpers: ordering, fallback, and caption labels."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import TypeVar

T = TypeVar("T")

# Newest first — used by the mouse (epitome) site selectors.
MOUSE_AVAILABLE_VERSIONS = ["v_0.03", "v_0.02", "v_0.01"]

# Individual single-cell objects (mouse datasets, tumour datasets, Other species
# atlases) are large and browsed one at a time, so they are evicted quickly.
SINGLE_CELL_CACHE_TTL_SECONDS = 10 * 60
SINGLE_CELL_CACHE_MAX_ENTRIES = 5

# Main analysis tables for the current release stay cached for the life of the
# server; tables loaded for an older release are dropped after this many seconds.
OLD_VERSION_TTL_SECONDS = 20 * 60

_RESOLVED_KEY = "_resolved_data_versions"


def version_sort_key(version: str) -> tuple[int, ...]:
    normalized = version.removeprefix("v_").replace(".", "_")
    parts: list[int] = []
    for part in normalized.split("_"):
        if part.isdigit():
            parts.append(int(part))
    return tuple(parts) if parts else (0,)


def sorted_versions(versions: Sequence[str], *, reverse: bool = True) -> list[str]:
    return sorted(versions, key=version_sort_key, reverse=reverse)


def version_candidates(
    requested: str,
    available: Sequence[str] | None = None,
) -> list[str]:
    """Versions to try for ``requested``, newest-compatible first then lower.

    Always includes ``requested`` first, then every lower version from
    ``available`` (or ``MOUSE_AVAILABLE_VERSIONS``).
    """
    pool = list(available) if available is not None else list(MOUSE_AVAILABLE_VERSIONS)
    ordered = sorted_versions(pool, reverse=True)
    if requested not in ordered:
        ordered = sorted_versions([*ordered, requested], reverse=True)
    req_key = version_sort_key(requested)
    lower_or_equal = [v for v in ordered if version_sort_key(v) <= req_key]
    # Ensure requested is first even if pool order differs.
    rest = [v for v in lower_or_equal if v != requested]
    return [requested, *rest]


def format_version_label(requested: str, resolved: str | None = None) -> str:
    """Caption fragment: ``v_0.03`` or ``Fall back to v_0.02``."""
    actual = resolved or requested
    if actual != requested:
        return f"Fall back to {actual}"
    return actual


def _store() -> dict[str, str]:
    try:
        import streamlit as st

        if _RESOLVED_KEY not in st.session_state:
            st.session_state[_RESOLVED_KEY] = {}
        return st.session_state[_RESOLVED_KEY]
    except Exception:
        if not hasattr(format_version_label, "_fallback_store"):
            format_version_label._fallback_store = {}  # type: ignore[attr-defined]
        return format_version_label._fallback_store  # type: ignore[attr-defined]


def record_resolved_version(loader_key: str, requested: str, resolved: str) -> None:
    store = _store()
    store[f"{loader_key}:{requested}"] = resolved


def get_resolved_version(
    requested: str,
    loader_key: str | None = None,
    loader_keys: Sequence[str] | None = None,
) -> str:
    """Version actually used, considering only the declared loaders.

    A plot may combine several loaders (e.g. a matrix plus curation); the
    caption should report the oldest version among them. Loaders that are
    unrelated to the plot are ignored — the app warms every loader at startup,
    so guessing across all of them mislabels pages whose own data is current.
    """
    store = _store()
    keys = [key for key in (loader_key, *(loader_keys or ())) if key]
    if not keys:
        return requested
    resolved = [
        store[f"{key}:{requested}"] for key in keys if f"{key}:{requested}" in store
    ]
    if not resolved:
        return requested
    return min(resolved, key=version_sort_key)


def resolve_versioned_path(
    build_path: Callable[[str], str],
    version: str,
    available: Sequence[str] | None = None,
    exists: Callable[[str], bool] = os.path.exists,
) -> tuple[str | None, str | None]:
    """First usable path across ``version`` then lower versions.

    For assets (figures, exports, download bundles) that are not routed through a
    data loader. ``exists`` can be narrowed to something stricter than "the path is
    there", e.g. "the directory holds at least one ``.h5ad``". Returns
    ``(None, None)`` when no version qualifies.
    """
    for candidate in version_candidates(version, available):
        path = build_path(candidate)
        if exists(path):
            return path, candidate
    return None, None


def load_with_version_fallback(
    loader: Callable[..., T],
    version: str,
    *args,
    available: Sequence[str] | None = None,
    loader_key: str = "data",
    **kwargs,
) -> T:
    """Call ``loader(candidate, ...)`` trying ``version`` then lower versions."""
    errors: list[str] = []
    for candidate in version_candidates(version, available):
        try:
            result = loader(candidate, *args, **kwargs)
            record_resolved_version(loader_key, version, candidate)
            return result
        except Exception as exc:  # noqa: BLE001 — intentional soft fallback
            errors.append(f"{candidate}: {exc}")
    detail = "; ".join(errors) if errors else "no candidates"
    raise FileNotFoundError(
        f"No data for version {version} (or lower). Attempts: {detail}"
    )
