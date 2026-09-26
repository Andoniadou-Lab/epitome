"""Configuration for Other Atlas data under ``other_data/``."""

from __future__ import annotations

from pathlib import Path

from config import Config
from modules.versioning import version_candidates, version_sort_key

OTHER_MIN_VERSION = "v_0.05"


class OtherConfig:
    BASE_PATH = Config.BASE_PATH
    OTHER_ROOT = BASE_PATH / "other_data"

    @classmethod
    def species_atlases_dir(cls, version: str) -> Path:
        return cls.OTHER_ROOT / "species_atlases" / version

    @classmethod
    def phylogeny_dir(cls, version: str) -> Path:
        return cls.OTHER_ROOT / "phylogeny" / version


def list_other_versions(min_version: str = OTHER_MIN_VERSION) -> list[str]:
    found: set[str] = set()
    min_key = version_sort_key(min_version)
    for sub in ("species_atlases", "phylogeny"):
        root = OtherConfig.OTHER_ROOT / sub
        if not root.is_dir():
            continue
        for path in root.iterdir():
            if path.is_dir() and path.name.startswith("v_") and version_sort_key(path.name) >= min_key:
                found.add(path.name)
    return sorted(found, key=version_sort_key, reverse=True)


def other_version_candidates(requested: str) -> list[str]:
    available = list_other_versions()
    if not available:
        available = [requested]
    return version_candidates(requested, available)
