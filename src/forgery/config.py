"""Project configuration loading and path resolution."""

from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config.yaml"


@lru_cache(maxsize=4)
def _load(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def load_config(path: str | Path | None = None) -> dict:
    """Load config.yaml (cached). Relative paths resolve against the project root.
    Returns a deep copy so callers may mutate it."""
    return copy.deepcopy(_load(str(resolve(path) if path else DEFAULT_CONFIG)))


def resolve(rel: str | Path) -> Path:
    """Resolve a project-relative path to an absolute one."""
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p
