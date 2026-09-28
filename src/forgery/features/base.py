"""Common interface every feature module implements.

A module turns one RGB image into
  * an evidence map  - HxW float32 in [0, 1], higher = more suspicious (or None), and
  * a dict of scalar features used by the image-level classifier.
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np


@dataclass
class FeatureOutput:
    evidence_map: np.ndarray | None
    features: dict[str, float] = field(default_factory=dict)

    def validate(self, shape: tuple[int, int], module: str) -> None:
        m = self.evidence_map
        if m is not None:
            if not isinstance(m, np.ndarray) or not np.issubdtype(m.dtype, np.floating):
                raise ValueError(f"{module}: evidence map must be a float array, got "
                                 f"{getattr(m, 'dtype', type(m))}")
            if m.shape != shape:
                raise ValueError(f"{module}: evidence map shape {m.shape} != image shape {shape}")
            if not np.isfinite(m).all():
                raise ValueError(f"{module}: evidence map contains NaN/inf")
            if m.min() < 0 or m.max() > 1:
                raise ValueError(f"{module}: evidence map must lie in [0, 1]")
        for k, v in self.features.items():
            if isinstance(v, bool) or not isinstance(v, numbers.Real) or not np.isfinite(v):
                raise ValueError(f"{module}: feature {k} must be a finite real number, got {v!r}")


class FeatureModule:
    """Base class. Subclasses set `name`/`category` and implement `extract`."""

    name: ClassVar[str] = "base"
    category: ClassVar[str] = ""   # point / histogram / spatial / frequency / edge

    def __init__(self, **params):
        self.params = params

    def extract(self, img: np.ndarray) -> FeatureOutput:  # pragma: no cover - interface
        raise NotImplementedError


REGISTRY: dict[str, type[FeatureModule]] = {}


def register(cls: type[FeatureModule]) -> type[FeatureModule]:
    if cls.name in REGISTRY and REGISTRY[cls.name] is not cls:
        raise KeyError(f"feature module {cls.name!r} registered twice")
    REGISTRY[cls.name] = cls
    return cls


def build_modules(names: list[str] | None = None, params: dict | None = None) -> list[FeatureModule]:
    """Instantiate registered modules (all, in registration order, if names is None)."""
    params = params or {}
    names = list(REGISTRY) if names is None else names
    missing = [n for n in names if n not in REGISTRY]
    if missing:
        raise KeyError(f"unknown feature modules: {missing}; registered: {list(REGISTRY)}")
    return [REGISTRY[n](**params.get(n, {})) for n in names]

