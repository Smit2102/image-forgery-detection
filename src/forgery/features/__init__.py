"""Feature modules. Importing this package registers every implemented module, in pipeline order."""

from .base import REGISTRY, FeatureModule, FeatureOutput, build_modules, register
from . import ela, histogram, noise, jpeg_ghost, dct_dq, edges, copymove  # noqa: F401  (registration)

__all__ = ["REGISTRY", "FeatureModule", "FeatureOutput", "build_modules", "register"]
