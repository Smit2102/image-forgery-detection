"""Histogram processing on the ELA map.

The raw ELA residual is low-contrast, so it is first histogram-equalised. The image is then cut into
overlapping patches, and each patch's histogram is compared with the image's reference histogram (the
mean over textured, unclipped patches) using the chi-square distance. Patches whose error distribution
differs from the rest stand out.
"""

from __future__ import annotations

import cv2
import numpy as np

from .base import FeatureModule, FeatureOutput, register
from .common import gray, inconsistency_stats, to_evidence
from .ela import ela_residual


def equalize(resid: np.ndarray) -> np.ndarray:
    """Contrast-stretch the residual to uint8 (99.5th percentile -> 255), then equalise its histogram."""
    hi = max(float(np.percentile(resid, 99.5)), 1e-3)
    u8 = np.clip(resid / hi * 255.0, 0, 255).astype(np.uint8)
    return cv2.equalizeHist(u8)


def chi2(h: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Chi-square distance between each row of h and the reference histogram g (both normalised)."""
    return 0.5 * ((h - g) ** 2 / (h + g + 1e-12)).sum(axis=-1)


@register
class HistogramChi2(FeatureModule):
    name = "histogram"
    category = "histogram"

    def __init__(self, quality: int = 90, patch: int = 32, bins: int = 16, min_std: float = 2.0):
        super().__init__(quality=quality, patch=patch, bins=bins, min_std=min_std)
        self.quality, self.patch, self.bins, self.min_std = quality, patch, bins, min_std

    def maps(self, img: np.ndarray) -> dict[str, np.ndarray]:
        eq = equalize(ela_residual(img, self.quality))
        q = (eq.astype(np.int32) * self.bins) // 256                     # bin index per pixel
        p, s = self.patch, self.patch // 2                                # 50 % overlapping patches
        h, w = q.shape
        ny, nx = (h - p) // s + 1, (w - p) // s + 1
        if ny < 2 or nx < 2:
            return {"equalized": eq, "distance": np.zeros((1, 1)), "valid": np.zeros((1, 1), bool)}
        ys, xs = np.arange(ny) * s, np.arange(nx) * s
        y0, x0 = np.meshgrid(ys, xs, indexing="ij")

        def box(ii):  # patch sums from an integral image (leading dims = image, trailing = channels)
            return ii[y0 + p, x0 + p] - ii[y0, x0 + p] - ii[y0 + p, x0] + ii[y0, x0]

        onehot = np.eye(self.bins, dtype=np.float32)[q]                   # (H, W, bins)
        hist = box(np.pad(onehot.cumsum(0).cumsum(1), ((1, 0), (1, 0), (0, 0)))) / (p * p)
        g = gray(img).astype(np.float64)
        s1 = box(np.pad(g.cumsum(0).cumsum(1), ((1, 0), (1, 0))))
        s2 = box(np.pad((g * g).cumsum(0).cumsum(1), ((1, 0), (1, 0))))
        mean = s1 / (p * p)
        std = np.sqrt(np.maximum(s2 / (p * p) - mean ** 2, 0))
        valid = (std >= self.min_std) & (mean > 4) & (mean < 251)        # skip flat / clipped patches
        ref = hist[valid].mean(axis=0) if valid.any() else hist.reshape(-1, self.bins).mean(axis=0)
        return {"equalized": eq, "distance": chi2(hist, ref), "valid": valid}

    def extract(self, img: np.ndarray) -> FeatureOutput:
        m = self.maps(img)
        d, v = m["distance"], m["valid"]
        feats = {**inconsistency_stats(d, valid=v),
                 "global_mean_distance": float(d[v].mean()) if v.any() else 0.0}
        if d.size < 4:
            return FeatureOutput(None, feats)
        # patch (i, j) covers pixels [i*s, i*s+p): place it on a stride-s grid, then shift by
        # (p - s) / 2 so each value sits at its patch centre
        h, w = img.shape[:2]
        off = (self.patch - self.patch // 2) // 2
        e = to_evidence(d, (h, w), self.patch // 2, valid=v)
        e = np.pad(e, ((off, 0), (off, 0)), mode="edge")[:h, :w]
        return FeatureOutput(np.ascontiguousarray(e), feats)
