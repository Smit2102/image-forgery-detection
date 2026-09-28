"""Edge-sharpness consistency (edge detection).

Canny finds edges. At each edge pixel, sharpness is the gradient magnitude divided by the local contrast
(max - min grey level in a 5x5 window), i.e. how steep the edge is relative to its height. Photographs
blur edges consistently (same lens, focus and processing), so a pasted object often has edges that are
unnaturally sharp (hard cut-out) or softer (feathered/resampled) than the rest. The per-block median
sharpness is compared across the image.
"""

from __future__ import annotations

import warnings

import cv2
import numpy as np

from .base import FeatureModule, FeatureOutput, register
from .common import blocks, gray, inconsistency_stats, to_evidence


@register
class EdgeSharpness(FeatureModule):
    name = "edges"
    category = "edge"

    def __init__(self, low: int = 50, high: int = 150, block: int = 16, min_edge_pixels: int = 6):
        super().__init__(low=low, high=high, block=block, min_edge_pixels=min_edge_pixels)
        self.low, self.high, self.block, self.min_edge_pixels = low, high, block, min_edge_pixels

    def maps(self, img: np.ndarray) -> dict:
        g = gray(img)
        g8 = np.clip(g, 0, 255).astype(np.uint8)
        edges = cv2.Canny(g8, self.low, self.high, L2gradient=True) > 0
        grad = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)) / 8.0
        k = np.ones((5, 5), np.uint8)
        contrast = cv2.dilate(g, k) - cv2.erode(g, k)
        sharp = np.where(edges, grad / (contrast + 1.0), np.nan).astype(np.float32)
        eb = blocks(edges, self.block)
        sb = blocks(sharp, self.block)
        n_edge = eb.sum(axis=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN blocks -> NaN, handled below
            med = np.nanmedian(np.where(n_edge[..., None] > 0, sb, 0.0), axis=2)
        med = np.where(n_edge >= self.min_edge_pixels, med, np.nan)
        return {"edges": edges, "sharpness": sharp, "block_sharpness": med,
                "edge_density": n_edge / (self.block ** 2)}

    def extract(self, img: np.ndarray) -> FeatureOutput:
        m = self.maps(img)
        s = m["block_sharpness"]
        valid = np.isfinite(s)
        feats = {**inconsistency_stats(s, valid=valid, prefix="sharp_"),
                 **inconsistency_stats(-s, valid=valid, prefix="soft_"),
                 "local_frac_edge_blocks": float(valid.mean()) if s.size else 0.0,
                 "global_edge_density": float(m["edges"].mean()),
                 "global_median_sharpness": float(np.nanmedian(s)) if valid.any() else 0.0}
        if s.size < 4 or valid.sum() < 4:
            return FeatureOutput(np.zeros(img.shape[:2], np.float32), feats)
        return FeatureOutput(to_evidence(s, img.shape[:2], self.block, valid=valid, signed=False), feats)
