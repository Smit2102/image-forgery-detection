"""Local noise-level map (spatial filtering).

A high-pass filter (3x3 Laplacian, or Immerkaer's kernel, which cancels edges and gradients better)
removes most image content and leaves sensor/compression noise. The noise level of each block is
estimated robustly (1.4826 x median absolute residual) from its *non-edge* pixels only: pixels whose
gradient magnitude exceeds a global percentile are masked, because edges and texture leak content into
the residual, and so are pixels in perfectly flat 3x3 neighbourhoods, which carry no noise. Blocks with too few usable pixels, and flat or clipped blocks (the common validity mask),
which carry no measurable noise, are excluded. A region pasted from another camera, or smoothed/resampled during editing, has a
different noise level from its surroundings.
"""

from __future__ import annotations

import warnings

import cv2
import numpy as np

from .base import FeatureModule, FeatureOutput, register
from .common import blocks, gray, inconsistency_stats, to_evidence, valid_blocks

KERNELS = {
    "laplacian": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32),
    # Immerkaer (1996): difference of two Laplacians, zero response to linear intensity ramps
    "immerkaer": np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32),
}


@register
class NoiseLevel(FeatureModule):
    name = "noise"
    category = "spatial"

    def __init__(self, kernel: str = "immerkaer", block: int = 8, edge_quantile: float = 0.6,
                 min_pixels: int = 16, min_std: float = 1.0):
        super().__init__(kernel=kernel, block=block, edge_quantile=edge_quantile, min_pixels=min_pixels,
                         min_std=min_std)
        if kernel not in KERNELS:
            raise ValueError(f"unknown kernel {kernel!r}")
        self.kernel, self.block, self.edge_quantile, self.min_pixels = kernel, block, edge_quantile, min_pixels
        self.min_std = min_std

    def maps(self, img: np.ndarray) -> dict[str, np.ndarray]:
        g = gray(img)
        resid = cv2.filter2D(g, cv2.CV_32F, KERNELS[self.kernel], borderType=cv2.BORDER_REFLECT)
        grad = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
        usable = grad <= np.quantile(grad, self.edge_quantile)
        usable &= (g > 4) & (g < 251)                                     # clipped pixels hold no noise
        # perfectly flat 3x3 neighbourhoods (e.g. a synthetic or clipped patch) hold no sensor noise
        # either; excluding them per pixel also covers blocks that straddle the edge of such a patch
        mean3 = cv2.blur(g, (3, 3))
        usable &= (cv2.blur(g * g, (3, 3)) - mean3 * mean3) > 0.01
        rb, ub = blocks(np.abs(resid), self.block), blocks(usable, self.block)
        n = ub.sum(axis=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN blocks -> NaN, handled below
            med = np.nanmedian(np.where(ub, rb, np.nan), axis=2)
        smooth = (n >= self.min_pixels) & valid_blocks(img, self.block, self.min_std)
        sigma = np.where(smooth, 1.4826 * np.nan_to_num(med), np.nan)
        return {"residual": resid, "usable": usable, "log_sigma": np.log(sigma + 0.1), "smooth": smooth}

    def extract(self, img: np.ndarray) -> FeatureOutput:
        m = self.maps(img)
        ls, smooth = m["log_sigma"], m["smooth"]
        feats = {**inconsistency_stats(ls, valid=smooth),
                 # low-noise outliers matter as much as high-noise ones (e.g. a smoothed paste)
                 **inconsistency_stats(-ls, valid=smooth, prefix="low_"),
                 "local_frac_usable_blocks": float(smooth.mean()) if smooth.size else 0.0,
                 "global_log_sigma": float(np.median(ls[smooth])) if smooth.any() else 0.0}
        if ls.size < 4:
            return FeatureOutput(None, feats)
        return FeatureOutput(to_evidence(np.where(smooth, ls, np.nan), img.shape[:2], self.block, valid=smooth,
                                         signed=False), feats)
