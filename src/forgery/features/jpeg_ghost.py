"""JPEG ghosts (Farid, 2009).

Re-save the image at a ladder of JPEG qualities and measure, per block, the squared difference at each
quality. A block's difference curve has its minimum near the quality it was last compressed at.
Content pasted from a differently compressed source has its minimum elsewhere: a "ghost". Each block's
curve is min-max normalised, and its distance from the image's median curve is the evidence.
"""

from __future__ import annotations

import numpy as np

from ..io import jpeg_roundtrip
from .base import FeatureModule, FeatureOutput, register
from .common import block_mean, inconsistency_stats, to_evidence, valid_blocks

DEFAULT_QUALITIES = (50, 55, 60, 65, 70, 75, 80, 85, 90, 95)


@register
class JPEGGhost(FeatureModule):
    name = "jpeg_ghost"
    category = "point"

    def __init__(self, qualities: tuple[int, ...] = DEFAULT_QUALITIES, block: int = 16, min_std: float = 2.0):
        qualities = tuple(int(q) for q in qualities)
        super().__init__(qualities=qualities, block=block, min_std=min_std)
        self.qualities, self.block, self.min_std = qualities, block, min_std

    def maps(self, img: np.ndarray) -> dict[str, np.ndarray]:
        if img.shape[0] < self.block or img.shape[1] < self.block:   # smaller than one block
            z = np.zeros((0, 0))
            return {"curves": np.zeros((0, 0, len(self.qualities))), "deviation": z,
                    "valid": z.astype(bool), "argmin_q": z}
        x = img.astype(np.float32)
        curves = np.stack([block_mean(((x - jpeg_roundtrip(img, q).astype(np.float32)) ** 2).mean(axis=2),
                                      self.block) for q in self.qualities], axis=-1)       # (nby, nbx, nq)
        lo, hi = curves.min(axis=-1, keepdims=True), curves.max(axis=-1, keepdims=True)
        norm = (curves - lo) / (hi - lo + 1e-6)
        valid = valid_blocks(img, self.block, self.min_std)
        ref = norm[valid] if valid.any() else norm.reshape(-1, norm.shape[-1])
        median_curve = np.median(ref.reshape(-1, norm.shape[-1]), axis=0)
        return {"curves": curves, "deviation": np.abs(norm - median_curve).mean(axis=-1), "valid": valid,
                "argmin_q": np.asarray(self.qualities)[curves.argmin(axis=-1)]}

    def extract(self, img: np.ndarray) -> FeatureOutput:
        m = self.maps(img)
        dev, amin, v = m["deviation"], m["argmin_q"], m["valid"]
        vals, counts = np.unique(amin[v], return_counts=True)
        mode_q = vals[counts.argmax()] if vals.size else 0
        feats = {**inconsistency_stats(dev, valid=v),
                 "local_frac_argmin_off": float((np.abs(amin[v] - mode_q) >= 10).mean()) if v.any() else 0.0,
                 "global_mode_quality": float(mode_q),
                 "global_mean_deviation": float(dev[v].mean()) if v.any() else 0.0}
        if dev.size < 4:
            return FeatureOutput(None, feats)
        return FeatureOutput(to_evidence(dev, img.shape[:2], self.block, valid=v), feats)
