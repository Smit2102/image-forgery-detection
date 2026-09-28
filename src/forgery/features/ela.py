"""Error Level Analysis (point processing).

Re-save the image as JPEG at quality q and take the per-pixel absolute difference. Content that was
compressed differently from the rest re-compresses at a different error level. Because ELA also grows
with local texture, the map is divided by the block's texture (grey-level standard deviation) so that
busy regions are not mistaken for tampering.
"""

from __future__ import annotations

import numpy as np

from ..io import jpeg_roundtrip
from .base import FeatureModule, FeatureOutput, register
from .common import block_mean, blocks, gray, inconsistency_stats, to_evidence, valid_blocks


def ela_residual(img: np.ndarray, quality: int) -> np.ndarray:
    """Mean absolute RGB difference between the image and its JPEG re-save (float32, >= 0)."""
    diff = np.abs(img.astype(np.int16) - jpeg_roundtrip(img, quality).astype(np.int16))
    return diff.mean(axis=2).astype(np.float32)


@register
class ELA(FeatureModule):
    name = "ela"
    category = "point"

    def __init__(self, quality: int = 90, block: int = 8, texture_c: float = 4.0, min_std: float = 2.0):
        super().__init__(quality=quality, block=block, texture_c=texture_c, min_std=min_std)
        self.quality, self.block, self.texture_c, self.min_std = quality, block, texture_c, min_std

    def maps(self, img: np.ndarray) -> dict[str, np.ndarray]:
        """Intermediate results, for figures: residual, block ELA, block texture, normalised block ELA."""
        resid = ela_residual(img, self.quality)
        b = self.block
        ela_b = block_mean(resid, b)
        tex_b = blocks(gray(img), b).std(axis=2)
        return {"residual": resid, "ela_block": ela_b, "texture_block": tex_b,
                "norm_block": ela_b / (tex_b + self.texture_c),
                "valid": valid_blocks(img, b, self.min_std)}

    def extract(self, img: np.ndarray) -> FeatureOutput:
        m = self.maps(img)
        if min(m["ela_block"].shape) < 2:
            return FeatureOutput(None, {**inconsistency_stats(np.zeros((1, 1)), prefix="raw_"),
                                        **inconsistency_stats(np.zeros((1, 1)), prefix="norm_"),
                                        "global_mean": 0.0, "global_p95": 0.0})
        v = m["valid"]
        feats = {**inconsistency_stats(m["ela_block"], valid=v, prefix="raw_"),
                 **inconsistency_stats(m["norm_block"], valid=v, prefix="norm_"),
                 "global_mean": float(m["residual"].mean()),
                 "global_p95": float(np.percentile(m["residual"], 95))}
        return FeatureOutput(to_evidence(m["norm_block"], img.shape[:2], self.block, valid=v), feats)
