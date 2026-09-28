"""Copy-move detectors: mirrored copies (keypoint and block) and the DCT zigzag order."""

import cv2
import numpy as np
import pytest

from forgery.features.copymove import BlockCopyMove, KeypointCopyMove, _zigzag
from forgery.io import jpeg_roundtrip


def texture(seed=0, h=256, w=320, noise=0.0) -> np.ndarray:
    """Multi-scale blurred-noise texture (uint8 RGB), about 128 +- 50."""
    rng = np.random.default_rng(seed)
    acc = np.zeros((h, w), np.float32)
    for sigma, amp in ((24, 60), (8, 30), (2, 15)):
        acc += amp * cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), sigma) * sigma / 2
    acc = 128 + 50 * acc / (acc.std() + 1e-6)
    rgb = np.stack([acc, acc * 0.9 + 12, acc * 0.8 + 25], axis=-1)
    if noise:
        rgb += rng.normal(0, noise, rgb.shape)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def mirrored_copy(w=320) -> np.ndarray:
    """A 64 x 80 patch, flipped left-right, pasted with a vertical offset of 80 (a glide reflection)."""
    img = jpeg_roundtrip(texture(w=w), 90).copy()
    img[100:164, 180:260] = img[20:84, 20:100][:, ::-1]
    return jpeg_roundtrip(img, 85)


def test_keypoint_mirrored_copy_with_vertical_offset():
    f = KeypointCopyMove().extract(mirrored_copy()).features
    assert f["local_n_inliers"] >= 4
    assert f["local_frac_mirrored"] > 0.5


@pytest.mark.parametrize("w", [320, 323])
def test_block_mirrored_copy_any_width(w):
    f = BlockCopyMove().extract(mirrored_copy(w)).features
    assert f["local_mirrored_votes"] >= 8


def test_zigzag_matches_jpeg():
    assert _zigzag(10, 8) == [(0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2), (2, 1), (3, 0)]
    full = _zigzag(64, 8)
    assert len(set(full)) == 64 and full[-1] == (7, 7)
