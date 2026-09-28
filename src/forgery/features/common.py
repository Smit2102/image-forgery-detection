"""Helpers shared by the feature modules.

Every module reduces its evidence to a *block map* (one value per image block) and derives
  * `local_*` features: within-image inconsistency of that map (robust z-scores, outlier clusters,
    spatial autocorrelation, mean z of the top 5 % of blocks). These compare a region with the rest of the same image, so they are
    largely immune to the global compression-history shortcut measured in Phase 2.
  * `global_*` features: absolute image-wide levels. Kept separate so Phase 5 can test them as
    potential shortcuts.
"""

from __future__ import annotations

import cv2
import numpy as np

EPS = 1e-6


def gray(img: np.ndarray) -> np.ndarray:
    """JPEG luminance (ITU-R BT.601) as float32 in [0, 255]."""
    x = img.astype(np.float32)
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]


def blocks(x: np.ndarray, b: int) -> np.ndarray:
    """Non-overlapping b x b blocks, cropped to a multiple of b: shape (H//b, W//b, b*b)."""
    h, w = (x.shape[0] // b) * b, (x.shape[1] // b) * b
    return x[:h, :w].reshape(h // b, b, w // b, b).transpose(0, 2, 1, 3).reshape(h // b, w // b, b * b)


def block_mean(x: np.ndarray, b: int) -> np.ndarray:
    return blocks(x, b).mean(axis=2)


def valid_blocks(img: np.ndarray, b: int, min_std: float = 2.0, lo: float = 4.0, hi: float = 251.0) -> np.ndarray:
    """Blocks that carry usable evidence: not flat (grey std >= min_std) and not clipped to black or
    white (block mean inside (lo, hi)). Flat and saturated areas re-compress with no error and hold
    no noise, so without this mask any textured object on a plain background looks "anomalous"."""
    g = blocks(gray(img), b)
    return (g.std(axis=2) >= min_std) & (g.mean(axis=2) > lo) & (g.mean(axis=2) < hi)


def robust_z(m: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """(m - median) / (1.4826 * MAD), statistics taken over `valid` entries; clipped to [-50, 50]."""
    v = m[valid] if valid is not None else m.ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.zeros_like(m, dtype=np.float64)
    med = np.median(v)
    mad = 1.4826 * np.median(np.abs(v - med))
    # floor the scale so near-constant maps (MAD ~ 0) do not turn tiny deviations into huge z
    scale = max(mad, EPS * (abs(med) + 1.0), 0.1 * np.std(v))
    return np.clip((m - med) / scale, -50, 50)


def morans_i(m: np.ndarray) -> float:
    """Moran's I with rook (4-neighbour) weights: > 0 when high values cluster spatially."""
    if m.size < 4 or min(m.shape) < 2:
        return 0.0
    x = m - m.mean()
    den = float((x * x).sum())
    if den <= EPS:
        return 0.0
    num = float((x[:, 1:] * x[:, :-1]).sum() + (x[1:, :] * x[:-1, :]).sum())
    w = x[:, 1:].size + x[1:, :].size
    return (m.size / w) * num / den


def largest_cluster_frac(mask: np.ndarray) -> float:
    """Size of the largest 8-connected component of `mask`, as a fraction of all blocks."""
    if not mask.any():
        return 0.0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return float(stats[1:, cv2.CC_STAT_AREA].max()) / mask.size


def inconsistency_stats(block_map: np.ndarray, valid: np.ndarray | None = None, prefix: str = "") -> dict:
    """Scale-free statistics of how much one part of the map stands out from the rest."""
    m = np.where(np.isfinite(block_map), block_map, np.nan)
    if valid is None:
        valid = np.isfinite(m)
    valid = valid & np.isfinite(m)
    if valid.sum() < 4:
        return {f"{prefix}local_{k}": 0.0 for k in ("z_max", "z_p99", "frac_z3", "cluster_z25", "moran", "top5_z")}
    z = robust_z(m, valid)
    zv = z[valid]
    filled = np.where(valid, m, np.median(m[valid]))
    top = np.sort(zv)[-max(1, int(0.05 * valid.sum())):]
    return {
        f"{prefix}local_z_max": float(zv.max()),
        f"{prefix}local_z_p99": float(np.percentile(zv, 99)),
        f"{prefix}local_frac_z3": float((zv > 3).mean()),
        f"{prefix}local_cluster_z25": largest_cluster_frac((z > 2.5) & valid),
        f"{prefix}local_moran": morans_i(filled),
        f"{prefix}local_top5_z": float(top.mean()),        # scale-free, also for signed / log maps
    }


def to_evidence(block_map: np.ndarray, shape: tuple[int, int], b: int, valid: np.ndarray | None = None,
                z_cap: float = 6.0, signed: bool = True) -> np.ndarray:
    """Robust-z map clipped to [0, z_cap] / z_cap, upsampled to the image size (float32 in [0, 1]).

    `b` is the block size in pixels; the map covers the top-left (H//b*b, W//b*b) region and the
    remainder is padded with edge values.
    """
    z = robust_z(block_map, valid if valid is not None else np.isfinite(block_map))
    z = np.nan_to_num(z, nan=0.0)
    if valid is not None:
        z = np.where(valid, z, 0.0)
    if not signed:
        z = np.abs(z)
    e = (np.clip(z, 0, z_cap) / z_cap).astype(np.float32)
    return upsample(e, shape, b)


def upsample(block_map: np.ndarray, shape: tuple[int, int], b: int, interpolation=cv2.INTER_LINEAR) -> np.ndarray:
    """Resize a block map (one value per b x b block) to the full image size, float32 in [0, 1]."""
    h, w = shape
    nby, nbx = block_map.shape
    up = cv2.resize(block_map.astype(np.float32), (nbx * b, nby * b), interpolation=interpolation)
    up = np.pad(up, ((0, h - up.shape[0]), (0, w - up.shape[1])), mode="edge")
    return np.clip(up, 0.0, 1.0).astype(np.float32)
