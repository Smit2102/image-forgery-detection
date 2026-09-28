"""Copy-move detection: keypoint matching and block matching.

Copy-move forgery duplicates a region inside the same image, so the image is matched against itself,
and also against its horizontal mirror, because CASIA copy-moves are often flipped copies.

* Keypoints (corner/blob detection): keypoints come from SIFT's DoG detector (scale- and
  rotation-invariant) or from Harris corners, and get SIFT descriptors. Each keypoint is matched
  against all others, and against descriptors computed at the mirrored positions of the flipped image,
  with the generalised 2-nearest-neighbour test (g2NN; Amerini et al. 2011). Pairs closer than
  `min_shift` pixels are dropped. Every pair is oriented left point first. RANSAC keeps pairs
  consistent with one affine transform, separately for direct and mirrored matches.
  Limitation: one RANSAC per mode recovers only one copied region per mode (the dominant transform);
  a second direct or second mirrored copy with a different transform counts as outliers.
* Blocks (Fridrich et al. 2003 style): every overlapping 16x16 block is described by its
  low-frequency DCT coefficients. The blocks of the image and of its mirror are sorted together
  lexicographically, near-identical neighbours in sort order are paired, and shift vectors are voted
  on. For a mirrored copy the shift measured in (image, mirror) coordinates is constant. This works in
  the flat, corner-free regions where keypoints fail. Nearly flat blocks are skipped because they
  match everything.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.fft import dctn

from .base import FeatureModule, FeatureOutput, register
from .common import gray


def _point_map(points: np.ndarray, shape: tuple[int, int], sigma: float) -> np.ndarray:
    m = np.zeros(shape, np.float32)
    if len(points):
        xy = np.clip(np.rint(points).astype(int), 0, [shape[1] - 1, shape[0] - 1])
        m[xy[:, 1], xy[:, 0]] = 1.0
        m = cv2.GaussianBlur(m, (0, 0), sigma)
        m /= m.max() + 1e-12
    return m


@register
class KeypointCopyMove(FeatureModule):
    name = "cm_keypoint"
    category = "corner"

    def __init__(self, detector: str = "sift", contrast: float = 0.01, max_corners: int = 1500,
                 harris_quality: float = 0.001, patch: float = 16.0, ratio: float = 0.6,
                 min_shift: float = 20.0, ransac_px: float = 3.0, min_inliers: int = 4, mirror: bool = True):
        if detector not in ("sift", "harris"):
            raise ValueError(f"unknown detector {detector!r}")
        super().__init__(detector=detector, contrast=contrast, max_corners=max_corners,
                         harris_quality=harris_quality, patch=patch, ratio=ratio, min_shift=min_shift,
                         ransac_px=ransac_px, min_inliers=min_inliers, mirror=mirror)
        self.detector, self.contrast, self.max_corners = detector, contrast, max_corners
        self.harris_quality, self.patch, self.ratio, self.min_shift = harris_quality, patch, ratio, min_shift
        self.ransac_px, self.min_inliers, self.mirror = ransac_px, min_inliers, mirror
        self._sift = None

    @property
    def sift(self):
        if self._sift is None:
            self._sift = cv2.SIFT_create(contrastThreshold=self.contrast)
        return self._sift

    def _keypoints(self, g8: np.ndarray) -> list:
        if self.detector == "sift":
            kps = sorted(self.sift.detect(g8, None), key=lambda k: -k.response)
            return kps[: self.max_corners]
        c = cv2.goodFeaturesToTrack(g8, self.max_corners, self.harris_quality, 3, useHarrisDetector=True, k=0.04)
        return [] if c is None else [cv2.KeyPoint(float(x), float(y), self.patch) for x, y in c.reshape(-1, 2)]

    def match(self, img: np.ndarray) -> dict:
        g8 = np.clip(gray(img), 0, 255).astype(np.uint8)
        out = {"n_keypoints": 0, "src": np.zeros((0, 2)), "dst": np.zeros((0, 2)),
               "inliers": np.zeros(0, bool), "mirrored": np.zeros(0, bool)}
        kps = self._keypoints(g8)
        if len(kps) < 3:
            return out
        kps, desc = self.sift.compute(g8, kps)
        if desc is None or len(kps) < 3:
            return out
        n, w = len(kps), g8.shape[1]
        pts = np.array([kp.pt for kp in kps])
        targets = [(desc, False)]
        if self.mirror:   # descriptors of the same points, seen in the mirrored image
            fk = [cv2.KeyPoint(w - 1 - kp.pt[0], kp.pt[1], kp.size,
                               (180.0 - kp.angle) % 360 if kp.angle >= 0 else -1, kp.response, kp.octave)
                  for kp in kps]
            fk, fdesc = self.sift.compute(np.ascontiguousarray(g8[:, ::-1]), fk)
            if fdesc is not None and len(fk) == n:
                targets.append((fdesc, True))
        out["n_keypoints"] = n
        pairs = {}
        for target, mirrored in targets:        # g2NN run separately for direct and mirrored matching
            knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc, target, k=min(10, n))
            for i, ms in enumerate(knn):
                ms = [m for m in ms if m.trainIdx != i]         # drop the self (or self-mirror) match
                for a, b in zip(ms, ms[1:]):                    # accept while d_j / d_{j+1} < ratio
                    if a.distance >= self.ratio * max(b.distance, 1e-6):
                        break
                    j = a.trainIdx
                    if np.hypot(*(pts[i] - pts[j])) <= self.min_shift:
                        continue
                    # orient every pair left point first. A translation has a direction, and a
                    # mirrored copy with a vertical offset is a glide reflection, whose inverse is a
                    # different transform, so orienting by index would split its pairs in two.
                    i2, j2 = (i, j) if tuple(pts[i]) <= tuple(pts[j]) else (j, i)
                    pairs[(i2, j2, int(mirrored))] = True
        if not pairs:
            return out
        idx = np.array(list(pairs), dtype=int)
        src, dst, mir = pts[idx[:, 0]], pts[idx[:, 1]], idx[:, 2].astype(bool)
        inliers = np.zeros(len(idx), bool)
        for group in (~mir, mir):                               # one transform per matching mode
            if group.sum() >= 3:
                _, mask = cv2.estimateAffine2D(src[group], dst[group], method=cv2.RANSAC,
                                               ransacReprojThreshold=self.ransac_px)
                if mask is not None and mask.sum() >= self.min_inliers:
                    inliers[np.flatnonzero(group)[mask.ravel().astype(bool)]] = True
        out.update(src=src, dst=dst, inliers=inliers, mirrored=mir)
        return out

    def extract(self, img: np.ndarray) -> FeatureOutput:
        m = self.match(img)
        h, w = img.shape[:2]
        inl = m["inliers"]
        n_match, n_in = len(m["src"]), int(inl.sum())
        pts = np.vstack([m["src"][inl], m["dst"][inl]]) if n_in else np.zeros((0, 2))
        hull = 0.0
        if n_in >= 3:
            hull = sum(cv2.contourArea(cv2.convexHull(p.astype(np.float32)))
                       for p in (m["src"][inl], m["dst"][inl])) / (h * w)
        shift = np.hypot(*(m["dst"][inl] - m["src"][inl]).T).mean() / np.hypot(h, w) if n_in else 0.0
        feats = {"local_n_matches": float(n_match), "local_n_inliers": float(n_in),
                 "local_inlier_frac": n_in / n_match if n_match else 0.0,
                 "local_log_inliers": float(np.log1p(n_in)), "local_hull_frac": float(hull),
                 "local_shift": float(shift),
                 "local_frac_mirrored": float(m["mirrored"][inl].mean()) if n_in else 0.0,
                 "global_n_keypoints": float(m["n_keypoints"])}
        return FeatureOutput(_point_map(pts, (h, w), sigma=6.0), feats)


def _zigzag(n: int, size: int) -> list[tuple[int, int]]:
    """First `n` (row u, column v) indices of the JPEG zigzag scan of a `size` x `size` block.

    Anti-diagonal s = u + v is walked with u decreasing when s is even and u increasing when s is
    odd, giving (0,0), (0,1), (1,0), (2,0), (1,1), (0,2), ...
    """
    order = sorted(((u, v) for u in range(size) for v in range(size)),
                   key=lambda t: (t[0] + t[1], t[0] if (t[0] + t[1]) % 2 else -t[0]))
    return order[:n]


@register
class BlockCopyMove(FeatureModule):
    name = "cm_block"
    category = "frequency"

    def __init__(self, block: int = 16, stride: int = 4, n_coef: int = 9, tol: float = 1.5,
                 min_std: float = 4.0, neighbors: int = 5, min_shift: float = 24.0, min_votes: int = 8,
                 mirror: bool = True):
        super().__init__(block=block, stride=stride, n_coef=n_coef, tol=tol, min_std=min_std,
                         neighbors=neighbors, min_shift=min_shift, min_votes=min_votes, mirror=mirror)
        self.block, self.stride, self.n_coef, self.tol = block, stride, n_coef, tol
        self.min_std, self.neighbors, self.min_shift, self.min_votes = min_std, neighbors, min_shift, min_votes
        self.mirror = mirror
        self._zz = _zigzag(n_coef, block)

    def _describe(self, g: np.ndarray, x_offset: int = 0):
        b, s = self.block, self.stride
        if g.shape[0] < b or g.shape[1] < b + x_offset:         # image smaller than one block
            return np.zeros((0, self.n_coef)), np.zeros(0, int), np.zeros(0, int), 0
        win = sliding_window_view(g, (b, b))[::s, x_offset::s]
        ny, nx = win.shape[:2]
        ys, xs = np.meshgrid(np.arange(ny) * s, np.arange(nx) * s + x_offset, indexing="ij")
        win, ys, xs = win.reshape(-1, b, b), ys.ravel(), xs.ravel()
        keep = win.std(axis=(1, 2)) >= self.min_std
        if not keep.any():
            return np.zeros((0, self.n_coef)), ys[:0], xs[:0], len(keep)
        c = dctn(win[keep], axes=(1, 2), norm="ortho")
        feat = np.stack([c[:, u, v] for u, v in self._zz], axis=1) / b     # per-pixel scale
        return feat, ys[keep], xs[keep], len(keep)

    def match(self, img: np.ndarray) -> dict:
        g = gray(img).astype(np.float32)
        h, w = g.shape
        b = self.block
        fa, ya, xa, n_total = self._describe(g)
        res = {"blocks": np.zeros((0, 2), int), "votes": 0, "n_shifts": 0, "n_blocks": int(len(fa)),
               "n_total": n_total, "mirrored_votes": 0}
        if len(fa) < 2:
            return res
        sets = [(fa, ya, xa, 0)]
        if self.mirror:
            # start the mirrored grid at (w - b) % stride so that a mirrored block at x maps to the
            # original column w - b - x, which lies on the image's own block grid
            fb, yb, xb, _ = self._describe(np.ascontiguousarray(g[:, ::-1]), x_offset=(w - b) % self.stride)
            sets.append((fb, yb, xb, 1))
        f = np.vstack([s[0] for s in sets])
        y = np.concatenate([s[1] for s in sets])
        x = np.concatenate([s[2] for s in sets])
        src = np.concatenate([np.full(len(s[1]), s[3]) for s in sets])
        xo = np.where(src == 1, w - b - x, x)                 # block position in original coordinates
        order = np.lexsort(np.rint(f / self.tol).T[::-1])
        f, y, x, xo, src = f[order], y[order], x[order], xo[order], src[order]
        cand = []
        for d in range(1, self.neighbors + 1):
            i = np.arange(len(f) - d)
            j = i + d
            close = np.abs(f[j] - f[i]).max(axis=1) <= self.tol
            far = np.hypot(y[j] - y[i], xo[j] - xo[i]) >= self.min_shift
            keep = close & far & ((src[i] == 0) & (src[j] == 0) | (src[i] != src[j]))
            cand.append(np.stack([i[keep], j[keep]], axis=1))
        pairs = np.concatenate(cand)
        if not len(pairs):
            return res
        swap = src[pairs[:, 0]] == 1                          # cross pairs: image block first
        pairs[swap] = pairs[swap][:, ::-1]
        a, c = pairs[:, 0], pairs[:, 1]
        mirrored = src[c] == 1
        dy, dx = y[c] - y[a], x[c] - x[a]                     # (image, mirror) coordinates for cross pairs
        flip = ~mirrored & ((dx < 0) | ((dx == 0) & (dy < 0)))  # direct pairs: canonical direction
        dy, dx = np.where(flip, -dy, dy), np.where(flip, -dx, dx)
        key = mirrored.astype(np.int64) * 10 ** 10 + (dy // 2 + 5000) * 100000 + (dx // 2 + 5000)
        uniq, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
        sig = counts[inv] >= self.min_votes
        pos = np.concatenate([np.stack([y[a[sig]], xo[a[sig]]], 1), np.stack([y[c[sig]], xo[c[sig]]], 1)])
        is_m = uniq >= 10 ** 10
        res.update(blocks=np.unique(pos, axis=0) if len(pos) else res["blocks"], votes=int(counts.max()),
                   n_shifts=int((counts >= self.min_votes).sum()),
                   mirrored_votes=int(counts[is_m].max()) if is_m.any() else 0)
        return res

    def extract(self, img: np.ndarray) -> FeatureOutput:
        m = self.match(img)
        h, w = img.shape[:2]
        cover = np.zeros((h, w), np.float32)
        b = self.block
        for yy, xx in m["blocks"]:
            cover[yy:yy + b, xx:xx + b] = 1.0
        feats = {"local_max_votes": float(m["votes"]), "local_log_votes": float(np.log1p(m["votes"])),
                 "local_mirrored_votes": float(m["mirrored_votes"]),
                 "local_n_shifts": float(m["n_shifts"]), "local_area_frac": float(cover.mean()),
                 "global_frac_textured_blocks": m["n_blocks"] / max(1, m["n_total"])}
        return FeatureOutput(cover, feats)
