"""Phase 6: fuse the eight evidence maps into one tampered-region mask.

Every module's evidence map is reduced to 8x8-pixel *cells*. A cell is described by the 8 evidence
values and their 3x3-cell neighbourhood means (context). A classifier trained on cells of training
images (label: at least half of the cell is inside the ground-truth mask) turns this into a per-cell
tampering probability. The probability map is upsampled to full resolution, thresholded, cleaned with
morphological opening / closing, and components smaller than a minimum area are removed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

CELL = 8


def cells(m: np.ndarray, cell: int = CELL) -> np.ndarray:
    """Average a full-resolution map over cell x cell pixels (cropped to a multiple of `cell`)."""
    h, w = (m.shape[0] // cell) * cell, (m.shape[1] // cell) * cell
    if h == 0 or w == 0:
        return np.zeros((max(1, h // cell), max(1, w // cell)), np.float32)
    return m[:h, :w].astype(np.float32).reshape(h // cell, cell, w // cell, cell).mean(axis=(1, 3))


def evidence_stack(maps: dict[str, np.ndarray | None], names: list[str], shape: tuple[int, int]) -> np.ndarray:
    """(n_modules, H // CELL, W // CELL) stack; a module without a map contributes zeros."""
    empty = np.zeros(shape, np.float32)
    return np.stack([cells(maps.get(n) if maps.get(n) is not None else empty) for n in names])


def cell_features(stack: np.ndarray) -> np.ndarray:
    """Per-cell features: the evidence values and their 3x3 neighbourhood means -> (n_cells, 2 * n_mod)."""
    ctx = np.stack([cv2.blur(s.astype(np.float32), (3, 3), borderType=cv2.BORDER_REFLECT) for s in stack])
    return np.concatenate([stack, ctx]).reshape(2 * stack.shape[0], -1).T


def upsample(prob: np.ndarray, shape: tuple[int, int], cell: int = CELL) -> np.ndarray:
    """Cell map -> full image size (bilinear), edge-padded where the image is not a multiple of `cell`."""
    h, w = shape
    ch, cw = prob.shape
    up = cv2.resize(prob.astype(np.float32), (cw * cell, ch * cell), interpolation=cv2.INTER_LINEAR)
    return np.pad(up, ((0, max(0, h - up.shape[0])), (0, max(0, w - up.shape[1]))), mode="edge")[:h, :w]


def postprocess(prob: np.ndarray, threshold: float, open_k: int, close_k: int, min_area: float) -> np.ndarray:
    """Threshold, morphological opening then closing (elliptical kernels), drop small components."""
    m = (prob >= threshold).astype(np.uint8)
    if open_k > 1:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k)))
    if close_k > 1:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)))
    if min_area > 0 and m.any():
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        keep = np.zeros(n, bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area * m.size
        m = keep[lab].astype(np.uint8)
    return m.astype(bool)


def scores(pred: np.ndarray, truth: np.ndarray) -> dict:
    """Pixel-level IoU, F1 and Matthews correlation of a predicted mask against the ground truth."""
    p, t = pred.ravel(), truth.ravel()
    tp = float((p & t).sum())
    fp = float((p & ~t).sum())
    fn = float((~p & t).sum())
    tn = float(p.size) - tp - fp - fn
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den else 0.0
    return {"iou": iou, "f1": f1, "mcc": float(mcc), "pred_area": float(p.mean())}


@dataclass
class Localizer:
    """Fitted fusion model + post-processing; `predict(maps, shape)` -> (probability map, mask)."""

    names: list[str]
    model: object                       # has predict_proba on cell features (None -> mean of maps)
    threshold: float = 0.5
    open_k: int = 1
    close_k: int = 1
    min_area: float = 0.0
    meta: dict = field(default_factory=dict)

    def cell_prob(self, stack: np.ndarray) -> np.ndarray:
        # the fusion model was trained on stacks stored as float16: round the same way at inference
        stack = np.asarray(stack, dtype=np.float16).astype(np.float32)
        if self.model is None:
            return stack.mean(axis=0)
        return self.model.predict_proba(cell_features(stack))[:, 1].reshape(stack.shape[1:])

    def predict(self, maps: dict[str, np.ndarray | None], shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        prob = upsample(self.cell_prob(evidence_stack(maps, self.names, shape)), shape)
        return prob, postprocess(prob, self.threshold, self.open_k, self.close_k, self.min_area)
