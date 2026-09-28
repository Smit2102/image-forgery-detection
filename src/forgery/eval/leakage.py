"""Shortcut ("leakage") baselines: how far can non-localised cues go on CASIA v2.0?

Feature families, all computed globally over the whole image (none localises anything):
  * metadata     - container format, JPEG quantisation table, bytes per pixel
  * dimensions   - width, height, aspect ratio, megapixels
  * naive_ela    - statistics of one error-level-analysis residual
  * dct_dq       - global histograms of quantised 8x8-DCT coefficients (double-quantisation traces)
  * ghost_curve  - global re-compression error at a ladder of JPEG qualities
The last two capture *compression history*, which survives re-encoding. If these classify well
under a protocol, a forensic feature must beat them to show it detects tampering.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from ..io import bytes_metadata, file_metadata, ijg_table, jpeg_encode, jpeg_roundtrip, load_rgb, resample

DQ_POSITIONS = ((0, 1), (1, 0), (1, 1), (0, 2), (2, 0))
GHOST_QUALITIES = (60, 65, 70, 75, 80, 85, 90, 95)
FEATURE_VERSION = 3   # bump when feature definitions change (invalidates the cache)

FEATURE_SETS = {
    "metadata": ["is_jpeg", "is_tiff", "is_bmp", "jpeg_quality", "qtable_luma_dc",
                 "qtable_luma_mean", "qtable_standard", "bytes_per_pixel"],
    "dimensions": ["width", "height", "aspect", "megapixels"],
    "naive_ela": ["ela_mean", "ela_std", "ela_p95", "ela_p99"],
    "dct_dq": [f"dq{u}{v}_{s}" for u, v in DQ_POSITIONS for s in ("p0", "p1", "p2", "even_odd", "period")],
    "ghost_curve": [f"ghost_{q}" for q in GHOST_QUALITIES],
}
FEATURE_SETS["compression_history"] = FEATURE_SETS["dct_dq"] + FEATURE_SETS["ghost_curve"]
FEATURE_SETS["all"] = [c for k in ("metadata", "dimensions", "naive_ela", "dct_dq", "ghost_curve")
                       for c in FEATURE_SETS[k]]


def naive_ela_stats(img: np.ndarray, quality: int) -> dict:
    resid = np.abs(img.astype(np.int16) - jpeg_roundtrip(img, quality).astype(np.int16)).mean(axis=2)
    return {"ela_mean": float(resid.mean()), "ela_std": float(resid.std()),
            "ela_p95": float(np.percentile(resid, 95)), "ela_p99": float(np.percentile(resid, 99))}


def block_dct_luma(img: np.ndarray) -> np.ndarray:
    """8x8 orthonormal DCT of the JPEG luminance (Y - 128), grid-aligned: shape (nby, nbx, 8, 8)."""
    rgb = img.astype(np.float64)
    y = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2] - 128.0
    h, w = (y.shape[0] // 8) * 8, (y.shape[1] // 8) * 8
    blocks = y[:h, :w].reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3)
    return dctn(blocks, axes=(2, 3), norm="ortho")


def dct_dq_stats(img: np.ndarray, luma_table: np.ndarray | None) -> dict:
    """Histogram statistics of quantised DCT coefficients at a few low frequencies.

    Coefficients are divided by the image's last quantisation step (1 if it was never JPEG),
    so a double-quantised image shows periodic gaps/peaks in the integer histogram.
    """
    coef = block_dct_luma(img)
    table = None if luma_table is None else np.asarray(luma_table, dtype=np.float64).reshape(8, 8)
    out = {}
    for u, v in DQ_POSITIONS:
        step = 1.0 if table is None else table[u, v]
        k = np.rint(coef[..., u, v].ravel() / step).astype(np.int64)
        a = np.abs(k)
        hist = np.bincount(np.clip(a, 0, 16), minlength=17)[1:16].astype(np.float64)   # |k| = 1..15
        spec = np.abs(np.fft.rfft(hist / max(hist.sum(), 1.0)))
        out.update({f"dq{u}{v}_p0": float((a == 0).mean()), f"dq{u}{v}_p1": float((a == 1).mean()),
                    f"dq{u}{v}_p2": float((a == 2).mean()),
                    f"dq{u}{v}_even_odd": float(hist[1::2].sum() / (hist[0::2].sum() + 1.0)),
                    f"dq{u}{v}_period": float(spec[1:].max() / (spec[0] + 1e-9))})
    return out


def ghost_curve(img: np.ndarray) -> dict:
    """Mean absolute re-compression error at each quality of the ladder (a global JPEG ghost)."""
    x = img.astype(np.int16)
    return {f"ghost_{q}": float(np.abs(x - jpeg_roundtrip(img, q).astype(np.int16)).mean())
            for q in GHOST_QUALITIES}


def _meta_row(meta: dict) -> dict:
    fmt = meta["format"]
    w, h = meta["width"], meta["height"]
    return {"is_jpeg": float(fmt == "JPEG"), "is_tiff": float(fmt == "TIFF"), "is_bmp": float(fmt == "BMP"),
            "jpeg_quality": meta["jpeg_quality"], "qtable_luma_dc": meta["qtable_luma_dc"],
            "qtable_luma_mean": meta["qtable_luma_mean"], "qtable_standard": float(meta["qtable_standard"]),
            "bytes_per_pixel": meta["bytes_per_pixel"],
            "width": w, "height": h, "aspect": w / h, "megapixels": w * h / 1e6}


def _decode(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def _file_luma_table(path: str) -> np.ndarray | None:
    with Image.open(path) as im:
        q = getattr(im, "quantization", None)
        return np.asarray(q[0], dtype=np.float64) if (im.format == "JPEG" and q) else None


def _row(protocol: str, meta: dict, img: np.ndarray, table, ela_quality: int) -> dict:
    return {"protocol": protocol, **_meta_row(meta), **naive_ela_stats(img, ela_quality),
            **dct_dq_stats(img, table), **ghost_curve(img)}


def image_features(path: str, b_qualities: list[int], ela_quality: int, subsampling: str,
                   resample_cfg: dict | None = None) -> list[dict]:
    """Rows for protocol A (as released), B<q> (re-encoded at q) and optionally R<q> (resampled, then
    re-encoded) for one image."""
    img = load_rgb(path)
    rows = [_row("A", file_metadata(path), img, _file_luma_table(path), ela_quality)]
    for q in b_qualities:
        data = jpeg_encode(img, q, subsampling)
        rows.append(_row(f"B{q}", bytes_metadata(data), _decode(data), ijg_table(q), ela_quality))
    if resample_cfg:
        q = resample_cfg["jpeg_quality"]
        data = jpeg_encode(resample(img, resample_cfg["factor"]), q, resample_cfg["subsampling"])
        rows.append(_row(f"R{q}", bytes_metadata(data), _decode(data), ijg_table(q), ela_quality))
    return rows


def grouped_cv(X: np.ndarray, y: np.ndarray, folds: np.ndarray, n_trees: int, seed: int) -> list[dict]:
    """Random-forest scores on each held-out grouped fold."""
    out = []
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        clf = RandomForestClassifier(n_estimators=n_trees, min_samples_leaf=2, class_weight="balanced",
                                     n_jobs=-1, random_state=seed)
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        pred = (p >= 0.5).astype(int)
        out.append({"fold": int(f), "n_test": int(te.sum()),
                    "balanced_accuracy": balanced_accuracy_score(y[te], pred),
                    "accuracy": accuracy_score(y[te], pred),
                    "roc_auc": roc_auc_score(y[te], p),
                    "majority_accuracy": max(y[te].mean(), 1 - y[te].mean())})
    return out


def run_study(feats: pd.DataFrame, protocols: dict[str, pd.Series], n_trees: int, seed: int,
              feature_sets: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """feats: one row per (path, protocol source). protocols: name -> boolean row filter."""
    feature_sets = feature_sets or FEATURE_SETS
    records = []
    for pname, rows in protocols.items():
        d = feats[rows]
        if d.empty:
            raise ValueError(f"protocol {pname!r} selects no rows")
        y, folds = d["label"].to_numpy(), d["cv_fold"].to_numpy()
        for fs, cols in feature_sets.items():
            for r in grouped_cv(d[cols].to_numpy(dtype=float), y, folds, n_trees, seed):
                records.append({"protocol": pname, "feature_set": fs, **r})
    return pd.DataFrame(records)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    g = results.groupby(["protocol", "feature_set"], sort=False)
    return g.agg(balanced_accuracy_mean=("balanced_accuracy", "mean"),
                 balanced_accuracy_std=("balanced_accuracy", "std"),
                 roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"),
                 accuracy_mean=("accuracy", "mean"), majority_accuracy=("majority_accuracy", "mean"),
                 n_images=("n_test", "sum")).reset_index()
