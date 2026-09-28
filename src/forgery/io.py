"""Image / mask loading, protocol-aware re-encoding and JPEG metadata."""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# IJG standard luminance quantisation table (natural row-major order, as Pillow returns it).
_STD_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
], dtype=np.int64)


def ijg_table(quality: int) -> np.ndarray:
    """Luminance table libjpeg writes for `quality` (integer arithmetic, as in jpeg_quality_scaling)."""
    scale = 5000 // quality if quality < 50 else 200 - 2 * quality
    return np.clip((_STD_LUMA * scale + 50) // 100, 1, 255).astype(np.float64)


_IJG_TABLES = {q: ijg_table(q) for q in range(1, 101)}


def is_standard_ijg(luma_table) -> bool:
    """True if the table is exactly an IJG (libjpeg) table for some quality factor."""
    t = np.asarray(luma_table, dtype=np.float64)
    return any(np.array_equal(t, v) for v in _IJG_TABLES.values())


def estimate_jpeg_quality(luma_table) -> int:
    """Closest IJG quality factor (1-100) for a luminance quantisation table."""
    t = np.asarray(luma_table, dtype=np.float64)
    return min(_IJG_TABLES, key=lambda q: float(np.abs(_IJG_TABLES[q] - t).sum()))


def load_rgb(path: str | Path) -> np.ndarray:
    """Load any CASIA image (JPEG/TIFF/BMP) as an HxWx3 uint8 RGB array."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()


def jpeg_encode(img: np.ndarray, quality: int, subsampling: str = "4:2:0") -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=int(quality), subsampling=subsampling)
    return buf.getvalue()


def jpeg_roundtrip(img: np.ndarray, quality: int, subsampling: str = "4:2:0") -> np.ndarray:
    """Encode to JPEG in memory and decode again (deterministic)."""
    with Image.open(io.BytesIO(jpeg_encode(img, quality, subsampling))) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()


def resample(img: np.ndarray, factor: float) -> np.ndarray:
    """Downscale with area interpolation (the protocol-R resampling step)."""
    h, w = img.shape[:2]
    size = (max(1, round(w * factor)), max(1, round(h * factor)))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def apply_protocol(img: np.ndarray, protocol: str, cfg: dict) -> np.ndarray:
    """Transform already-loaded RGB pixels according to an evaluation protocol.

    A / C: unchanged.
    B:     re-encoded once to JPEG at a fixed quality (removes container / quantisation-table shortcuts).
    R:     downscaled, then re-encoded (additionally erases most global compression history).
    """
    if protocol in ("A", "C"):
        return img
    if protocol == "B":
        b = cfg["protocols"]["B"]
        return jpeg_roundtrip(img, b["jpeg_quality"], b["subsampling"])
    if protocol == "R":
        r = cfg["protocols"]["R"]
        return jpeg_roundtrip(resample(img, r["factor"]), r["jpeg_quality"], r["subsampling"])
    raise ValueError(f"unknown protocol {protocol!r}")


def load_image(path: str | Path, protocol: str = "A", cfg: dict | None = None) -> np.ndarray:
    """Load an image under an evaluation protocol (see apply_protocol)."""
    if cfg is None:
        from .config import load_config
        cfg = load_config()
    return apply_protocol(load_rgb(path), protocol, cfg)


def protocol_mask(mask: np.ndarray, protocol: str, cfg: dict) -> np.ndarray:
    """Bring a ground-truth mask to the geometry of the protocol's image (nearest-neighbour for R)."""
    if protocol != "R":
        return mask
    h, w = mask.shape
    f = cfg["protocols"]["R"]["factor"]
    size = (max(1, round(w * f)), max(1, round(h * f)))
    return cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)


def load_mask(path: str | Path, threshold: int = 127) -> np.ndarray:
    """Load a ground-truth mask as a boolean HxW array (True = tampered)."""
    with Image.open(path) as im:
        return np.asarray(im.convert("L")) > threshold


def file_metadata(path: str | Path) -> dict:
    """Container-level facts: format, size, JPEG quantisation info, bytes per pixel."""
    path = Path(path)
    with Image.open(path) as im:
        w, h = im.size
        fmt = (im.format or "").upper()
        meta = {"format": fmt, "width": w, "height": h, "mode": im.mode,
                "jpeg_quality": -1, "qtable_luma_dc": -1, "qtable_luma_mean": -1.0,
                "qtable_standard": False, "n_qtables": 0}
        if fmt == "JPEG" and getattr(im, "quantization", None):
            luma = np.asarray(im.quantization[0], dtype=np.float64)
            meta.update(jpeg_quality=estimate_jpeg_quality(luma),
                        qtable_luma_dc=int(luma[0]),
                        qtable_luma_mean=float(luma.mean()),
                        qtable_standard=is_standard_ijg(luma),
                        n_qtables=len(im.quantization))
    meta["file_bytes"] = path.stat().st_size
    meta["bytes_per_pixel"] = meta["file_bytes"] / (w * h)
    return meta


def bytes_metadata(data: bytes) -> dict:
    """Same as file_metadata but for an in-memory encoded image (used for protocol B)."""
    with Image.open(io.BytesIO(data)) as im:
        w, h = im.size
        luma = np.asarray(im.quantization[0], dtype=np.float64)
        return {"format": "JPEG", "width": w, "height": h,
                "jpeg_quality": estimate_jpeg_quality(luma),
                "qtable_luma_dc": int(luma[0]), "qtable_luma_mean": float(luma.mean()),
                "qtable_standard": is_standard_ijg(luma), "n_qtables": len(im.quantization),
                "file_bytes": len(data), "bytes_per_pixel": len(data) / (w * h)}
