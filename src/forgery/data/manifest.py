"""Build the per-image manifest (one row per CASIA v2.0 image)."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..config import ROOT, load_config, resolve
from ..io import file_metadata, load_mask, load_rgb
from .naming import mask_tamper_id, parse_authentic, parse_tampered

IMAGE_EXTS = {".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
AREA_BUCKETS = [(0.0, 0.01, "<1%"), (0.01, 0.05, "1-5%"), (0.05, 1.01, ">5%")]
LETTER_TYPE = {"S": "copy-move", "D": "splicing"}

# Columns that must stay strings when the CSV is read back (ids with leading zeros, hex hashes).
STR_COLUMNS = ["path", "image_id", "host_id", "donor_id", "group_id", "tamper_id",
               "flags", "mask_path", "pixel_md5", "dhash", "category"]


def read_manifest(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    return pd.read_csv(resolve(cfg["paths"]["manifest"]), dtype={c: str for c in STR_COLUMNS})


def area_bucket(frac: float) -> str | None:
    if frac is None or not np.isfinite(frac):
        return None
    for lo, hi, name in AREA_BUCKETS:
        if lo <= frac < hi:
            return name
    return None


def dhash(img: np.ndarray) -> str:
    """64-bit difference hash (hex) used to find near-duplicate images."""
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (9, 8), interpolation=cv2.INTER_AREA)
    return np.packbits((g[:, 1:] > g[:, :-1]).ravel()).tobytes().hex()


def pixel_similarity(a: np.ndarray, b: np.ndarray, tol: int = 10) -> float:
    """Share of pixels whose grey levels differ by <= tol (NaN if the sizes differ)."""
    if a.shape != b.shape:
        return float("nan")
    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.int16)
    gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.int16)
    return float((np.abs(ga - gb) <= tol).mean())


def mask_name_relation(image_stem: str, mask_name: str) -> str:
    """How a mask file name differs from its image name."""
    body = Path(mask_name).stem
    if not body.endswith("_gt"):
        return "odd_suffix"
    body = body[: -len("_gt")]
    if body == image_stem:
        return "identical"
    a, b = image_stem.split("_"), body.split("_")
    if len(a) != len(b):
        return "other"
    diff = {i for i, (x, y) in enumerate(zip(a, b)) if x != y}
    if diff == {1}:
        return "letter_only"
    if diff & {5, 6}:
        return "ids"
    return "other"


def _rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def _list_images(d: Path, prefix: str) -> list[Path]:
    return sorted(p for p in d.iterdir()
                  if p.name.startswith(prefix) and p.suffix.lower() in IMAGE_EXTS)


def _describe(path: Path, mask_path: Path | None, host_path: Path | None, donor_path: Path | None,
              cfg: dict) -> dict:
    rec = file_metadata(path)
    img = load_rgb(path)
    rec["pixel_md5"] = hashlib.md5(img.tobytes()).hexdigest()
    rec["dhash"] = dhash(img)
    if mask_path is None:
        return rec

    # How much of the image comes from each named source? The first id (host) supplies most pixels
    # in the vast majority of cases. When the second id does ("donor_dominant": a large pasted region,
    # or ids swapped in the file name) the image must share that donor's split group.
    host = load_rgb(host_path) if host_path else None
    donor = load_rgb(donor_path) if donor_path and donor_path != host_path else host
    rec["host_similarity"] = pixel_similarity(img, host) if host is not None else np.nan
    rec["donor_similarity"] = pixel_similarity(img, donor) if donor is not None else np.nan
    dominant = (np.isfinite(rec["donor_similarity"]) and rec["donor_similarity"] > 0.5
                and not (rec["donor_similarity"] <= rec["host_similarity"]))
    rec["donor_dominant"] = bool(dominant and donor_path != host_path)

    mcfg = cfg["masks"]
    mask = load_mask(mask_path, mcfg["binarize_threshold"])
    h, w = img.shape[:2]
    rec["mask_height"], rec["mask_width"] = mask.shape
    if mask.shape == (h, w):
        status = "ok"
    elif mask.shape == (w, h):
        status = "transposed"
    else:
        status = "size_mismatch"
    frac = float(mask.mean())
    if status == "ok" and frac == 0.0:
        status = "empty"
    elif status == "ok" and frac > 0.95:
        status = "full"
    rec["mask_status"] = status
    rec["tampered_frac"] = frac
    rec["mask_components"] = int(cv2.connectedComponents(mask.astype(np.uint8))[0] - 1)

    # Sanity check: pixels that differ from the background image should mostly lie inside the mask.
    # The background ("mask reference") is the named image that best matches the *unmasked* area -
    # normally the host, but the donor when the ids in the file name are swapped.
    rec["host_diff_recall"] = rec["host_diff_precision"] = np.nan
    rec["mask_reference"] = None
    if status == "ok":
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        best = None
        for name, cand in (("host", host), ("donor", donor if donor_path != host_path else None)):
            if cand is None or cand.shape != img.shape:
                continue
            gc = cv2.cvtColor(cand, cv2.COLOR_RGB2GRAY).astype(np.float32)
            outside = float((np.abs(g - gc)[~mask] <= 10).mean()) if (~mask).any() else 0.0
            if best is None or outside > best[0]:
                best = (outside, name, gc)
        if best is not None and best[0] > 0.5:   # the reference must explain the unmasked area
            _, rec["mask_reference"], gc = best
            diff = cv2.GaussianBlur(np.abs(g - gc), (5, 5), 0) > mcfg["host_diff_threshold"]
            inter = float((diff & mask).sum())
            rec["host_diff_recall"] = inter / max(mask.sum(), 1)
            rec["host_diff_precision"] = inter / max(diff.sum(), 1)
    return rec


def _majority(votes: list[str]) -> str:
    return Counter(votes).most_common(1)[0][0]


def build_manifest(cfg: dict | None = None, n_jobs: int | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    paths = cfg["paths"]
    au_dir, tp_dir, mask_dir = (resolve(paths[k]) for k in ("authentic_dir", "tampered_dir", "mask_dir"))

    au_files = _list_images(au_dir, "Au_")
    tp_files = _list_images(tp_dir, "Tp_")
    masks_by_tid: dict[str, Path] = {}
    for m in sorted(mask_dir.glob("*_gt*.png")):
        tid = mask_tamper_id(m.name)
        if tid in masks_by_tid:
            raise ValueError(f"two masks share tamper id {tid}: {masks_by_tid[tid].name}, {m.name}")
        masks_by_tid[tid] = m

    rows, jobs = [], []
    au_by_id: dict[str, Path] = {}
    for p in au_files:
        info = parse_authentic(p.name)
        au_by_id[info["image_id"]] = p
        rows.append({"path": _rel(p), "label": 0, "label_name": "authentic",
                     "forgery_type": "none", "forgery_type_name": "none", "forgery_type_mask": "none",
                     "forgery_type_ids": "none", "category": info["category"], "image_id": info["image_id"],
                     "host_id": info["image_id"], "donor_id": None, "group_id": info["image_id"],
                     "flags": None, "tamper_id": None, "mask_path": None, "mask_name_relation": None})
        jobs.append((p, None, None, None))

    for p in tp_files:
        info = parse_tampered(p.name)
        mask = masks_by_tid.get(info["tamper_id"])
        mask_letter = mask.name.split("_")[1] if mask else None
        ids_type = "copy-move" if info["host_id"] == info["donor_id"] else "splicing"
        votes = [info["forgery_type"], ids_type] + ([LETTER_TYPE[mask_letter]] if mask_letter in LETTER_TYPE else [])
        rows.append({"path": _rel(p), "label": 1, "label_name": "tampered",
                     "forgery_type": _majority(votes), "forgery_type_name": info["forgery_type"],
                     "forgery_type_mask": LETTER_TYPE.get(mask_letter), "forgery_type_ids": ids_type,
                     "category": info["category"], "image_id": info["stem"], "host_id": info["host_id"],
                     "donor_id": info["donor_id"], "group_id": info["host_id"], "flags": info["flags"],
                     "tamper_id": info["tamper_id"], "mask_path": _rel(mask) if mask else None,
                     "mask_name_relation": mask_name_relation(info["stem"], mask.name) if mask else None,
                     "mask_has_placeholder_id": bool(mask and "xxx" in mask.name)})
        jobs.append((p, mask, au_by_id.get(info["host_id"]), au_by_id.get(info["donor_id"])))

    n_jobs = n_jobs or cfg["n_jobs"]
    described = Parallel(n_jobs=n_jobs, batch_size=64)(
        delayed(_describe)(p, m, h, d, cfg) for p, m, h, d in jobs)
    df = pd.concat([pd.DataFrame(rows), pd.DataFrame(described)], axis=1)

    df["ext"] = df["path"].str.rsplit(".", n=1).str[-1].str.lower()
    tp = df["label"] == 1
    df.loc[tp & df["mask_path"].isna(), "mask_status"] = "missing"
    df["mask_valid"] = tp & (df["mask_status"] == "ok")
    df["area_bucket"] = [area_bucket(f) if v else None for f, v in zip(df["tampered_frac"], df["mask_valid"])]
    df["donor_dominant"] = df["donor_dominant"].fillna(False).astype(bool)
    df["host_in_dataset"] = df["host_id"].isin(au_by_id.keys())
    df["donor_in_dataset"] = df["donor_id"].isin(au_by_id.keys()) | df["donor_id"].isna()
    df["forgery_type_votes_agree"] = ~tp | (
        (df["forgery_type_name"] == df["forgery_type_ids"])
        & (df["forgery_type_mask"].isna() | (df["forgery_type_mask"] == df["forgery_type_ids"])))

    # Exact pixel duplicates (a tampered file identical to an authentic one would be label noise).
    dup = df.groupby("pixel_md5")["label"].agg(["size", "nunique"])
    df["dup_count"] = df["pixel_md5"].map(dup["size"]).astype(int)
    df["dup_cross_label"] = df["pixel_md5"].map(dup["nunique"] > 1)
    df["mask_has_placeholder_id"] = df["mask_has_placeholder_id"].fillna(False).astype(bool)
    return df

