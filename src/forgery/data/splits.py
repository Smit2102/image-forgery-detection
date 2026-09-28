"""Leakage-safe train/val/test splits and grouped CV folds.

Split groups are built with union-find over:
  * the host image id  (an authentic image and every tampered image built on it),
  * exact pixel duplicates (196 authentic pairs are saved under two ids), and
  * verified near duplicates: dHash Hamming distance <= `near_dup_hamming` AND
    32x32 grey thumbnails with mean abs difference < `near_dup_max_mad` and
    correlation > `near_dup_min_corr`. dHash alone gives many false matches on
    smooth images; the thumbnail check removes them. About 480 tampered images
    are near-identical to an authentic image other than their named host.
  * the donor image, when the second-named image supplies most of a tampered image's pixels
    (`donor_dominant`, 23 images: a large pasted region in 18, ids swapped in the name in 5).
Every split group lands entirely in one split, and within the dev set entirely in one CV fold.
"""

from __future__ import annotations

import hashlib

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from ..config import load_config, resolve
from ..io import load_rgb


def read_splits(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    return pd.read_csv(resolve(cfg["paths"]["splits"]), dtype={"path": str, "split": str, "split_group": str})


class _UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def near_duplicate_pairs(dhashes: pd.Series, max_hamming: int, chunk: int = 2048) -> list[tuple[int, int]]:
    """Index pairs (i < j) whose dHash Hamming distance is <= max_hamming."""
    h = np.array([int(x, 16) for x in dhashes], dtype=np.uint64)
    pairs = []
    for start in range(0, len(h), chunk):
        block = h[start:start + chunk]
        dist = np.bitwise_count(block[:, None] ^ h[None, :])
        ii, jj = np.nonzero(dist <= max_hamming)
        ii = ii + start
        keep = ii < jj
        pairs.extend(zip(ii[keep].tolist(), jj[keep].tolist()))
    return pairs


def _thumb(path: str) -> np.ndarray:
    g = cv2.cvtColor(load_rgb(resolve(path)), cv2.COLOR_RGB2GRAY)
    return cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float64)


def verified_near_duplicates(df: pd.DataFrame, max_hamming: int, max_mad: float,
                             min_corr: float) -> list[tuple[int, int]]:
    """dHash candidates confirmed by a thumbnail comparison (positional indices into df)."""
    gids, md5 = df["group_id"].to_numpy(), df["pixel_md5"].to_numpy()
    cand = [(i, j) for i, j in near_duplicate_pairs(df["dhash"], max_hamming)
            if gids[i] != gids[j] and md5[i] != md5[j]]
    paths, cache = df["path"].to_numpy(), {}

    def thumb(i):
        if i not in cache:
            cache[i] = _thumb(paths[i])
        return cache[i]

    out = []
    for i, j in cand:
        a, b = thumb(i), thumb(j)
        if np.abs(a - b).mean() < max_mad and np.corrcoef(a.ravel(), b.ravel())[0, 1] > min_corr:
            out.append((i, j))
    return out


def build_split_groups(df: pd.DataFrame, near_dup_hamming: int = 8, near_dup_max_mad: float = 5.0,
                       near_dup_min_corr: float = 0.95) -> tuple[pd.Series, dict]:
    df = df.reset_index(drop=True)
    uf = _UnionFind(df["group_id"].unique())
    n_exact = n_near = 0
    for _, ids in df.groupby("pixel_md5")["group_id"]:
        ids = ids.unique()
        for other in ids[1:]:
            n_exact += uf.find(ids[0]) != uf.find(other)
            uf.union(ids[0], other)
    gids = df["group_id"].to_numpy()
    near = verified_near_duplicates(df, near_dup_hamming, near_dup_max_mad, near_dup_min_corr)
    for i, j in near:
        if uf.find(gids[i]) != uf.find(gids[j]):
            n_near += 1
            uf.union(gids[i], gids[j])
    n_donor = 0
    for host, donor in df.loc[df["donor_dominant"], ["group_id", "donor_id"]].itertuples(index=False):
        if donor in uf.parent and uf.find(host) != uf.find(donor):
            n_donor += 1
            uf.union(host, donor)
    groups = df["group_id"].map(uf.find)
    sizes = groups.value_counts()
    info = {"n_host_groups": int(df["group_id"].nunique()), "n_split_groups": int(groups.nunique()),
            "merges_exact_duplicate": int(n_exact), "near_duplicate_pairs": len(near),
            "merges_near_duplicate": int(n_near), "merges_donor_dominant": int(n_donor),
            "largest_split_group": int(sizes.max()),
            "near_dup_hamming": near_dup_hamming, "near_dup_max_mad": near_dup_max_mad,
            "near_dup_min_corr": near_dup_min_corr}
    return groups.rename("split_group"), info


def stratum(df: pd.DataFrame) -> pd.Series:
    """Stratification key: forgery type x file format (authentic_jpg, copy-move_tif, ...)."""
    return df["forgery_type"] + "_" + df["ext"]


def make_splits(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    scfg, seed = cfg["splits"], cfg["seed"]
    groups, info = build_split_groups(df, scfg["near_dup_hamming"], scfg["near_dup_max_mad"],
                                      scfg["near_dup_min_corr"])
    y = stratum(df)

    n_bins = scfg["n_split_bins"]
    n_test, n_val = round(n_bins * scfg["test"]), round(n_bins * scfg["val"])
    bins = np.empty(len(df), dtype=int)
    for b, (_, idx) in enumerate(StratifiedGroupKFold(n_bins, shuffle=True, random_state=seed)
                                 .split(df, y, groups)):
        bins[idx] = b
    split = np.where(bins < n_test, "test", np.where(bins < n_test + n_val, "val", "train"))

    cv_fold = np.full(len(df), -1, dtype=int)
    dev = np.flatnonzero(split != "test")
    for f, (_, idx) in enumerate(StratifiedGroupKFold(scfg["n_cv_folds"], shuffle=True, random_state=seed)
                                 .split(dev, y.iloc[dev], groups.iloc[dev])):
        cv_fold[dev[idx]] = f

    out = pd.DataFrame({"path": df["path"], "split": split, "cv_fold": cv_fold,
                        "split_group": groups.to_numpy()})
    return out, info


def frozen_test_hash(splits: pd.DataFrame) -> str:
    """SHA-256 over the sorted test-set paths; stored so the frozen test set can be verified."""
    test = sorted(splits.loc[splits["split"] == "test", "path"])
    return hashlib.sha256("\n".join(test).encode()).hexdigest()


def full_split_hash(splits: pd.DataFrame) -> str:
    """SHA-256 over every (path, split, cv_fold) row, so silent train/val/fold changes are caught too."""
    rows = sorted(f"{p}\t{s}\t{f}" for p, s, f in zip(splits["path"], splits["split"], splits["cv_fold"]))
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def donor_merge_component(df: pd.DataFrame, groups: pd.Series) -> int:
    """Size of the largest group if every tampered image were also merged with its donor."""
    uf = _UnionFind(groups.unique())
    group_of = dict(zip(df["image_id"], groups))
    for g, donor in zip(groups[df["label"] == 1], df.loc[df["label"] == 1, "donor_id"]):
        if donor in group_of:
            uf.union(g, group_of[donor])
    return int(groups.map(uf.find).value_counts().max())
