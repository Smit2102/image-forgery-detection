"""Phase 6: extract cell-level evidence stacks for localisation (train + val only; test is untouched).

For each image: the 8 evidence maps reduced to 8x8-pixel cells (float16) and, for tampered images
with a valid mask, the fraction of each cell inside the mask. One .npz per image under
data/maps/<protocol>/<fingerprint>/, so an interrupted run resumes where it stopped.
Images: every tampered train image with a valid mask, a random sample of 1,000 authentic train images
(fusion training), and every val image (tuning + evaluation).
Usage:  python scripts/extract_maps.py [--protocols R B]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from forgery.config import load_config, resolve
from forgery.data.manifest import read_manifest
from forgery.data.splits import read_splits
from forgery.features import build_modules
from forgery.io import load_image, load_mask, protocol_mask
from forgery.localize.fusion import cells, evidence_stack
from forgery.pipeline import feature_code_sha1, module_params


def map_dir(cfg: dict, protocol: str):
    fusion_src = resolve("src/forgery/localize/fusion.py").read_bytes()
    key = hashlib.sha1(json.dumps([module_params(cfg, protocol), cfg["protocols"].get(protocol),
                                   feature_code_sha1(), hashlib.sha1(fusion_src).hexdigest()],
                                  sort_keys=True, default=str).encode()).hexdigest()[:12]
    return resolve("data/maps") / protocol / key


def select(df: pd.DataFrame, seed: int, limit: int | None = None) -> pd.DataFrame:
    tr, va = df[df.split == "train"], df[df.split == "val"]
    tt, au = tr[(tr.label == 1) & tr.mask_valid], tr[tr.label == 0].sample(1000, random_state=seed)
    if limit:                                  # small subset for smoke tests
        tt, au = tt.sample(min(limit, len(tt)), random_state=seed), au.iloc[:limit // 2]
        va = pd.concat([va[va.label == 1].sample(limit, random_state=seed), va[va.label == 0].iloc[:limit // 2]])
    return pd.concat([tt, au, va]).reset_index(drop=True)


def _one(row: dict, protocol: str, cfg: dict, out_dir) -> str:
    out = out_dir / f"{row['image_id']}.npz"
    if out.exists():
        return "skip"
    cv2.setNumThreads(1)
    modules = build_modules(params=module_params(cfg, protocol))
    img = load_image(resolve(row["path"]), protocol, cfg)
    maps = {m.name: m.extract(img).evidence_map for m in modules}
    stack = evidence_stack(maps, [m.name for m in modules], img.shape[:2]).astype(np.float16)
    frac = np.zeros(stack.shape[1:], np.float16)
    if row["label"] == 1 and row["mask_valid"]:
        frac = cells(protocol_mask(load_mask(resolve(row["mask_path"])), protocol, cfg).astype(np.float32)).astype(np.float16)
    tmp = out.with_name(out.stem + ".tmp.npz")          # atomic: a half-written file is never "done"
    np.savez_compressed(tmp, stack=stack, frac=frac, names=np.array([m.name for m in modules]))
    tmp.replace(out)
    return "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocols", nargs="+", default=["R", "B"])
    ap.add_argument("--limit", type=int, default=None, help="smoke test on a small subset")
    args = ap.parse_args()
    cfg = load_config()
    df = read_manifest(cfg).merge(read_splits(cfg), on="path")
    sel = select(df, cfg["seed"], args.limit)
    assert not (sel.split == "test").any()
    for p in args.protocols:
        out_dir = map_dir(cfg, p)
        if args.limit:
            out_dir = out_dir.with_name(out_dir.name + f"_smoke{args.limit}")
        out_dir.mkdir(parents=True, exist_ok=True)
        sel[["path", "image_id", "split", "label"]].to_csv(out_dir / "index.csv", index=False)
        t0 = time.time()
        rows = sel[["path", "image_id", "label", "mask_valid", "mask_path"]].to_dict("records")
        res = Parallel(n_jobs=cfg["n_jobs"], batch_size=8)(delayed(_one)(r, p, cfg, out_dir) for r in rows)
        print(f"[{p}] {res.count('done')} new, {res.count('skip')} already there, {time.time() - t0:.0f}s -> {out_dir}",
              flush=True)


if __name__ == "__main__":
    main()
