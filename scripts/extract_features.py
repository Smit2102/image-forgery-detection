"""Extract all feature-module features for the dev set (train + val) under one or more protocols.

Output: data/features/features_<protocol>.csv (one row per image; `_with_test` suffix when the test
set is included) + a fingerprint sidecar, so stale
features are recomputed when module code or parameters change.
The frozen test set is excluded unless --include-test is given (reserved for Phase 7).
Usage:  python scripts/extract_features.py [--protocols A B R] [--include-test]
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from forgery.config import load_config, resolve
from forgery.data.manifest import read_manifest
from forgery.data.splits import read_splits
from forgery.features import build_modules
import cv2

from forgery.io import load_image
from forgery.pipeline import module_params

FEATURE_SRC = Path(__file__).resolve().parents[1] / "src" / "forgery"


def fingerprint(cfg: dict, protocol: str, paths: list[str]) -> dict:
    code = hashlib.sha1()
    for f in sorted([*(FEATURE_SRC / "features").glob("*.py"), FEATURE_SRC / "io.py"]):
        code.update(f.read_bytes())
    return {"features": module_params(cfg, protocol), "protocol": protocol,
            "protocol_cfg": cfg["protocols"].get(protocol), "code_sha1": code.hexdigest(),
            "paths_sha1": hashlib.sha1("\n".join(sorted(paths)).encode()).hexdigest()}


def _one(path: str, protocol: str, cfg: dict) -> dict:
    cv2.setNumThreads(1)                       # one thread per worker: avoids oversubscribing the cores
    modules = build_modules(params=module_params(cfg, protocol))
    img = load_image(resolve(path), protocol, cfg)
    rec = {"path": path}
    for m in modules:
        t0 = time.perf_counter()
        out = m.extract(img)
        rec[f"time_ms.{m.name}"] = 1000 * (time.perf_counter() - t0)
        rec.update({f"{m.name}.{k}": float(v) for k, v in out.features.items()})
    return rec


def extract(cfg: dict, protocol: str, paths: list[str], force: bool = False, suffix: str = "") -> pd.DataFrame:
    out_dir = resolve("data/features")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv, meta = out_dir / f"features_{protocol}{suffix}.csv", out_dir / f"features_{protocol}{suffix}.json"
    fp = fingerprint(cfg, protocol, paths)
    if csv.exists() and meta.exists() and not force and json.loads(meta.read_text()) == fp:
        print(f"[{protocol}] up to date")
        return pd.read_csv(csv, dtype={"path": str})
    t0 = time.time()
    # work in chunks saved to disk, so an interrupted run (shutdown, crash) resumes at the last chunk
    part_dir = out_dir / f".partial_{protocol}{suffix}_{hashlib.sha1(json.dumps(fp, sort_keys=True, default=str).encode()).hexdigest()[:12]}"
    part_dir.mkdir(exist_ok=True)
    chunk = 500
    for i in range(0, len(paths), chunk):
        part = part_dir / f"{i // chunk:04d}.csv"
        if part.exists():
            continue
        rows = Parallel(n_jobs=cfg["n_jobs"], batch_size=8)(delayed(_one)(p, protocol, cfg) for p in paths[i:i + chunk])
        pd.DataFrame(rows).to_csv(part, index=False)
        print(f"[{protocol}] {min(i + chunk, len(paths)):,}/{len(paths):,} images ({time.time() - t0:.0f}s)", flush=True)
    df = pd.concat([pd.read_csv(f, dtype={"path": str}) for f in sorted(part_dir.glob("*.csv"))], ignore_index=True)
    assert list(df.path) == list(paths), "chunk files do not match the image list"
    assert np.isfinite(df.drop(columns="path").to_numpy(float)).all(), "non-finite feature values"
    df.to_csv(csv, index=False)
    meta.write_text(json.dumps(fp, indent=2, default=str))
    shutil.rmtree(part_dir)
    print(f"[{protocol}] {len(df):,} images, {df.shape[1] - 1} columns in {time.time() - t0:.0f}s")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocols", nargs="+", default=["A", "B", "R"])
    ap.add_argument("--include-test", action="store_true", help="Phase 7 only")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    df = read_manifest(cfg).merge(read_splits(cfg), on="path")
    keep = df if args.include_test else df[df.split != "test"]
    for p in args.protocols:
        extract(cfg, p, sorted(keep.path), args.force, suffix="_with_test" if args.include_test else "")


if __name__ == "__main__":
    main()
