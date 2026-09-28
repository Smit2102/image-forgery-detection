"""Phase 8 input: ELA images for the CNN (every image; one lossless PNG per image and protocol).

ELA = |x - JPEG_90(x)| per RGB channel, scaled x10 and clipped to 0..255, computed on the image after
its protocol (A: as released, B: re-encoded, R: resampled + re-encoded), at native resolution.
Output: data/ela/<protocol>/<image_id>.png; existing files are skipped, so an interrupted run resumes.
Usage:  python scripts/make_ela.py [--protocols R B A]
"""

from __future__ import annotations

import argparse
import inspect
import hashlib
import json
import time

import numpy as np
from joblib import Parallel, delayed
from PIL import Image

from forgery.config import load_config, resolve
from forgery.data.manifest import read_manifest
from forgery import cnn
from forgery.cnn import ela_image
from forgery import io
from forgery.io import load_image


def _one(path: str, image_id: str, protocol: str, cfg: dict, out_dir) -> str:
    out = out_dir / f"{image_id}.png"
    if out.exists():
        return "skip"
    tmp = out.with_suffix(".tmp.png")
    Image.fromarray(ela_image(load_image(resolve(path), protocol, cfg))).save(tmp)
    tmp.replace(out)
    return "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocols", nargs="+", default=["R", "B", "A"])
    args = ap.parse_args()
    cfg = load_config()
    df = read_manifest(cfg)
    meta = {"ela_quality": cnn.ELA_QUALITY, "ela_scale": cnn.ELA_SCALE,
            "ela_image_sha1": hashlib.sha1(inspect.getsource(ela_image).encode()).hexdigest(),
            "protocol_code_sha1": hashlib.sha1("".join(inspect.getsource(f) for f in (
                io.apply_protocol, io.resample, io.jpeg_encode, io.jpeg_roundtrip)).encode()).hexdigest(),
            "protocols": cfg["protocols"]}
    meta_file = resolve("data/ela") / "meta.json"
    if meta_file.exists() and json.loads(meta_file.read_text()) != json.loads(json.dumps(meta)):
        raise SystemExit("data/ela was built with other ELA settings: delete it and run again")
    meta_file.parent.mkdir(parents=True, exist_ok=True)
    meta_file.write_text(json.dumps(meta, indent=2))
    for p in args.protocols:
        out_dir = resolve("data/ela") / p
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        res = Parallel(n_jobs=cfg["n_jobs"], batch_size=16)(
            delayed(_one)(r.path, r.image_id, p, cfg, out_dir) for r in df.itertuples())
        print(f"[{p}] {res.count('done')} new, {res.count('skip')} existing ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
