"""Checks that the CASIA v2.0 dataset is where the project expects it (run this first when reproducing results).

    python scripts/check_dataset.py

The project reads images through data/manifest.csv, whose paths are relative to the project folder:
    CASIA2/Au/...                      7,491 authentic images
    CASIA2/Tp/...                      5,123 tampered images
    CASIA2/CASIA 2 Groundtruth/...     5,123 ground-truth masks
Exit status 0 = everything found.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from forgery.config import load_config, resolve

IMAGE_EXT = {".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".png"}
EXPECTED = {"Au": 7491, "Tp": 5123, "CASIA 2 Groundtruth": 5123}


def count_images(folder: Path) -> int:
    return sum(1 for f in folder.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXT and not f.name.startswith("."))


def find_candidates(base: Path) -> list[Path]:
    """Folders (up to 3 levels below `base`) that contain both an 'Au' and a 'Tp' sub-folder."""
    found = []
    for d in [base, *base.glob("*"), *base.glob("*/*"), *base.glob("*/*/*")]:
        if d.is_dir() and (d / "Au").is_dir() and (d / "Tp").is_dir():
            found.append(d)
    return found


def main() -> int:
    cfg = load_config()
    root = resolve(cfg["paths"]["casia_root"])
    print(f"project folder : {resolve('.')}")
    print(f"dataset folder : {root}")
    ok = True
    if not root.is_dir():
        print("\nNOT FOUND: the dataset folder does not exist.")
        for c in find_candidates(resolve(".")):
            print(f"  a folder that looks like CASIA v2.0 exists at: {c}\n  -> move or link it to: {root}")
        print("\nExpected layout:\n  CASIA2/Au/                    (authentic images)\n  CASIA2/Tp/                    (tampered images)"
              "\n  CASIA2/CASIA 2 Groundtruth/   (ground-truth masks, *_gt.png)")
        return 1

    for name, expected in EXPECTED.items():
        d = root / name
        if not d.is_dir():
            print(f"MISSING folder: {d}"); ok = False; continue
        n = count_images(d)
        flag = "ok" if n == expected else f"WARNING: expected {expected:,}"
        ok &= n == expected
        print(f"{name:22s}: {n:6,} image files  [{flag}]")

    manifest = pd.read_csv(resolve(cfg["paths"]["manifest"]), dtype=str)
    missing = [p for p in manifest.path if not resolve(p).is_file()]
    masks = manifest[manifest.mask_path.notna()]
    missing_masks = [p for p in masks.mask_path if not resolve(p).is_file()]
    print(f"manifest       : {len(manifest):,} images listed, {len(missing):,} not found on disk")
    print(f"masks          : {len(masks):,} listed, {len(missing_masks):,} not found on disk")
    for label, lst in (("image", missing), ("mask", missing_masks)):
        for p in lst[:5]:
            print(f"   missing {label}: {p}")
    ok &= not missing and not missing_masks
    print("\nDATASET OK" if ok else "\nPROBLEM FOUND: see the lines above. If your copy uses another layout, "
          "re-create the manifest with  python scripts/build_manifest.py  (it reads the folders in config.yaml).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
