"""Command-line entry point:  python detect.py <image> [--protocol B|R|A] [--out results/detect]

Runs every feature module through forgery.pipeline.detect(), classifies the image with the Phase-5
model for that protocol (models/forgery_<protocol>.joblib) if it exists, localises the suspected region
with models/localizer_<protocol>.joblib, and writes <out>/<image stem>/report.json, plus mask.png and
overlay.png when a localiser is available.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import joblib
import numpy as np

from PIL import Image

from forgery.config import load_config
from forgery.explain import ImageError, prepare_image
from forgery.features import REGISTRY
from forgery.io import load_image
from forgery.pipeline import detect


def save_mask_and_overlay(image: Path, res, protocol: str, cfg: dict, out_dir: Path) -> None:
    """mask.png (white = suspected region) and overlay.png (region tinted red), at the analysed size."""
    img = load_image(image, protocol, cfg)
    Image.fromarray((res.mask * 255).astype(np.uint8)).save(out_dir / "mask.png")
    over = img.copy()
    over[res.mask] = (0.45 * over[res.mask] + 0.55 * np.array([255, 0, 0])).astype(np.uint8)
    Image.fromarray(over).save(out_dir / "overlay.png")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("image", type=Path)
    ap.add_argument("--protocol", default=None, choices=["A", "B", "R"],
                    help="default: the main protocol chosen in Phase 5 (models/MAIN_PROTOCOL)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output folder (default: <project>/results/detect)")
    args = ap.parse_args(argv)
    if not args.image.exists():
        ap.error(f"image not found: {args.image}")

    tmp = tempfile.TemporaryDirectory(prefix="forgery_detect_")
    try:                                   # full decode + size limits; EXIF rotation and 16-bit handled here
        analysed_path, notes = prepare_image(args.image, tmp.name)
    except ImageError as exc:
        ap.error(f"cannot analyse {args.image}: {exc}")
    for n in notes:
        print(f"note: the image was {n} before analysis")

    cfg = load_config()
    models = Path(__file__).resolve().parent / "models"
    main_file = models / "MAIN_PROTOCOL"
    main = main_file.read_text().strip() if main_file.exists() else "none"
    if main not in ("A", "B", "R", "none"):
        ap.error(f"models/MAIN_PROTOCOL contains {main!r}; expected B, R or none")
    if args.protocol is None and main == "none":
        ap.error("no main protocol was established in Phase 5; choose one with --protocol B|R")
    protocol = args.protocol or main
    model = joblib.load(models / f"forgery_{protocol}.joblib") if (models / f"forgery_{protocol}.joblib").exists() else None
    type_model = joblib.load(models / f"type_{protocol}.joblib") if (models / f"type_{protocol}.joblib").exists() else None
    loc_file = models / f"localizer_{protocol}.joblib"
    localizer = joblib.load(loc_file) if loc_file.exists() else None
    res = detect(analysed_path, protocol=protocol, cfg=cfg, model=model, type_model=type_model, localizer=localizer)
    res.path = str(args.image)
    base = args.out.resolve() if args.out else Path(__file__).resolve().parent / "results" / "detect"
    out_dir = base / args.image.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(res.to_json(indent=2))
    if res.mask is not None:
        save_mask_and_overlay(analysed_path, res, protocol, cfg, out_dir)
    tmp.cleanup()

    verdict = (f"{res.verdict.upper()} (P(tampered) = {res.confidence:.2f})" if res.verdict
               else f"no verdict (no trained model for protocol {protocol})")
    kind = f" · likely {res.likely_type}" if res.likely_type else ""
    if res.mask is not None and res.mask.any():
        kind += f" · suspected region {100 * res.mask.mean():.1f}% of the image"
    print(f"{args.image.name} -> {verdict}{kind} · protocol {protocol} · modules: {len(REGISTRY)} · "
          f"features: {len(res.features)} · see {out_dir / 'report.json'}")
    return res


if __name__ == "__main__":
    main()
