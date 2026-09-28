"""Phase 6b: effect of gating the localisation mask with the Phase-5 image classifier.

detect() shows the mask only when the image verdict is not "authentic" (P(tampered) > 0.5 - band).
Using the out-of-fold Phase-5 probabilities of the validation images, mapped through the deployed
model's isotonic calibrator (detect() gates on calibrated probabilities), this measures:
  * authentic images: share with any predicted region, mean predicted area (with / without gating);
  * tampered images: mean pixel F1 when a gated-out image counts as an empty mask (F1 = 0).
Note: the final localiser's post-processing was tuned on all validation images, so the ungated
tampered-image F1 here is slightly optimistic; the out-of-sample F1 is in localization.md.
Usage:  python scripts/phase6_gating.py [--protocols R B]
"""

from __future__ import annotations

import argparse
import importlib.util
import json

import joblib
import numpy as np
import pandas as pd

from forgery.config import load_config, resolve
from forgery.data.manifest import read_manifest
from forgery.data.splits import read_splits
from forgery.io import load_mask, protocol_mask
from forgery.localize.fusion import CELL, postprocess, scores, upsample

OUT = resolve("results/phase6")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocols", nargs="+", default=["R", "B"])
    args = ap.parse_args()
    cfg = load_config()
    spec = importlib.util.spec_from_file_location("extract_maps", resolve("scripts/extract_maps.py"))
    em = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(em)
    meta = read_manifest(cfg).merge(read_splits(cfg), on="path")
    out, lines = {}, ["# Phase 6b: gating the mask with the image classifier (validation images)", ""]
    for P in args.protocols:
        loc = joblib.load(resolve(f"models/localizer_{P}.joblib"))
        band = joblib.load(resolve(f"models/forgery_{P}.joblib")).meta["uncertain_half_width"]
        oof = pd.read_csv(resolve(f"results/phase5/oof_{P}.csv"), dtype={"path": str})[["path", "forensic_all|rf"]]
        val = meta[meta.split == "val"].merge(oof, on="path", validate="one_to_one")
        # detect() gates on the *calibrated* probability. Map the out-of-fold raw forest scores through
        # the deployed model's isotonic calibrator (fitted on grouped out-of-fold dev scores), then apply
        # the deployed band. This is an approximation of fully out-of-fold calibrated scores.
        model = joblib.load(resolve(f"models/forgery_{P}.joblib"))
        iso = model.estimator.calibrated_classifiers_[0].calibrators[0]
        val["p_cal"] = iso.predict(val["forensic_all|rf"].to_numpy())
        val["shown"] = val["p_cal"] > 0.5 - band
        d = em.map_dir(cfg, P)
        rows = []
        for r in val.itertuples():
            z = np.load(d / f"{r.image_id}.npz")
            stack = z["stack"].astype(np.float32)
            if r.label == 1 and r.mask_valid:
                truth = protocol_mask(load_mask(resolve(r.mask_path)), P, cfg)
                shape = truth.shape
            else:
                truth, shape = None, (stack.shape[1] * CELL, stack.shape[2] * CELL)
            prob = upsample(loc.cell_prob(stack), shape)
            mask = postprocess(prob, loc.threshold, loc.open_k, loc.close_k, loc.min_area)
            rec = {"label": r.label, "shown": bool(r.shown), "area": float(mask.mean())}
            if truth is not None:
                rec["f1"] = scores(mask, truth)["f1"]
            rows.append(rec)
        df = pd.DataFrame(rows)
        au, tp = df[df.label == 0], df[df.f1.notna()]
        res = {
            "band": band,
            "authentic_any_region_ungated": float((au.area > 0).mean()),
            "authentic_any_region_gated": float(((au.area > 0) & au.shown).mean()),
            "authentic_mean_area_ungated": float(au.area.mean()),
            "authentic_mean_area_gated": float((au.area * au.shown).mean()),
            "tampered_shown": float(tp.shown.mean()),
            "tampered_f1_ungated": float(tp.f1.mean()),
            "tampered_f1_gated": float((tp.f1 * tp.shown).mean()),
            "n_authentic": len(au), "n_tampered": len(tp)}
        out[P] = res
        lines += [f"## Protocol {P} (uncertain band 0.5 ± {band})", "",
                  pd.DataFrame([res]).T.rename(columns={0: "value"}).round(4).to_markdown(), ""]
    (OUT / "gating.json").write_text(json.dumps(out, indent=2))
    (OUT / "gating.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
