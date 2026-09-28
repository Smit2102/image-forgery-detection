"""Phase 2b: shortcut-baseline ("leakage") study on the dev set (train + val, grouped 5-fold CV).

The frozen test set is never touched here.
Usage:  python scripts/run_leakage_study.py [--recompute]
"""

from __future__ import annotations

import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from forgery.config import load_config, resolve
from forgery.data.manifest import read_manifest
from forgery.data.splits import read_splits
from forgery.eval import leakage
from forgery.eval.leakage import FEATURE_SETS, image_features, run_study, summarize

OUT = resolve("results/phase2")
SET_LABEL = {"metadata": "File metadata", "dimensions": "Image dimensions", "naive_ela": "Naive global ELA",
             "dct_dq": "Global DCT double-quant.", "ghost_curve": "Global ghost curve",
             "compression_history": "Compression history", "all": "All shortcuts"}
PLOT_SETS = ["metadata", "dimensions", "naive_ela", "compression_history", "all"]


def cache_fingerprint(cfg) -> dict:
    return {"feature_version": leakage.FEATURE_VERSION,
            "b_qualities": cfg["protocols"]["B_quality_sensitivity"],
            "subsampling": cfg["protocols"]["B"]["subsampling"],
            "naive_ela_quality": cfg["leakage"]["naive_ela_quality"],
            "resample": {k: cfg["protocols"]["R"][k] for k in ("factor", "jpeg_quality", "subsampling")},
            "dq_positions": [list(p) for p in leakage.DQ_POSITIONS],
            "ghost_qualities": list(leakage.GHOST_QUALITIES)}


def compute_features(cfg, dev: pd.DataFrame, recompute: bool) -> pd.DataFrame:
    cache_dir = resolve(cfg["paths"]["cache_dir"])
    cache, meta = cache_dir / "leakage_features.csv", cache_dir / "leakage_features.json"
    fp = cache_fingerprint(cfg)
    if cache.exists() and meta.exists() and not recompute:
        feats = pd.read_csv(cache, dtype={"path": str})
        if json.loads(meta.read_text()) == fp and set(feats.path) == set(dev.path):
            return feats
        print("feature cache is stale (config/paths changed) - recomputing")
    t0 = time.time()
    rows = Parallel(n_jobs=cfg["n_jobs"], batch_size=16)(
        delayed(image_features)(str(resolve(p)), fp["b_qualities"], fp["naive_ela_quality"],
                                fp["subsampling"], fp["resample"]) for p in dev.path)
    feats = pd.DataFrame([{"path": p, **r} for p, rs in zip(dev.path, rows) for r in rs])
    cache_dir.mkdir(parents=True, exist_ok=True)
    feats.to_csv(cache, index=False)
    meta.write_text(json.dumps(fp, indent=2))
    print(f"features for {len(dev):,} images in {time.time() - t0:.0f}s")
    return feats


def plot(summary: pd.DataFrame, b_main: str, r_name: str):
    order = [("A", "A: as released", "#E45756"), ("C", "C: JPEG-only subset", "#F58518"),
             (b_main, f"B: re-encoded (Q={b_main[1:]})", "#4C78A8"),
             (r_name, f"R: resampled + re-encoded (Q={r_name[1:]})", "#72B7B2")]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    width = 0.2
    for k, (p, label, c) in enumerate(order):
        s = summary[summary.protocol == p].set_index("feature_set").reindex(PLOT_SETS)
        x = np.arange(len(PLOT_SETS)) + (k - 1.5) * width
        ax.bar(x, s.balanced_accuracy_mean, width, yerr=s.balanced_accuracy_std, label=label, color=c, capsize=2)
        for xi, v in zip(x, s.balanced_accuracy_mean):
            ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=6.5)
    ax.axhline(0.5, color="grey", ls="--", lw=0.8)
    ax.text(-0.5, 0.505, "chance", color="grey", fontsize=8, ha="left", va="bottom")
    ax.set_xticks(np.arange(len(PLOT_SETS)), [SET_LABEL[s] for s in PLOT_SETS], fontsize=9)
    ax.set_xlim(-0.55, len(PLOT_SETS) - 0.45)
    ax.set_ylim(0.4, 1.05)
    ax.set_ylabel("balanced accuracy (5-fold grouped CV)")
    ax.set_title("Global shortcut features alone vs. evaluation protocol", pad=30)
    ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_leakage.png", dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()
    cfg = load_config()

    df = read_manifest(cfg).merge(read_splits(cfg), on="path", validate="one_to_one")
    dev = df[df.split != "test"].reset_index(drop=True)
    assert (dev.cv_fold >= 0).all()

    feats = compute_features(cfg, dev, args.recompute)
    feats = feats.merge(dev[["path", "label", "cv_fold", "ext", "forgery_type"]], on="path",
                        validate="many_to_one")
    b_main = f"B{cfg['protocols']['B']['jpeg_quality']}"
    r_name = f"R{cfg['protocols']['R']['jpeg_quality']}"
    is_a, is_b = feats.protocol == "A", feats.protocol == b_main
    protocols = {"A": is_a, "C": is_a & (feats.ext == "jpg")}
    for q in cfg["protocols"]["B_quality_sensitivity"]:
        protocols[f"B{q}"] = feats.protocol == f"B{q}"
    protocols[r_name] = feats.protocol == r_name
    # Per-format views of the main protocol: which source format still leaks after re-encoding?
    protocols[f"{b_main}-jpeg-sources"] = is_b & (feats.ext == "jpg")
    protocols[f"{b_main}-au-vs-tiff"] = is_b & ((feats.label == 0) | (feats.ext == "tif"))

    results = run_study(feats, protocols, cfg["leakage"]["rf_trees"], cfg["seed"])
    summary = summarize(results)
    results.to_csv(OUT / "leakage_folds.csv", index=False)
    summary.to_csv(OUT / "leakage_summary.csv", index=False)
    plot(summary, b_main, r_name)

    fmt = lambda m, s: f"{m:.3f} ± {s:.3f}"
    table = summary.assign(
        feature_set=summary.feature_set.map(SET_LABEL),
        balanced_accuracy=[fmt(m, s) for m, s in zip(summary.balanced_accuracy_mean, summary.balanced_accuracy_std)],
        roc_auc=[fmt(m, s) for m, s in zip(summary.roc_auc_mean, summary.roc_auc_std)],
        majority_accuracy=summary.majority_accuracy.round(3),
    )[["protocol", "feature_set", "n_images", "balanced_accuracy", "roc_auc", "majority_accuracy"]]
    wide = summary.pivot(index="feature_set", columns="protocol", values="balanced_accuracy_mean") \
        .reindex(index=list(FEATURE_SETS), columns=list(protocols)).rename(index=SET_LABEL).round(3)
    md = ["# Phase 2: shortcut-baseline (leakage) study", "",
          f"Dev set only (train + val, {len(dev):,} images), grouped 5-fold CV, random forest "
          f"({cfg['leakage']['rf_trees']} trees, balanced class weights). Test set untouched.", "",
          "Protocols: **A** as released; **C** JPEG files only (both classes); **B<q>** every image re-encoded "
          f"once to JPEG at quality q (main: {b_main}); **{r_name}** downscaled x{cfg['protocols']['R']['factor']} "
          "then re-encoded; the last two rows restrict the main protocol to one source format.", "",
          "## Balanced accuracy overview (mean of 5 folds)", "", wide.to_markdown(), "",
          "## Full table", "", table.to_markdown(index=False), ""]
    (OUT / "leakage_table.md").write_text("\n".join(md))
    (OUT / "leakage_summary.json").write_text(json.dumps(summary.round(4).to_dict(orient="records"), indent=2))
    print("\n".join(md[:9]))


if __name__ == "__main__":
    main()
