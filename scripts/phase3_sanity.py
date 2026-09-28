"""Phase 3: sanity check of every feature module on 20 hand-picked validation images + figures.

For each tampered image the evidence map is scored against the ground-truth mask with a pixel-level
ROC-AUC (0.5 = map unrelated to the mask). Authentic images report their mean evidence.
Validation images only; the test set is never touched.
Usage:  python scripts/phase3_sanity.py [--protocols B R]
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from forgery.config import load_config, resolve
from forgery.data.manifest import read_manifest
from forgery.data.splits import read_splits
from forgery.features import build_modules
from forgery.features.dct_dq import ZIGZAG, block_dct, estimate_period, estimate_step
from forgery.features.common import gray
from forgery.features.copymove import KeypointCopyMove
from forgery.features.ela import ela_residual
from forgery.features.histogram import equalize
from forgery.io import load_image, load_mask, protocol_mask
from forgery.pipeline import module_params

OUT = resolve("results/phase3")
LABEL = {"ela": "ELA (texture-norm.)", "histogram": "Patch hist. chi²", "noise": "Noise level",
         "jpeg_ghost": "JPEG ghost", "dct_dq": "DCT double-quant.", "edges": "Edge sharpness",
         "cm_keypoint": "Copy-move (keypoints)", "cm_block": "Copy-move (blocks)"}


def pick(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    val = df[df.split == "val"]
    rng = np.random.default_rng(seed)
    parts = []
    for t, n in (("copy-move", 8), ("splicing", 8)):
        pool = val[(val.forgery_type == t) & val.mask_valid & val.tampered_frac.between(0.01, 0.5)]
        parts.append(pool.iloc[rng.choice(len(pool), n, replace=False)])
    au = val[val.label == 0]
    parts.append(au.iloc[rng.choice(len(au), 4, replace=False)])
    return pd.concat(parts)


def run(sample: pd.DataFrame, protocol: str, cfg, modules) -> tuple[pd.DataFrame, dict]:
    rows, maps = [], {}
    for r in sample.itertuples():
        img = load_image(resolve(r.path), protocol, cfg)
        mask = protocol_mask(load_mask(resolve(r.mask_path)), protocol, cfg) if r.label == 1 else None
        maps[r.path] = {"img": img, "mask": mask}
        for m in modules:
            out = m.extract(img)
            e = out.evidence_map
            maps[r.path][m.name] = e
            rec = {"protocol": protocol, "path": r.path, "type": r.forgery_type, "module": m.name,
                   "mean_evidence": float(e.mean()) if e is not None else np.nan}
            if mask is not None and e is not None:
                rec["pixel_auc"] = roc_auc_score(mask.ravel(), e.ravel()) if e.std() > 0 else 0.5
                rec["in_minus_out"] = float(e[mask].mean() - e[~mask].mean())
            rows.append(rec)
    return pd.DataFrame(rows), maps


def overview_figure(sample, maps, protocol, names):
    chosen = [sample[sample.forgery_type == "splicing"].iloc[0], sample[sample.forgery_type == "splicing"].iloc[1],
              sample[sample.forgery_type == "copy-move"].iloc[0], sample[sample.label == 0].iloc[0]]
    fig, axes = plt.subplots(len(chosen), len(names) + 1, figsize=(2.1 * (len(names) + 1), 1.75 * len(chosen)))
    for i, r in enumerate(chosen):
        d = maps[r.path]
        axes[i, 0].imshow(d["img"])
        if d["mask"] is not None:
            axes[i, 0].contour(d["mask"], levels=[0.5], colors="red", linewidths=0.8)
        axes[i, 0].set_ylabel({"none": "authentic"}.get(r.forgery_type, r.forgery_type), fontsize=8)
        for j, n in enumerate(names, start=1):
            e = d[n]
            axes[i, j].imshow(e if e is not None else np.zeros(d["img"].shape[:2]), cmap="magma", vmin=0, vmax=1)
            if d["mask"] is not None:
                axes[i, j].contour(d["mask"], levels=[0.5], colors="cyan", linewidths=0.5)
            if i == 0:
                axes[i, j].set_title(LABEL[n], fontsize=8)
        axes[0, 0].set_title("image (mask in red)", fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle(f"Evidence maps of the 8 modules, protocol {protocol} (bright = suspicious; cyan = true mask)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / f"fig_modules_overview_{protocol}.png", dpi=130)
    plt.close(fig)


def ela_before_after(r, cfg):
    img = load_image(resolve(r.path), "B", cfg)
    resid = ela_residual(img, 90)
    fig, ax = plt.subplots(1, 3, figsize=(10, 2.8))
    ax[0].imshow(img), ax[0].set_title("original (protocol B)", fontsize=9)
    ax[1].imshow(np.clip(resid * 12, 0, 255).astype(np.uint8), cmap="gray"), ax[1].set_title("ELA residual ×12", fontsize=9)
    ax[2].imshow(equalize(resid), cmap="gray"), ax[2].set_title("after histogram equalisation", fontsize=9)
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "fig_ela_before_after.png", dpi=140)
    plt.close(fig)


def dct_histogram_figure(au_path, cfg):
    """DCT coefficient histogram of an authentic image: periodic double-quantisation pattern."""
    img = load_image(resolve(au_path), "B", cfg)
    coef = block_dct(gray(img).astype(np.float64))
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
    for ax, (u, v) in zip(axes, ZIGZAG[:3]):
        c = coef[..., u, v].ravel()
        q = estimate_step(c)
        k = np.rint(c / q).astype(int)
        p, power, _ = estimate_period(k)
        vals, cnt = np.unique(k[(np.abs(k) <= 20) & (k != 0)], return_counts=True)
        ax.bar(vals, cnt, width=0.8, color="#4C78A8")
        ax.set_title(f"DCT ({u},{v}): step {q}, period {p} (power {power:.2f})", fontsize=8)
        ax.set_xlabel("quantised value k", fontsize=8)
    fig.suptitle("Authentic image after protocol B: double quantisation leaves periodic peaks", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_dct_histogram.png", dpi=140)
    plt.close(fig)


def copymove_figure(r, cfg):
    img = load_image(resolve(r.path), "B", cfg)
    m = KeypointCopyMove(**cfg["features"].get("cm_keypoint", {})).match(img)
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.imshow(img)
    for (x0, y0), (x1, y1), ok in zip(m["src"], m["dst"], m["inliers"]):
        ax.plot([x0, x1], [y0, y1], color="lime" if ok else "orange", lw=0.7)
    ax.contour(load_mask(resolve(r.mask_path)), levels=[0.5], colors="red", linewidths=0.8)
    ax.set_title(f"g2NN matches: {len(m['src'])}, RANSAC inliers (green): {int(m['inliers'].sum())}", fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "fig_copymove_matches.png", dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocols", nargs="+", default=["B", "R"])
    args = ap.parse_args()
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    df = read_manifest(cfg).merge(read_splits(cfg), on="path")
    sample = pick(df, cfg["seed"])
    assert (sample.split == "val").all()
    names = [m.name for m in build_modules(params=cfg["features"])]

    frames = []
    for p in args.protocols:
        res, maps = run(sample, p, cfg, build_modules(params=module_params(cfg, p)))
        frames.append(res)
        overview_figure(sample, maps, p, names)
    res = pd.concat(frames)
    res.to_csv(OUT / "sanity_20.csv", index=False)

    tp = res[res.pixel_auc.notna()]
    tab = tp.groupby(["module", "protocol"]).pixel_auc.median().unstack().reindex(names)
    by_type = tp.groupby(["module", "type", "protocol"]).pixel_auc.median().unstack(["protocol", "type"]).reindex(names)
    frac = tp.assign(hit=tp.pixel_auc > 0.6).groupby(["module", "protocol"]).hit.mean().unstack().reindex(names)
    au = res[res.type == "none"].groupby(["module", "protocol"]).mean_evidence.mean().unstack().reindex(names)
    md = ["# Phase 3: module sanity check (20 validation images)", "",
          "8 copy-move + 8 splicing images (mask 1-50% of the frame) and 4 authentic images, fixed seed.", "",
          "## Median pixel ROC-AUC of the evidence map against the mask (0.5 = unrelated)", "",
          tab.round(3).to_markdown(), "",
          "## Share of tampered images with pixel AUC > 0.6", "", frac.round(2).to_markdown(), "",
          "## Median pixel AUC by forgery type", "", by_type.round(3).to_markdown(), "",
          "## Mean evidence on the 4 authentic images (lower = fewer false alarms)", "", au.round(3).to_markdown(), ""]
    (OUT / "sanity_20.md").write_text("\n".join(md))
    print("\n".join(md))

    ela_before_after(sample[sample.forgery_type == "splicing"].iloc[0], cfg)
    dct_histogram_figure(sample[sample.label == 0].iloc[0].path, cfg)
    copymove_figure(sample[sample.forgery_type == "copy-move"].iloc[0], cfg)


if __name__ == "__main__":
    main()
