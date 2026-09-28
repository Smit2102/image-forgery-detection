"""Phase 4: parameter sweeps for every feature module, under protocols B and R.

Coordinate ascent: parameters are swept one at a time (others fixed at their current best),
in the order listed in GRIDS. Each setting is scored on a fixed sample:
  * image AUC  - random forest on the module's *local* features, fit on 2,400 train images,
                 scored on 1,000 validation images;
  * pixel AUC  - mean pixel-level ROC-AUC of the evidence map against the mask on the validation
                 tampered images (copy-move modules: copy-move images only).
Selection score = mean of the two; a new value must beat the current one by > 0.002 to be adopted.
Parameters are chosen under protocol B and re-used under R (both are reported).
The test set is never touched.
Usage:  python scripts/sweep_params.py [--write-config]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from forgery.config import DEFAULT_CONFIG, load_config, resolve
from forgery.data.manifest import read_manifest
from forgery.data.splits import read_splits
from forgery.eval.metrics import local_columns, pixel_auc, rf_auc
from forgery.features import build_modules
import cv2

from forgery.io import load_image, load_mask, protocol_mask
from forgery.pipeline import module_params

OUT = resolve("results/phase4")
GRIDS = {
    "ela": {"quality": [75, 85, 90, 95], "block": [8, 16], "texture_c": [1.0, 4.0, 16.0]},
    "histogram": {"quality": [75, 90, 95], "patch": [16, 32, 64], "bins": [8, 16, 32]},
    "noise": {"kernel": ["laplacian", "immerkaer"], "block": [8, 16], "edge_quantile": [0.4, 0.6, 0.8],
              "min_std": [0.5, 1.0, 2.0]},
    "jpeg_ghost": {"block": [8, 16, 32], "min_std": [1.0, 2.0, 4.0]},
    "dct_dq": {"n_freq": [3, 6, 9, 15], "alpha": [0.001, 0.01, 0.05]},   # permutation-test level
    "edges": {"low": [30, 50, 80], "block": [16, 32]},
    "cm_keypoint": {"detector": ["harris", "sift"], "ratio": [0.5, 0.6, 0.7], "contrast": [0.005, 0.01, 0.04]},
    "cm_block": {"tol": [1.0, 1.5, 2.5], "min_std": [2.0, 4.0, 8.0], "min_votes": [5, 8, 15]},
}
LINKED = {("edges", "low"): ("high", lambda v: 3 * v)}      # Canny: keep high = 3 x low
# The sweep always starts from these Phase-3 defaults (not from config.yaml, which it rewrites),
# so re-running it reproduces the same choices.
START = {
    "ela": {"quality": 90, "block": 8, "texture_c": 4.0, "min_std": 2.0},
    "histogram": {"quality": 90, "patch": 32, "bins": 16, "min_std": 2.0},
    "noise": {"kernel": "immerkaer", "block": 8, "edge_quantile": 0.6, "min_pixels": 16, "min_std": 1.0},
    "jpeg_ghost": {"block": 16, "min_std": 2.0},
    "dct_dq": {"n_freq": 9, "alpha": 0.01},
    "edges": {"low": 50, "high": 150, "block": 16, "min_edge_pixels": 6},
    "cm_keypoint": {"detector": "sift", "contrast": 0.01, "ratio": 0.6, "min_shift": 20.0, "ransac_px": 3.0,
                    "min_inliers": 4, "mirror": True},
    "cm_block": {"block": 16, "stride": 4, "n_coef": 9, "tol": 1.5, "min_std": 4.0, "min_votes": 8, "mirror": True},
}


def sample(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    def take(d, n):
        return d.iloc[rng.choice(len(d), min(n, len(d)), replace=False)]

    tr, va = df[df.split == "train"], df[df.split == "val"]
    train = pd.concat([take(tr[tr.label == 0], 1200), take(tr[tr.label == 1], 1200)])
    val = pd.concat([take(va[va.label == 0], 500), take(va[va.label == 1], 500)])
    return train.reset_index(drop=True), val.reset_index(drop=True)


def _one(path: str, mask_path: str | None, protocol: str, cfg: dict, name: str, params: dict) -> dict:
    cv2.setNumThreads(1)
    extra = module_params(cfg, protocol).get(name, {})           # protocol-specific additions
    module = build_modules([name], {name: {**params, **{k: v for k, v in extra.items() if k not in params}}})[0]
    img = load_image(resolve(path), protocol, cfg)
    out = module.extract(img)
    rec = {f"{name}.{k}": v for k, v in out.features.items()}
    if isinstance(mask_path, str):
        rec["pixel_auc"] = pixel_auc(out.evidence_map, protocol_mask(load_mask(resolve(mask_path)), protocol, cfg))
    return rec


def _code_hash() -> str:
    """Hash of the code a cached sweep result depends on (modules, image I/O, metrics)."""
    src = resolve("src/forgery")
    h = hashlib.sha1()
    for f in sorted([*(src / "features").glob("*.py"), src / "io.py", src / "eval" / "metrics.py"]):
        h.update(f.read_bytes())
    return h.hexdigest()


def evaluate(name, params, protocol, train, val, cfg, cache_dir) -> dict:
    key = hashlib.sha1(json.dumps([name, params, protocol, cfg["protocols"].get(protocol), _code_hash(),
                                   list(train.path), list(val.path)],
                                  sort_keys=True, default=str).encode()).hexdigest()[:16]
    f = cache_dir / f"{name}_{protocol}_{key}.csv"
    both = pd.concat([train.assign(part="train"), val.assign(part="val")], ignore_index=True)
    if f.exists():
        feats = pd.read_csv(f)
    else:
        masks = [m if (p == "val" and v) else None for m, p, v in zip(both.mask_path, both.part, both.mask_valid)]
        rows = Parallel(n_jobs=cfg["n_jobs"], batch_size=16)(
            delayed(_one)(p, m, protocol, cfg, name, params) for p, m in zip(both.path, masks))
        feats = pd.DataFrame(rows)
        feats.to_csv(f, index=False)
    d = pd.concat([both.reset_index(drop=True), feats], axis=1)
    cols = local_columns([c for c in feats.columns if c.startswith(name + ".")])
    tr, va = d[d.part == "train"], d[d.part == "val"]
    img_auc = rf_auc(tr, va, cols, cfg["seed"])
    target = va[va.mask_valid & va.pixel_auc.notna()] if "pixel_auc" in va else va.iloc[:0]
    if name.startswith("cm_"):
        target = target[target.forgery_type == "copy-move"]
    pix = float(target.pixel_auc.mean()) if len(target) else 0.5
    return {"image_auc": img_auc, "pixel_auc": pix, "score": (img_auc + pix) / 2,
            "pixel_auc_hit_rate": float((target.pixel_auc > 0.6).mean()) if len(target) else 0.0}


def sweep_module(name, cfg, train, val, cache_dir, protocol="B"):
    best = dict(START[name])
    records = []
    base = evaluate(name, best, protocol, train, val, cfg, cache_dir)
    for param, values in GRIDS[name].items():
        scores = {}
        for v in values:
            trial = {**best, param: v}
            if (name, param) in LINKED:
                other, fn = LINKED[(name, param)]
                trial[other] = fn(v)
            r = evaluate(name, trial, protocol, train, val, cfg, cache_dir)
            scores[v] = (r, trial)
            records.append({"module": name, "protocol": protocol, "param": param, "value": str(v), **r})
        current = best.get(param)
        cur_score = scores[current][0]["score"] if current in scores else base["score"]
        top = max(scores, key=lambda v: scores[v][0]["score"])
        if scores[top][0]["score"] > cur_score + 0.002:
            best = scores[top][1]
        base = scores.get(best.get(param), (base,))[0]
    return best, records


def plot(records: pd.DataFrame):
    mods = list(GRIDS)
    fig, axes = plt.subplots(len(mods), 3, figsize=(11, 2.1 * len(mods)))
    for i, m in enumerate(mods):
        params = list(GRIDS[m])
        for j in range(3):
            ax = axes[i, j]
            if j >= len(params):
                ax.axis("off")
                continue
            d = records[(records.module == m) & (records.param == params[j]) & (records.protocol == "B")]
            x = np.arange(len(d))
            ax.plot(x, d.image_auc, "o-", label="image AUC", color="#4C78A8")
            ax.plot(x, d.pixel_auc, "s--", label="pixel AUC", color="#E45756")
            ax.set_xticks(x, d.value, fontsize=7)
            ax.set_title(f"{m}: {params[j]}", fontsize=8)
            ax.tick_params(axis="y", labelsize=7)
            ax.axhline(0.5, color="grey", lw=0.5, ls=":")
            if i == 0 and j == 0:
                ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Phase 4 parameter sweeps (protocol B, validation sample)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_sweeps.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-config", action="store_true", help="store the chosen parameters in config.yaml")
    ap.add_argument("--modules", nargs="+", default=list(GRIDS),
                    help="modules to sweep; with --write-config the others keep their current config values")
    args = ap.parse_args()
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    cache_dir = resolve(cfg["paths"]["cache_dir"]) / "sweeps"
    cache_dir.mkdir(parents=True, exist_ok=True)

    df = read_manifest(cfg).merge(read_splits(cfg), on="path")
    train, val = sample(df[df.split != "test"], cfg["seed"])
    assert set(train.split) == {"train"} and set(val.split) == {"val"}

    chosen, records, finals = {}, [], []
    for name in args.modules:
        t0 = time.time()
        best, rec = sweep_module(name, cfg, train, val, cache_dir)
        chosen[name] = best
        records += rec
        for p in ("B", "R"):
            finals.append({"module": name, "protocol": p, "default": json.dumps(START[name]),
                           "chosen": json.dumps(best),
                           **evaluate(name, best, p, train, val, cfg, cache_dir),
                           **{f"default_{k}": v for k, v in
                              evaluate(name, START[name], p, train, val, cfg, cache_dir).items()}})
        print(f"{name}: {best}  ({time.time() - t0:.0f}s)", flush=True)

    records, finals = pd.DataFrame(records), pd.DataFrame(finals)
    if set(args.modules) != set(GRIDS) and (OUT / "sweep_records.csv").exists():
        # partial re-sweep: keep the earlier rows / choices of the modules not re-swept
        old_r, old_f = pd.read_csv(OUT / "sweep_records.csv"), pd.read_csv(OUT / "sweep_final.csv")
        records = pd.concat([old_r[~old_r.module.isin(args.modules)], records], ignore_index=True)
        finals = pd.concat([old_f[~old_f.module.isin(args.modules)], finals], ignore_index=True)
        chosen = {**json.loads((OUT / "chosen_params.json").read_text()), **chosen}
        order = {m: i for i, m in enumerate(GRIDS)}
        records = records.sort_values("module", key=lambda c: c.map(order), kind="stable")
        finals = finals.sort_values("module", key=lambda c: c.map(order), kind="stable")
    records.to_csv(OUT / "sweep_records.csv", index=False)
    finals.to_csv(OUT / "sweep_final.csv", index=False)
    (OUT / "chosen_params.json").write_text(json.dumps(chosen, indent=2))
    plot(records)

    if args.write_config:
        text = DEFAULT_CONFIG.read_text()
        start, end = text.index("features:\n"), text.index("\n# Verdict policy")
        block = "features:\n" + "".join(f"  {k}: {yaml.safe_dump(v, default_flow_style=True, width=10**6).strip()}\n"
                                        for k, v in {**cfg["features"], **chosen}.items())
        DEFAULT_CONFIG.write_text(text[:start] + block + text[end:])
        print("config.yaml updated")


if __name__ == "__main__":
    main()
