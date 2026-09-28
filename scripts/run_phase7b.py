"""Phase 7b: robustness, synthetic "own photo" forgeries, and a cross-dataset test.

All models are the frozen deployed ones (models/forgery_<P>, models/localizer_<P>); nothing is tuned here.
  * robustness: every test image is degraded *before* the protocol (JPEG re-compression, resizing,
    blur, noise, a simulated social-media upload); the tampered-region mask follows any geometry change.
  * synthetic:  forgeries generated from test-set authentic images with exactly known masks - hard-edged
    splices, Poisson-blended (seamless) splices, plain and rotated/scaled copy-moves - plus untouched
    authentic images, each also in a "social" version (downscale + JPEG Q70).
  * external:   MICC-F220 (Amerini et al., IEEE TIFS 2011; 110 copy-move forgeries + 110 originals,
    image-level labels only), downloaded to data/external/ for non-commercial research use.
Per-condition results are cached in data/cache/phase7b/, so an interrupted run resumes.
Usage:  python scripts/run_phase7b.py [--parts robustness synthetic external] [--protocols R B]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time

import cv2
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from forgery.config import load_config, resolve
from forgery.data.manifest import read_manifest
from forgery.data.splits import read_splits
from forgery.eval.classify import group_bootstrap
from forgery.features import build_modules
from forgery.io import apply_protocol, jpeg_roundtrip, load_mask, load_rgb, protocol_mask
from forgery.localize.fusion import scores
from forgery.pipeline import feature_code_sha1, module_params

OUT = resolve("results/phase7b")
CACHE = resolve("data/cache/phase7b")
SYN = resolve("data/synthetic")

CONDITIONS = {                     # family -> list of (label, parameter)
    "clean": [("clean", None)],
    "jpeg": [(f"JPEG Q{q}", q) for q in (95, 85, 75, 65, 50)],
    "resize": [(f"resize x{f}", f) for f in (0.9, 0.75, 0.5)],
    "blur": [(f"blur s={s}", s) for s in (0.5, 1.0, 1.5)],
    "noise": [(f"noise s={s}", s) for s in (2, 5, 10)],
    "social": [("social (x0.8 + JPEG Q70)", None)],
}


# --------------------------------------------------------------- degradations ---

def degrade(img: np.ndarray, mask: np.ndarray | None, family: str, param, seed: int):
    if family == "jpeg":
        return jpeg_roundtrip(img, param), mask
    if family in ("resize", "social"):
        f = param if family == "resize" else 0.8
        size = (max(1, round(img.shape[1] * f)), max(1, round(img.shape[0] * f)))
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        if mask is not None:
            mask = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
        return (jpeg_roundtrip(img, 70) if family == "social" else img), mask
    if family == "blur":
        return cv2.GaussianBlur(img, (0, 0), param), mask
    if family == "noise":
        rng = np.random.default_rng(seed)
        return np.clip(np.rint(img + rng.normal(0, param, img.shape)), 0, 255).astype(np.uint8), mask
    return img, mask


# ------------------------------------------------------------------ analysis ---

_MODELS: dict = {}


def _models(P: str, cfg: dict):
    if P not in _MODELS:
        model = joblib.load(resolve(f"models/forgery_{P}.joblib"))
        _MODELS[P] = (build_modules(params=module_params(cfg, P)), model,
                      model.estimator.calibrated_classifiers_[0].estimator,
                      joblib.load(resolve(f"models/localizer_{P}.joblib")))
    return _MODELS[P]


def analyse(img: np.ndarray, mask: np.ndarray | None, P: str, cfg: dict) -> dict:
    """Frozen pipeline on a pixel array: protocol, 8 modules, raw forest score, calibrated score, mask."""
    cv2.setNumThreads(1)
    modules, model, forest, loc = _models(P, cfg)
    x = apply_protocol(img, P, cfg)
    feats, maps = {}, {}
    for m in modules:
        o = m.extract(x)
        maps[m.name] = o.evidence_map
        feats.update({f"{m.name}.{k}": float(v) for k, v in o.features.items()})
    row = np.array([[feats[n] for n in model.feature_names_]])
    rec = {"p_raw": float(forest.predict_proba(row)[0, 1]), "p_cal": float(model.predict_proba(row)[0, 1]),
           "cm_inliers": feats["cm_keypoint.local_n_inliers"]}
    if mask is not None:
        _, pred = loc.predict(maps, x.shape[:2])
        truth = protocol_mask(mask, P, cfg)
        if truth.shape == pred.shape and 0 < truth.mean() < 1:
            rec.update({k: v for k, v in scores(pred, truth).items() if k in ("f1", "iou")})
    return rec


def _job(item: dict, P: str, family: str, param, cfg: dict) -> dict:
    img = load_rgb(resolve(item["path"]))
    mask = load_mask(resolve(item["mask_path"])) if isinstance(item.get("mask_path"), str) else None
    seed = int(hashlib.md5(item["path"].encode()).hexdigest()[:8], 16)
    img, mask = degrade(img, mask, family, param, seed)
    return {"path": item["path"], **analyse(img, mask, P, cfg)}


def run_condition(items: pd.DataFrame, P: str, family: str, label: str, param, cfg: dict, tag: str) -> pd.DataFrame:
    key = hashlib.sha1(json.dumps([tag, P, label, feature_code_sha1(), module_params(cfg, P), sorted(items.path)],
                                  sort_keys=True, default=str).encode()).hexdigest()[:12]
    f = CACHE / f"{tag}_{P}_{key}.csv"
    if f.exists():
        return pd.read_csv(f, dtype={"path": str})
    recs = items[["path", "mask_path"]].to_dict("records")
    res = pd.DataFrame(Parallel(n_jobs=cfg["n_jobs"], batch_size=8)(delayed(_job)(r, P, family, param, cfg) for r in recs))
    res.to_csv(f, index=False)
    return res


def summarise(res: pd.DataFrame, items: pd.DataFrame, band: float, seed: int, n_boot: int = 500) -> dict:
    d = items.merge(res, on="path", validate="one_to_one")
    y, g = d.label.to_numpy(), d.group.to_numpy()
    lo, hi = 0.5 - band, 0.5 + band
    judged = (d.p_cal >= hi) | (d.p_cal <= lo)
    out = {"n": len(d), "auc": roc_auc_score(y, d.p_raw) if len(set(y)) == 2 else np.nan,
           "balanced_accuracy@0.5": balanced_accuracy_score(y, d.p_raw >= 0.5) if len(set(y)) == 2 else np.nan,
           "coverage": float(judged.mean()),
           "accuracy_judged": float(((d.p_cal >= 0.5).astype(int) == y)[judged].mean()) if judged.any() else np.nan,
           "detected_tampered": float((d.p_cal[y == 1] >= hi).mean()) if (y == 1).any() else np.nan,
           "false_alarm_authentic": float((d.p_cal[y == 0] >= hi).mean()) if (y == 0).any() else np.nan}
    if len(set(y)) == 2:
        out["auc_ci"] = group_bootstrap(y, d.p_raw.to_numpy(), g, roc_auc_score, n_boot, seed)
    if "f1" in d:
        out["loc_f1"], out["loc_iou"], out["n_loc"] = float(d.f1.mean()), float(d.iou.mean()), int(d.f1.notna().sum())
    return out


# ---------------------------------------------------------------- the three parts ---

def robustness(cfg, test: pd.DataFrame, protocols) -> pd.DataFrame:
    rows = []
    for P in protocols:
        band = joblib.load(resolve(f"models/forgery_{P}.joblib")).meta["uncertain_half_width"]
        for family, conds in CONDITIONS.items():
            for label, param in conds:
                t0 = time.time()
                res = run_condition(test, P, family, label, param, cfg, "robust")
                rows.append({"protocol": P, "family": family, "condition": label, "param": param,
                             **summarise(res, test, band, cfg["seed"])})
                print(f"  [{P}] {label:26s} AUC {rows[-1]['auc']:.3f}  F1 {rows[-1].get('loc_f1', np.nan):.3f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def _blob(h: int, w: int, area: float, rng) -> np.ndarray:
    """Random smooth, object-like region covering about `area` of an h x w frame."""
    noise = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), min(h, w) / 8)
    cy, cx = rng.integers(h // 4, 3 * h // 4), rng.integers(w // 4, 3 * w // 4)
    yy, xx = np.mgrid[:h, :w]
    ry, rx = np.sqrt(area * h * w / np.pi) * rng.uniform(0.7, 1.3, 2)
    field = 1 - ((yy - cy) / ry) ** 2 - ((xx - cx) / rx) ** 2 + 0.6 * noise / (noise.std() + 1e-6)
    m = field > 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    return (lab == 1 + stats[1:, cv2.CC_STAT_AREA].argmax()) if n > 1 else m


def _rel(p) -> str:
    """Project-relative path when possible (resolve() accepts both forms)."""
    try:
        return str(p.relative_to(resolve(".")))
    except ValueError:
        return str(p)


def make_synthetic(cfg, test: pd.DataFrame, n_per_type: int = 120) -> pd.DataFrame:
    """Forgeries from test-set authentic images with known masks; saved losslessly (PNG) with their masks."""
    index = SYN / "index.csv"
    if index.exists():
        return pd.read_csv(index, dtype={"path": str, "mask_path": str})
    SYN.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])
    shuffled = test[test.label == 0].sample(frac=1, random_state=cfg["seed"]).path.tolist()
    untouched, au = shuffled[:2 * n_per_type], shuffled[2 * n_per_type:]   # disjoint: never used as host/donor
    kinds = ["splice (hard edge)", "splice (seamless)", "copy-move (shift)", "copy-move (rotate+scale)"]
    rows, k = [], 0
    for kind in kinds:
        made = 0
        while made < n_per_type:
            host, donor = load_rgb(resolve(au[k % len(au)])), load_rgb(resolve(au[(k + 1) % len(au)]))
            k += 2
            h, w = host.shape[:2]
            blob = _blob(h, w, rng.uniform(0.02, 0.12), rng)
            if blob.sum() < 64:                               # degenerate random region: draw again
                continue
            ys, xs = np.nonzero(blob)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            bh, bw = y1 - y0, x1 - x0
            if bh < 16 or bw < 16:
                continue
            out, mask = host.copy(), np.zeros((h, w), bool)
            if kind.startswith("splice"):
                if donor.shape[0] < bh or donor.shape[1] < bw:
                    continue
                dy, dx = rng.integers(0, donor.shape[0] - bh + 1), rng.integers(0, donor.shape[1] - bw + 1)
                patch, pm = donor[dy:dy + bh, dx:dx + bw], blob[y0:y1, x0:x1]
                if kind.endswith("(seamless)"):
                    center = (int(x0 + bw // 2), int(y0 + bh // 2))
                    try:
                        full = np.zeros_like(host)
                        full[y0:y1, x0:x1] = patch
                        out = cv2.seamlessClone(full, host, (blob * 255).astype(np.uint8), center, cv2.NORMAL_CLONE)
                    except cv2.error:
                        continue
                    # Poisson blending also alters pixels around the region: the true tampered area is
                    # the pasted region plus every pixel changed by more than 2 grey levels
                    changed = np.abs(out.astype(np.int16) - host.astype(np.int16)).max(axis=2) > 2
                    mask = blob | changed
                else:
                    out[y0:y1, x0:x1][pm] = patch[pm]
                    mask = blob
                union = None
            else:
                ang, sc = (rng.uniform(-20, 20), rng.uniform(0.85, 1.15)) if "rotate" in kind else (0.0, 1.0)
                ty, tx = rng.integers(-h // 3, h // 3), rng.integers(-w // 3, w // 3)
                if abs(ty) < bh // 2 and abs(tx) < bw // 2:
                    continue
                M = cv2.getRotationMatrix2D((float(x0 + bw / 2), float(y0 + bh / 2)), ang, sc)
                M[:, 2] += (tx, ty)
                warped = cv2.warpAffine(host, M, (w, h), flags=cv2.INTER_LINEAR)
                wmask = cv2.warpAffine(blob.astype(np.uint8), M, (w, h), flags=cv2.INTER_NEAREST).astype(bool)
                if wmask.sum() < 0.6 * blob.sum():            # copied region fell off the frame
                    continue
                out[wmask] = warped[wmask]
                mask = wmask                                  # the pasted copy is the tampered region
                union = wmask | (blob & ~wmask)               # source region (where still visible) + copy
            name = f"syn_{len(rows):04d}"
            Image.fromarray(out).save(SYN / f"{name}.png")
            Image.fromarray((mask * 255).astype(np.uint8)).save(SYN / f"{name}_gt.png")
            union_path = None
            if union is not None:
                Image.fromarray((union * 255).astype(np.uint8)).save(SYN / f"{name}_gt_union.png")
                union_path = _rel(SYN / f"{name}_gt_union.png")
            rows.append({"path": _rel(SYN / f"{name}.png"), "mask_path": _rel(SYN / f"{name}_gt.png"),
                         "union_mask_path": union_path, "label": 1, "kind": kind, "group": name,
                         "area": float(mask.mean())})
            made += 1
    for j, p in enumerate(untouched):                         # untouched authentic images (never hosts/donors)
        name = f"syn_au_{j:04d}"
        Image.fromarray(load_rgb(resolve(p))).save(SYN / f"{name}.png")
        rows.append({"path": _rel(SYN / f"{name}.png"), "mask_path": None, "union_mask_path": None,
                     "label": 0, "kind": "authentic", "group": name, "area": 0.0})
    df = pd.DataFrame(rows)
    df.to_csv(index, index=False)
    return df


def synthetic(cfg, syn: pd.DataFrame, protocols) -> pd.DataFrame:
    rows = []
    for P in protocols:
        band = joblib.load(resolve(f"models/forgery_{P}.joblib")).meta["uncertain_half_width"]
        for family, label in (("clean", "as made"), ("social", "social (x0.8 + JPEG Q70)")):
            res = run_condition(syn, P, family, label, None, cfg, "synthetic")
            cm = syn[syn.union_mask_path.notna()].assign(mask_path=lambda d: d.union_mask_path)
            union = run_condition(cm, P, family, label, None, cfg, "synthetic_union")[["path", "f1"]]
            for kind in [k for k in syn.kind.unique() if k != "authentic"] + ["all"]:
                sub = syn[(syn.kind == kind) | (syn.label == 0)] if kind != "all" else syn
                row = {"protocol": P, "version": label, "kind": kind, **summarise(res, sub, band, cfg["seed"])}
                if kind.startswith("copy-move"):
                    row["loc_f1_source_and_copy"] = float(union[union.path.isin(sub.path)].f1.mean())
                rows.append(row)
            print(f"  [{P}] synthetic {label}: AUC(all) {rows[-1]['auc']:.3f}  F1 {rows[-1].get('loc_f1', np.nan):.3f}",
                  flush=True)
    return pd.DataFrame(rows)


def external(cfg, protocols) -> pd.DataFrame:
    root = resolve("data/external/MICC-F220")
    gt = pd.read_csv(root / "groundtruthDB_220.txt", sep=r"\s+", header=None, names=["file", "label"])
    items = pd.DataFrame({"path": [str((root / f).relative_to(resolve("."))) for f in gt.file],
                          "mask_path": None, "label": gt.label.astype(int),
                          # originals and their forgeries share a group (CRW_4853_scale.jpg, CRW_4853tamp176.jpg)
                          "group": gt.file.str.extract(r"^(.+?)_?(?:tamp\d+|scale)\.[A-Za-z]+$", expand=False)
                          .fillna(gt.file)})
    rows = []
    for P in protocols:
        band = joblib.load(resolve(f"models/forgery_{P}.joblib")).meta["uncertain_half_width"]
        res = run_condition(items, P, "clean", "MICC-F220", None, cfg, "external")
        d = items.merge(res, on="path")
        rows.append({"protocol": P, "dataset": "MICC-F220", **summarise(res, items, band, cfg["seed"]),
                     "share_with_cm_inliers_tampered": float((d.cm_inliers[d.label == 1] > 0).mean()),
                     "share_with_cm_inliers_original": float((d.cm_inliers[d.label == 0] > 0).mean())})
        print(f"  [{P}] MICC-F220: AUC {rows[-1]['auc']:.3f}", flush=True)
    return pd.DataFrame(rows)


# -------------------------------------------------------------------- figures ---

def fig_robustness(rob: pd.DataFrame):
    fams = [f for f in CONDITIONS if f != "clean"]
    fig, axes = plt.subplots(2, len(fams), figsize=(3.2 * len(fams), 5.6), sharey="row")
    colors = {"R": "#4C78A8", "B": "#E45756"}
    for j, fam in enumerate(fams):
        for P in rob.protocol.unique():
            d = pd.concat([rob[(rob.protocol == P) & (rob.family == "clean")],
                           rob[(rob.protocol == P) & (rob.family == fam)]])
            x = np.arange(len(d))
            for i, (metric, lab) in enumerate((("auc", "image AUC"), ("loc_f1", "pixel F1"))):
                axes[i, j].plot(x, d[metric], "o-", color=colors.get(P, "k"), label=f"protocol {P}")
                axes[i, j].set_xticks(x, ["clean"] + [c.split(" ", 1)[-1] if fam != "social" else "social"
                                                      for c in d.condition.iloc[1:]], fontsize=7, rotation=30)
                if j == 0:
                    axes[i, j].set_ylabel(lab)
        axes[0, j].set_title(fam, fontsize=9)
    axes[0, 0].axhline(0.5, color="grey", ls=":", lw=0.8)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("Robustness of the frozen detector on the test set (degradation applied before the protocol)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "fig_robustness.png", dpi=140)
    plt.close(fig)


def fig_synthetic_examples(cfg, syn: pd.DataFrame, P: str = "R"):
    """For each forgery kind, the example whose pixel F1 is closest to that kind's median (representative)."""
    loc = joblib.load(resolve(f"models/localizer_{P}.joblib"))
    modules = build_modules(params=module_params(cfg, P))
    res = run_condition(syn, P, "clean", "as made", None, cfg, "synthetic")
    d = syn.merge(res, on="path")
    kinds = [k for k in syn.kind.unique() if k != "authentic"]
    fig, axes = plt.subplots(len(kinds), 4, figsize=(11, 2.3 * len(kinds)))
    for i, kind in enumerate(kinds):
        dk = d[(d.kind == kind) & d.f1.notna()]
        r = dk.iloc[(dk.f1 - dk.f1.median()).abs().argmin()]
        img, truth = load_rgb(resolve(r.path)), load_mask(resolve(r.mask_path))
        x = apply_protocol(img, P, cfg)
        maps = {m.name: m.extract(x).evidence_map for m in modules}
        prob, pred = loc.predict(maps, x.shape[:2])
        t = protocol_mask(truth, P, cfg)
        for ax, im, title in zip(axes[i], (img, t, prob, pred),
                                 (f"{kind} (P(tampered) {r.p_cal:.2f})", "true region", "fused probability",
                                  f"predicted mask (F1 {scores(pred, t)['f1']:.2f}; median {dk.f1.median():.2f})")):
            ax.imshow(im, cmap=None if im.ndim == 3 else ("gray" if im.dtype == bool else "magma"))
            ax.set_title(title, fontsize=7)
            ax.axis("off")
    fig.suptitle(f"Synthetic forgeries from test-set authentic images: median-F1 example of each kind (protocol {P})",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "fig_synthetic_examples.png", dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------- main ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["robustness", "synthetic", "external"])
    ap.add_argument("--protocols", nargs="+", default=["R", "B"])
    args = ap.parse_args()
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    df = read_manifest(cfg).merge(read_splits(cfg), on="path")
    test = df[df.split == "test"].copy()
    test["group"] = test.split_group
    test["mask_path"] = test.mask_path.where(test.mask_valid, None)
    test = test[["path", "mask_path", "label", "group", "forgery_type", "area_bucket"]].reset_index(drop=True)
    with resolve("results/phase7/test_runs.log").open("a") as fh:     # post-hoc reuse of the test split
        fh.write(json.dumps({"event": "posthoc_phase7b", "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                             "parts": args.parts, "protocols": args.protocols,
                             "script_sha1": hashlib.sha1(resolve("scripts/run_phase7b.py").read_bytes()).hexdigest(),
                             "note": "frozen models, evaluation only; nothing tuned or fed back"}) + "\n")
    summary_file = OUT / "summary.json"
    summary = json.loads(summary_file.read_text()) if summary_file.exists() else {}   # keep other parts
    if "robustness" in args.parts:
        rob = robustness(cfg, test, args.protocols)
        rob.to_csv(OUT / "robustness.csv", index=False)
        fig_robustness(rob)
        summary["robustness"] = rob.round(4).to_dict(orient="records")
    if "synthetic" in args.parts:
        syn = make_synthetic(cfg, test)
        sy = synthetic(cfg, syn, args.protocols)
        sy.to_csv(OUT / "synthetic.csv", index=False)
        fig_synthetic_examples(cfg, syn)
        summary["synthetic"] = sy.round(4).to_dict(orient="records")
    if "external" in args.parts:
        ex = external(cfg, args.protocols)
        ex.to_csv(OUT / "external.csv", index=False)
        summary["external"] = ex.round(4).to_dict(orient="records")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
