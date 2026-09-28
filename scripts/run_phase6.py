"""Phase 6: localisation - fuse the eight evidence maps into a tampered-region mask.

Data: cell-level evidence stacks from scripts/extract_maps.py (train + val; the test set is untouched).
  * Fusion models are trained on cells of *training* images (tampered with a valid mask + 1,000
    authentic): mean of maps (no training), logistic regression, gradient boosting (HGB).
  * Post-processing (threshold, opening, closing, minimum component area) is tuned on one half of the
    validation images and scored on the other half, then the halves swap (2-fold, grouped), so every
    reported validation number is out-of-sample.
  * Metrics per tampered image with a valid mask: pixel IoU, F1, MCC, and threshold-free pixel AUC
    (on every second pixel). Baselines: best single module, "whole image", and the oracle (best
    threshold per image, no morphology: a reference for how much a per-image threshold could gain).
  * Post-processing search: threshold first (no morphology), then morphology at that threshold.
  * The main fusion method and the "best single module" are picked by the reported out-of-sample F1,
    which slightly favours both (see the report).
  * Authentic validation images: share with any predicted region and mean predicted area.
The final localiser (fusion trained on train, post-processing tuned on all of val) is saved to
models/localizer_<protocol>.joblib and used by detect().
Usage:  python scripts/run_phase6.py [--protocols R B]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from forgery.config import load_config, resolve
from forgery.data.manifest import read_manifest
from forgery.data.splits import read_splits
from forgery.io import load_image, load_mask, protocol_mask
from forgery.localize.fusion import CELL, Localizer, cell_features, postprocess, scores, upsample
from forgery.pipeline import feature_code_sha1, resolved_params

OUT = resolve("results/phase6")
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)
MORPH = [(o, c, a) for o in (1, 5, 9) for c in (1, 9, 17) for a in (0.0, 0.002, 0.01)]
METHOD_LABEL = {"hgb": "Gradient boosting (cells)", "logreg": "Logistic regression (cells)",
                "mean": "Mean of the 8 maps", "best_single": "Best single module", "whole": "Whole image"}


def load_stacks(out_dir, rows: pd.DataFrame) -> dict:
    data = {}
    for r in rows.itertuples():
        z = np.load(out_dir / f"{r.image_id}.npz")
        data[r.image_id] = (z["stack"].astype(np.float32), z["frac"].astype(np.float32))
    return data, list(np.load(out_dir / f"{rows.image_id.iloc[0]}.npz")["names"])


def training_cells(train: pd.DataFrame, stacks: dict, seed: int, per_image: int = 400):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for r in train.itertuples():
        stack, frac = stacks[r.image_id]
        f = cell_features(stack)
        lab = (frac.ravel() >= 0.5).astype(int)
        pos, neg = np.flatnonzero(lab == 1), np.flatnonzero(lab == 0)
        take = np.concatenate([rng.choice(pos, min(per_image, pos.size), replace=False),
                               rng.choice(neg, min(per_image, neg.size), replace=False)])
        X.append(f[take])
        y.append(lab[take])
    return np.concatenate(X), np.concatenate(y)


def fit_models(X, y, seed) -> dict:
    models = {"mean": None,
              "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
              # no early stopping: its hold-out would be random cells, not whole images
              "hgb": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
                                                    class_weight="balanced", early_stopping=False,
                                                    random_state=seed)}
    for name, m in models.items():
        if m is not None:
            t0 = time.time()
            m.fit(X, y)
            print(f"  fitted {name} on {len(y):,} cells ({time.time() - t0:.0f}s)", flush=True)
    return models


def cell_prob(method: str, model, stack: np.ndarray, module_idx: int | None = None) -> np.ndarray:
    if method == "best_single":
        return stack[module_idx]
    return Localizer([], model).cell_prob(stack)


def _score_image(prob_full, truth, open_k, close_k, min_area, thresholds):
    return [scores(postprocess(prob_full, t, open_k, close_k, min_area), truth)["f1"] for t in thresholds]


def tune(probs: dict, truths: dict, ids: list) -> tuple[float, int, int, float]:
    """Threshold (no morphology) first, then morphology at that threshold; objective = mean F1."""
    f1 = np.mean([_score_image(probs[i], truths[i], 1, 1, 0.0, THRESHOLDS) for i in ids], axis=0)
    t = float(THRESHOLDS[int(f1.argmax())])
    best, best_score = (1, 1, 0.0), -1.0
    for o, c, a in MORPH:
        s = np.mean([scores(postprocess(probs[i], t, o, c, a), truths[i])["f1"] for i in ids])
        if s > best_score:
            best, best_score = (o, c, a), s
    return (t, *best)


def evaluate(probs, truths, ids, params) -> pd.DataFrame:
    t, o, c, a = params
    rows = []
    for i in ids:
        s = scores(postprocess(probs[i], t, o, c, a), truths[i])
        p, g = probs[i][::2, ::2].ravel(), truths[i][::2, ::2].ravel()
        s["pixel_auc"] = roc_auc_score(g, p) if 0 < g.mean() < 1 and np.ptp(p) > 0 else 0.5
        s["oracle_f1"] = max(scores(probs[i] >= th, truths[i])["f1"] for th in THRESHOLDS)
        rows.append({"image_id": i, **s})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocols", nargs="+", default=["R", "B"])
    ap.add_argument("--smoke", type=int, default=None, help="use the --limit N smoke-test maps; outputs to data/cache/phase6_smoke<N>")
    args = ap.parse_args()
    cfg = load_config()
    global OUT
    if args.smoke:
        OUT = resolve(f"data/cache/phase6_smoke{args.smoke}")
    seed = cfg["seed"]
    OUT.mkdir(parents=True, exist_ok=True)
    import importlib.util
    spec = importlib.util.spec_from_file_location("extract_maps", resolve("scripts/extract_maps.py"))
    em = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(em)

    meta = read_manifest(cfg).merge(read_splits(cfg), on="path")
    summary, all_rows = {}, []
    for P in args.protocols:
        t_start = time.time()
        out_dir = em.map_dir(cfg, P)
        if args.smoke:
            out_dir = out_dir.with_name(out_dir.name + f"_smoke{args.smoke}")
        index = pd.read_csv(out_dir / "index.csv", dtype={"path": str, "image_id": str})
        df = index.merge(meta.drop(columns=["split", "label", "image_id"]), on="path", validate="one_to_one")
        assert not (df.split == "test").any()
        stacks, names = load_stacks(out_dir, df)
        train, val = df[df.split == "train"], df[df.split == "val"]
        vt = val[(val.label == 1) & val.mask_valid].reset_index(drop=True)
        va = val[val.label == 0].reset_index(drop=True)

        X, y = training_cells(train, stacks, seed)
        models = fit_models(X, y, seed)

        # full-resolution ground truth for the evaluated images
        truths = {}
        for r in vt.itertuples():
            img_shape = load_image(resolve(r.path), P, cfg).shape[:2]
            m = protocol_mask(load_mask(resolve(r.mask_path)), P, cfg)
            assert m.shape == img_shape
            truths[r.image_id] = m
        shapes = {i: truths[i].shape for i in truths}
        for r in va.itertuples():
            h, w = stacks[r.image_id][0].shape[1:]
            shapes[r.image_id] = (h * CELL, w * CELL)

        # 2-fold grouped split of the validation images: tune on one half, score the other
        half = {g: int(hashlib_int(g) % 2) for g in val.split_group.unique()}
        vt["half"], va["half"] = vt.split_group.map(half), va.split_group.map(half)

        method_rows, per_image, chosen = [], {}, {}
        candidates = ["hgb", "logreg", "mean"] + [f"single:{n}" for n in names]
        for method in candidates:
            base, idx = (method, None) if not method.startswith("single:") else ("best_single", names.index(method[7:]))
            model = models.get(base)
            probs = {i: upsample(cell_prob(base, model, stacks[i][0], idx), shapes[i])
                     for i in list(truths) + list(va.image_id)}
            parts, params_per_half = [], {}
            for h in (0, 1):
                tune_ids = list(vt[vt.half == h].image_id)
                test_ids = list(vt[vt.half != h].image_id)
                params = tune(probs, truths, tune_ids)
                params_per_half[h] = params
                ev = evaluate(probs, truths, test_ids, params)
                au_ids = list(va[va.half != h].image_id)
                fp_area = [postprocess(probs[i], *params).mean() for i in au_ids]
                ev["half"] = 1 - h
                parts.append((ev, fp_area))
            ev = pd.concat([p[0] for p in parts]).merge(vt[["image_id", "forgery_type", "area_bucket", "split_group"]],
                                                        on="image_id")
            fp = np.concatenate([p[1] for p in parts])
            per_image[method] = ev
            chosen[method] = params_per_half
            method_rows.append({"protocol": P, "method": method, "f1": ev.f1.mean(), "iou": ev.iou.mean(),
                                "mcc": ev.mcc.mean(), "pixel_auc": ev.pixel_auc.mean(),
                                "oracle_f1": ev.oracle_f1.mean(),
                                "authentic_any_region": float((fp > 0).mean()),
                                "authentic_mean_area": float(fp.mean()), "n_tampered": len(ev),
                                "n_authentic": len(fp)})
            print(f"  [{P}] {method:24s} F1 {ev.f1.mean():.3f}  IoU {ev.iou.mean():.3f}  "
                  f"pixel-AUC {ev.pixel_auc.mean():.3f}", flush=True)
        mt = pd.DataFrame(method_rows)
        singles = mt[mt.method.str.startswith("single:")]
        best_single = singles.loc[singles.f1.idxmax(), "method"]
        whole = np.array([scores(np.ones_like(truths[i]), truths[i])["f1"] for i in vt.image_id])
        mt = pd.concat([mt, pd.DataFrame([{"protocol": P, "method": "whole", "f1": whole.mean(),
                                           "iou": np.mean([truths[i].mean() for i in vt.image_id]),
                                           "authentic_any_region": 1.0, "authentic_mean_area": 1.0,
                                           "n_tampered": len(vt)}])], ignore_index=True)
        main_method = mt[mt.method.isin(["hgb", "logreg", "mean"])].set_index("method").f1.idxmax()
        ev = per_image[main_method]
        g = ev.split_group.to_numpy()
        ci = {k: mean_ci(ev[k].to_numpy(), g, seed) for k in ("f1", "iou", "mcc")}
        by_area = ev.groupby("area_bucket")[["f1", "iou", "mcc", "pixel_auc"]].mean().reindex(["<1%", "1-5%", ">5%"])
        by_type = ev.groupby("forgery_type")[["f1", "iou", "mcc", "pixel_auc"]].mean()

        # final localiser: fusion from train, post-processing tuned on all tampered val images
        base = main_method
        probs = {i: upsample(cell_prob(base, models.get(base), stacks[i][0]), shapes[i]) for i in truths}
        final_params = tune(probs, truths, list(truths))
        loc = Localizer(names, models.get(base), *final_params,
                        meta={"protocol": P, "method": base, "feature_params": resolved_params(cfg, P),
                              "feature_code_sha1": feature_code_sha1(),
                              "trained_on": "train split cells; post-processing tuned on val"})
        if not args.smoke:
            resolve("models").mkdir(exist_ok=True)
            joblib.dump(loc, resolve(f"models/localizer_{P}.joblib"), compress=3)

        # figures
        fig_methods(mt, P, best_single)
        fig_examples(ev, vt, P, cfg, loc)
        mt.to_csv(OUT / f"methods_{P}.csv", index=False)
        ev.to_csv(OUT / f"per_image_{P}.csv", index=False)
        summary[P] = {"main_method": main_method, "best_single": best_single,
                      "final_params": dict(zip(["threshold", "open_k", "close_k", "min_area"], final_params)),
                      "chosen_per_half": {m: {str(h): list(p) for h, p in v.items()} for m, v in chosen.items()
                                          if m in ("hgb", "logreg", "mean")},
                      "ci95": ci, "by_area": by_area.round(4).to_dict(orient="index"),
                      "by_type": by_type.round(4).to_dict(orient="index"),
                      "minutes": round((time.time() - t_start) / 60, 1)}
        all_rows.append((P, mt, by_area, by_type, ci, main_method, best_single, final_params))

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    write_md(all_rows)


def hashlib_int(s: str) -> int:
    """Stable integer hash of a split group (assigns validation groups to the two tuning halves)."""
    return int(hashlib.md5(str(s).encode()).hexdigest(), 16)


def mean_ci(values: np.ndarray, groups: np.ndarray, seed: int, n: int = 1000) -> tuple[float, float]:
    """95 % percentile CI of the mean, resampling whole split groups with replacement."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    sums, counts = np.bincount(inv, weights=values), np.bincount(inv)
    draws = rng.integers(0, len(uniq), size=(n, len(uniq)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def fig_methods(mt, P, best_single):
    keep = ["hgb", "logreg", "mean", best_single, "whole"]
    d = mt.set_index("method").reindex(keep)
    labels = [METHOD_LABEL.get(m, f"Best single: {m[7:]}") for m in keep]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(keep))
    ax.bar(x - 0.2, d.f1, 0.4, label="pixel F1", color="#4C78A8")
    ax.bar(x + 0.2, d.iou, 0.4, label="IoU", color="#72B7B2")
    for xi, v in zip(x, d.f1):
        ax.text(xi - 0.2, v + 0.005, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x, labels, fontsize=8, rotation=12)
    ax.set_ylabel("mean over tampered val images")
    ax.set_title(f"Localisation (protocol {P}; post-processing tuned on the other half of val)", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / f"fig_methods_{P}.png", dpi=150)
    plt.close(fig)


def fig_examples(ev, vt, P, cfg, loc):
    """Best, median and worst F1 cases (2 each) + 2 random: image, truth, probability, final mask."""
    ev = ev.sort_values("f1")
    rng = np.random.default_rng(cfg["seed"])
    ids = list(ev.image_id.iloc[-2:]) + list(ev.image_id.iloc[len(ev) // 2 - 1:len(ev) // 2 + 1]) + \
        list(ev.image_id.iloc[:2]) + list(rng.choice(ev.image_id, 2, replace=False))
    from forgery.features import build_modules
    from forgery.pipeline import module_params
    modules = build_modules(params=module_params(cfg, P))
    fig, axes = plt.subplots(len(ids), 4, figsize=(10, 2.1 * len(ids)))
    for k, i in enumerate(ids):
        r = vt[vt.image_id == i].iloc[0]
        img = load_image(resolve(r.path), P, cfg)
        maps = {m.name: m.extract(img).evidence_map for m in modules}
        prob, mask = loc.predict(maps, img.shape[:2])
        truth = protocol_mask(load_mask(resolve(r.mask_path)), P, cfg)
        f1 = scores(mask, truth)["f1"]
        for ax, im, title in zip(axes[k], (img, truth, prob, mask),
                                 (f"{r.forgery_type}, area {100 * r.tampered_frac:.1f}%", "ground truth",
                                  "fused probability", f"final mask (F1 {f1:.2f})")):
            ax.imshow(im, cmap=None if im.ndim == 3 else ("magma" if im.dtype != bool else "gray"),
                      vmin=0, vmax=1 if im.ndim == 2 else None)
            ax.set_title(title, fontsize=7)
            ax.axis("off")
    fig.suptitle(f"Localisation examples, protocol {P}: 2 best, 2 median, 2 worst, 2 random (val)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(OUT / f"fig_examples_{P}.png", dpi=110)
    plt.close(fig)


def write_md(all_rows):
    md = ["# Phase 6: localisation", "",
          "Tampered validation images with a valid mask; post-processing tuned on one half of val and scored on the "
          "other (2-fold, grouped), so the per-method numbers are out-of-sample (the choice of main method and best "
          "single module is made on these numbers). Fusion models trained on train-split cells. The final "
          "post-processing below is tuned on all val images. Pixel AUC on every second pixel. Test set untouched.", ""]
    for P, mt, by_area, by_type, ci, main_method, best_single, final_params in all_rows:
        tab = mt.copy()
        tab["method"] = [METHOD_LABEL.get(m, f"single: {m[7:]}") for m in tab.method]
        md += [f"## Protocol {P}", "", f"Main method: **{METHOD_LABEL[main_method]}**; best single module: "
               f"{best_single[7:]}; final post-processing (threshold, open, close, min area): {final_params}", "",
               f"95 % CI (group bootstrap) of the main method: {({k: [round(v[0], 3), round(v[1], 3)] for k, v in ci.items()})}",
               "", tab.round(3).to_markdown(index=False), "", "### By tampered area", "",
               by_area.round(3).to_markdown(), "", "### By forgery type", "", by_type.round(3).to_markdown(), ""]
    (OUT / "localization.md").write_text("\n".join(md))
    print("\n".join(md[:12]))


if __name__ == "__main__":
    main()
