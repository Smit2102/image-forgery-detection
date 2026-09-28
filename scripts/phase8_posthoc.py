"""Phase 8 post-hoc analysis: reads saved scores only (never re-evaluates the test set).

  * Size groups. The pre-registered size check (area terciles, run_phase8.size_analysis) is invalid: 78 % of the
    test images are 384 x 256 (either orientation) and the tie was broken by row order, which is authentic-first, so the
    terciles were confounded with the label. Here images are grouped by their natural size instead (smaller than /
    exactly / larger than 384 x 256), on test and on validation (selection runs, seed 42), with CIs; per group also
    per-type AUCs (with n), the mean CNN logit of authentic images, and Spearman(CNN score, tile count).
  * Protocol A, JPEG only: does the CNN separate tampered JPEGs from authentic JPEGs with standard IJG tables only,
    or also from those with non-standard tables?
  * The CNN and the rank average against every Phase-7 classical model (RF and SVM), overall and on splicing.
  * Multiplicity of the four pre-registered paired primaries: two-sided group-bootstrap p-values with Holm.
Usage:  python scripts/phase8_posthoc.py
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from forgery import cnn
from forgery.config import load_config, resolve
from forgery.data.manifest import read_manifest
from forgery.data.splits import read_splits
from forgery.eval.classify import group_bootstrap

STANDARD = 384 * 256
OUT = resolve("results/phase8")


def _ci(y, s, g, seed) -> list[float]:
    return list(group_bootstrap(y, np.asarray(s), g, roc_auc_score, 1000, seed))


def boot_delta(y, a, b, g, seed, n=1000) -> dict:
    """Paired group-bootstrap of AUC(a) - AUC(b): point, 95 % CI and two-sided p (share of resamples crossing 0)."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(g, return_inverse=True)
    members = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    ds = []
    for _ in range(n):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(uniq), len(uniq))])
        if len(np.unique(y[idx])) == 2:
            ds.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    ds = np.asarray(ds)
    p = min(1.0, 2 * min((ds <= 0).mean(), (ds >= 0).mean()))
    return {"delta_auc": float(roc_auc_score(y, a) - roc_auc_score(y, b)),
            "ci": [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))], "p_two_sided": float(p)}


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adj, running = np.empty_like(p), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def size_groups(d: pd.DataFrame, cols: list[str], seed: int, split: str, P: str) -> list[dict]:
    px = d.width * d.height
    rows = []
    for name, sel in (("smaller than 384x256", px < STANDARD), ("384x256", px == STANDARD),
                      ("larger than 384x256", px > STANDARD)):
        s = d[sel]
        y, g = s.label.to_numpy(), s.split_group.to_numpy()
        row = {"split": split, "protocol": P, "size_group": name, "n": len(s), "n_tampered": int(y.sum()),
               "share_tiff": float((s.format == "TIFF").mean()),
               "mean_cnn_logit_authentic": float(s.cnn[y == 0].mean()),
               "spearman_cnn_vs_tiles": float(spearmanr(s.cnn, s.n_tiles).statistic) if s.n_tiles.nunique() > 1
               else None}
        for c in cols:
            row[f"{c}_auc"] = float(roc_auc_score(y, s[c]))
            row[f"{c}_ci"] = _ci(y, s[c], g, seed)
        if "classical" in cols:
            dl = boot_delta(y, s.cnn.to_numpy(), s.classical.to_numpy(), g, seed)
            row["cnn_minus_classical"], row["delta_ci"] = dl["delta_auc"], dl["ci"]
        for t in ("copy-move", "splicing"):
            sub = (y == 0) | (s.forgery_type == t).to_numpy()
            row[f"n_{t}"] = int((s.forgery_type == t).sum())
            for c in cols:
                row[f"{c}_auc_{t}"] = float(roc_auc_score(y[sub], s[c][sub])) if row[f"n_{t}"] else None
        rows.append(row)
    return rows


def main():
    cfg, seed = load_config(), load_config()["seed"]
    man = read_manifest(cfg).merge(read_splits(cfg), on="path")
    meta = man[["path", "image_id", "width", "height", "format", "forgery_type", "qtable_standard", "split_group"]]
    out = {"created": datetime.now().isoformat(timespec="seconds"), "note": "post-hoc; saved scores only",
           "script_sha1": hashlib.sha1(resolve("scripts/phase8_posthoc.py").read_bytes()).hexdigest(),
           "inputs_sha1": {}}
    rows, test = [], {}
    for P in ("R", "B", "A"):
        f = OUT / f"test_scores_cnn_{P}.csv"
        out["inputs_sha1"][f.name] = hashlib.sha1(f.read_bytes()).hexdigest()
        d = pd.read_csv(f, dtype={"path": str}, float_precision="round_trip").merge(meta.drop(columns="split_group"), on="path", validate="one_to_one")
        test[P] = d
        rows += size_groups(d, ["cnn", "classical", "combination"], seed, "test", P)
        # validation: the train-only selection model (seed 42) and its selected pooling
        sel = pd.read_csv(OUT / "selection.csv").query("protocol == @P and seed == 42").iloc[0]
        v = pd.read_csv(OUT / f"val_scores_{P}_s42.csv", dtype={"path": str}).merge(meta, on="path", validate="one_to_one")
        v["cnn"] = v[f"cnn_{sel.aggregator}"]
        v["n_tiles"] = [cnn.n_tiles(Image.open(resolve("data/ela") / P / f"{i}.png").size[::-1]) for i in v.image_id]
        rows += size_groups(v, ["cnn"], seed, "val", P)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "posthoc_size_groups.csv", index=False)
    out["size_groups"] = res.to_dict(orient="records")

    # protocol A, JPEG only: authentic with standard IJG tables vs non-standard tables, each against tampered JPEGs
    d = test["A"][test["A"].format == "JPEG"]
    tp = d[d.label == 1]
    out["A_jpeg_only"] = {}
    for name, au in (("authentic_standard_ijg", d[(d.label == 0) & d.qtable_standard]),
                     ("authentic_non_standard", d[(d.label == 0) & ~d.qtable_standard.astype(bool)])):
        s = pd.concat([au, tp])
        out["A_jpeg_only"][name] = {"n_authentic": len(au), "n_tampered": len(tp),
                                    "cnn_auc": float(roc_auc_score(s.label, s.cnn)),
                                    "classical_auc": float(roc_auc_score(s.label, s.classical))}

    # the CNN and the rank average against every Phase-7 classical model (RF and SVM; overall and splicing only)
    out["vs_phase7_models"] = []
    for P in ("R", "B"):
        d = test[P].merge(pd.read_csv(resolve(f"results/phase7/run_1/test_scores_{P}.csv"), dtype={"path": str}),
                          on="path", validate="one_to_one")
        y, g = d.label.to_numpy(), d.split_group.to_numpy()
        spl = ((y == 0) | (d.forgery_type == "splicing")).to_numpy()
        for col in [c for c in d.columns if "|" in c]:
            out["vs_phase7_models"].append({
                "protocol": P, "model": col, "auc": float(roc_auc_score(y, d[col])),
                "auc_splicing": float(roc_auc_score(y[spl], d[col][spl])),
                "cnn_minus_model": boot_delta(y, d.cnn.to_numpy(), d[col].to_numpy(), g, seed),
                "rank_average_minus_model": boot_delta(y, d.combination.to_numpy(), d[col].to_numpy(), g, seed)})
    out["cnn_auc_splicing"] = {P: float(roc_auc_score(test[P].label[(test[P].label == 0) | (test[P].forgery_type == "splicing")],
                                                      test[P].cnn[(test[P].label == 0) | (test[P].forgery_type == "splicing")]))
                               for P in ("R", "B")}

    # the four pre-registered paired primaries: bootstrap p-values, Holm
    prim = []
    for P in ("R", "B"):
        d = test[P]
        y, g = d.label.to_numpy(), d.split_group.to_numpy()
        for name, a in (("CNN - classical", d.cnn), ("rank average - classical", d.combination)):
            prim.append({"protocol": P, "comparison": name, **boot_delta(y, a.to_numpy(), d.classical.to_numpy(), g, seed)})
    adj = holm(np.array([r["p_two_sided"] for r in prim]))
    for r, a in zip(prim, adj):
        r["p_holm"] = float(a)
    out["primaries"] = prim
    (OUT / "posthoc.json").write_text(json.dumps(out, indent=2, default=float))
    with pd.option_context("display.width", 220, "display.max_columns", 40):
        print(res[["split", "protocol", "size_group", "n", "n_tampered", "cnn_auc", "classical_auc",
                   "cnn_minus_classical", "mean_cnn_logit_authentic", "spearman_cnn_vs_tiles"]].round(3).to_string(index=False))
    print(json.dumps({"A_jpeg_only": out["A_jpeg_only"], "primaries": prim}, indent=1))
    for r in out["vs_phase7_models"]:
        print(f"{r['protocol']} {r['model']:24s} AUC {r['auc']:.3f} splicing {r['auc_splicing']:.3f} | CNN - model "
              f"{r['cnn_minus_model']['delta_auc']:+.3f} {np.round(r['cnn_minus_model']['ci'], 3)} | rank avg - model "
              f"{r['rank_average_minus_model']['delta_auc']:+.3f} {np.round(r['rank_average_minus_model']['ci'], 3)}")
    print("CNN splicing AUC", out["cnn_auc_splicing"])


if __name__ == "__main__":
    main()
