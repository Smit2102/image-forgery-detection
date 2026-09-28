"""Phase 7: the one-time evaluation on the frozen test set.

Nothing here is tuned on test images. Everything that was chosen during development is re-used as-is:
  * classifiers: refit on all dev images (train + val) with the hyper-parameters selected most often in
    Phase 5 (nested CV), then applied to test; the deployed calibrated models (models/forgery_<P>)
    give verdicts with their frozen uncertain band;
  * localisation: the deployed localisers (models/localizer_<P>) with their frozen post-processing;
    the baselines (mean of maps, best single module) use post-processing tuned on all of val.
Pre-registered main protocol: R (Phase 5 rule). Every run is appended to results/phase7/test_runs.log
with the code fingerprint, so any re-run after a bug fix is visible.
Usage:  python scripts/run_phase7.py            (the real, logged test run)
        python scripts/run_phase7.py --dry-run  (same code on train -> 400 val images; never touches test)
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from datetime import datetime

import cv2
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import fisher_exact
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)

from forgery.config import load_config, resolve
from forgery.data.manifest import read_manifest
from forgery.data.splits import frozen_test_hash, read_splits
from forgery.eval import classify as C
from forgery.eval.leakage import image_features
from forgery.features.common import gray, valid_blocks
from forgery.io import load_image, load_mask, protocol_mask
from forgery.localize.fusion import CELL, postprocess, scores, upsample
from forgery.pipeline import feature_code_sha1

OUT = resolve("results/phase7")
N_BOOT = 1000
TAG = "_test"            # suffix for test-set caches; "_dry" in a dry run


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, resolve(f"scripts/{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


p5, p6, ef, em, ls = (_load(n) for n in ("run_phase5", "run_phase6", "extract_features", "extract_maps",
                                          "run_leakage_study"))


# ----------------------------------------------------------------- bookkeeping ---

def check_frozen(cfg, splits):
    frozen = {t[1]: t[0] for t in (l.split() for l in resolve(cfg["paths"]["test_hash"]).read_text().splitlines())
              if len(t) >= 2}
    assert frozen_test_hash(splits) == frozen["test"], "the test split differs from the frozen one"


def _sha1(path) -> str:
    return hashlib.sha1(resolve(path).read_bytes()).hexdigest()


def audit(cfg) -> dict:
    """Fingerprints of everything that determines the test numbers: all project code, config, the frozen
    split, the models, and the Phase-5/6 inputs they were built from."""
    code = sorted([*resolve("src/forgery").rglob("*.py"), *resolve("scripts").glob("*.py"), resolve("config.yaml")])
    h = hashlib.sha1()
    for f in code:
        h.update(str(f.relative_to(resolve("."))).encode())
        h.update(f.read_bytes())
    inputs = [cfg["paths"]["test_hash"], "results/phase5/results.csv", "results/phase6/summary.json",
              *(f"data/features/features_{P}.csv" for P in ("A", "B", "R")),
              *(f"models/{m}_{P}.joblib" for m in ("forgery", "type", "localizer") for P in ("B", "R"))]
    return {"code_sha1": h.hexdigest(), "n_code_files": len(code), "feature_code_sha1": feature_code_sha1(),
            "inputs_sha1": {f: _sha1(f) for f in inputs}}


def preflight(cfg, chosen: pd.DataFrame) -> None:
    """Everything that can be checked without the test set, *before* a run number is used up."""
    from forgery.pipeline import resolved_params
    fp_leak = ls.cache_fingerprint(cfg)
    assert json.loads(resolve(cfg["paths"]["cache_dir"]).joinpath("leakage_features.json").read_text()) == fp_leak, \
        "dev shortcut-feature cache is stale"
    for P in ("A", "B", "R"):
        stored = json.loads(resolve(f"data/features/features_{P}.json").read_text())
        current = ef.fingerprint(cfg, P, [])
        assert (stored["code_sha1"], stored["features"], stored["protocol_cfg"]) == \
               (current["code_sha1"], current["features"], current["protocol_cfg"]), f"dev features {P} are stale"
        plan = [(s, "rf") for s in ("forensic_local", "forensic_all", "shortcuts", "forensic+shortcuts")]
        if P in ("B", "R"):
            plan += [(s, "svm") for s in ("forensic_all", "shortcuts", "forensic+shortcuts")]
            for m in ("forgery", "type", "localizer"):
                meta = joblib.load(resolve(f"models/{m}_{P}.joblib")).meta
                assert meta["feature_params"] == resolved_params(cfg, P), f"{m}_{P}: feature parameters differ"
                assert meta["feature_code_sha1"] == feature_code_sha1(), f"{m}_{P}: feature code differs"
            assert (em.map_dir(cfg, P) / "index.csv").exists(), f"val maps for {P} missing"
        for s_, m_ in plan:
            assert len(chosen[(chosen.protocol == P) & (chosen.feature_set == s_) & (chosen.model == m_)]) == 1, \
                f"no Phase-5 hyper-parameters for {P}/{s_}/{m_}"


def log_run(cfg, audit_rec: dict) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = OUT / "test_runs.log"
    starts = [json.loads(l) for l in runs.open()] if runs.exists() else []
    rec = {"event": "start", "time": datetime.now().isoformat(timespec="seconds"),
           "run_number": 1 + sum(1 for r in starts if r.get("event") == "start"), **audit_rec}
    with runs.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def log_done(run: dict, run_dir) -> None:
    rec = {"event": "done", "time": datetime.now().isoformat(timespec="seconds"), "run_number": run["run_number"],
           "summary_sha1": hashlib.sha1((run_dir / "summary.json").read_bytes()).hexdigest(),
           "results_sha1": hashlib.sha1((run_dir / "test_classification.csv").read_bytes()).hexdigest()}
    with resolve("results/phase7/test_runs.log").open("a") as fh:     # main log (OUT is the run folder here)
        fh.write(json.dumps(rec) + "\n")


def _clean(o):
    """JSON-safe: NaN / inf -> None, numpy scalars -> Python."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


# -------------------------------------------------------------------- features ---

def shortcut_features(cfg, paths: list[str]) -> pd.DataFrame:
    """Phase-2 global shortcut features for the test images (same code and settings as the dev cache)."""
    fp = ls.cache_fingerprint(cfg)
    cache, meta = (resolve(f"data/cache/leakage_features{TAG}.csv"), resolve(f"data/cache/leakage_features{TAG}.json"))
    if cache.exists() and meta.exists() and json.loads(meta.read_text()) == fp:
        feats = pd.read_csv(cache, dtype={"path": str})
        if set(feats.path) == set(paths):
            return feats
    rows = Parallel(n_jobs=cfg["n_jobs"], batch_size=8)(
        delayed(image_features)(str(resolve(p)), fp["b_qualities"], fp["naive_ela_quality"], fp["subsampling"],
                                fp["resample"]) for p in paths)
    feats = pd.DataFrame([{"path": p, **r} for p, rs in zip(paths, rows) for r in rs])
    feats.to_csv(cache, index=False)
    meta.write_text(json.dumps(fp, indent=2))
    return feats


def frame(cfg, P: str, rows: pd.DataFrame, feats: pd.DataFrame, sc: pd.DataFrame) -> pd.DataFrame:
    tag = {"A": "A", "B": f"B{cfg['protocols']['B']['jpeg_quality']}", "R": f"R{cfg['protocols']['R']['jpeg_quality']}"}[P]
    s = sc[sc.protocol == tag].drop(columns="protocol")
    s = s.rename(columns={c: f"shortcut.{c}" for c in s.columns if c != "path"})
    df = rows.merge(feats, on="path", validate="one_to_one").merge(s, on="path", validate="one_to_one")
    assert len(df) == len(rows)
    return df.reset_index(drop=True)


# -------------------------------------------------------------- classification ---

def fit_predict(dev, test, cols, model: str, params: dict, seed: int) -> np.ndarray:
    if model == "rf":
        est = C.forest(seed, max_features=params.get("max_features", "sqrt"),
                       min_samples_leaf=params.get("min_samples_leaf", 1))
    else:
        est = C.svm(len(cols)).set_params(**params)
    est.fit(dev[cols].to_numpy(float), dev.label)
    return C.scores_of(est, test[cols].to_numpy(float))


def metrics(y, s, thr, groups, seed) -> dict:
    pred = (s >= thr).astype(int)
    auc_ci = C.group_bootstrap(y, s, groups, roc_auc_score, N_BOOT, seed)
    ba_ci = C.group_bootstrap(y, pred, groups, balanced_accuracy_score, N_BOOT, seed)
    return {"auc": roc_auc_score(y, s), "auc_ci_low": auc_ci[0], "auc_ci_high": auc_ci[1],
            "balanced_accuracy": balanced_accuracy_score(y, pred), "ba_ci_low": ba_ci[0], "ba_ci_high": ba_ci[1],
            "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred)}


def classification(cfg, P, dev, test, chosen: pd.DataFrame, seed):
    sets = p5.feature_sets(dev)
    plan = [(s, "rf") for s in sets] + ([(s, "svm") for s in ("forensic_all", "shortcuts", "forensic+shortcuts")]
                                        if P in ("B", "R") else [])
    y, g = test.label.to_numpy(), test.split_group.to_numpy()
    rows, scores_ = [], pd.DataFrame({"path": test.path})
    for s, model in plan:
        prev = chosen[(chosen.protocol == P) & (chosen.feature_set == s) & (chosen.model == model)]
        params = p5.most_common(json.loads(prev.chosen_params.iloc[0]))
        sc = fit_predict(dev, test, sets[s], model, params, seed)
        scores_[f"{s}|{model}"] = sc
        rows.append({"protocol": P, "feature_set": s, "model": model, "params": json.dumps(params),
                     **metrics(y, sc, C.threshold_of(model), g, seed)})
    res = pd.DataFrame(rows)
    deltas = []
    for m in ("rf", "svm"):
        for new in ("forensic+shortcuts", "forensic_all", "forensic_local"):
            a, b = f"{new}|{m}", f"shortcuts|{m}"
            if a in scores_ and b in scores_:
                deltas.append({"protocol": P, "model": m, "comparison": f"{new} - shortcuts",
                               **C.paired_auc_delta(y, scores_[a].to_numpy(), scores_[b].to_numpy(), g, N_BOOT, seed)})
    return res, pd.DataFrame(deltas), scores_


def deployed(cfg, P, test, seed) -> dict:
    """The deployed calibrated model: verdicts with the frozen band, calibration on test."""
    model = joblib.load(resolve(f"models/forgery_{P}.joblib"))
    X = test[model.feature_names_].to_numpy(float)
    p = model.predict_proba(X)[:, 1]
    band = model.meta["uncertain_half_width"]
    y = test.label.to_numpy()
    lo, hi = 0.5 - band, 0.5 + band
    judged = (p >= hi) | (p <= lo)                     # same rule as pipeline.verdict_from_probability
    pred = (p >= 0.5).astype(int)
    _, ece, brier = C.reliability(y, p)
    out = {"band": band, "coverage": float(judged.mean()),
           "accuracy_judged": float((pred[judged] == y[judged]).mean()),
           "balanced_accuracy_judged": float(balanced_accuracy_score(y[judged], pred[judged])),
           "ece": ece, "brier": brier,
           # same forest as the forensic_all|rf row; differs from it only through isotonic ties
           "auc_of_calibrated_output": roc_auc_score(y, p),
           "verdicts": {"tampered": int((p >= hi).sum()), "authentic": int((p <= lo).sum()),
                        "uncertain": int((~judged).sum())}}
    return out, p


# ---------------------------------------------------------------- localisation ---

def test_maps(cfg, P, rows: pd.DataFrame):
    out_dir = em.map_dir(cfg, P).with_name(em.map_dir(cfg, P).name + TAG)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows[["path", "image_id", "split", "label"]].to_csv(out_dir / "index.csv", index=False)
    recs = rows[["path", "image_id", "label", "mask_valid", "mask_path"]].to_dict("records")
    Parallel(n_jobs=cfg["n_jobs"], batch_size=8)(delayed(em._one)(r, P, cfg, out_dir) for r in recs)
    return out_dir


def val_tuned_baselines(cfg, P, meta, names) -> dict:
    """Post-processing for the mean-of-maps and best-single baselines, tuned on all tampered val images."""
    d = em.map_dir(cfg, P)
    idx = pd.read_csv(d / "index.csv", dtype={"path": str, "image_id": str})
    vt = idx[idx.split == "val"].merge(meta[["path", "mask_valid", "mask_path"]], on="path")
    vt = vt[(vt.label == 1) & vt.mask_valid]
    stacks, _ = p6.load_stacks(d, vt)
    truths = {r.image_id: protocol_mask(load_mask(resolve(r.mask_path)), P, cfg) for r in vt.itertuples()}
    single = json.loads(resolve("results/phase6/summary.json").read_text())[P]["best_single"]
    out = {}
    for method, idx_m in (("mean", None), (single, names.index(single[7:]))):
        base = "mean" if method == "mean" else "best_single"
        probs = {i: upsample(p6.cell_prob(base, None, stacks[i][0], idx_m), truths[i].shape) for i in truths}
        out[method] = (base, idx_m, p6.tune(probs, truths, list(truths)))
    return out


def localisation(cfg, P, meta, test, p_cal, seed):
    loc = joblib.load(resolve(f"models/localizer_{P}.joblib"))
    band = joblib.load(resolve(f"models/forgery_{P}.joblib")).meta["uncertain_half_width"]
    rows = test[(test.label == 0) | test.mask_valid].copy()
    rows["p_cal"] = p_cal[rows.index]
    d = test_maps(cfg, P, rows)
    stacks, names = p6.load_stacks(d, rows)
    baselines = val_tuned_baselines(cfg, P, meta, names)
    per, fp = [], []
    for r in rows.itertuples():
        stack = stacks[r.image_id][0]
        if r.label == 1:
            truth = protocol_mask(load_mask(resolve(r.mask_path)), P, cfg)
            shape = truth.shape
            assert (shape[0] // CELL, shape[1] // CELL) == stack.shape[1:], f"mask / map geometry differ: {r.path}"
        else:
            truth, shape = None, (stack.shape[1] * CELL, stack.shape[2] * CELL)
        prob = upsample(loc.cell_prob(stack), shape)
        mask = postprocess(prob, loc.threshold, loc.open_k, loc.close_k, loc.min_area)
        shown = r.p_cal > 0.5 - band
        if truth is None:
            fp.append({"image_id": r.image_id, "area": float(mask.mean()), "shown": bool(shown)})
            continue
        rec = {"image_id": r.image_id, "forgery_type": r.forgery_type, "area_bucket": r.area_bucket,
               "split_group": r.split_group, "shown": bool(shown), **scores(mask, truth)}
        pp, gg = prob[::2, ::2].ravel(), truth[::2, ::2].ravel()
        rec["pixel_auc"] = roc_auc_score(gg, pp) if 0 < gg.mean() < 1 and np.ptp(pp) > 0 else 0.5
        rec["whole_f1"] = scores(np.ones_like(truth), truth)["f1"]
        for method, (base, idx_m, prm) in baselines.items():
            bp = upsample(p6.cell_prob(base, None, stack, idx_m), shape)
            rec[f"{method}_f1"] = scores(postprocess(bp, *prm), truth)["f1"]
        per.append(rec)
    per, fp = pd.DataFrame(per), pd.DataFrame(fp)
    g = per.split_group.to_numpy()
    summary = {"n_tampered": len(per), "n_authentic": len(fp),
               "f1": per.f1.mean(), "iou": per.iou.mean(), "mcc": per.mcc.mean(), "pixel_auc": per.pixel_auc.mean(),
               **{f"{k}_ci": p6.mean_ci(per[k].to_numpy(), g, seed) for k in ("f1", "iou", "mcc")},
               "whole_image_f1": per.whole_f1.mean(),
               **{f"{m.replace('single:', 'single_')}_f1": per[f"{m}_f1"].mean() for m in baselines},
               "authentic_any_region_ungated": float((fp.area > 0).mean()),
               "authentic_any_region_gated": float(((fp.area > 0) & fp.shown).mean()),
               "authentic_mean_area_ungated": float(fp.area.mean()),
               "tampered_shown": float(per.shown.mean()), "tampered_f1_gated": float((per.f1 * per.shown).mean()),
               "by_area": per.groupby("area_bucket")[["f1", "iou", "pixel_auc"]].mean().round(4).to_dict(orient="index"),
               "by_type": per.groupby("forgery_type")[["f1", "iou", "pixel_auc"]].mean().round(4).to_dict(orient="index")}
    return summary, per


# -------------------------------------------------------------------- failures ---

def failure_tags(cfg, P, test: pd.DataFrame) -> pd.DataFrame:
    def one(path):
        img = load_image(resolve(path), P, cfg)
        g = gray(img)
        grad = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
        return {"path": path, "texture": float(grad.mean()), "flat_share": float(1 - valid_blocks(img, 8).mean()),
                "saturated_share": float(((g <= 4) | (g >= 251)).mean())}
    return pd.DataFrame(Parallel(n_jobs=cfg["n_jobs"], batch_size=16)(delayed(one)(p) for p in test.path))


def failures(cfg, P, test, p_cal, band) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Outcomes with the deployed band (uncertain images are excluded from both error and correct sets)."""
    t = test.assign(p=p_cal).merge(failure_tags(cfg, P, test), on="path", validate="one_to_one").copy()
    assert list(t.path) == list(test.path)
    t["verdict"] = np.where(t.p >= 0.5 + band, "tampered", np.where(t.p <= 0.5 - band, "authentic", "uncertain"))
    t["outcome"] = np.select(
        [(t.label == 1) & (t.verdict == "tampered"), (t.label == 1) & (t.verdict == "authentic"),
         (t.label == 0) & (t.verdict == "authentic"), (t.label == 0) & (t.verdict == "tampered")],
        ["TP", "FN", "TN", "FP"], "uncertain")
    t["keypoints"] = t["cm_keypoint.global_n_keypoints"]
    t["texture_tercile"] = pd.qcut(t.texture, 3, labels=["low", "mid", "high"])
    t["keypoint_tercile"] = pd.qcut(t.keypoints.rank(method="first"), 3, labels=["low", "mid", "high"])
    t["flat_>30%"] = t.flat_share > 0.3
    t["saturated_>5%"] = t.saturated_share > 0.05
    rows = []
    for err, ok, label in (("FN", "TP", "missed tampered (FN) vs detected (TP)"),
                           ("FP", "TN", "false alarm (FP) vs correct authentic (TN)")):
        e, c = t[t.outcome == err], t[t.outcome == ok]
        tags = {"texture: low": lambda d: d.texture_tercile == "low", "texture: high": lambda d: d.texture_tercile == "high",
                "flat area > 30 %": lambda d: d["flat_>30%"], "saturated > 5 %": lambda d: d["saturated_>5%"],
                "keypoints: lowest tercile": lambda d: d.keypoint_tercile == "low"}
        if err == "FN":
            tags.update({"area < 1 %": lambda d: d.area_bucket == "<1%", "area > 5 %": lambda d: d.area_bucket == ">5%",
                         "splicing": lambda d: d.forgery_type == "splicing",
                         "JPEG source": lambda d: d.ext == "jpg", "TIFF source": lambda d: d.ext == "tif"})
        for tag, fn in tags.items():
            ke, kc = int(fn(e).sum()), int(fn(c).sum())
            pe, pc = (ke / len(e) if len(e) else np.nan), (kc / len(c) if len(c) else np.nan)
            p_fisher = fisher_exact([[ke, len(e) - ke], [kc, len(c) - kc]])[1] if len(e) and len(c) else np.nan
            rows.append({"comparison": label, "tag": tag, "share_in_errors": pe, "share_in_correct": pc,
                         "ratio": (pe / pc) if pc else (np.inf if pe else np.nan), "fisher_p": p_fisher,
                         "n_errors": len(e), "n_correct": len(c)})
    return t, pd.DataFrame(rows)


def fig_failures(cfg, P, t: pd.DataFrame, seed: int):
    """Fixed slots: row 1 most confident missed forgeries, row 2 random other missed forgeries,
    row 3 most confident false alarms, row 4 random other false alarms (empty slots stay blank)."""
    rng = np.random.default_rng(seed)
    fn, fp = t[t.outcome == "FN"].sort_values("p"), t[t.outcome == "FP"].sort_values("p", ascending=False)

    def head_and_random(d):
        head, rest = d.head(4), d.iloc[4:]
        return head, rest.iloc[rng.choice(len(rest), min(4, len(rest)), replace=False)] if len(rest) else rest

    slots = [*head_and_random(fn), *head_and_random(fp)]
    fig, axes = plt.subplots(4, 4, figsize=(12, 9))
    for row, d in enumerate(slots):
        for col in range(4):
            ax = axes[row, col]
            ax.axis("off")
            if col >= len(d):
                continue
            r = d.iloc[col]
            img = load_image(resolve(r.path), P, cfg)
            ax.imshow(img)
            if r.label == 1 and r.mask_valid:
                ax.contour(protocol_mask(load_mask(resolve(r.mask_path)), P, cfg), levels=[0.5], colors="red",
                           linewidths=0.8)
            what = f"{r.outcome}: {r.forgery_type}, {r.area_bucket}" if r.label == 1 else f"{r.outcome}: authentic"
            ax.set_title(f"{what}\nP(tampered) {r.p:.2f} · texture {r.texture:.0f}", fontsize=7)
    fig.suptitle(f"Test-set failures, protocol {P}. Row 1: most confident missed forgeries; row 2: random other "
                 "missed forgeries (red = true region); row 3: most confident false alarms; row 4: random other "
                 "false alarms", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / f"fig_failures_{P}.png", dpi=110)
    plt.close(fig)


def fig_roc(dev_oof: pd.Series, dev_y, test_s, test_y, P):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for s, y, lab, c in ((dev_oof, dev_y, "dev (out-of-fold)", "#9D9D9D"), (test_s, test_y, "test", "#4C78A8")):
        f, t, _ = roc_curve(y, s)
        ax.plot(f, t, color=c, lw=1.6, label=f"{lab}: AUC {roc_auc_score(y, s):.3f}")
    ax.plot([0, 1], [0, 1], ":", color="grey", lw=0.8)
    ax.set_xlabel("false-positive rate")
    ax.set_ylabel("true-positive rate")
    ax.set_title(f"Forensic features, random forest, protocol {P}", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / f"fig_roc_test_{P}.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------ main ---

def main():
    global OUT, TAG, N_BOOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    seed = cfg["seed"]
    t0 = time.time()
    meta = read_manifest(cfg)
    splits = read_splits(cfg)
    check_frozen(cfg, splits)
    df = meta.merge(splits, on="path", validate="one_to_one")
    if args.dry_run:        # train plays "dev", 400 val images play "test"; the test split is never read
        OUT, TAG, N_BOOT = resolve("data/cache/phase7_dry"), "_dry", 100
        OUT.mkdir(parents=True, exist_ok=True)
        va = df[df.split == "val"]
        dev = df[df.split == "train"].reset_index(drop=True)
        test = pd.concat([va[va.label == 0].sample(200, random_state=seed),
                          va[va.label == 1].sample(200, random_state=seed)]).reset_index(drop=True)
        run = {"run_number": 0, "time": datetime.now().isoformat(timespec="seconds"),
               "feature_code_sha1": feature_code_sha1(), "eval_code_sha1": "dry-run"}
    else:
        preflight(cfg, pd.read_csv(resolve("results/phase5/results.csv")))
        run = log_run(cfg, audit(cfg))
        OUT = OUT / f"run_{run['run_number']}"
        OUT.mkdir(parents=True, exist_ok=True)
        dev, test = df[df.split != "test"].reset_index(drop=True), df[df.split == "test"].reset_index(drop=True)
    assert not (dev.split == "test").any()
    print(f"test run #{run['run_number']} ({run['time']})", flush=True)
    test_paths = sorted(test.path)
    test = test.set_index("path").loc[test_paths].reset_index()

    sc_dev = pd.read_csv(resolve(cfg["paths"]["cache_dir"]) / "leakage_features.csv", dtype={"path": str})
    sc_dev = sc_dev[sc_dev.path.isin(dev.path)]
    sc_test = shortcut_features(cfg, test_paths)
    chosen = pd.read_csv(resolve("results/phase5/results.csv"))
    oof = {P: pd.read_csv(resolve(f"results/phase5/oof_{P}.csv"), dtype={"path": str}) for P in ("R", "B", "A")}

    results, deltas, summary = [], [], {"run": run, "protocols": {}}
    per_images = {}
    for P in ("R", "B", "A"):
        ft = ef.extract(cfg, P, test_paths, suffix=TAG)
        fd = pd.read_csv(resolve(f"data/features/features_{P}.csv"), dtype={"path": str})
        fd = fd[fd.path.isin(dev.path)]
        d, t = frame(cfg, P, dev, fd, sc_dev), frame(cfg, P, test, ft, sc_test)
        res, dl, sc = classification(cfg, P, d, t, chosen, seed)
        results.append(res)
        deltas.append(dl)
        entry = {}
        if P in ("B", "R"):
            dep, p_cal = deployed(cfg, P, t, seed)
            entry["deployed"] = dep
            entry["breakdown"] = p5.per_type_and_bucket(t, sc["forensic_all|rf"].to_numpy(), 0.5) \
                .round(4).to_dict(orient="records")
            mc = C.mcnemar(t.label.to_numpy(), (sc["forensic_all|rf"] >= 0.5).astype(int).to_numpy(),
                           (sc["forensic_all|svm"] >= 0.0).astype(int).to_numpy())
            entry["mcnemar_rf_vs_svm"] = mc
            loc_sum, per = localisation(cfg, P, meta, t, p_cal, seed)
            entry["localisation"] = loc_sum
            per_images[P] = per
            per.to_csv(OUT / f"localisation_per_image_{P}.csv", index=False)
            if P == "R":
                tf, enrich = failures(cfg, P, t, p_cal, dep["band"])
                tf[["path", "label", "forgery_type", "area_bucket", "ext", "p", "verdict", "outcome", "texture",
                    "flat_share", "saturated_share", "keypoints"]].to_csv(OUT / "test_outcomes_R.csv", index=False)
                enrich.to_csv(OUT / "failure_enrichment_R.csv", index=False)
                entry["outcomes"] = tf.outcome.value_counts().to_dict()
                fig_failures(cfg, P, tf, seed)
                o = oof[P].set_index("path")
                dd = d[d.path.isin(o.index)]
                fig_roc(o.loc[dd.path, "forensic_all|rf"].to_numpy(), dd.label, sc["forensic_all|rf"], t.label, P)
        sc.to_csv(OUT / f"test_scores_{P}.csv", index=False)
        summary["protocols"][P] = entry
        print(f"  [{P}] done ({time.time() - t0:.0f}s)", flush=True)

    results, deltas = pd.concat(results), pd.concat(deltas)
    results.to_csv(OUT / "test_classification.csv", index=False)
    deltas.to_csv(OUT / "test_added_value.csv", index=False)
    summary["minutes"] = round((time.time() - t0) / 60, 1)
    summary["test_counts"] = test.label.map({0: "authentic", 1: "tampered"}).value_counts().to_dict()
    summary["test_split_sha256"] = frozen_test_hash(splits)
    summary["n_bootstrap"] = N_BOOT
    (OUT / "summary.json").write_text(json.dumps(_clean(summary), indent=2))
    write_md(results, deltas, summary)
    if not args.dry_run:
        log_done(run, OUT)


def write_md(results, deltas, summary):
    fmt = lambda a, l, h: f"{a:.3f} [{l:.3f}, {h:.3f}]"
    tab = results.assign(auc_95ci=[fmt(a, l, h) for a, l, h in zip(results.auc, results.auc_ci_low, results.auc_ci_high)],
                         balanced_acc_95ci=[fmt(a, l, h) for a, l, h in
                                            zip(results.balanced_accuracy, results.ba_ci_low, results.ba_ci_high)])
    tab = tab[["protocol", "feature_set", "model", "auc_95ci", "balanced_acc_95ci", "precision", "recall", "f1"]]
    r = summary["run"]
    md = ["# Phase 7: test-set results (frozen models, evaluated once)", "",
          f"Run #{r['run_number']} at {r['time']}; code fingerprint {r.get('code_sha1', r.get('eval_code_sha1', ''))[:12]}; "
          f"test split {summary['test_split_sha256'][:12]}; test images {summary['test_counts']}. "
          f"95 % CIs from {summary['n_bootstrap']:,} group-bootstrap resamples of the test set.", "",
          "The `forensic_all | rf` row is the deployed model's own forest (refit on all dev = identical); the "
          "deployed model adds isotonic calibration, whose output AUC is reported separately.", "",
          "## Classification", "", tab.round(3).to_markdown(index=False), "",
          "## Added value over the shortcuts (paired group bootstrap)", "", deltas.round(4).to_markdown(index=False), ""]
    for P, e in summary["protocols"].items():
        if "deployed" not in e:
            continue
        md += [f"## Protocol {P}: deployed model, localisation", "",
               f"- Deployed model: {json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in e['deployed'].items()})}",
               f"- McNemar RF vs SVM: {e['mcnemar_rf_vs_svm']}",
               f"- Localisation: {json.dumps(_clean({k: (round(v, 4) if isinstance(v, float) else v) for k, v in e['localisation'].items() if k not in ('by_area', 'by_type')}))}",
               "", "### Localisation by tampered area", "", pd.DataFrame(e["localisation"]["by_area"]).T.round(3).to_markdown(),
               "", "### Localisation by forgery type", "", pd.DataFrame(e["localisation"]["by_type"]).T.round(3).to_markdown(),
               "", "### Breakdown (forensic RF)", "", pd.DataFrame(e["breakdown"]).to_markdown(index=False), ""]
        if "outcomes" in e:
            md += [f"- Outcomes with the deployed band: {e['outcomes']}", ""]
    (OUT / "test_results.md").write_text("\n".join(md))
    print("\n".join(md[:30]))


if __name__ == "__main__":
    main()
