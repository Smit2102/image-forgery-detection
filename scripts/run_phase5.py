"""Phase 5: fusion + classification, ablations, statistics, calibration, SHAP, final models.

Dev set only (train + val, grouped 5-fold CV from Phase 2). Test set untouched.
Feature sets (per protocol):
  forensic_local  - within-image inconsistency features of the 8 modules
  forensic_all    - local + global module features
  shortcuts       - the Phase-2 global shortcut features (metadata, size, naive ELA, compression history)
  forensic+shortcuts
Main-protocol rule (fixed before looking at the results): the main protocol is R if, under R, adding
the forensic features to the shortcuts raises AUC with a 95 % CI excluding 0; otherwise B if the
same holds under B; otherwise both are reported without a main protocol.
Usage:  python scripts/run_phase5.py [--protocols B R A] [--n-boot 1000]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold

from forgery.config import load_config, resolve
from forgery.data.manifest import read_manifest
from forgery.data.splits import read_splits
from forgery.eval import classify as C
from forgery.pipeline import resolved_params

OUT = resolve("results/phase5")
MODELS = resolve("models")
SET_LABEL = {"forensic_local": "Forensic (local only)", "forensic_all": "Forensic (local + global)",
             "shortcuts": "Shortcuts (Phase 2)", "forensic+shortcuts": "Forensic + shortcuts"}


def load(cfg, protocol: str, dev: pd.DataFrame) -> pd.DataFrame:
    feats = pd.read_csv(resolve(f"data/features/features_{protocol}.csv"), dtype={"path": str})
    sc = pd.read_csv(resolve(cfg["paths"]["cache_dir"]) / "leakage_features.csv", dtype={"path": str})
    tag = {"A": "A", "B": f"B{cfg['protocols']['B']['jpeg_quality']}",
           "R": f"R{cfg['protocols']['R']['jpeg_quality']}"}[protocol]
    sc = sc[sc.protocol == tag].drop(columns="protocol")
    sc = sc.rename(columns={c: f"shortcut.{c}" for c in sc.columns if c != "path"})
    df = dev.merge(feats, on="path", validate="one_to_one").merge(sc, on="path", validate="one_to_one")
    assert len(df) == len(dev), "feature files do not cover the dev set"
    return df.reset_index(drop=True)


def feature_sets(df) -> dict[str, list[str]]:
    fa = C.module_columns(df.columns)
    return {"forensic_local": C.module_columns(df.columns, local_only=True), "forensic_all": fa,
            "shortcuts": C.shortcut_columns(), "forensic+shortcuts": fa + C.shortcut_columns()}


def pooled(y, s, thr, groups, n_boot, seed) -> dict:
    pred = (s >= thr).astype(int)
    auc_ci = C.group_bootstrap(y, s, groups, roc_auc_score, n_boot, seed)
    ba_ci = C.group_bootstrap(y, pred, groups, balanced_accuracy_score, n_boot, seed)
    return {"pooled_auc": roc_auc_score(y, s), "auc_ci_low": auc_ci[0], "auc_ci_high": auc_ci[1],
            "pooled_balanced_accuracy": balanced_accuracy_score(y, pred), "ba_ci_low": ba_ci[0], "ba_ci_high": ba_ci[1]}


def most_common(chosen: list[dict]) -> dict:
    """The hyper-parameter setting selected in most outer folds."""
    return json.loads(Counter(json.dumps(c, sort_keys=True) for c in chosen).most_common(1)[0][0])


def rf_factory(params: dict, seed: int):
    return lambda: C.forest(seed, max_features=params.get("max_features", "sqrt"),
                            min_samples_leaf=params.get("min_samples_leaf", 1))


# ----------------------------------------------------------------- analyses ---

def _checkpoint(protocol: str, s: str, model: str) -> "Path":
    """Cache file for one nested-CV run, keyed on the feature fingerprint, so an interrupted Phase 5
    resumes after the last finished run instead of starting again."""
    fp = hashlib.sha1(resolve(f"data/features/features_{protocol}.json").read_bytes()).hexdigest()[:12]
    d = resolve("data/cache/phase5")
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{protocol}_{s.replace('+', 'plus')}_{model}_{fp}.joblib"


def main_runs(df, protocol, cfg, n_boot, with_svm) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    seed, sets = cfg["seed"], feature_sets(df)
    y, g = df.label.to_numpy(), df.split_group.to_numpy()
    plan = [(s, "rf") for s in sets]
    if with_svm:
        plan += [(s, "svm") for s in ("forensic_all", "shortcuts", "forensic+shortcuts")]
    rows, oof, chosen = [], pd.DataFrame({"path": df.path}), {}
    for s, model in plan:
        t0 = time.time()
        ck = _checkpoint(protocol, s, model)
        if ck.exists():
            r = joblib.load(ck)
        else:
            r = C.nested_cv(df, sets[s], model, seed)
            joblib.dump(r, ck)
        oof[f"{s}|{model}"] = r["oof"]
        chosen[f"{s}|{model}"] = r["chosen"]
        rows.append({"protocol": protocol, "feature_set": s, "model": model, "n_features": len(sets[s]),
                     **C.summarize_folds(r["folds"]), **pooled(y, r["oof"], r["threshold"], g, n_boot, seed),
                     "chosen_params": json.dumps(r["chosen"])})
        print(f"  [{protocol}] {s:20s} {model}: AUC {rows[-1]['roc_auc_mean']:.3f} ± {rows[-1]['roc_auc_std']:.3f}"
              f"  ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows), oof, chosen


def added_value(df, oof, protocol, models, n_boot, seed) -> pd.DataFrame:
    y, g = df.label.to_numpy(), df.split_group.to_numpy()
    rows = []
    for m in models:
        for new, old in (("forensic+shortcuts", "shortcuts"), ("forensic_all", "shortcuts"),
                         ("forensic_local", "shortcuts")):
            if f"{new}|{m}" in oof and f"{old}|{m}" in oof:
                rows.append({"protocol": protocol, "model": m, "comparison": f"{new} - {old}",
                             **C.paired_auc_delta(y, oof[f"{new}|{m}"].to_numpy(), oof[f"{old}|{m}"].to_numpy(),
                                                  g, n_boot, seed)})
    return pd.DataFrame(rows)


def ablations(df, protocol, params, seed) -> pd.DataFrame:
    cols = C.module_columns(df.columns)
    make = rf_factory(params, seed)
    full = roc_auc_score(df.label, C.fixed_cv(df, cols, make)["oof"])
    rows = [{"protocol": protocol, "variant": "all modules", "module": "-", "auc": full}]
    for m in C.MODULES:
        alone = C.fixed_cv(df, C.module_columns(df.columns, [m]), make)["oof"]
        local = C.module_columns(df.columns, [m], local_only=True)
        alone_local = C.fixed_cv(df, local, make)["oof"] if local else np.full(len(df), 0.5)
        loo = C.fixed_cv(df, C.module_columns(df.columns, [x for x in C.MODULES if x != m]), make)["oof"]
        rows += [{"protocol": protocol, "variant": "module alone", "module": m, "auc": roc_auc_score(df.label, alone)},
                 {"protocol": protocol, "variant": "module alone (local only)", "module": m,
                  "auc": roc_auc_score(df.label, alone_local)},
                 {"protocol": protocol, "variant": "leave one out", "module": m, "auc": roc_auc_score(df.label, loo),
                  "drop_vs_all": full - roc_auc_score(df.label, loo)}]
    return pd.DataFrame(rows)


def copy_move_study(df, protocol, params, seed, sweep_final) -> pd.DataFrame:
    d = df[(df.label == 0) | (df.forgery_type == "copy-move")].reset_index(drop=True)
    make = rf_factory(params, seed)
    rows = []
    for name, mods in (("keypoint", ["cm_keypoint"]), ("block", ["cm_block"]), ("both", ["cm_keypoint", "cm_block"]),
                       ("all modules", C.MODULES)):
        s = C.fixed_cv(d, C.module_columns(d.columns, mods), make)["oof"]
        rows.append({"protocol": protocol, "features": name, "auc_copymove_vs_authentic": roc_auc_score(d.label, s)})
    for m in ("cm_keypoint", "cm_block"):
        f = sweep_final[(sweep_final.module == m) & (sweep_final.protocol == protocol)]
        if len(f):
            rows.append({"protocol": protocol, "features": f"{m} localisation (Phase 4 val sample)",
                         "pixel_auc_copymove": float(f.pixel_auc.iloc[0]),
                         "share_pixel_auc_gt_0.6": float(f.pixel_auc_hit_rate.iloc[0])})
    return pd.DataFrame(rows)


def per_type_and_bucket(df, s, thr) -> pd.DataFrame:
    rows = []
    au = df.label == 0
    for t in ("copy-move", "splicing"):
        sub = au | (df.forgery_type == t)
        rows.append({"group": f"{t} vs authentic", "auc": roc_auc_score(df.label[sub], s[sub]),
                     "n_tampered": int((df.forgery_type == t).sum())})
    for b in ("<1%", "1-5%", ">5%"):
        m = df.area_bucket == b
        rows.append({"group": f"tampered area {b}", "detection_rate": float((s[m] >= thr).mean()),
                     "auc_vs_authentic": roc_auc_score(df.label[au | m], s[au | m]), "n_tampered": int(m.sum())})
    for e in ("jpg", "tif"):
        m = (df.label == 1) & (df.ext == e)
        rows.append({"group": f"tampered {e.upper()} source", "detection_rate": float((s[m] >= thr).mean()),
                     "auc_vs_authentic": roc_auc_score(df.label[au | m], s[au | m]), "n_tampered": int(m.sum())})
    rows.append({"group": "authentic false-positive rate", "detection_rate": float((s[au] >= thr).mean()),
                 "n_tampered": 0})
    return pd.DataFrame(rows)


def calibration(df, cols, params, oof_raw, seed) -> dict:
    y = df.label.to_numpy()
    cal = C.calibrated_cv(df, cols, rf_factory(params, seed), seed)
    t_raw, ece_raw, brier_raw = C.reliability(y, oof_raw)
    t_cal, ece_cal, brier_cal = C.reliability(y, cal)
    cov = C.coverage_curve(y, cal)
    return {"oof_calibrated": cal, "raw": (t_raw, ece_raw, brier_raw), "cal": (t_cal, ece_cal, brier_cal),
            "coverage": cov}


def choose_band(cov: pd.DataFrame, target: float = 0.85, min_coverage: float = 0.5) -> float:
    """Smallest half-width reaching `target` accuracy while still judging >= min_coverage of images;
    otherwise the width that maximises accuracy subject to the coverage floor."""
    ok = cov[(cov.coverage >= min_coverage)]
    hit = ok[ok.accuracy >= target]
    return float(hit.half_width.min() if len(hit) else ok.loc[ok.accuracy.idxmax(), "half_width"])


def shap_analysis(df, cols, params, seed, protocol):
    rf = rf_factory(params, seed)().fit(df[cols].to_numpy(float), df.label)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), min(600, len(df)), replace=False)
    X = df.iloc[idx][cols]
    sv = shap.TreeExplainer(rf).shap_values(X.to_numpy(float))
    sv = sv[1] if isinstance(sv, list) else (sv[..., 1] if sv.ndim == 3 else sv)
    imp = pd.Series(np.abs(sv).mean(0), index=cols).sort_values(ascending=False)
    by_module = imp.groupby(lambda c: c.split(".", 1)[0]).sum().sort_values(ascending=False)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    top = imp.head(20)[::-1]
    ax[0].barh(top.index, top.values, color="#4C78A8")
    ax[0].set_title("Top 20 features: mean |SHAP|", fontsize=10)
    ax[0].tick_params(axis="y", labelsize=7)
    ax[1].bar(by_module.index, by_module.values, color="#F58518")
    ax[1].set_title("Summed mean |SHAP| per module", fontsize=10)
    ax[1].tick_params(axis="x", rotation=40, labelsize=8)
    fig.suptitle(f"What the random forest relies on (protocol {protocol}, forensic features)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / f"fig_shap_{protocol}.png", dpi=140)
    plt.close(fig)
    plt.figure()
    shap.summary_plot(sv, X, max_display=15, show=False, plot_size=(8, 5))
    plt.title(f"SHAP summary (protocol {protocol})", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT / f"fig_shap_beeswarm_{protocol}.png", dpi=130)
    plt.close("all")
    return imp, by_module


def final_models(df, cols, params, seed, protocol, band, cfg):
    X, y, g = df[cols].to_numpy(float), df.label.to_numpy(), df.split_group.to_numpy()
    splits = list(GroupKFold(5).split(X, y, g))
    # ensemble=False: one forest on all dev data, isotonic map fitted on grouped out-of-fold predictions
    cal = CalibratedClassifierCV(rf_factory(params, seed)(), method="isotonic", cv=splits, ensemble=False).fit(X, y)
    MODELS.mkdir(exist_ok=True)
    fingerprint = json.loads(resolve(f"data/features/features_{protocol}.json").read_text())
    common = {"protocol": protocol, "feature_params": resolved_params(cfg, protocol),
              "feature_code_sha1": fingerprint["code_sha1"],
              "trained_on": "all dev images (train+val): predictions on dev images are in-sample"}
    joblib.dump(C.FittedModel(cal, cols, {**common, "rf_params": params, "uncertain_half_width": band,
                                          "n": len(df)}),
                MODELS / f"forgery_{protocol}.joblib", compress=3)
    tp = df[df.label == 1].reset_index(drop=True)
    yt = (tp.forgery_type == "copy-move").astype(int).to_numpy()
    oof = C.fixed_cv(tp.assign(label=yt), cols, rf_factory(params, seed))["oof"]
    type_model = rf_factory(params, seed)().fit(tp[cols].to_numpy(float), yt)
    joblib.dump(C.FittedModel(type_model, cols, {**common, "positive_class": "copy-move"}),
                MODELS / f"type_{protocol}.joblib", compress=3)
    return roc_auc_score(yt, oof)


# ------------------------------------------------------------------- figures ---

def fig_roc(df, oof, protocol):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for s, c in (("shortcuts", "#9D9D9D"), ("forensic_local", "#72B7B2"), ("forensic_all", "#4C78A8"),
                 ("forensic+shortcuts", "#E45756")):
        k = f"{s}|rf"
        if k in oof:
            fpr, tpr, _ = roc_curve(df.label, oof[k])
            ax.plot(fpr, tpr, color=c, lw=1.5, label=f"{SET_LABEL[s]} ({roc_auc_score(df.label, oof[k]):.3f})")
    ax.plot([0, 1], [0, 1], ":", color="grey", lw=0.8)
    ax.set_xlabel("false-positive rate")
    ax.set_ylabel("true-positive rate")
    ax.set_title(f"Out-of-fold ROC, random forest, protocol {protocol}", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / f"fig_roc_{protocol}.png", dpi=150)
    plt.close(fig)


def fig_ablation(abl: pd.DataFrame):
    prots = list(abl.protocol.unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    w = 0.8 / len(prots)
    for k, p in enumerate(prots):
        a = abl[(abl.protocol == p) & (abl.variant == "module alone")].set_index("module").reindex(C.MODULES)
        l = abl[(abl.protocol == p) & (abl.variant == "leave one out")].set_index("module").reindex(C.MODULES)
        x = np.arange(len(C.MODULES)) + (k - (len(prots) - 1) / 2) * w
        axes[0].bar(x, a.auc, w, label=f"protocol {p}")
        axes[1].bar(x, l.drop_vs_all, w, label=f"protocol {p}")
    axes[0].axhline(0.5, color="grey", ls=":", lw=0.8)
    axes[0].set_ylim(0.45, 1.0)
    axes[0].set_title("AUC of each module alone", fontsize=10)
    axes[1].axhline(0, color="grey", lw=0.8)
    axes[1].set_title("AUC lost when the module is left out", fontsize=10)
    for ax in axes:
        ax.set_xticks(np.arange(len(C.MODULES)), C.MODULES, rotation=35, fontsize=8)
        ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_ablation.png", dpi=150)
    plt.close(fig)


def fig_calibration(cal: dict, protocol: str, band: float):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
    for key, lab, c in (("raw", "random forest (raw)", "#9D9D9D"), ("cal", "isotonic-calibrated", "#4C78A8")):
        t, ece, brier = cal[key]
        ax[0].plot(t.mean_pred, t.frac_pos, "o-", color=c, label=f"{lab}: ECE {ece:.3f}, Brier {brier:.3f}")
    ax[0].plot([0, 1], [0, 1], ":", color="grey")
    ax[0].set_xlabel("predicted P(tampered)")
    ax[0].set_ylabel("observed share tampered")
    ax[0].set_title(f"Reliability (out-of-fold, protocol {protocol})", fontsize=10)
    ax[0].legend(fontsize=7, frameon=False)
    cov = cal["coverage"]
    ax[1].plot(cov.coverage, cov.accuracy, "o-", color="#E45756", ms=3)
    sel = cov[cov.half_width == band].iloc[0]
    ax[1].scatter([sel.coverage], [sel.accuracy], s=80, facecolors="none", edgecolors="k",
                  label=f"chosen band 0.5 ± {band:.2f}: {100 * sel.coverage:.0f}% judged, "
                        f"accuracy {sel.accuracy:.3f}")
    ax[1].set_xlabel("share of images judged (outside the uncertain band)")
    ax[1].set_ylabel("accuracy on judged images")
    ax[1].invert_xaxis()
    ax[1].set_title("Accuracy vs coverage", fontsize=10)
    ax[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / f"fig_calibration_{protocol}.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------- main ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocols", nargs="+", default=["B", "R", "A"])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--reuse-cv", action="store_true",
                    help="reuse the nested-CV outputs (results/oof/added_value/ablations/copy-move CSVs) of an "
                         "earlier run and redo only calibration, bands, SHAP, breakdowns and final models")
    args = ap.parse_args()
    cfg = load_config()
    seed = cfg["seed"]
    OUT.mkdir(parents=True, exist_ok=True)
    dev = read_manifest(cfg).merge(read_splits(cfg), on="path")
    dev = dev[dev.split != "test"].reset_index(drop=True)
    sweep_final = pd.read_csv(resolve("results/phase4/sweep_final.csv"))

    results, deltas, abls, cms, oofs, chosen_all, data = [], [], [], [], {}, {}, {}
    for p in args.protocols:
        df = load(cfg, p, dev)
        data[p] = df
        if args.reuse_cv:            # nested-CV outputs of an earlier run of this script (same code and data)
            saved = json.loads((OUT / "features_fingerprints.json").read_text())
            current = json.loads(resolve(f"data/features/features_{p}.json").read_text())
            assert saved.get(p) == current, f"saved CV outputs were made from different {p} features"
            prev = pd.read_csv(OUT / "results.csv")
            oof = pd.read_csv(OUT / f"oof_{p}.csv", dtype={"path": str})
            assert list(oof.path) == list(df.path), "saved out-of-fold file does not match the dev set"
            prev = prev[prev.protocol == p]
            results.append(prev)
            oofs[p] = oof
            chosen_all[p] = {f"{r.feature_set}|{r.model}": json.loads(r.chosen_params) for r in prev.itertuples()}
            continue
        res, oof, chosen = main_runs(df, p, cfg, args.n_boot, with_svm=p in ("B", "R"))
        results.append(res)
        oofs[p], chosen_all[p] = oof, chosen
        oof.to_csv(OUT / f"oof_{p}.csv", index=False)
        deltas.append(added_value(df, oof, p, ["rf", "svm"], args.n_boot, seed))
        fig_roc(df, oof, p)
        if p in ("B", "R"):
            params = most_common(chosen["forensic_all|rf"])
            abls.append(ablations(df, p, params, seed))
            cms.append(copy_move_study(df, p, params, seed, sweep_final))
    results = pd.concat(results)
    if args.reuse_cv:
        deltas = pd.read_csv(OUT / "added_value.csv")
        abl, cm = pd.read_csv(OUT / "ablations.csv"), pd.read_csv(OUT / "copy_move_study.csv")
    else:
        deltas, abl, cm = pd.concat(deltas), pd.concat(abls), pd.concat(cms)
        results.to_csv(OUT / "results.csv", index=False)
        (OUT / "features_fingerprints.json").write_text(json.dumps(
            {p: json.loads(resolve(f"data/features/features_{p}.json").read_text()) for p in data}, indent=2))
        deltas.to_csv(OUT / "added_value.csv", index=False)
        abl.to_csv(OUT / "ablations.csv", index=False)
        cm.to_csv(OUT / "copy_move_study.csv", index=False)
    fig_ablation(abl)

    # main protocol by the pre-registered rule
    def genuine(p):
        d = deltas[(deltas.protocol == p) & (deltas.model == "rf") & (deltas.comparison == "forensic+shortcuts - shortcuts")]
        return len(d) and d.ci_low.iloc[0] > 0
    main_p = "R" if genuine("R") else ("B" if genuine("B") else None)
    report_p = main_p or "B"          # protocol whose statistics are tabulated when no main protocol exists

    # calibration + uncertain band, per deployable protocol
    bands, cals = {}, {}
    for p in ("B", "R"):
        if p not in data:
            continue
        d = data[p]
        prm = most_common(chosen_all[p]["forensic_all|rf"])
        cals[p] = calibration(d, C.module_columns(d.columns), prm, oofs[p]["forensic_all|rf"].to_numpy(), seed)
        bands[p] = choose_band(cals[p]["coverage"])
        fig_calibration(cals[p], p, bands[p])
        cals[p]["coverage"].to_csv(OUT / f"coverage_{p}.csv", index=False)

    # statistics on the main (or report) protocol
    df, oof = data[report_p], oofs[report_p]
    cols = C.module_columns(df.columns)
    params = most_common(chosen_all[report_p]["forensic_all|rf"])
    mc = C.mcnemar(df.label.to_numpy(), (oof["forensic_all|rf"] >= 0.5).astype(int).to_numpy(),
                   (oof["forensic_all|svm"] >= 0.0).astype(int).to_numpy())
    breakdown = per_type_and_bucket(df, oof["forensic_all|rf"].to_numpy(), 0.5)
    breakdown.to_csv(OUT / f"breakdown_{report_p}.csv", index=False)
    cal, band = cals[report_p], bands[report_p]
    at_band = {p: cals[p]["coverage"].set_index("half_width").loc[bands[p]].round(4).to_dict() for p in cals}
    imp, by_module = shap_analysis(df, cols, params, seed, report_p)
    imp.rename("mean_abs_shap").to_csv(OUT / f"shap_importance_{report_p}.csv")
    type_auc = {p: final_models(data[p], C.module_columns(data[p].columns),
                                most_common(chosen_all[p]["forensic_all|rf"]), seed, p, bands[p], cfg)
                for p in ("B", "R") if p in data}

    (MODELS / "MAIN_PROTOCOL").write_text((main_p or "none") + "\n")
    summary = {"main_protocol": main_p, "report_protocol": report_p, "mcnemar_rf_vs_svm": mc,
               "uncertain_half_width": bands, "at_band": at_band,
               "ece_raw": cal["raw"][1], "brier_raw": cal["raw"][2],
               "ece_calibrated": cal["cal"][1], "brier_calibrated": cal["cal"][2],
               "type_classifier_auc_copymove_vs_splicing": type_auc,
               "shap_by_module": by_module.round(4).to_dict(), "rf_params": params,
               "rf_params_per_protocol": {p: most_common(chosen_all[p]["forensic_all|rf"]) for p in chosen_all},
               "reused_cv": bool(args.reuse_cv)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    fmt = lambda m, s: f"{m:.3f} ± {s:.3f}"
    tab = results.assign(
        feature_set=results.feature_set.map(SET_LABEL),
        auc=[fmt(m, s) for m, s in zip(results.roc_auc_mean, results.roc_auc_std)],
        balanced_acc=[fmt(m, s) for m, s in zip(results.balanced_accuracy_mean, results.balanced_accuracy_std)],
        pooled_auc_95ci=[f"{a:.3f} [{l:.3f}, {h:.3f}]" for a, l, h in
                         zip(results.pooled_auc, results.auc_ci_low, results.auc_ci_high)],
        f1=[fmt(m, s) for m, s in zip(results.f1_mean, results.f1_std)],
    )[["protocol", "feature_set", "model", "n_features", "auc", "pooled_auc_95ci", "balanced_acc", "f1"]]
    md = ["# Phase 5: classification", "",
          f"Dev set (train + val, {len(dev):,} images), grouped 5-fold CV with nested grouped 3-fold model "
          "selection; 95 % CIs from 1,000 group-bootstrap resamples. Test set untouched.", "",
          f"**Main protocol (pre-registered rule): {main_p or 'none - both reported'}**", "",
          "## Classification", "", tab.to_markdown(index=False), "",
          "## Added value over the shortcut features (paired group bootstrap)", "",
          deltas.round(4).to_markdown(index=False), "",
          f"## Statistics on protocol {report_p}", "",
          f"- McNemar, RF vs SVM (forensic features): {mc}",
          f"- Calibration: ECE {cal['raw'][1]:.3f} -> {cal['cal'][1]:.3f} after isotonic calibration; "
          f"Brier {cal['raw'][2]:.3f} -> {cal['cal'][2]:.3f}",
          "- Uncertain band per protocol (chosen on out-of-fold calibrated predictions; accuracy and balanced "
          f"accuracy on the judged images): {at_band}", "",
          "### Breakdown (forensic features, RF, out-of-fold)", "", breakdown.round(3).to_markdown(index=False), "",
          "## Ablations (RF, fixed parameters)", "",
          abl.pivot_table(index="module", columns=["protocol", "variant"], values="auc").round(3).to_markdown(), "",
          "## Copy-move: keypoint vs block matching", "", cm.round(3).to_markdown(index=False), "",
          "## SHAP: summed mean |SHAP| per module", "", by_module.round(4).to_markdown(), "",
          f"## Type classifier (copy-move vs splicing, tampered only): AUC {type_auc}", ""]
    (OUT / "classification.md").write_text("\n".join(md))
    print("\n".join(md[:14]))


if __name__ == "__main__":
    main()
