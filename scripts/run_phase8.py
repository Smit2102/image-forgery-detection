"""Phase 8: a deep-learning baseline (ELA + ResNet-18) against the classical pipeline.

Stages, run in this order (each is checkpointed and resumes after an interruption):
  select  train on the train split; choose the epoch and the tile pooling (mean / max) by validation AUC.
          Protocols R, B, A with seed 42; protocol R also with seeds 43 and 44 (seed spread, validation only).
  refit   retrain on all development images (train + val) for the selected number of epochs with the same
          recipe -> models/cnn_<P>.pt; then write the pre-registered test comparisons
          (results/phase8/preregistration.json, with the model hashes).
  test    ONE evaluation on the test split, logged in results/phase7/test_runs.log: AUC, paired difference to the
          classical forest of run 1, a rank-average combination, JPEG-only subset, three degradations, MICC-F220.
Validation is a single held-out split (not 5-fold CV as for the classical models): one CNN run costs ~15 min.

Usage:  python scripts/run_phase8.py --stage select|refit|test
        python scripts/run_phase8.py --smoke        (all three stages on small subsets; "test" = 150 val images,
                                                     classical scores = Phase-5 out-of-fold; never reads test)
Prerequisite: python scripts/make_ela.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from forgery import cnn  # noqa: E402
from forgery.config import load_config, resolve  # noqa: E402
from forgery.data.manifest import read_manifest  # noqa: E402
from forgery.data.splits import read_splits  # noqa: E402
from forgery.eval.classify import group_bootstrap, paired_auc_delta  # noqa: E402
from forgery.io import apply_protocol, load_rgb  # noqa: E402

PROTOCOLS = ("R", "B", "A")
SELECT_RUNS = [("R", 42), ("B", 42), ("A", 42), ("R", 43), ("R", 44)]
PRIMARY_SEED = 42
MAX_EPOCHS, PATIENCE, BATCH, LR, WD = 15, 4, 64, 3e-4, 1e-4
CLASSICAL = "forensic_all|rf"          # the headline classical model of Phase 7 (refit on all dev images)
CLASSICAL_PLUS = "forensic+shortcuts|rf"   # the same with the shortcut features (naive ELA, size, ...) allowed
DEGRADATIONS = [("jpeg", 75, "JPEG Q75"), ("resize", 0.5, "resize x0.5"), ("social", None, "social (x0.8 + JPEG Q70)")]
ELA_DIR = resolve("data/ela")


class Paths:
    def __init__(self, smoke: bool):
        tag = "phase8_smoke" if smoke else "phase8"
        self.out = resolve(f"results/{tag}")
        self.cache = resolve(f"data/cache/{tag}")
        self.models = self.cache / "models" if smoke else resolve("models")
        for p in (self.out, self.cache, self.models):
            p.mkdir(parents=True, exist_ok=True)

    def model(self, P: str) -> Path:
        return self.models / f"cnn_{P}.pt"


def sha1(p: Path) -> str:
    return hashlib.sha1(Path(p).read_bytes()).hexdigest()


_P7B = None


def _phase7b():
    """scripts/run_phase7b.py as a module (its degradations and cached classical results are reused)."""
    global _P7B
    if _P7B is not None:
        return _P7B
    spec = importlib.util.spec_from_file_location("run_phase7b", resolve("scripts/run_phase7b.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_phase7b"] = mod
    spec.loader.exec_module(mod)
    _P7B = mod
    return mod


def load_elas(df: pd.DataFrame, P: str) -> list[np.ndarray]:
    return Parallel(n_jobs=8, prefer="threads")(delayed(cnn.load_ela)(ELA_DIR / P / f"{i}.png") for i in df.image_id)


def _ela_job(path: str, P: str, cfg: dict, family: str | None, param, seed: int) -> np.ndarray:
    img = load_rgb(resolve(path))
    if family is not None:
        img, _ = _phase7b().degrade(img, None, family, param, seed)
    return cnn.ela_image(apply_protocol(img, P, cfg))


def compute_elas(paths, P: str, cfg: dict, family=None, param=None) -> list[np.ndarray]:
    seeds = [int(hashlib.md5(p.encode()).hexdigest()[:8], 16) for p in paths]   # same noise seed as Phase 7b
    return Parallel(n_jobs=cfg["n_jobs"], batch_size=8)(
        delayed(_ela_job)(p, P, cfg, family, param, s) for p, s in zip(paths, seeds))


# ------------------------------------------------------------------ training ---

def train_run(name: str, P: str, seed: int, train: pd.DataFrame, val: pd.DataFrame | None, n_epochs: int,
              paths: Paths, dev) -> pd.DataFrame:
    """Train one network; checkpoint every epoch (resumes); with val: early stopping, best.pt by val AUC."""
    run_dir = paths.cache / name
    run_dir.mkdir(parents=True, exist_ok=True)
    hist_file, last, best = run_dir / "history.csv", run_dir / "last.pt", run_dir / "best.pt"
    best_at = lambda e: run_dir / f"best_e{e}.pt"          # noqa: E731  (epoch-stamped: safe to resume)
    if (run_dir / "DONE").exists():
        return pd.read_csv(hist_file)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = cnn.build_model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    steps = math.ceil(len(train) / BATCH)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS * steps)   # same schedule for refits
    history: list[dict] = []
    if last.exists():
        ck = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        rng.bit_generator.state = ck["rng"]
        history = ck["history"]
        print(f"  [{name}] resuming after epoch {len(history)}", flush=True)
    t0 = time.time()
    x_train = load_elas(train, P)
    y_train = train.label.to_numpy()
    x_val = load_elas(val, P) if val is not None else None
    print(f"  [{name}] loaded {len(x_train)} train / {0 if val is None else len(val)} val ELA images "
          f"({time.time() - t0:.0f}s)", flush=True)
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    def stopped() -> bool:                                   # early stopping, also checked after a resume
        if x_val is None or not history:
            return False
        return len(history) - (int(np.argmax([h["val_auc"] for h in history])) + 1) >= PATIENCE

    while len(history) < n_epochs and not stopped():
        t0 = time.time()
        rec = {"epoch": len(history) + 1,
               "loss": cnn.train_epoch(model, opt, sched, x_train, y_train, rng, dev, BATCH, pos_weight)}
        if x_val is not None:
            s = cnn.predict(model, x_val, dev)
            rec.update({f"val_auc_{a}": roc_auc_score(val.label, s[a]) for a in cnn.AGGREGATORS})
            rec["val_auc"] = max(rec[f"val_auc_{a}"] for a in cnn.AGGREGATORS)
            if rec["val_auc"] > max([h["val_auc"] for h in history], default=-1):
                _atomic_save({"model": model.state_dict()}, best_at(rec["epoch"]))
        rec["seconds"] = time.time() - t0
        history.append(rec)
        _atomic_save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                      "rng": rng.bit_generator.state, "history": history}, last)
        pd.DataFrame(history).to_csv(hist_file, index=False)
        print(f"  [{name}] epoch {rec['epoch']:2d} loss {rec['loss']:.4f} "
              + (f"val AUC mean {rec['val_auc_mean']:.3f} max {rec['val_auc_max']:.3f} " if x_val is not None else "")
              + f"({rec['seconds']:.0f}s)", flush=True)
    if x_val is None:
        _atomic_save({"model": model.state_dict()}, best)
    else:                                                    # the recorded best epoch's weights -> best.pt
        src = best_at(int(np.argmax([h["val_auc"] for h in history])) + 1)
        if src.exists():
            src.replace(best)
        elif not best.exists():
            raise FileNotFoundError(f"{src}: weights of the best epoch are missing")
        for f in run_dir.glob("best_e*.pt"):
            f.unlink()
    (run_dir / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
    return pd.DataFrame(history)


def _atomic_save(obj, path: Path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def selected(history: pd.DataFrame) -> dict:
    """Best (epoch, pooling) by validation AUC; ties -> earlier epoch, then mean pooling."""
    best = None
    for h in history.itertuples():
        for a in cnn.AGGREGATORS:
            v = getattr(h, f"val_auc_{a}")
            if best is None or v > best["val_auc"]:
                best = {"epoch": int(h.epoch), "aggregator": a, "val_auc": float(v)}
    return best


# -------------------------------------------------------------------- stages ---

def stage_select(dev_df, paths, dev, smoke):
    train, val = dev_df[dev_df.split == "train"], dev_df[dev_df.split == "val"]
    rows = []
    for P, seed in SELECT_RUNS:
        h = train_run(f"select_{P}_s{seed}", P, seed, train, val, 2 if smoke else MAX_EPOCHS, paths, dev)
        rows.append({"protocol": P, "seed": seed, **selected(h), "epochs_run": len(h)})
        # validation scores of the selected epoch, kept for later analysis
        model = cnn.build_model(pretrained=False).to(dev)
        model.load_state_dict(torch.load(paths.cache / f"select_{P}_s{seed}" / "best.pt", map_location=dev)["model"])
        s = cnn.predict(model, load_elas(val, P), dev)
        pd.DataFrame({"path": val.path.to_numpy(), "label": val.label.to_numpy(), **{f"cnn_{a}": s[a] for a in s}}) \
            .to_csv(paths.out / f"val_scores_{P}_s{seed}.csv", index=False)
    sel = pd.DataFrame(rows)
    sel.to_csv(paths.out / "selection.csv", index=False)
    print(sel.to_string(index=False), flush=True)
    fig_training_curves(paths)
    return sel


def stage_refit(dev_df, paths, dev, smoke):
    sel = pd.read_csv(paths.out / "selection.csv")
    models = {}
    for P in PROTOCOLS:
        s = sel[(sel.protocol == P) & (sel.seed == PRIMARY_SEED)].iloc[0]
        name = f"refit_{P}_e{int(s.epoch)}"
        train_run(name, P, PRIMARY_SEED, dev_df, None, int(s.epoch), paths, dev)
        state = torch.load(paths.cache / name / "best.pt", map_location="cpu")["model"]
        meta = {"protocol": P, "epochs": int(s.epoch), "aggregator": s.aggregator, "selection_val_auc": float(s.val_auc),
                "seed": PRIMARY_SEED, "trained_on": "train+val", "n_train": len(dev_df), **_constants(),
                "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
        _atomic_save({"model": state, "meta": meta}, paths.model(P))
        models[P] = {"file": str(paths.model(P).relative_to(resolve("."))), "sha1": sha1(paths.model(P)), **meta}
        print(f"  refit {P}: {meta['epochs']} epochs, pooling {meta['aggregator']} -> {paths.model(P).name}", flush=True)
    prereg = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": "written before the CNN sees the test split; the test stage refuses to run if a model changed",
        "models": models,
        "script_sha1": sha1(resolve("scripts/run_phase8.py")),
        "selection_csv_sha1": sha1(paths.out / "selection.csv"),
        "ela_meta": json.loads((ELA_DIR / "meta.json").read_text()),
        "classical_comparator": f"{CLASSICAL} test scores of results/phase7/run_1 (Phase 7 run 1, frozen)",
        "primary": ["test AUC of the protocol-R CNN, with a 95% group-bootstrap CI",
                    f"paired test delta AUC, CNN - classical ({CLASSICAL}), protocols R and B",
                    "paired delta AUC, rank-average(CNN, classical) - classical, protocols R and B"],
        "secondary": [f"paired delta AUC, CNN - {CLASSICAL_PLUS} (classical forensic + shortcut features: the CNN "
                      "can see naive-ELA level and, through max pooling, image size)",
                      "protocol A: CNN vs shortcuts (does an ELA-CNN learn the format leak?)",
                      "JPEG-only subset (authentic JPEG vs tampered JPEG) and per forgery type: AUCs with paired deltas",
                      "image size: AUC of the tile count alone, and CNN / classical AUC within tile-count terciles "
                      "(validation and test)",
                      "Spearman correlation of the CNN and classical scores",
                      f"test AUC under clean + {[d[2] for d in DEGRADATIONS]} for R and B, paired vs classical; "
                      "share of images smaller than one tile (zero-padded); analysis variant without padding",
                      "MICC-F220 AUC for R and B, paired vs classical"],
        "combination": "mean of the two scores' within-set ranks (fixed, not tuned; uses no labels, but is "
                       "transductive, i.e. not a per-image score one could deploy)",
        "not_reported": "balanced accuracy of the CNN: its logit is uncalibrated (max pooling, class weights) and no "
                        "threshold was fixed on validation",
    }
    (paths.out / "preregistration.json").write_text(json.dumps(prereg, indent=2))
    return prereg


def _constants() -> dict:
    return {"crop": cnn.CROP, "stride": cnn.STRIDE, "ela_quality": cnn.ELA_QUALITY, "ela_scale": cnn.ELA_SCALE,
            "cnn_code_sha1": cnn.code_sha1()}


def _auc_ci(y, s, g, seed, n_boot) -> dict:
    lo, hi = group_bootstrap(y, s, g, roc_auc_score, n_boot, seed)
    return {"auc": float(roc_auc_score(y, s)), "ci": [lo, hi]}


def rank_mean(*scores) -> np.ndarray:
    return np.mean([rankdata(s) / len(s) for s in scores], axis=0)


def _cache_key_file(items: pd.DataFrame, P: str, label: str, cfg: dict, tag: str) -> Path:
    """File name used by run_phase7b.run_condition (same formula) -> lets the preflight check for cache hits."""
    from forgery.pipeline import feature_code_sha1, module_params
    key = hashlib.sha1(json.dumps([tag, P, label, feature_code_sha1(), module_params(cfg, P), sorted(items.path)],
                                  sort_keys=True, default=str).encode()).hexdigest()[:12]
    return resolve("data/cache/phase7b") / f"{tag}_{P}_{key}.csv"


def micc_items() -> pd.DataFrame:
    root = resolve("data/external/MICC-F220")
    gt = pd.read_csv(root / "groundtruthDB_220.txt", sep=r"\s+", header=None, names=["file", "label"])
    return pd.DataFrame({"path": [str((root / f).relative_to(resolve("."))) for f in gt.file], "mask_path": None,
                         "label": gt.label.astype(int),       # same grouping as Phase 7b
                         "group": gt.file.str.extract(r"^(.+?)_?(?:tamp\d+|scale)\.[A-Za-z]+$", expand=False)
                         .fillna(gt.file)})


def preflight(test_df, paths, smoke, cfg) -> dict:
    """Everything the test stage needs, checked BEFORE the test split is used (and logged)."""
    prereg = json.loads((paths.out / "preregistration.json").read_text())
    problems = []
    for P, m in prereg["models"].items():
        if sha1(resolve(m["file"])) != m["sha1"]:
            problems.append(f"model {m['file']} changed after the pre-registration")
        meta = torch.load(resolve(m["file"]), map_location="cpu", weights_only=False)["meta"]
        if {k: meta[k] for k in _constants()} != _constants():
            problems.append(f"cnn_{P}: CNN/ELA constants or cnn.py changed since the refit")
    if json.loads((ELA_DIR / "meta.json").read_text()) != prereg["ela_meta"]:
        problems.append("data/ela/meta.json differs from the pre-registration")
    for P in PROTOCOLS:
        missing = [i for i in test_df.image_id if not (ELA_DIR / P / f"{i}.png").exists()]
        if missing:
            problems.append(f"{len(missing)} ELA images missing for protocol {P}")
        c = _classical_scores(P, test_df, smoke)
        if c[[CLASSICAL, CLASSICAL_PLUS, "shortcuts|rf"]].isna().any().any():
            problems.append(f"classical scores missing for some images (protocol {P})")
    if not smoke:                     # smoke evaluates val images: the classical results are computed on the fly
        items = test_df[["path", "mask_path"]]
        for P in ("R", "B"):
            for family, param, label in [("clean", None, "clean")] + DEGRADATIONS:
                if not _cache_key_file(items, P, label, cfg, "robust").exists():
                    problems.append(f"Phase-7b cache missing: robust {P} {label}")
            if not _cache_key_file(micc_items(), P, "MICC-F220", cfg, "external").exists():
                problems.append(f"Phase-7b cache missing: external {P}")
    if problems:
        raise SystemExit("preflight failed:\n  " + "\n  ".join(problems))
    print("  preflight OK", flush=True)
    return prereg


def _log(event: dict):
    with resolve("results/phase7/test_runs.log").open("a") as fh:
        fh.write(json.dumps({**event, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")


def size_analysis(y, shapes, scores: dict) -> dict:
    """Is the score just image size? AUC of the tile count / pixel count alone, and AUCs within area terciles."""
    n_tiles = np.array([cnn.n_tiles(s) for s in shapes])
    area = np.array([s[0] * s[1] for s in shapes], float)
    # Ties (most images are 384 x 256) are broken at random. The test run of 2026-09-27 broke them by row order,
    # which is authentic-first, so its tercile AUCs in summary.json are confounded with the label and are not
    # reported (see report/KNOWN_ISSUES.md; scripts/phase8_posthoc.py groups by natural size instead).
    tiebreak = np.random.default_rng(0).permutation(len(area))
    strata = pd.qcut(pd.Series(area + tiebreak / (10 * len(area))).rank(method="first"), 3, labels=False).to_numpy()
    out = {"auc_tile_count_alone": float(roc_auc_score(y, n_tiles)), "auc_area_alone": float(roc_auc_score(y, area)),
           "strata": []}
    for k in range(3):
        m = strata == k
        out["strata"].append({"pixels_range": [int(area[m].min()), int(area[m].max())], "n": int(m.sum()),
                              **{name: float(roc_auc_score(y[m], s[m])) for name, s in scores.items()}})
    for name in scores:
        out[f"within_strata_mean_{name}"] = float(np.average([r[name] for r in out["strata"]],
                                                             weights=[r["n"] for r in out["strata"]]))
    return out


def _val_size_analysis(dev_df, paths) -> dict:
    """The same size check on validation (selection runs, seed 42): computed before looking at test numbers."""
    out = {}
    val = dev_df[dev_df.split == "val"]
    for P in PROTOCOLS:
        f = paths.out / f"val_scores_{P}_s{PRIMARY_SEED}.csv"
        if not f.exists():
            continue
        agg = pd.read_csv(paths.out / "selection.csv").query("protocol == @P and seed == @PRIMARY_SEED").aggregator.iloc[0]
        v = val[["path", "image_id"]].merge(pd.read_csv(f, dtype={"path": str}), on="path", validate="one_to_one")
        shapes = [Image.open(ELA_DIR / P / f"{i}.png").size[::-1] for i in v.image_id]
        out[P] = size_analysis(v.label.to_numpy(), shapes, {"cnn": v[f"cnn_{agg}"].to_numpy()})
    return out


def stage_test(dev_df, test_df, paths, dev, smoke, cfg):
    prereg = preflight(test_df, paths, smoke, cfg)
    n_boot, seed = (200 if smoke else 1000), cfg["seed"]
    summary = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "smoke": smoke,
               "val_size_analysis": _val_size_analysis(dev_df, paths), "protocols": {}}
    if not smoke:
        _log({"event": "phase8_cnn_test_start", "script_sha1": sha1(resolve("scripts/run_phase8.py")),
              "cnn_code_sha1": cnn.code_sha1(), "models_sha1": {P: m["sha1"] for P, m in prereg["models"].items()},
              "note": "new models (CNN) evaluated once on test; classical models untouched; nothing fed back"})
    y, g = test_df.label.to_numpy(), test_df.split_group.to_numpy()
    jpeg = (test_df.format == "JPEG").to_numpy()
    rows = []
    for P in PROTOCOLS:
        ck = torch.load(paths.model(P), map_location=dev, weights_only=False)
        model = cnn.build_model(pretrained=False).to(dev)
        model.load_state_dict(ck["model"])
        agg = ck["meta"]["aggregator"]
        elas = load_elas(test_df, P)
        s_all = cnn.predict(model, elas, dev)
        classical = _classical_scores(P, test_df, smoke)
        s_cnn, s_cls = s_all[agg], classical[CLASSICAL].to_numpy()
        s_plus, s_short = classical[CLASSICAL_PLUS].to_numpy(), classical["shortcuts|rf"].to_numpy()
        s_comb = rank_mean(s_cnn, s_cls)
        assert len(s_cnn) == len(s_cls) == len(y)
        n_t = np.array([cnn.n_tiles(e.shape) for e in elas])
        pd.DataFrame({"path": test_df.path.to_numpy(), "label": y, "split_group": g, "n_tiles": n_t,
                      **{f"cnn_{a}": v for a, v in s_all.items()}, "cnn": s_cnn, "classical": s_cls,
                      "classical_plus_shortcuts": s_plus, "shortcuts": s_short, "combination": s_comb}) \
            .to_csv(paths.out / f"test_scores_cnn_{P}.csv", index=False)
        res = {"aggregator": agg, "epochs": ck["meta"]["epochs"],
               "auc": {k: _auc_ci(y, v, g, seed, n_boot) for k, v in
                       (("cnn", s_cnn), ("classical", s_cls), ("combination", s_comb),
                        ("classical_plus_shortcuts", s_plus), ("shortcuts", s_short))},
               "cnn_minus_classical": paired_auc_delta(y, s_cnn, s_cls, g, n_boot, seed),
               "combination_minus_classical": paired_auc_delta(y, s_comb, s_cls, g, n_boot, seed),
               "cnn_minus_classical_plus_shortcuts": paired_auc_delta(y, s_cnn, s_plus, g, n_boot, seed),
               "cnn_minus_shortcuts": paired_auc_delta(y, s_cnn, s_short, g, n_boot, seed),
               "spearman_cnn_classical": float(spearmanr(s_cnn, s_cls).statistic),
               "size_analysis": size_analysis(y, [e.shape for e in elas], {"cnn": s_cnn, "classical": s_cls}),
               "subsets": {}}
        subsets = {"jpeg_only": jpeg,
                   **{f"authentic_vs_{t}": (y == 0) | (test_df.forgery_type == t).to_numpy()
                      for t in ("copy-move", "splicing")}}
        for name, m in subsets.items():
            if len(set(y[m])) < 2:
                continue
            res["subsets"][name] = {"n": int(m.sum()), "n_tampered": int(y[m].sum()),
                                    "cnn": _auc_ci(y[m], s_cnn[m], g[m], seed, n_boot),
                                    "classical": _auc_ci(y[m], s_cls[m], g[m], seed, n_boot),
                                    "cnn_minus_classical": paired_auc_delta(y[m], s_cnn[m], s_cls[m], g[m], n_boot, seed)}
        summary["protocols"][P] = res
        d = res["cnn_minus_classical"]
        rows.append({"protocol": P, **{f"{k}_auc": v["auc"] for k, v in res["auc"].items()},
                     "delta_cnn_classical": d["delta_auc"], "delta_ci_low": d["ci_low"], "delta_ci_high": d["ci_high"],
                     "delta_comb_classical": res["combination_minus_classical"]["delta_auc"]})
        print(f"  [{P}] CNN {res['auc']['cnn']['auc']:.3f}  classical {res['auc']['classical']['auc']:.3f}  "
              f"combination {res['auc']['combination']['auc']:.3f}  delta {d['delta_auc']:+.3f} "
              f"[{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]", flush=True)
        if P != "A":
            res["robustness"] = _robustness(P, model, agg, test_df, cfg, dev, n_boot, seed)
            if not smoke:
                res["micc_f220"] = _micc(P, model, agg, cfg, dev, n_boot, seed)
    pd.DataFrame(rows).to_csv(paths.out / "test_comparison.csv", index=False)
    (paths.out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    fig_roc(paths, test_df)
    write_markdown(paths, summary)
    if not smoke:
        _log({"event": "phase8_cnn_test_complete", "summary_sha1": sha1(paths.out / "summary.json")})
    return summary


def _classical_scores(P: str, test_df: pd.DataFrame, smoke: bool) -> pd.DataFrame:
    f = resolve(f"results/phase5/oof_{P}.csv") if smoke else resolve(f"results/phase7/run_1/test_scores_{P}.csv")
    c = pd.read_csv(f, dtype={"path": str})
    return test_df[["path"]].merge(c, on="path", how="left", validate="one_to_one")


def _score_variants(model, elas, agg, dev) -> dict:
    s = cnn.predict(model, elas, dev)[agg]
    padded = np.array([cnn.is_padded(e.shape) for e in elas])
    s_unpadded = cnn.predict(model, elas, dev, pad=False)[agg] if padded.any() else s
    return {"cnn": s, "cnn_unpadded": s_unpadded, "padded": padded}


def _robustness(P, model, agg, test_df, cfg, dev, n_boot, seed) -> list[dict]:
    p7b = _phase7b()
    items = test_df[["path", "mask_path"]].copy()
    y, g = test_df.label.to_numpy(), test_df.split_group.to_numpy()
    out = []
    for family, param, label in [("clean", None, "clean")] + DEGRADATIONS:
        t0 = time.time()
        v = _score_variants(model, compute_elas(test_df.path.tolist(), P, cfg, None if family == "clean" else family,
                                                param), agg, dev)
        cls = items.merge(p7b.run_condition(items, P, family, label, param, cfg, "robust"), on="path",
                          validate="one_to_one").p_raw.to_numpy()
        assert len(cls) == len(y) == len(v["cnn"])
        out.append({"condition": label, "cnn": _auc_ci(y, v["cnn"], g, seed, n_boot),
                    "classical_auc": float(roc_auc_score(y, cls)),
                    "combination_auc": float(roc_auc_score(y, rank_mean(v["cnn"], cls))),
                    "cnn_minus_classical": paired_auc_delta(y, v["cnn"], cls, g, n_boot, seed),
                    "share_padded": float(v["padded"].mean()),
                    "cnn_unpadded_auc": float(roc_auc_score(y, v["cnn_unpadded"]))})
        print(f"  [{P}] {label:26s} CNN {out[-1]['cnn']['auc']:.3f} (unpadded {out[-1]['cnn_unpadded_auc']:.3f}, "
              f"{out[-1]['share_padded']:.0%} padded)  classical {out[-1]['classical_auc']:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    return out


def _micc(P, model, agg, cfg, dev, n_boot, seed) -> dict:
    items = micc_items()
    v = _score_variants(model, compute_elas(items.path.tolist(), P, cfg), agg, dev)
    cls = items.merge(_phase7b().run_condition(items, P, "clean", "MICC-F220", None, cfg, "external"),
                      on="path", validate="one_to_one").p_raw.to_numpy()
    y, g = items.label.to_numpy(), items.group.to_numpy()
    assert len(cls) == len(y) == len(v["cnn"])
    out = {"cnn": _auc_ci(y, v["cnn"], g, seed, n_boot), "classical_auc": float(roc_auc_score(y, cls)),
           "combination_auc": float(roc_auc_score(y, rank_mean(v["cnn"], cls))),
           "cnn_minus_classical": paired_auc_delta(y, v["cnn"], cls, g, n_boot, seed),
           "share_padded": float(v["padded"].mean()), "cnn_unpadded_auc": float(roc_auc_score(y, v["cnn_unpadded"]))}
    print(f"  [{P}] MICC-F220: CNN {out['cnn']['auc']:.3f}  classical {out['classical_auc']:.3f}", flush=True)
    return out


# ------------------------------------------------------------------- outputs ---

def fig_training_curves(paths: Paths):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    colors = {"R": "#4C78A8", "B": "#E45756", "A": "#54A24B"}
    for P, seed in SELECT_RUNS:
        f = paths.cache / f"select_{P}_s{seed}" / "history.csv"
        if f.exists():
            h = pd.read_csv(f)
            ax.plot(h.epoch, h.val_auc, "o-" if seed == PRIMARY_SEED else ":", ms=3, color=colors[P],
                    label=f"{P}, seed {seed}", alpha=1 if seed == PRIMARY_SEED else 0.6)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation AUC (best pooling)")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("ELA-CNN training (train split; validation split for selection)", fontsize=9)
    fig.tight_layout()
    fig.savefig(paths.out / "fig_training_curves.png", dpi=140)
    plt.close(fig)


def fig_roc(paths: Paths, test_df: pd.DataFrame):
    fig, axes = plt.subplots(1, len(PROTOCOLS), figsize=(4 * len(PROTOCOLS), 3.8))
    for ax, P in zip(axes, PROTOCOLS):
        d = pd.read_csv(paths.out / f"test_scores_cnn_{P}.csv")
        for col, lab, c in (("cnn", "ELA-CNN", "#E45756"), ("classical", "classical forest", "#4C78A8"),
                            ("combination", "rank average", "#54A24B"), ("shortcuts", "shortcuts", "#999999")):
            fpr, tpr, _ = roc_curve(d.label, d[col])
            ax.plot(fpr, tpr, color=c, lw=1.4 if col != "shortcuts" else 1, ls="-" if col != "shortcuts" else "--",
                    label=f"{lab} ({roc_auc_score(d.label, d[col]):.3f})")
        ax.plot([0, 1], [0, 1], color="grey", lw=0.6, ls=":")
        ax.set_title(f"protocol {P}", fontsize=9)
        ax.set_xlabel("false positive rate")
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    axes[0].set_ylabel("true positive rate")
    fig.suptitle("Test ROC: ELA-CNN vs the classical pipeline", fontsize=10)
    fig.tight_layout()
    fig.savefig(paths.out / "fig_roc_cnn_vs_classical.png", dpi=140)
    plt.close(fig)


def fig_ela_examples(dev_df: pd.DataFrame, paths: Paths, cfg: dict):
    """Validation images: an authentic JPEG and a tampered TIFF, ELA as released (A) and after protocol R."""
    val = dev_df[dev_df.split == "val"]
    if not ((val.label == 1) & (val.format == "TIFF")).any() or not ((val.label == 0) & (val.format == "JPEG")).any():
        return
    picks = [val[(val.label == 0) & (val.format == "JPEG")].iloc[0], val[(val.label == 1) & (val.format == "TIFF")].iloc[0]]
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.4))
    for row, r in zip(axes, picks):
        img = load_rgb(resolve(r.path))
        for ax, im, title in zip(row, (img, cnn.load_ela(ELA_DIR / "A" / f"{r.image_id}.png"),
                                       cnn.load_ela(ELA_DIR / "R" / f"{r.image_id}.png")),
                                 (f"{r.label_name} ({r.format})", "ELA, files as released (A)",
                                  "ELA after protocol R")):
            ax.imshow(im)
            ax.set_title(title, fontsize=8)
            ax.axis("off")
    fig.suptitle("ELA inputs of the CNN (validation images)", fontsize=10)
    fig.tight_layout()
    fig.savefig(paths.out / "fig_ela_examples.png", dpi=120)
    plt.close(fig)


def write_markdown(paths: Paths, s: dict):
    f = lambda v: f"{v['auc']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]"          # noqa: E731
    d = lambda v: f"{v['delta_auc']:+.3f} [{v['ci_low']:+.3f}, {v['ci_high']:+.3f}]"  # noqa: E731
    L = ["# Phase 8: ELA-CNN vs classical (test split)", "", f"Created {s['created']}" + (" (SMOKE)" if s["smoke"] else ""),
         "", "| Protocol | CNN | classical | CNN - classical | rank average | combination - classical | "
         "classical + shortcuts | CNN - (classical + shortcuts) | shortcuts |", "|---|---|---|---|---|---|---|---|---|"]
    for P, r in s["protocols"].items():
        a = r["auc"]
        L.append(f"| {P} | {f(a['cnn'])} | {f(a['classical'])} | {d(r['cnn_minus_classical'])} | "
                 f"{f(a['combination'])} | {d(r['combination_minus_classical'])} | {f(a['classical_plus_shortcuts'])} | "
                 f"{d(r['cnn_minus_classical_plus_shortcuts'])} | {f(a['shortcuts'])} |")
    L += ["", "## Subsets (AUC; CNN - classical)", ""]
    for P, r in s["protocols"].items():
        for name, v in r["subsets"].items():
            L.append(f"- {P} {name} (n={v['n']}): CNN {f(v['cnn'])}, classical {f(v['classical'])}, "
                     f"delta {d(v['cnn_minus_classical'])}")
    L += ["", "## Image size (area terciles, pre-registered check)", ""]
    for P, r in s["protocols"].items():
        z = r["size_analysis"]
        L.append(f"- {P}: tile count alone AUC {z['auc_tile_count_alone']:.3f}, pixel count alone "
                 f"{z['auc_area_alone']:.3f}; within-area-tercile mean AUC CNN "
                 f"{z['within_strata_mean_cnn']:.3f}, classical {z['within_strata_mean_classical']:.3f}; "
                 f"Spearman(CNN, classical) {r['spearman_cnn_classical']:.2f}")
    L += ["", "## Degradations and MICC-F220", ""]
    for P, r in s["protocols"].items():
        for rb in r.get("robustness", []) + ([{"condition": "MICC-F220", **r["micc_f220"]}] if "micc_f220" in r else []):
            L.append(f"- {P} {rb['condition']}: CNN {f(rb['cnn'])} (unpadded {rb['cnn_unpadded_auc']:.3f}; "
                     f"{rb['share_padded']:.0%} of images smaller than a tile), classical {rb['classical_auc']:.3f}, "
                     f"delta {d(rb['cnn_minus_classical'])}, rank average {rb['combination_auc']:.3f}")
    (paths.out / "test_results.md").write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------------- main ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["select", "refit", "test", "all"], default="all")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    paths = Paths(args.smoke)
    dev = cnn.device()
    df = read_manifest(cfg).merge(read_splits(cfg), on="path")
    df["mask_path"] = df.mask_path.where(df.mask_valid, None)
    dev_df = df[df.split.isin(["train", "val"])].reset_index(drop=True)
    if args.smoke:          # small subsets; val images play "test"; the test split is never read
        rs = cfg["seed"]
        train = dev_df[dev_df.split == "train"].sample(320, random_state=rs)
        val = dev_df[dev_df.split == "val"].sample(300, random_state=rs)
        dev_df, test_df = pd.concat([train, val.iloc[:150]]).reset_index(drop=True), val.iloc[150:].reset_index(drop=True)
    else:
        test_df = df[df.split == "test"].reset_index(drop=True)
    if not args.smoke and args.stage == "all":
        raise SystemExit("run the stages one by one (--stage select, refit, then test)")
    print(f"device {dev}; dev {len(dev_df)} images; out {paths.out}", flush=True)
    if args.stage in ("select", "all"):
        stage_select(dev_df, paths, dev, args.smoke)
        fig_ela_examples(dev_df, paths, cfg)
    if args.stage in ("refit", "all"):
        stage_refit(dev_df, paths, dev, args.smoke)
    if args.stage in ("test", "all"):
        stage_test(dev_df, test_df, paths, dev, args.smoke, cfg)


if __name__ == "__main__":
    main()
