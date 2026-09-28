"""Phase 5: image-level classification, ablations and statistics.

All cross-validation uses the grouped dev folds from Phase 2 (`cv_fold`); inner model selection uses
GroupKFold over `split_group`, so no image group is ever on both sides of a split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..features import REGISTRY
from .leakage import FEATURE_SETS as SHORTCUT_SETS

MODULES = list(REGISTRY)


# ------------------------------------------------------------------ columns ---

def module_columns(cols, modules=None, local_only: bool = False) -> list[str]:
    modules = MODULES if modules is None else modules
    out = [c for c in cols if c.split(".", 1)[0] in modules and not c.startswith("time_ms.")]
    if local_only:
        out = [c for c in out if "local_" in c.split(".", 1)[1]]
    return out


def shortcut_columns() -> list[str]:
    return [f"shortcut.{c}" for c in SHORTCUT_SETS["all"]]


# ------------------------------------------------------------------- models ---

def svm(n_features: int, C: float = 10.0, gamma_factor: float = 1.0) -> Pipeline:
    return Pipeline([("scale", StandardScaler()),
                     ("svm", SVC(C=C, gamma=gamma_factor / max(n_features, 1), class_weight="balanced"))])


def forest(seed: int, max_features="sqrt", min_samples_leaf: int = 1, n_estimators: int = 300,
           n_jobs: int = -1) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=n_estimators, max_features=max_features,
                                  min_samples_leaf=min_samples_leaf, class_weight="balanced",
                                  n_jobs=n_jobs, random_state=seed)


def grid(model: str, n_features: int, seed: int):
    """(estimator, parameter grid, GridSearchCV n_jobs) for 'svm' or 'rf'."""
    if model == "svm":
        return (svm(n_features), {"svm__C": [1.0, 10.0, 100.0],
                                  "svm__gamma": [f / max(n_features, 1) for f in (0.3, 1.0, 3.0)]}, -1)
    if model == "rf":
        return forest(seed), {"max_features": ["sqrt", 0.3], "min_samples_leaf": [1, 4]}, 1
    raise ValueError(model)


def scores_of(est, X) -> np.ndarray:
    """Continuous score: P(tampered) for probabilistic models, SVM decision value otherwise."""
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)[:, 1]
    return est.decision_function(X)


def threshold_of(model: str) -> float:
    return 0.0 if model == "svm" else 0.5


def fold_metrics(y, s, thr) -> dict:
    pred = (s >= thr).astype(int)
    return {"roc_auc": roc_auc_score(y, s), "balanced_accuracy": balanced_accuracy_score(y, pred),
            "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred)}


def nested_cv(df: pd.DataFrame, cols: list[str], model: str, seed: int, inner_splits: int = 3) -> dict:
    """Outer: Phase-2 grouped folds. Inner: GridSearchCV with GroupKFold on split_group."""
    X, y = df[cols].to_numpy(float), df["label"].to_numpy()
    groups, folds = df["split_group"].to_numpy(), df["cv_fold"].to_numpy()
    oof = np.full(len(df), np.nan)
    rows, chosen = [], []
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        est, params, n_jobs = grid(model, len(cols), seed)
        inner = list(GroupKFold(inner_splits).split(X[tr], y[tr], groups[tr]))
        gs = GridSearchCV(est, params, scoring="roc_auc", cv=inner, n_jobs=n_jobs, refit=True)
        gs.fit(X[tr], y[tr])
        oof[te] = scores_of(gs.best_estimator_, X[te])
        chosen.append(gs.best_params_)
        rows.append({"fold": int(f), **fold_metrics(y[te], oof[te], threshold_of(model))})
    return {"oof": oof, "folds": pd.DataFrame(rows), "chosen": chosen, "threshold": threshold_of(model)}


def fixed_cv(df: pd.DataFrame, cols: list[str], make, threshold: float = 0.5) -> dict:
    """Grouped-fold CV with a fixed model factory (used for ablations)."""
    X, y, folds = df[cols].to_numpy(float), df["label"].to_numpy(), df["cv_fold"].to_numpy()
    oof = np.full(len(df), np.nan)
    rows = []
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        est = make().fit(X[tr], y[tr])
        oof[te] = scores_of(est, X[te])
        rows.append({"fold": int(f), **fold_metrics(y[te], oof[te], threshold)})
    return {"oof": oof, "folds": pd.DataFrame(rows), "threshold": threshold}


def calibrated_cv(df: pd.DataFrame, cols: list[str], make, seed: int) -> np.ndarray:
    """Out-of-fold probabilities of an isotonic-calibrated model (calibration fitted inside each fold)."""
    X, y = df[cols].to_numpy(float), df["label"].to_numpy()
    groups, folds = df["split_group"].to_numpy(), df["cv_fold"].to_numpy()
    oof = np.full(len(df), np.nan)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        inner = list(GroupKFold(3).split(X[tr], y[tr], groups[tr]))
        cal = CalibratedClassifierCV(make(), method="isotonic", cv=inner).fit(X[tr], y[tr])
        oof[te] = cal.predict_proba(X[te])[:, 1]
    return oof


# ---------------------------------------------------------------- statistics ---

def group_bootstrap(y, s, groups, stat, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95 % percentile CI of stat(y, s) resampling whole split groups with replacement."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    vals = []
    for _ in range(n):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(uniq), len(uniq))])
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(stat(y[idx], s[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_auc_delta(y, s_new, s_old, groups, n: int = 1000, seed: int = 0) -> dict:
    """AUC(s_new) - AUC(s_old) with a paired group-bootstrap 95 % CI."""
    point = roc_auc_score(y, s_new) - roc_auc_score(y, s_old)
    lo, hi = group_bootstrap(y, np.stack([s_new, s_old], 1), groups,
                             lambda yy, ss: roc_auc_score(yy, ss[:, 0]) - roc_auc_score(yy, ss[:, 1]), n, seed)
    return {"delta_auc": float(point), "ci_low": lo, "ci_high": hi}


def mcnemar(y, pred_a, pred_b) -> dict:
    """Exact McNemar test on the discordant pairs of two classifiers' decisions."""
    a_ok, b_ok = pred_a == y, pred_b == y
    b, c = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return {"a_right_b_wrong": b, "a_wrong_b_right": c, "p_value": float(p)}


def reliability(y, p, bins: int = 10) -> tuple[pd.DataFrame, float, float]:
    """Reliability table, expected calibration error and Brier score."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    tab = pd.DataFrame({"bin": idx, "p": p, "y": y}).groupby("bin").agg(
        mean_pred=("p", "mean"), frac_pos=("y", "mean"), n=("y", "size")).reset_index()
    ece = float((tab.n / len(p) * (tab.mean_pred - tab.frac_pos).abs()).sum())
    return tab, ece, float(brier_score_loss(y, p))


def coverage_curve(y, p, widths=np.round(np.arange(0, 0.41, 0.02), 2)) -> pd.DataFrame:
    """Accuracy on the images still judged (|p - 0.5| >= w, i.e. outside the open uncertain band) vs. their share."""
    rows = []
    for w in widths:
        keep = np.abs(p - 0.5) >= w      # same boundary as pipeline.verdict_from_probability
        pred = (p >= 0.5).astype(int)
        rows.append({"half_width": float(w), "coverage": float(keep.mean()),
                     "accuracy": float((pred[keep] == y[keep]).mean()) if keep.any() else np.nan,
                     "balanced_accuracy": float(balanced_accuracy_score(y[keep], pred[keep]))
                     if keep.any() and len(np.unique(y[keep])) == 2 else np.nan})
    return pd.DataFrame(rows)


def summarize_folds(folds: pd.DataFrame) -> dict:
    out = {}
    for c in folds.columns.drop("fold"):
        out[f"{c}_mean"], out[f"{c}_std"] = float(folds[c].mean()), float(folds[c].std())
    return out


class FittedModel:
    """What detect() needs: predict_proba + the feature order (`feature_names_`)."""

    def __init__(self, estimator, feature_names: list[str], meta: dict | None = None):
        self.estimator, self.feature_names_, self.meta = estimator, list(feature_names), meta or {}

    def predict_proba(self, X):
        return self.estimator.predict_proba(np.asarray(X, dtype=float))

