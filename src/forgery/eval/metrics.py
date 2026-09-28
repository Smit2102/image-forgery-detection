"""Shared evaluation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def pixel_auc(evidence: np.ndarray | None, mask: np.ndarray, step: int = 2) -> float:
    """ROC-AUC of an evidence map against a boolean mask (every `step`-th pixel). 0.5 if uninformative."""
    if evidence is None:
        return 0.5
    e, m = evidence[::step, ::step].ravel(), mask[::step, ::step].ravel()
    if m.all() or not m.any() or np.ptp(e) == 0:
        return 0.5
    return float(roc_auc_score(m, e))


def local_columns(cols) -> list[str]:
    """Feature columns describing within-image inconsistency (see features/common.py)."""
    return [c for c in cols if "local_" in c.split(".", 1)[-1]]


def rf_auc(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], seed: int, n_trees: int = 200) -> float:
    """Train a random forest on `train[cols]` and return ROC-AUC on `test` (label column: 'label')."""
    if not cols:
        return 0.5
    clf = RandomForestClassifier(n_estimators=n_trees, min_samples_leaf=2, class_weight="balanced",
                                 n_jobs=-1, random_state=seed)
    clf.fit(train[cols].to_numpy(float), train["label"])
    return float(roc_auc_score(test["label"], clf.predict_proba(test[cols].to_numpy(float))[:, 1]))
