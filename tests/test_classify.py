"""Phase-5 classification helpers on a small synthetic data set (fast smoke + correctness checks)."""

import importlib.util

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from forgery.config import ROOT
from forgery.eval import classify as C


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, n)
    df = pd.DataFrame({"label": y, "split_group": [f"g{i // 2}" for i in range(n)],
                       "path": [f"p{i}" for i in range(n)]})
    df["cv_fold"] = df.split_group.map({g: i % 5 for i, g in enumerate(df.split_group.unique())})
    df["forgery_type"] = np.where(y == 1, rng.choice(["copy-move", "splicing"], n), "none")
    df["area_bucket"] = np.where(y == 1, rng.choice(["<1%", "1-5%", ">5%"], n), None)
    df["ext"] = np.where(y == 1, rng.choice(["jpg", "tif"], n), "jpg")
    for m in C.MODULES:
        df[f"{m}.local_signal"] = y * 1.0 + rng.normal(0, 1.5, n)
        df[f"{m}.global_level"] = rng.normal(0, 1, n)
    for c in C.shortcut_columns():
        df[c] = rng.normal(0, 1, n)
    return df


def test_columns(data):
    assert len(C.module_columns(data.columns)) == 2 * len(C.MODULES)
    assert all("local_" in c for c in C.module_columns(data.columns, local_only=True))
    assert C.module_columns(data.columns, ["ela"]) == ["ela.local_signal", "ela.global_level"]


@pytest.mark.parametrize("model", ["rf", "svm"])
def test_nested_cv_is_out_of_fold_and_learns(data, model):
    r = C.nested_cv(data, C.module_columns(data.columns), model, seed=0)
    assert np.isfinite(r["oof"]).all() and len(r["chosen"]) == 5
    assert roc_auc_score(data.label, r["oof"]) > 0.8


def test_shortcut_noise_is_near_chance(data):
    r = C.fixed_cv(data, C.shortcut_columns(), lambda: C.forest(0, n_estimators=100))
    assert abs(roc_auc_score(data.label, r["oof"]) - 0.5) < 0.12


def test_calibration_and_statistics(data):
    cols = C.module_columns(data.columns)
    p = C.calibrated_cv(data, cols, lambda: C.forest(0, n_estimators=100), 0)
    assert np.isfinite(p).all() and (0 <= p).all() and (p <= 1).all()
    tab, ece, brier = C.reliability(data.label.to_numpy(), p)
    assert 0 <= ece < 0.3 and 0 <= brier < 0.3 and tab.n.sum() == len(data)
    cov = C.coverage_curve(data.label.to_numpy(), p)
    assert cov.coverage.iloc[0] == 1 and cov.coverage.is_monotonic_decreasing
    y, g = data.label.to_numpy(), data.split_group.to_numpy()
    lo, hi = C.group_bootstrap(y, p, g, roc_auc_score, n=200)
    assert lo <= roc_auc_score(y, p) <= hi
    d = C.paired_auc_delta(y, p, np.random.default_rng(1).random(len(y)), g, n=200)
    assert d["delta_auc"] > 0 and d["ci_low"] > 0
    mc = C.mcnemar(y, y.copy(), 1 - y)
    assert mc["a_right_b_wrong"] == len(y) and mc["p_value"] < 1e-6


def test_phase5_script_helpers(data, tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("run_phase5", ROOT / "scripts" / "run_phase5.py")
    p5 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p5)
    monkeypatch.setattr(p5, "OUT", tmp_path)
    monkeypatch.setattr(p5, "MODELS", tmp_path / "models")
    params = {"max_features": "sqrt", "min_samples_leaf": 1}
    abl = p5.ablations(data, "B", params, 0)
    assert set(abl.variant) == {"all modules", "module alone", "module alone (local only)", "leave one out"}
    sweep = pd.DataFrame({"module": ["cm_keypoint"], "protocol": ["B"], "pixel_auc": [0.7], "pixel_auc_hit_rate": [0.5]})
    cm = p5.copy_move_study(data, "B", params, 0, sweep)
    assert len(cm) == 5
    s = C.fixed_cv(data, C.module_columns(data.columns), p5.rf_factory(params, 0))["oof"]
    br = p5.per_type_and_bucket(data, s, 0.5)
    assert "authentic false-positive rate" in set(br.group)
    cal = p5.calibration(data, C.module_columns(data.columns), params, s, 0)
    band = p5.choose_band(cal["coverage"])
    assert 0 <= band <= 0.4
    p5.fig_calibration(cal, "B", band)
    imp, by_module = p5.shap_analysis(data, C.module_columns(data.columns), params, 0, "B")
    assert set(by_module.index) == set(C.MODULES)
    monkeypatch.setattr(p5, "resolve", lambda rel: tmp_path / "fp.json" if str(rel).endswith(".json") else rel)
    (tmp_path / "fp.json").write_text('{"code_sha1": "test"}')
    from forgery.config import load_config
    auc = p5.final_models(data, C.module_columns(data.columns), params, 0, "B", band, load_config())
    assert 0 <= auc <= 1 and (tmp_path / "models" / "forgery_B.joblib").exists()
    assert p5.most_common([{"a": 1}, {"a": 2}, {"a": 1}]) == {"a": 1}
