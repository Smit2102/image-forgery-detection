import json

import numpy as np
import pytest
from PIL import Image

from forgery.features import REGISTRY, FeatureModule, FeatureOutput, build_modules, register
from forgery.pipeline import detect, verdict_from_probability


class _MeanModule(FeatureModule):
    name = "_test_mean"
    category = "point"

    def extract(self, img):
        g = img.mean(axis=2) / 255.0
        return FeatureOutput(evidence_map=g.astype(np.float32), features={"mean": float(g.mean())})


class _BadModule(FeatureModule):
    name = "_test_bad"

    def extract(self, img):
        return FeatureOutput(evidence_map=np.full(img.shape[:2], 2.0), features={})


class _Model:
    feature_names_ = ["_test_mean.mean"]

    def predict_proba(self, x):
        return np.array([[0.1, 0.9]])


@pytest.fixture
def image_path(tmp_path, rgb):
    p = tmp_path / "img.tif"
    Image.fromarray(rgb).save(p)
    return p


def test_detect_runs_modules(image_path, cfg):
    res = detect(image_path, modules=[_MeanModule()], cfg=cfg)
    assert res.shape == (64, 96)
    assert set(res.features) == {"_test_mean.mean"}
    assert res.maps["_test_mean"].shape == (64, 96)
    assert res.verdict is None
    d = json.loads(res.to_json())
    assert d["features"]["_test_mean.mean"] == pytest.approx(res.features["_test_mean.mean"])


def test_detect_with_model(image_path, cfg):
    res = detect(image_path, modules=[_MeanModule()], model=_Model(), cfg=cfg)
    assert res.verdict == "tampered" and res.confidence == pytest.approx(0.9)


def test_invalid_evidence_map_rejected(image_path, cfg):
    with pytest.raises(ValueError, match="must lie in"):
        detect(image_path, modules=[_BadModule()], cfg=cfg)


class _IntMapModule(FeatureModule):
    name = "_test_intmap"

    def extract(self, img):
        return FeatureOutput(evidence_map=np.zeros(img.shape[:2], dtype=np.uint8), features={})


class _StringFeatureModule(FeatureModule):
    name = "_test_strfeat"

    def extract(self, img):
        return FeatureOutput(evidence_map=None, features={"x": "high"})


@pytest.mark.parametrize("module", [_IntMapModule(), _StringFeatureModule()])
def test_contract_violations_raise_value_error(image_path, cfg, module):
    with pytest.raises(ValueError):
        detect(image_path, modules=[module], cfg=cfg)


def test_detect_with_sklearn_estimator(image_path, cfg):
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    X = pd.DataFrame({"_test_mean.mean": [0.1, 0.2, 0.8, 0.9]})
    clf = LogisticRegression().fit(X, [0, 0, 1, 1])
    res = detect(image_path, modules=[_MeanModule()], model=clf, cfg=cfg)
    assert res.verdict in {"tampered", "authentic", "uncertain"} and 0 <= res.confidence <= 1


def test_verdict_band(cfg):
    assert verdict_from_probability(0.9, cfg) == "tampered"
    assert verdict_from_probability(0.5, cfg) == "uncertain"
    assert verdict_from_probability(0.1, cfg) == "authentic"


def test_registry():
    register(_MeanModule)
    try:
        assert [m.name for m in build_modules(["_test_mean"])] == ["_test_mean"]
        with pytest.raises(KeyError):
            build_modules(["does_not_exist"])
    finally:
        REGISTRY.pop("_test_mean", None)


def test_model_meta_mismatch_is_rejected(image_path, cfg):
    from forgery.pipeline import resolved_params

    class _Meta(_Model):
        def __init__(self, meta):
            self.meta = meta

    good = _Meta({"protocol": "B", "feature_params": resolved_params(cfg, "B")})
    detect(image_path, protocol="B", modules=[_MeanModule()], model=good, cfg=cfg)
    with pytest.raises(ValueError, match="protocol"):
        detect(image_path, protocol="R", modules=[_MeanModule()], model=good, cfg=cfg)
    bad = _Meta({"protocol": "B", "feature_params": {**resolved_params(cfg, "B"), "ela": {"quality": 1}}})
    with pytest.raises(ValueError, match="feature parameters"):
        detect(image_path, protocol="B", modules=[_MeanModule()], model=bad, cfg=cfg)


def test_detect_with_localizer(image_path, cfg):
    from forgery.localize.fusion import Localizer
    loc = Localizer(["_test_mean"], None, threshold=0.5)          # mean of maps, no trained model
    res = detect(image_path, modules=[_MeanModule()], cfg=cfg, localizer=loc)
    assert res.mask is not None and res.mask.shape == res.shape and res.mask.dtype == bool
    assert res.maps["fused"].shape == res.shape
    res = detect(image_path, modules=[_MeanModule()], cfg=cfg, localizer=loc, model=_Model())
    assert res.verdict == "tampered" and res.mask is not None


def test_localization_helpers():
    from forgery.localize.fusion import cell_features, cells, postprocess, scores, upsample
    m = np.zeros((64, 96), np.float32)
    m[16:48, 32:64] = 1.0
    c = cells(m)
    assert c.shape == (8, 12) and c[3, 5] == 1.0 and c[0, 0] == 0.0
    up = upsample(c, (64, 96))
    assert up.shape == (64, 96)
    mask = postprocess(up, 0.5, 3, 3, 0.001)
    s = scores(mask, m > 0.5)
    assert s["f1"] > 0.9 and s["iou"] > 0.8
    assert cell_features(np.stack([c, c])).shape == (96, 4)
    assert scores(np.zeros((4, 4), bool), np.ones((4, 4), bool))["f1"] == 0.0


def test_mask_zeroed_when_verdict_authentic(image_path, cfg):
    from forgery.localize.fusion import Localizer

    class _Low(_Model):
        def predict_proba(self, x):
            return np.array([[0.95, 0.05]])

    loc = Localizer(["_test_mean"], None, threshold=0.0)             # would mark everything
    res = detect(image_path, modules=[_MeanModule()], cfg=cfg, localizer=loc, model=_Low())
    assert res.verdict == "authentic" and not res.mask.any() and not res.maps["fused"].any()


def test_upsample_non_multiple_of_cell():
    from forgery.localize.fusion import cells, upsample
    m = np.zeros((67, 101), np.float32)
    m[16:40, 24:56] = 1.0
    up = upsample(cells(m), m.shape)
    assert up.shape == (67, 101)
    ys, xs = np.nonzero(up > 0.5)
    assert abs(ys.mean() - 27.5) < 1 and abs(xs.mean() - 39.5) < 1
