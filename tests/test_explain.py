from pathlib import Path

import numpy as np
import pytest

from forgery import explain as E

MODELS = Path(__file__).resolve().parents[1] / "models"
needs_models = pytest.mark.skipif(not (MODELS / "forgery_R.joblib").exists(), reason="trained models missing")


@pytest.fixture(scope="module")
def models():
    return E.load_models("R")


def test_overlay_only_changes_mask():
    img = np.full((10, 12, 3), 100, np.uint8)
    m = np.zeros((10, 12), bool)
    m[2:4, 3:5] = True
    out = E.overlay(img, m)
    assert (out[~m] == 100).all() and (out[m] != 100).any()
    assert (E.overlay(img, None) == img).all()


@needs_models
def test_every_feature_has_a_readable_description(models):
    for n in models.forgery.feature_names_:
        d = E.describe_feature(n)
        assert ":" in d and "_" not in d.split(":", 1)[1], (n, d)


@needs_models
def test_shap_contributions_add_up(models, tmp_path):
    from PIL import Image
    rng = np.random.default_rng(0)
    x = (rng.random((256, 384, 3)) * 60 + np.linspace(0, 150, 384)[None, :, None]).astype(np.uint8)
    f = tmp_path / "x.png"
    Image.fromarray(x).save(f)
    res = E.analyse(f, models)
    full = E.explain(models, res.features, top=len(models.forgery.feature_names_))
    names = models.forgery.feature_names_
    row = np.array([[res.features[n] for n in names]])
    p_forest = models.forest.predict_proba(row)[0, 1]
    import shap
    base = np.ravel(shap.TreeExplainer(models.forest).expected_value)[-1]
    assert full.contribution.sum() + base == pytest.approx(p_forest, abs=1e-6)
    assert res.verdict in ("tampered", "authentic", "uncertain")
    assert "P(tampered)" in E.verdict_text(res, models)
    pdf = E.pdf_report(res, E.analysed_image(f, "R"), models, full.head(10), "x.png")
    assert pdf[:4] == b"%PDF" and len(pdf) > 10_000
    d = E.report_dict(res, models, full.head(5), "x.png")
    assert d["image"] == "x.png" and len(d["top_evidence"]) == 5


def test_prepare_image_passthrough_and_fixes(tmp_path):
    from PIL import Image
    rng = np.random.default_rng(2)
    rgb = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    Image.fromarray(rgb).save(tmp_path / "ok.png")
    p, notes = E.prepare_image(tmp_path / "ok.png", tmp_path)
    assert p == tmp_path / "ok.png" and notes == []                  # normal files are analysed as they are

    g16 = (np.linspace(0, 65535, 120 * 160).reshape(120, 160)).astype(np.uint16)
    Image.fromarray(g16).save(tmp_path / "g16.png")
    p, notes = E.prepare_image(tmp_path / "g16.png", tmp_path)
    a = np.asarray(Image.open(p))
    assert notes and a.dtype == np.uint8 and a.min() == 0 and a.max() == 255   # not clipped to white

    im = Image.fromarray(rgb)
    ex = im.getexif()
    ex[274] = 6                                                       # rotate 90 degrees
    im.save(tmp_path / "rot.jpg", exif=ex)
    p, notes = E.prepare_image(tmp_path / "rot.jpg", tmp_path)
    assert notes and Image.open(p).size == (120, 160)
    mtime = p.stat().st_mtime_ns
    p2, _ = E.prepare_image(tmp_path / "rot.jpg", tmp_path)       # same input: same file, not rewritten
    assert p2 == p and p2.stat().st_mtime_ns == mtime


@pytest.mark.parametrize("make", ["truncated", "text", "tiny", "huge"])
def test_prepare_image_rejects(tmp_path, make):
    from PIL import Image
    f = tmp_path / "bad.jpg"
    if make == "truncated":
        Image.fromarray(np.random.default_rng(0).integers(0, 255, (300, 300, 3), dtype=np.uint8)).save(f, quality=95)
        f.write_bytes(f.read_bytes()[:3000])
    elif make == "text":
        f.write_text("not an image")
    elif make == "tiny":
        Image.fromarray(np.zeros((40, 50, 3), np.uint8)).save(f)
    else:
        f = tmp_path / "huge.png"
        Image.new("L", (E.MAX_SIDE + 1, 10)).save(f)
    with pytest.raises(E.ImageError):
        E.prepare_image(f, tmp_path)


def test_describe_fixed_texts():
    assert "top 5 %" in E.describe_feature("ela.norm_local_top5_z")
    assert "high-noise side" in E.describe_feature("noise.local_z_max")
    assert "low-noise side" in E.describe_feature("noise.low_local_z_max")
