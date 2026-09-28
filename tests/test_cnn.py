import numpy as np
import pytest
import torch

from forgery import cnn

CPU = torch.device("cpu")


@pytest.mark.parametrize("shape", [(120, 90), (192, 192), (256, 384), (193, 500)])
def test_tiles_cover_whole_image(shape):
    h, w = shape
    x = np.arange(h * w * 3, dtype=np.int64).reshape(h, w, 3) % 251 + 1
    t = cnn.tiles(x.astype(np.uint8))
    assert t.shape[1:] == (cnn.CROP, cnn.CROP, 3)
    covered = np.zeros((max(h, cnn.CROP), max(w, cnn.CROP)), bool)
    ys, xs = cnn._starts(max(h, cnn.CROP)), cnn._starts(max(w, cnn.CROP))
    assert len(t) == len(ys) * len(xs)
    for y in ys:
        for c in xs:
            covered[y:y + cnn.CROP, c:c + cnn.CROP] = True
    assert covered.all()


def test_pad_is_zero_and_keeps_content():
    x = np.full((100, 150, 3), 7, np.uint8)
    p = cnn.pad_to_crop(x)
    assert p.shape == (cnn.CROP, cnn.CROP, 3)
    assert (p[:100, :150] == 7).all() and p[100:].max() == 0 and p[:, 150:].max() == 0


def test_random_crop_shape_and_bounds():
    rng = np.random.default_rng(0)
    for shape in [(100, 100, 3), (300, 200, 3), (192, 192, 3)]:
        assert cnn.random_crop(np.ones(shape, np.uint8), rng).shape == (cnn.CROP, cnn.CROP, 3)


class _SumModel(torch.nn.Module):
    """Tile logit = mean pixel value of the (normalised) tile: lets us check the pooling exactly."""

    def forward(self, x):
        return x.mean(dim=(1, 2, 3)).unsqueeze(1)


def test_predict_pools_per_image():
    rng = np.random.default_rng(1)
    images = [rng.integers(0, 255, s + (3,), dtype=np.uint8) for s in [(150, 150), (400, 300), (192, 600)]]
    out = cnn.predict(_SumModel(), images, CPU, batch=5)          # batch smaller than tiles: several flushes
    for i, x in enumerate(images):
        z = _SumModel()(cnn.to_input(cnn.tiles(x), CPU)).squeeze(1).numpy()
        assert out["mean"][i] == pytest.approx(z.mean(), rel=1e-5)
        assert out["max"][i] == pytest.approx(z.max(), rel=1e-5)


def test_ela_zero_for_stable_image():
    x = np.full((64, 64, 3), 128, np.uint8)                       # flat grey survives JPEG exactly
    assert cnn.ela_image(x).max() <= cnn.ELA_SCALE


def test_model_outputs_one_logit():
    m = cnn.build_model(pretrained=False).eval()
    with torch.no_grad():
        assert m(torch.zeros(2, 3, cnn.CROP, cnn.CROP)).shape == (2, 1)


def test_predict_batch_size_invariant_and_unpadded_variant():
    rng = np.random.default_rng(2)
    images = [rng.integers(0, 255, s + (3,), dtype=np.uint8) for s in [(100, 80), (300, 250), (150, 400), (60, 60)]]
    ref = cnn.predict(_SumModel(), images, CPU, batch=256)
    for b in (1, 3, 7):
        out = cnn.predict(_SumModel(), images, CPU, batch=b)
        for a in cnn.AGGREGATORS:
            np.testing.assert_allclose(out[a], ref[a], rtol=1e-5)
    unp = cnn.predict(_SumModel(), images, CPU, batch=3, pad=False)
    for i, x in enumerate(images):
        z = _SumModel()(cnn.to_input(cnn.tiles(x, pad=False), CPU)).squeeze(1).numpy()
        assert unp["max"][i] == pytest.approx(z.max(), rel=1e-5)
        if not cnn.is_padded(x.shape):                             # large images: identical to the default
            assert unp["mean"][i] == pytest.approx(ref["mean"][i], rel=1e-5)
    t = cnn.tiles(images[3], pad=False)
    assert t.shape == (1, 60, 60, 3)
    assert cnn.n_tiles(images[1].shape) == len(cnn.tiles(images[1]))


def test_random_crop_flips_about_half():
    rng = np.random.default_rng(3)
    x = np.zeros((192, 192, 3), np.uint8)
    x[:, 0] = 1
    flips = sum(cnn.random_crop(x, rng)[0, -1, 0] == 1 for _ in range(400))
    assert 150 < flips < 250


def test_ela_png_roundtrip(tmp_path):
    from PIL import Image
    x = np.random.default_rng(4).integers(0, 255, (70, 90, 3), dtype=np.uint8)
    e = cnn.ela_image(x)
    Image.fromarray(e).save(tmp_path / "e.png")
    assert (cnn.load_ela(tmp_path / "e.png") == e).all()
