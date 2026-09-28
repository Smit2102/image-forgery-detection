import io

import numpy as np
from PIL import Image

from forgery.io import (bytes_metadata, estimate_jpeg_quality, file_metadata, ijg_table, is_standard_ijg,
                        jpeg_encode, jpeg_roundtrip, load_image, load_mask, load_rgb)


def test_quality_estimate_recovers_pillow_quality(rgb):
    for q in (50, 75, 85, 90, 95):
        meta = bytes_metadata(jpeg_encode(rgb, q))
        assert meta["jpeg_quality"] == q
        assert meta["qtable_standard"]


def test_nonstandard_table_detected():
    table = np.full(64, 6.0)
    assert not is_standard_ijg(table)
    assert 1 <= estimate_jpeg_quality(table) <= 100


def test_roundtrip_is_deterministic(rgb):
    a, b = jpeg_roundtrip(rgb, 85), jpeg_roundtrip(rgb, 85)
    assert a.dtype == np.uint8 and a.shape == rgb.shape
    assert np.array_equal(a, b)


def test_load_rgb_handles_tiff_bmp_and_modes(tmp_path, rgb):
    for ext, mode in (("tif", "RGB"), ("bmp", "RGB"), ("png", "RGBA"), ("png", "L")):
        p = tmp_path / f"x_{mode}.{ext}"
        Image.fromarray(rgb).convert(mode).save(p)
        out = load_rgb(p)
        assert out.shape == rgb.shape and out.dtype == np.uint8


def test_protocols(tmp_path, rgb, cfg):
    p = tmp_path / "x.tif"
    Image.fromarray(rgb).save(p)
    assert np.array_equal(load_image(p, "A", cfg), rgb)
    b = load_image(p, "B", cfg)
    assert b.shape == rgb.shape and not np.array_equal(b, rgb)
    assert np.array_equal(b, jpeg_roundtrip(rgb, cfg["protocols"]["B"]["jpeg_quality"]))


def test_file_metadata(tmp_path, rgb):
    p = tmp_path / "x.jpg"
    Image.fromarray(rgb).save(p, quality=90)
    m = file_metadata(p)
    assert m["format"] == "JPEG" and m["jpeg_quality"] == 90 and m["width"] == 96
    t = tmp_path / "x.tif"
    Image.fromarray(rgb).save(t)
    assert file_metadata(t)["jpeg_quality"] == -1


def test_load_mask_binarizes(tmp_path):
    m = np.zeros((10, 10), np.uint8)
    m[2:5, 2:5] = 255
    m[0, 0] = 100   # below threshold
    p = tmp_path / "m.png"
    Image.fromarray(m).save(p)
    out = load_mask(p)
    assert out.dtype == bool and out.sum() == 9


def test_ijg_tables_match_libjpeg_for_every_quality(rgb):
    for q in range(1, 101):
        data = jpeg_encode(rgb, q)
        meta = bytes_metadata(data)
        assert meta["qtable_standard"], q
        assert meta["jpeg_quality"] == q
        with Image.open(io.BytesIO(data)) as im:
            assert np.array_equal(np.asarray(im.quantization[0], float), ijg_table(q)), q


def test_block_dct_matches_jpeg_quantisation():
    """A decoded JPEG's DCT coefficients divided by its own table sit close to integers; a wrong
    convention (transposed table, wrong scaling) does not."""
    import cv2
    from forgery.eval.leakage import block_dct_luma
    rng = np.random.default_rng(1)
    img = cv2.GaussianBlur(rng.integers(0, 256, size=(64, 96, 3)).astype(np.uint8), (5, 5), 1.5)
    coef = block_dct_luma(jpeg_roundtrip(img, 75, "4:4:4"))
    table = ijg_table(75).reshape(8, 8)

    def residual(t):
        q = coef / t
        return np.abs(q - np.rint(q))[..., :4, :4].mean()

    correct = residual(table)
    assert correct < 0.08
    assert residual(table.T) > correct + 0.03
    assert residual(np.full((8, 8), 16.0)) > correct + 0.1
    assert residual(table * 2) > correct + 0.1


def test_protocol_r_resamples_image_and_mask(tmp_path, rgb, cfg):
    from forgery.io import protocol_mask
    p = tmp_path / "x.tif"
    Image.fromarray(rgb).save(p)
    r = load_image(p, "R", cfg)
    f = cfg["protocols"]["R"]["factor"]
    assert r.shape == (round(64 * f), round(96 * f), 3)
    mask = np.zeros((64, 96), bool)
    mask[8:40, 16:48] = True
    m = protocol_mask(mask, "R", cfg)
    assert m.shape == r.shape[:2] and m.dtype == bool
    assert abs(m.mean() - mask.mean()) < 0.02
    assert protocol_mask(mask, "B", cfg) is mask
