"""Feature modules: interface contract on awkward inputs + detection of synthetic forgeries.

Each synthetic test plants one kind of manipulation in a known region and checks that the module's
evidence map is clearly higher inside that region than outside.
"""

import cv2
import numpy as np
import pytest

from forgery.features import REGISTRY, build_modules
from forgery.features.common import inconsistency_stats, robust_z, to_evidence
from forgery.io import jpeg_roundtrip

REGION = (slice(64, 128), slice(96, 176))          # y, x of the planted region in a 256 x 320 image


def texture(seed=0, h=256, w=320, noise=0.0) -> np.ndarray:
    """Natural-looking multi-scale texture (uint8 RGB)."""
    rng = np.random.default_rng(seed)
    acc = np.zeros((h, w), np.float32)
    for sigma, amp in ((24, 60), (8, 30), (2, 15)):
        acc += amp * cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), sigma) * sigma / 2
    acc = 128 + 50 * acc / (acc.std() + 1e-6)
    rgb = np.stack([acc, acc * 0.9 + 12, acc * 0.8 + 25], axis=-1)
    if noise:
        rgb += rng.normal(0, noise, rgb.shape)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def inside_vs_outside(e: np.ndarray) -> tuple[float, float]:
    m = np.zeros(e.shape, bool)
    m[REGION] = True
    return float(e[m].mean()), float(e[~m].mean())


def paste(bg: np.ndarray, fg: np.ndarray) -> np.ndarray:
    out = bg.copy()
    out[REGION] = fg[REGION]
    return out


# ---------------------------------------------------------------- contract ---

@pytest.mark.parametrize("name", list(REGISTRY))
@pytest.mark.parametrize("shape", [(256, 320), (37, 53), (8, 8), (200, 13)])
def test_contract_on_awkward_sizes(name, shape):
    img = texture(1, *shape) if min(shape) >= 8 else np.zeros((*shape, 3), np.uint8)
    m = build_modules([name])[0]
    out = m.extract(img)
    out.validate(shape, name)                    # float map in [0, 1] of the right shape, finite features
    assert out.features, name
    assert all(k.split("_", 1)[0] in ("local", "global") or "_local_" in k for k in out.features)


@pytest.mark.parametrize("name", list(REGISTRY))
def test_flat_and_black_images_give_no_evidence(name):
    for value in (0, 128):
        img = np.full((128, 160, 3), value, np.uint8)
        out = build_modules([name])[0].extract(img)
        out.validate((128, 160), name)
        if out.evidence_map is not None:
            assert out.evidence_map.max() <= 1e-6, (name, value)


@pytest.mark.parametrize("name", list(REGISTRY))
def test_deterministic(name):
    img = jpeg_roundtrip(texture(2), 85)
    a, b = build_modules([name])[0].extract(img), build_modules([name])[0].extract(img)
    assert a.features == b.features


def test_inconsistency_stats_detect_outlier_cluster():
    rng = np.random.default_rng(0)
    m = rng.normal(0, 1, (20, 30))
    base = inconsistency_stats(m)
    m[5:9, 10:15] += 8
    hit = inconsistency_stats(m)
    assert hit["local_z_max"] > base["local_z_max"] + 4
    assert hit["local_cluster_z25"] >= 20 / 600 - 1e-9 and hit["local_moran"] > base["local_moran"]
    assert robust_z(np.ones((3, 3))).max() == 0
    e = to_evidence(m, (160, 240), 8)
    assert e.shape == (160, 240) and e.dtype == np.float32 and 0 <= e.min() and e.max() <= 1


# ------------------------------------------------------- synthetic forgeries ---

PARAM_SOURCE = {"which": "default"}


@pytest.fixture(autouse=True, params=["default", "config"])
def param_source(request):
    """Run every synthetic-forgery test with the Phase-3 defaults and with the tuned config.yaml values."""
    PARAM_SOURCE["which"] = request.param
    yield request.param
    PARAM_SOURCE["which"] = "default"


def _evidence(name, img, **params):
    if PARAM_SOURCE["which"] == "config":
        from forgery.config import load_config
        params = {**load_config()["features"].get(name, {}), **params}
    return build_modules([name], {name: params})[0].extract(img)


def test_ela_and_histogram_find_differently_compressed_region():
    src = texture(3, noise=3)
    img = paste(jpeg_roundtrip(src, 60), src)           # region never compressed, background at Q60
    for name in ("ela", "histogram"):
        inside, outside = inside_vs_outside(_evidence(name, img).evidence_map)
        assert inside > outside + 0.1, (name, inside, outside)


def test_noise_finds_noisier_region():
    img = paste(texture(4, noise=2), texture(4, noise=10))
    inside, outside = inside_vs_outside(_evidence("noise", img).evidence_map)
    assert inside > outside + 0.2, (inside, outside)


def test_jpeg_ghost_finds_region_compressed_at_other_quality():
    src = texture(5, noise=3)
    img = jpeg_roundtrip(paste(jpeg_roundtrip(src, 90), jpeg_roundtrip(src, 60)), 95)
    inside, outside = inside_vs_outside(_evidence("jpeg_ghost", img).evidence_map)
    assert inside > outside + 0.1, (inside, outside)


def test_dct_dq_finds_single_compressed_region():
    """Under protocols B / R the last quality is known (last_quality=85): background compressed twice
    (Q60 then Q85), pasted region only once (Q85)."""
    src = texture(6, noise=4)
    img = jpeg_roundtrip(paste(jpeg_roundtrip(src, 60), src), 85)
    out = _evidence("dct_dq", img, last_quality=85)
    assert out.features["global_frac_periodic_freqs"] > 0
    inside, outside = inside_vs_outside(out.evidence_map)
    assert inside > outside + 0.05, (inside, outside)


def test_dct_blind_step_and_null_period():
    from forgery.features.dct_dq import block_dct, estimate_period, estimate_step, window_for
    from forgery.features.common import gray
    from forgery.io import ijg_table
    img = jpeg_roundtrip(texture(9, noise=4), 75)          # single compression, known table
    coef, table = block_dct(gray(img).astype(float)), ijg_table(75).reshape(8, 8)
    hits = [estimate_step(coef[..., u, v].ravel()) == table[u, v] for u, v in ((0, 1), (1, 0), (1, 1), (2, 0))]
    assert sum(hits) >= 3, hits
    rng = np.random.default_rng(0)                         # non-periodic histograms rarely pass
    passes = [estimate_period(np.rint(rng.laplace(0, 3, 4000)).astype(int))[0] > 1 for _ in range(100)]
    assert np.mean(passes) <= 0.1
    assert window_for(2.5, 40) == 5 and window_for(3.0, 40) == 3


def test_edges_find_hard_cut_out_in_soft_image():
    """Same kind of content everywhere (random rectangles); only the planted region is unblurred."""
    rng = np.random.default_rng(7)
    sharp = np.full((256, 320), 128, np.uint8)
    for _ in range(120):
        y, x = rng.integers(0, 240), rng.integers(0, 300)
        sharp[y:y + rng.integers(8, 30), x:x + rng.integers(8, 30)] = rng.integers(30, 225)
    soft = cv2.GaussianBlur(sharp, (0, 0), 1.5)
    img = paste(np.dstack([soft] * 3), np.dstack([sharp] * 3))
    inside, outside = inside_vs_outside(_evidence("edges", img).evidence_map)
    assert inside > outside + 0.1, (inside, outside)


@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("name", ["cm_keypoint", "cm_block"])
def test_copy_move_detects_plain_and_mirrored_copies(name, mirror):
    base = jpeg_roundtrip(texture(8, noise=2), 90)
    patch = base[20:84, 20:100]
    img = base.copy()
    img[REGION] = patch[:, ::-1] if mirror else patch
    img = jpeg_roundtrip(img, 85)
    out = _evidence(name, img)
    key = "local_n_inliers" if name == "cm_keypoint" else "local_max_votes"
    assert out.features[key] >= (4 if name == "cm_keypoint" else 8), out.features
    inside, outside = inside_vs_outside(out.evidence_map)
    assert inside > outside, (inside, outside)
    clean = _evidence(name, base)
    assert clean.features[key] < out.features[key]


def test_period_test_is_calibrated_on_wide_and_heavy_tailed_histograms():
    """QA regression: the old fixed threshold flagged 16-48 % of such non-periodic histograms."""
    from forgery.features.dct_dq import estimate_period
    rng = np.random.default_rng(3)
    draws = [rng.laplace(0, s, 1500) for s in (5, 12) for _ in range(40)]
    flagged = [estimate_period(np.rint(x).astype(int), alpha=0.01, seed=i)[0] > 1 for i, x in enumerate(draws)]
    assert np.mean(flagged) <= 0.06
    x = rng.laplace(0, 15, 8000)                                      # double quantisation 9 -> 4
    assert estimate_period(np.rint(np.rint(x / 9) * 9 / 4).astype(int), alpha=0.01)[0] > 1


def test_blind_step_large_steps():
    """QA regression: steps >= 20 were returned as q + 1 or q + 2."""
    from forgery.features.dct_dq import estimate_step
    rng = np.random.default_rng(4)
    for q in (20, 24, 30):
        c = np.rint(rng.laplace(0, 40, 3000) / q) * q + rng.normal(0, 0.5, 3000)
        assert estimate_step(c) == q


def test_noise_ignores_edge_of_flat_patch():
    """QA regression: blocks straddling a flat, uncompressed patch became strong low-noise outliers."""
    img = jpeg_roundtrip(texture(3, noise=3), 90).copy()
    img[37:101, 53:141] = (128, 120, 110)
    out = _evidence("noise", img)
    assert out.features["low_local_z_max"] < 8
