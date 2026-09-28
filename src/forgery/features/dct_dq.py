"""Localised DCT double quantisation (frequency domain), after Lin et al. (2009).

The image is split into the 8x8 blocks of the JPEG grid and transformed with the DCT. For each of the
first low-frequency AC positions:
  1. coefficients are divided by the last quantisation step. Under protocols B and R the last
     compression is known (the pipeline re-encodes at a fixed quality), so the exact IJG table is
     used (`last_quality`). Otherwise the step is estimated blindly: the largest step at which the
     coefficients sit on integer multiples as well as rounding noise allows;
  2. the histogram of quantised values is tested for periodicity: the log-histogram minus a smooth
     quadratic trend is projected on every candidate period (fractional, 2 .. kmax/2), and the largest
     normalised power is compared with its permutation null (residual shuffled across bins), giving a
     calibrated p-value; periodicity is declared at p <= alpha.
     Double compression with different steps leaves periodic peaks and gaps, while content that did
     not share that history, such as a pasted region on a misaligned grid, fills the histogram uniformly;
  3. every coefficient gets a posterior probability of *not* following the periodic pattern:
     P_t = 1/p against P_u = h(k) / (sum of h over the p bins around k).
Per-block log-odds are averaged over the positions. High values = likely tampered.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dctn

from ..io import ijg_table
from .base import FeatureModule, FeatureOutput, register
from .common import gray, inconsistency_stats, to_evidence

ZIGZAG = [(0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2), (2, 1), (3, 0), (4, 0), (3, 1), (2, 2),
          (1, 3), (0, 4), (0, 5), (1, 4), (2, 3), (3, 2), (4, 1), (5, 0)]


def block_dct(g: np.ndarray) -> np.ndarray:
    """Orthonormal 8x8 DCT of (Y - 128) on the JPEG grid: shape (nby, nbx, 8, 8)."""
    h, w = (g.shape[0] // 8) * 8, (g.shape[1] // 8) * 8
    b = (g[:h, :w] - 128.0).reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3)
    return dctn(b, axes=(2, 3), norm="ortho")


RESIDUAL_SIGMA = 0.5   # rounding noise of decoded pixels, in DCT units (Y <- RGB <- Y)


def step_score(c: np.ndarray, q: int, min_count: int = 50) -> float:
    """Mean cos(2 pi c / q) over coefficients that quantise to a non-zero value; NaN if too few."""
    nz = c[np.abs(c) >= 0.5 * q]
    return float(np.cos(2 * np.pi * nz / q).mean()) if nz.size >= min_count else np.nan


def estimate_step(c: np.ndarray, max_step: int = 60, ratio: float = 0.8, min_count: int = 50,
                  sigmas: tuple[float, ...] = (0.5, 1.0)) -> int:
    """Blind estimate of the last quantisation step (1 if none).

    A coefficient quantised with step q and then passed through pixel rounding (noise sigma in DCT
    units) scores E[cos] = exp(-2 pi^2 sigma^2 / q^2). The last step is the largest q that is a
    *local maximum* of the score (beats q - 1 and q + 1, so q + 1 is never taken for q) and reaches
    `ratio` of that expectation. Divisors of the step also qualify but are smaller; an earlier, coarser
    step of a double-compressed image only partly fits and falls below the ratio. The noise level is
    tried at 0.5 first and relaxed to 1.0 (saturated images) only if no step qualifies.
    """
    s = {q: step_score(c, q, min_count) for q in range(1, max_step + 2)}
    val = {q: (v if np.isfinite(v) else -1.0) for q, v in s.items()}
    for sigma in sigmas:
        for q in range(max_step, 1, -1):
            if val[q] < -0.5:
                continue
            local_max = val[q] > val[q + 1] and (q == 2 or val[q] > val[q - 1])
            if local_max and val[q] >= ratio * np.exp(-2 * np.pi ** 2 * sigma ** 2 / q ** 2):
                return q
    return 1


_BASIS_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def _period_basis(kmax: int, max_period: int) -> tuple[np.ndarray, np.ndarray]:
    key = (kmax, max_period)
    if key not in _BASIS_CACHE:
        periods = np.round(np.arange(2.0, min(max_period, kmax / 2) + 1e-9, 0.1), 1)
        j = np.arange(1, kmax + 1, dtype=np.float64)
        _BASIS_CACHE[key] = (periods, np.exp(-2j * np.pi * np.outer(1 / periods, j)))
    return _BASIS_CACHE[key]


def estimate_period(k: np.ndarray, max_period: int = 16, alpha: float = 0.01, n_perm: int = 200,
                    min_count: int = 200, seed: int = 0) -> tuple[float, float, np.ndarray]:
    """Period p of the |k| histogram (1 if none); p may be fractional (e.g. q1 / q2 = 2.5 bins).

    Residual r = log(1 + h) minus a quadratic trend. Statistic = the maximum over candidate periods
    p in [2, kmax / 2] (step 0.1) of |sum r_j exp(-2 pi i j / p)|^2 / sum r_j^2. Its null
    distribution is obtained per histogram by permuting r across bins (`n_perm` times), so the test
    is calibrated whatever the histogram's width or tails and however many periods are tried.
    Returns (p, p-value-based strength 1 - p_value, h); p = 1 when the permutation p-value > alpha.
    """
    a = np.abs(k[k != 0])
    if a.size < min_count:
        return 1, 0.0, np.zeros(0)
    kmax = int(min(60, max(9, np.percentile(a, 99))))
    h = np.bincount(np.clip(a, 0, kmax + 1), minlength=kmax + 2)[1:kmax + 1].astype(np.float64)
    j = np.arange(1, kmax + 1, dtype=np.float64)
    y = np.log1p(h)
    r = y - np.polyval(np.polyfit(j, y, 2), j)
    denom = float((r * r).sum())
    periods, basis = _period_basis(kmax, max_period)
    if denom <= 1e-12 or periods.size == 0:
        return 1, 0.0, h
    power = np.abs(basis @ r) ** 2 / denom
    best = int(power.argmax())
    rng = np.random.default_rng(seed)
    perms = np.stack([rng.permutation(r) for _ in range(n_perm)], axis=1)       # (kmax, n_perm)
    null_max = (np.abs(basis @ perms) ** 2 / denom).max(axis=0)
    p_value = (1 + int((null_max >= power[best]).sum())) / (n_perm + 1)
    if p_value > alpha:
        return 1, 1 - p_value, h
    return float(periods[best]), 1 - p_value, h


def window_for(p: float, kmax: int) -> int:
    """Smallest whole number of bins (<= kmax) spanning an (almost) whole number of periods of length p."""
    for m in range(1, 11):
        w = round(p * m)
        if w > kmax:
            break
        if abs(p * m - w) <= 0.15 and w >= 2:
            return int(w)
    return int(min(max(2, round(p)), max(2, kmax)))


def tamper_posterior(k: np.ndarray, h: np.ndarray, p: int) -> np.ndarray:
    """P(coefficient does not follow the periodic pattern) for each quantised value k (0 -> 0.5).

    `p` is the window length in bins (a whole number of periods, see window_for): content that follows
    the pattern is distributed like h inside the window, content that does not is uniform (1 / p)."""
    a = np.abs(k)
    kmax = h.size
    post = np.full(k.shape, 0.5)
    ok = (a >= 1) & (a <= kmax)
    # sum of h over a window of p bins centred on each bin
    hp = np.pad(h, (p, p))
    csum = np.concatenate([[0.0], np.cumsum(hp)])
    idx = np.arange(kmax) + p                      # position of bin |k|=i+1 in hp
    lo, hi = idx - p // 2, idx - p // 2 + p
    window = csum[hi] - csum[lo]
    p_u = h / np.maximum(window, 1.0)
    p_t = 1.0 / p
    bin_post = p_t / (p_t + p_u)
    post[ok] = bin_post[a[ok] - 1]
    return post


@register
class DCTDoubleQuant(FeatureModule):
    name = "dct_dq"
    category = "frequency"

    def __init__(self, n_freq: int = 9, alpha: float = 0.01, last_quality: int | None = None):
        super().__init__(n_freq=n_freq, alpha=alpha, last_quality=last_quality)
        self.n_freq, self.alpha, self.last_quality = n_freq, alpha, last_quality
        self._table = None if last_quality is None else ijg_table(int(last_quality)).reshape(8, 8)

    def maps(self, img: np.ndarray) -> dict:
        coef = block_dct(gray(img).astype(np.float64))
        nby, nbx = coef.shape[:2]
        logodds, informative = np.zeros((nby, nbx)), np.zeros((nby, nbx))
        steps, periods = [], []
        for u, v in ZIGZAG[: self.n_freq]:
            c = coef[..., u, v]
            q = int(self._table[u, v]) if self._table is not None else estimate_step(c.ravel())
            k = np.rint(c / q).astype(np.int64)
            p, _, h = estimate_period(k.ravel(), alpha=self.alpha)
            steps.append(q)
            periods.append(p)
            if p == 1:
                continue
            post = np.clip(tamper_posterior(k, h, window_for(p, h.size)), 0.02, 0.98)
            use = (k != 0) & (np.abs(k) <= h.size)          # values beyond the histogram carry no evidence
            logodds += np.where(use, np.log(post / (1 - post)), 0.0)
            informative += use
        score = np.where(informative > 0, logodds / np.maximum(informative, 1), np.nan)
        return {"score": score, "informative": informative, "steps": steps, "periods": periods}

    def extract(self, img: np.ndarray) -> FeatureOutput:
        m = self.maps(img)
        score = m["score"]
        valid = np.isfinite(score)
        n_periodic = sum(p > 1 for p in m["periods"])
        feats = {**inconsistency_stats(score, valid=valid),
                 "local_frac_blocks_positive": float((score[valid] > 0).mean()) if valid.any() else 0.0,
                 "global_frac_periodic_freqs": n_periodic / len(m["periods"]),
                 "global_mean_step": float(np.mean(m["steps"]))}
        if score.size < 4 or not valid.any():
            return FeatureOutput(np.zeros(img.shape[:2], np.float32), feats)
        return FeatureOutput(to_evidence(np.where(valid, score, np.nan), img.shape[:2], 8, valid=valid), feats)
