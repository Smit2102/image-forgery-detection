# V. Method: Forensic Feature Modules

*Code: `src/forgery/features/`. Figures: `results/phase3/fig_modules_overview_{B,R}.png`, `fig_ela_before_after.png`, `fig_dct_histogram.png`, `fig_copymove_matches.png`. Final parameters: Section VI.*

## A. Common design

Every module maps an RGB image to two outputs:
- an **evidence map** (float, [0, 1], image-sized; bright = suspicious), used for localisation
- a set of **scalar features**, used for image-level classification

Section IV showed that *global* image statistics on CASIA v2.0 largely measure compression history rather than tampering. Each module therefore reduces its evidence to a *block map* and describes it with **within-image inconsistency** statistics (`local_*`). All of these are scale-free, so they compare a region with the rest of the same image rather than with other images:
- the robust z-score of the most extreme block, and its 99th percentile; the z-score is (value − median) / (1.4826·MAD)
- the share of blocks with z > 3
- the size of the largest connected cluster of blocks with z > 2.5
- Moran's I spatial autocorrelation
- `local_top5_z`: the mean robust z-score of the top 5 % of blocks, which stays scale-free for signed and log-valued maps

Absolute image-wide levels are kept separately as `global_*` features, so Phase 5 can test whether they act as shortcuts.

**Flat and clipped regions carry no evidence.** A textured object on a plain black background has, trivially, a different error level and noise level from its surroundings. Blocks whose grey-level standard deviation is below a threshold, or whose mean is within 4 grey levels of black or white, are therefore excluded from the statistics and receive zero evidence. This was added after the Phase-3 sanity check showed four modules lighting up on an authentic studio photograph. All modules, including the noise module, use the same validity mask; before the QA fixes the noise module did not apply it.

## B. The eight modules

| # | Module | DIP category | Evidence | Main parameters |
|---|---|---|---|---|
| 1 | **Error Level Analysis** | point processing | \|I − JPEG_q(I)\| averaged per 8×8 block, divided by (block texture + c) so busy regions are not mistaken for tampering | q, block, c |
| 2 | **Patch-histogram χ²** | histogram processing | ELA residual contrast-stretched and **histogram-equalised**; χ² distance of each patch histogram to the image's reference histogram (the mean over textured patches) | q, patch, bins |
| 3 | **Noise level** | spatial filtering | Immerkaer high-pass kernel (or a 3×3 Laplacian); per-block robust σ = 1.4826 · median\|r\| over **non-edge pixels** (gradient ≤ a global quantile), since edges leak content into the residual; flat and clipped blocks are excluded via the common validity mask | kernel, block, edge quantile |
| 4 | **JPEG ghosts** (Farid 2009) | point processing | per-block squared re-compression error at Q = 50 … 95; each block's curve is min-max normalised; evidence = L1 distance from the image's median curve | block |
| 5 | **DCT double quantisation** (after Lin et al. 2009) | frequency domain | per 8×8 DCT frequency: last quantisation step (known or estimated, see below), test of the coefficient histogram for periodicity, and per-coefficient posterior of *not* following the periodic pattern; block log-odds averaged over frequencies | number of frequencies, period threshold |
| 6 | **Edge sharpness** | edge detection | Canny edges; at each edge pixel, sharpness = gradient / local contrast (5×5 max − min); block median; both over-sharp (hard cut-out) and over-soft (feathered) outliers count | Canny thresholds, block |
| 7 | **Copy-move: keypoints** | corner/keypoint detection | SIFT (or Harris) keypoints with SIFT descriptors; g2NN matching of the image against itself **and against its mirror image**; pairs < 20 px apart dropped; RANSAC affine fit, separately for direct and mirrored matches | detector, ratio, contrast |
| 8 | **Copy-move: blocks** | frequency domain | overlapping 16×16 blocks described by 9 low-frequency DCT coefficients; the blocks of the image and of its mirror are sorted lexicographically together; near-identical neighbours are paired and their shift vectors voted on | tolerance, min. block std, min. votes |

**DCT double quantisation.** Under protocols B and R the last compression is known (Q85), so the module uses the exact IJG Q85 quantisation table. A blind estimate of the last quantisation step is used only when it is unknown (protocol A, or arbitrary input files). Periodicity is tested on the log-histogram of each frequency's coefficients after removing a quadratic trend. The residual is projected on every candidate period from 2 to half the histogram range, including fractional periods such as q₁/q₂ = 2.5 bins. The largest normalised power is compared with its **permutation null**: the same maximum computed after shuffling the residual across bins, 200 times per histogram. This gives a calibrated p-value whatever the histogram's width or tails, and however many periods are tried. Periodicity is declared at p ≤ α, with α tuned in Section VI. On QA's test cases the false-positive rate at α = 0.01 is 0–3 % (0.4 % on real history-free images), against 16–48 % for the earlier fixed threshold. Genuinely double-compressed images are flagged at 75 % of frequencies. The per-coefficient posterior compares the observed histogram with a uniform one inside a window spanning a whole number of periods. The blind step estimator picks the largest step that is a local maximum of the integer-fit score and reaches 80 % of what pixel-rounding noise allows (σ = 0.5, relaxed to 1.0 for saturated images). When the first step is an exact multiple of the last (e.g. 9 and 3), double compression cannot be detected blindly at that frequency.

**Mirrored copies.** Several CASIA copy-moves are horizontally flipped duplicates, which neither SIFT descriptors nor block DCT features match directly. Both detectors therefore also match against the mirrored image. For keypoints, descriptors are recomputed at the mirrored keypoint positions. For blocks, the shift measured in (image, mirror) coordinates is constant for a mirrored copy, so ordinary shift voting still applies; the mirrored image's block grid is offset so that it aligns with the image's block grid.

**Pair orientation.** A translation has a direction, so direct keypoint pairs are oriented left-to-right before RANSAC. Unoriented pairs split one true copy into two opposite transforms, and RANSAC then rejected genuine copies. Mirrored pairs are oriented left-to-right as well: a mirrored copy with a vertical offset is a glide reflection, whose inverse is a different transform, so the same split would occur.

**Limitation.** One RANSAC fit is run per matching mode (direct and mirrored), so the keypoint detector recovers at most one copied region per mode.

## C. Verification

Each module has a unit test (`tests/test_features.py`) on a **synthetic forgery with a known region**:
- ELA and histogram: an uncompressed region inside a Q60 image
- noise: a noisier region
- JPEG ghost: a region previously saved at Q60 inside a Q90 image, re-saved at Q95
- DCT: a single-compressed region inside a double-compressed image
- edges: an unblurred region in a blurred image of the same content
- copy-move: plain and mirrored copies

In every case the evidence map is clearly higher inside the planted region, and a clean image yields no copy-move votes. Contract tests cover tiny (8×8), narrow (200×13), flat and black images, and check determinism. The synthetic tests run both with the Phase-3 defaults and with the tuned parameters from `config.yaml`.
