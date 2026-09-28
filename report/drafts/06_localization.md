# VIII. Localisation

*Source: `results/phase6/localization.md`, `methods_{R,B}.csv`, `per_image_{R,B}.csv`, `summary.json`, `gating.{md,json}`. Figures: `fig_methods_{R,B}.png`, `fig_examples_{R,B}.png`. Code: `src/forgery/localize/fusion.py`, `scripts/extract_maps.py`, `scripts/run_phase6.py`, `scripts/phase6_gating.py`.*

## A. Method

The eight evidence maps (Section V) are reduced to 8×8-pixel **cells**. A cell is described by 16 features: the eight evidence values and their 3×3-cell neighbourhood means, which give each cell some context. Three ways of fusing them are compared:
- the **mean** of the eight maps (no training)
- **logistic regression** on the 16 cell features
- **gradient boosting** (histogram-based, 300 iterations) on the 16 cell features

The two learned fusions are trained on cells of *training* images: every tampered training image with a valid mask, plus 1,000 authentic training images. A cell is labelled tampered when at least half of it lies inside the mask. Up to 400 tampered and 400 untampered cells are sampled per image, about 2.1–2.3 million cells in all.

The cell probability map is upsampled bilinearly to full resolution and then cleaned in three steps:
1. threshold at *t*
2. morphological **opening**, then **closing**, with elliptical kernels
3. removal of connected components smaller than a minimum share of the image

The post-processing parameters (t ∈ 0.05…0.95; opening ∈ {1, 5, 9}; closing ∈ {1, 9, 17}; minimum area ∈ {0, 0.2 %, 1 %}) are tuned to maximise mean pixel F1. The threshold is chosen first, without morphology; the morphology settings are then chosen at that threshold.

**Honest evaluation.** The validation images are split into two halves by split group. Post-processing is tuned on one half and scored on the other, and then the halves swap. The per-method numbers are therefore out-of-sample, and the fusion models never see validation images. One caveat: the main method and the "best single module" are picked by these same numbers, which slightly flatters both. The margins between the learned fusions are small, so this matters little.

**Metrics.** For every tampered validation image with a valid mask (769 images), we compute pixel IoU, F1 and MCC of the final mask, and the threshold-free pixel ROC-AUC of the probability map (computed on every second pixel). For authentic validation images (1,124), we report the share of images with any predicted region and the mean predicted area. Baselines:
- each single module's map, with the same post-processing tuning
- **whole image**: marking everything as tampered, which gives F1 = 2a / (1 + a) for tampered share a
- the **oracle** F1: the best threshold chosen separately for each image, without morphology. This shows how much a perfect per-image threshold could gain. It is a reference point, not a strict upper bound: post-processing beats it on some images.

## B. Results

**Table V: Localisation on tampered validation images (mean per image; out-of-sample post-processing)**

| Method | R: F1 | R: IoU | R: MCC | R: pixel AUC | B: F1 | B: IoU | B: MCC | B: pixel AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Gradient-boosted fusion** | **0.185** | 0.119 | **0.159** | **0.695** | **0.260** | **0.176** | **0.254** | **0.806** |
| Logistic-regression fusion | 0.184 | 0.121 | 0.155 | 0.686 | 0.237 | 0.158 | 0.224 | 0.786 |
| Mean of the 8 maps | 0.159 | 0.099 | 0.100 | 0.671 | 0.175 | 0.113 | 0.150 | 0.734 |
| Best single module (copy-move keypoints) | 0.152 | 0.102 | 0.130 | 0.620 | 0.179 | 0.121 | 0.158 | 0.644 |
| Whole image | 0.136 | 0.086 | – | – | 0.136 | 0.086 | – | – |
| *Oracle threshold (gradient boosting)* | *0.308* | – | – | – | *0.436* | – | – | – |

95 % group-bootstrap CIs for the gradient-boosted fusion:
- R: F1 [0.168, 0.200], IoU [0.108, 0.130], MCC [0.140, 0.176]
- B: F1 [0.237, 0.282], IoU [0.160, 0.193], MCC [0.230, 0.277]

**Findings.**
1. **Learned fusion is clearly better than any single cue.** Under R, gradient boosting reaches F1 0.185 against 0.152 for the best single module (copy-move keypoints) and 0.159 for the plain average. Under B the gap is wider (0.260 vs 0.179). Under R, no single module beats the trivial "whole image" baseline by much, while the combination does. Under B, copy-move keypoints alone already beat that baseline clearly.
2. **Protocol B localises better than R** (F1 0.260 vs 0.185; pixel AUC 0.806 vs 0.695). Under B the compression modules (DCT, ghosts) still see the pasted region's different compression history, and resampling removes most of it. For localisation this within-image evidence is genuine (Section VI, observation 5), unlike the global shortcut of Section IV.
3. **Absolute scores are low.** This is typical of classical methods on CASIA v2.0, where the median tampered region covers 3 % of the image. The oracle F1 (0.31 under R, 0.44 under B) shows that a perfect per-image threshold would raise F1 by about 1.7×. This suggests that much of the gap lies in choosing the threshold for each image rather than in how the pixels are ranked.
4. **By tampered area (R):**

   | Tampered area | F1 | Pixel AUC |
   |---|---:|---:|
   | < 1 % | 0.034 | 0.617 |
   | 1–5 % | 0.162 | 0.736 |
   | > 5 % | 0.309 | 0.706 |

   Very small regions are essentially not localised, although their pixel AUC is still above chance. Regions above 5 % reach F1 0.31. Under B the same buckets reach 0.10, 0.27 and 0.35.
5. **By forgery type (R):** copy-move F1 0.204, splicing 0.152 (B: 0.302 and 0.184), consistent with image-level detection (Section VII).
6. **Failure modes** (`fig_examples_R.png`):
   - tiny regions (< 1 %), where the probability peak lands elsewhere
   - large uniform pasted areas (sky, water), which carry little texture for any module
   - textured authentic regions that attract spurious evidence (which modules cause this is not broken down here)

## C. Gating with the image classifier

On authentic images the fused mask is rarely empty. Under R, with the final (deployed) post-processing, gradient boosting marks some region on 89 % of authentic validation images, with a mean area of 5.6 %. The out-of-sample figures in `methods_R.csv` are almost the same (88.8 %, 5.7 %). `detect()` therefore shows the mask only when the image-level verdict (Section VII) is not "authentic". `detect()` gates on the *calibrated* probability. Here the out-of-fold forest scores of the validation images are mapped through the deployed model's isotonic calibrator, then gated with the deployed band (`gating.md`). This approximates fully out-of-fold calibrated scores; the calibrator itself was fitted on grouped out-of-fold scores of the whole dev set, validation included.

| | Authentic images with a region: ungated → gated | Mean false area: ungated → gated | Tampered images shown | Tampered F1: ungated → gated |
|---|---|---|---:|---|
| R (band ± 0.24) | 89 % → **42 %** | 5.6 % → 3.7 % | 84 % | 0.186 → 0.175 |
| B (band ± 0.04) | 71 % → **10 %** | 2.5 % → 0.7 % | 76 % | 0.261 → 0.225 |

Gating removes a large share of false regions at a small cost in tampered-image F1: images judged authentic by mistake lose their mask. The gated tampered F1 values use the final post-processing, which was tuned on all validation images, so they are slightly optimistic compared with Table V.

## D. Deployed localiser

`models/localizer_{R,B}.joblib` holds the gradient-boosting fusion (trained on training cells) and the post-processing tuned on all validation images:
- R: t = 0.60, no opening, no closing, minimum area 0.2 %
- B: t = 0.70, no opening, closing 17, minimum area 0.2 %

`detect.py` writes `mask.png` and `overlay.png` next to `report.json`. The localiser reuses the evidence maps that `detect()` already computes, so localisation adds almost no run time.

## E. Limitations

- **Tuning and evaluation share validation images.** Post-processing tuning and evaluation use disjoint halves of the same validation set. The final deployed parameters are tuned on all of it, so only the test set (Phase 7) gives a fully independent estimate.
- **Cell resolution.** The 8×8-pixel cells limit boundary accuracy (not quantified separately).
- **Images without a valid mask.** Tampered images without a valid mask (19 in the whole dataset, 1 of them in the validation set) are excluded from localisation scoring.
