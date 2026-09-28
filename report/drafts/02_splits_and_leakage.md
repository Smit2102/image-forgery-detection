# IV. Splits, Protocols and the Shortcut Study

*Source numbers: `results/phase2/split_summary.{md,json}`, `results/phase2/leakage_table.md`, `results/phase2/leakage_summary.csv`. Figure: `results/phase2/fig_leakage.png`.*

## A. Leakage-safe splits

A tampered image shares most of its pixels with its host image. If the host is in the training set and the tampered version in the test set, the evaluation is optimistic. We therefore split by **groups**, built with union-find over four kinds of link:
- the host id
- exact pixel duplicates (196 merges)
- verified near duplicates (17 merges from 694 pairs)
- the second-named id, for the 23 images where that image supplies most of the pixels (large pastes or swapped ids; 16 merges)

A near-duplicate candidate needs a 64-bit dHash Hamming distance ≤ 8, and is confirmed only if the 32×32 grey thumbnails have a mean absolute difference < 5 and a correlation > 0.95. dHash alone matched many unrelated photos with smooth gradients; the thumbnail check removes those. The result is **7,262 split groups** of at most 69 images.

We use stratified group k-fold assignment (20 bins, seed 42) with strata of forgery type × file format. This gives **train 8,828 / val 1,894 / test 1,892 images (70.0/15.0/15.0%)**. Every stratum is within 0.23 percentage points of its target share, except the 54-image BMP stratum, which is within 1.67. The train+val ("dev") set is further divided into five grouped CV folds. No group crosses a split or a fold; this is asserted before the split is written and checked again by the test suite.

The split is frozen: SHA-256 hashes of the test file list and of the whole split table are stored, and the script refuses to change them without an explicit `--refreeze`. We re-froze once during Phase 2, before any model had seen the test set, after QA review corrected the forgery-type labels and added the donor-dominant links. The test set stays unused until the final evaluation.

One limitation remains. For 830 tampered images (828 splicing), the donor image is in a different split. The shared content is the pasted region: a median of 5.5% of the image, and 34.5% at the 90th percentile. We report this rather than prevent it. Merging every donor would create a single group of 4,027 images (31.9% of the dataset), which makes a stratified 70/15/15 split impossible.

## B. Evaluation protocols

- **A: as released.** Images exactly as distributed.
- **C: JPEG-only.** Protocol A restricted to JPEG files of both classes (a common "fix" in the literature).
- **B: re-encoded (Q=85).** Every image is decoded and re-encoded once to JPEG at quality 85, with 4:2:0 chroma subsampling. All images then share one container format and one quantisation table.
- **R: resampled + re-encoded ("strict").** Every image is downscaled by 0.75 with area interpolation, then re-encoded as in B. Masks are resized with nearest-neighbour interpolation.

## C. Shortcut study

We train a random forest (300 trees, balanced class weights) on feature families computed **globally over the whole image**. None of them localises anything:
- **file metadata:** format flags, quantisation-table statistics, standard-table flag, bytes per pixel
- **image dimensions:** width, height, aspect ratio, megapixels
- **naive global ELA:** mean, standard deviation and 95th/99th percentiles of one ELA residual at Q=90
- **compression history:**
  - global double-quantisation statistics: histograms of 8×8-DCT coefficients at five low frequencies, divided by the last quantisation step
  - a global JPEG-ghost curve: mean re-compression error at Q = 60, 65, …, 95

Scores are balanced accuracy on the five grouped dev folds (mean ± std; chance is 0.5), with ROC-AUC in brackets.

**Table II: Global shortcut features alone** (`fig_leakage.png`)

| Features | A: as released | C: JPEG only | B: re-encoded | R: resampled |
|---|---:|---:|---:|---:|
| File metadata | **0.990** (0.999) | **0.991** (0.996) | 0.508 (0.510) | 0.503 (0.509) |
| Image dimensions | 0.603 (0.642) | 0.597 (0.645) | 0.603 (0.642) | 0.604 (0.643) |
| Naive global ELA | 0.696 (0.778) | 0.678 (0.782) | 0.564 (0.597) | 0.539 (0.556) |
| Compression history | 0.928 (0.981) | 0.955 (0.990) | **0.789** (0.870) | 0.608 (0.683) |
| All shortcuts | 0.989 (0.999) | 0.986 (0.998) | **0.797** ± 0.007 (0.876) | **0.649** ± 0.011 (0.720) |

**Findings.**
1. **The benchmark as released can be solved almost perfectly without looking at the image content.** File metadata alone reaches 99.0% balanced accuracy (AUC 0.999).
2. **Restricting to JPEG files does not fix this.** The quantisation tables still separate the classes (99.1%).
3. **Re-encoding (B) removes the container/metadata shortcut** (0.508) **but not the compression history.** A global double-quantisation and ghost-curve classifier still reaches 0.789 under B. Each image carries its earlier compression history: authentic images were last saved as JPEGs with standard libjpeg tables (89% of them), while tampered images were last saved as editing-software JPEGs or as uncompressed TIFFs. One extra re-encode does not erase that history. The effect is strongest when both classes started as JPEG: 0.919 on JPEG sources alone, against 0.725 for authentic vs. TIFF-tampered.
4. **Resampling before re-encoding (R) erases most of it.** Compression history falls to 0.608, and all shortcuts together reach 0.649. Most of what remains is the image-size shortcut.
5. **Image size is a residual shortcut (0.60) that no protocol removes.** The forensic pipeline must never use raw image dimensions as features.
6. **Naive global ELA is weak once the container is controlled** (0.56 under B). Even under A it reaches only 0.70, and it gains nothing from TIFF files: it scores 0.68 under C, where no TIFFs remain. Its signal is compression history, not tampering.

**Consequences for the rest of the project.**
- **Bars to beat.** A forensic feature set shows real tampering evidence only if it clearly beats the shortcuts under the same protocol: **0.797 balanced accuracy (AUC 0.876) under B** and **0.649 (AUC 0.720) under R**. These are lower bounds on how far global cues can go; a stronger global probe might score higher. That is why we also measure the added value of the forensic features over these shortcuts.
- **Protocols in Phase 5.** We will report classification under both B and R. We will also measure the *added value* of the forensic features on top of the shortcut features, not just their standalone accuracy.
- **Localisation as the main metric.** Pixel-level localisation against the masks cannot be gamed by global shortcuts, so we treat it as the most trustworthy result.
- **Feature design (Phase 3).** Features will favour *within-image inconsistency*, i.e. a region compared with the rest of the same image, over absolute global statistics.

**Choice of re-encode quality for B** (sensitivity, all shortcuts): Q=75 gives 0.791, Q=85 gives 0.797, Q=95 gives 0.855. A high re-save quality preserves more of the old compression history. Qualities 75 and 85 suppress the shortcuts about equally, so we keep Q=85, which discards less image detail. Whether B or R is the better main protocol depends on how much *genuine* local evidence each keeps. Phase 4 measures this per module (Section VI), and Section VII makes the choice with a rule fixed before the classification results were seen.
