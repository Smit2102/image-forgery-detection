# VI. Parameter Selection

*Source: `results/phase4/sweep_records.csv`, `sweep_final.csv`, `chosen_params.json`. Figure: `results/phase4/fig_sweeps.png`. Script: `scripts/sweep_params.py`.*

## A. Procedure

Parameters were tuned **on training and validation images only** (the test set is untouched):
- a fixed random sample of 2,400 training images (1,200 authentic, 1,200 tampered) is used for fitting
- 1,000 validation images (500 / 500) are used for scoring

These 3,400 dev images later appear in the Phase-5 cross-validation folds, so the Phase-5 AUCs are not fully independent of this tuning, a source of mild optimism (see Section VII-H).

Each module is tuned on its own by coordinate ascent: one parameter at a time, the others held at their current best, always starting from the same Phase-3 defaults, so a re-run reproduces the same choices. A setting is scored by the mean of two numbers:
- **image AUC:** a random forest trained on the module's `local_*` features only, so global shortcuts cannot inflate the score
- **pixel AUC:** the mean pixel-level ROC-AUC of the evidence map against the ground-truth mask, over the validation tampered images; copy-move images only for the two copy-move modules

A new value is adopted only if it improves the score by more than 0.002. Parameters are chosen under protocol B and reused unchanged under R.

**Note on re-tuning.** After QA review, the DCT double-quantisation, noise and both copy-move modules were corrected (Section V). The shared statistics also changed slightly: `local_top5_z` replaced the top-5 % ratio, and the robust-z scale floor rose. All eight modules were then re-tuned from their defaults with the current code, so every row in the table below comes from that run.

## B. Chosen parameters

| Module | Chosen (changes from the default in bold) | Image AUC B / R | Pixel AUC B / R | Share of images with pixel AUC > 0.6 (B) |
|---|---|---:|---:|---:|
| ELA | **q = 85**, block 8, **texture c = 16** | 0.696 / 0.638 | 0.615 / 0.625 | 0.52 |
| Patch-histogram χ² | **q = 75, patch 16**, 16 bins | 0.571 / 0.511 | 0.589 / 0.563 | 0.44 |
| Noise level | Immerkaer, block 8, **edge quantile 0.8, min. block std 2** | 0.609 / 0.592 | 0.560 / 0.544 | 0.41 |
| JPEG ghosts | block 16, **min. std 4** | 0.745 / 0.602 | 0.645 / 0.542 | 0.58 |
| DCT double quantisation | **15 frequencies, α = 0.05** (permutation-test level) | 0.797 / 0.491 | 0.668 / 0.543 | 0.65 |
| Edge sharpness | **Canny 30 / 90**, block 16 | 0.610 / 0.575 | 0.574 / 0.568 | 0.46 |
| Copy-move keypoints | SIFT, ratio 0.6, contrast 0.01, mirror matching (defaults kept) | 0.744 / 0.728 | 0.719 / 0.683 | 0.56 |
| Copy-move blocks | tol 1.5, min. std 4, 8 votes, mirror matching (defaults kept) | 0.618 / 0.561 | 0.527 / 0.517 | 0.14 |

## C. Observations

1. **The sweeps mainly improved localisation.** The largest pixel-AUC gains over the defaults were ELA (0.551 → 0.615), noise (0.491 → 0.560) and DCT double quantisation (0.630 → 0.668). For ELA, the lower re-save quality (q = 85) had the largest effect (0.551 → 0.613); stronger texture normalisation (c = 16) added little on top (0.613 → 0.615). The gains were not uniform: a few chosen settings lowered one of the two scores. Because the selection criterion is their mean, a gain in one can offset a small loss in the other (e.g. for the noise and edge modules).
2. **Harris corners, proposed originally, are not a usable copy-move detector on these images.** On CASIA's small (384×256) images they yield few, poorly repeatable points: image AUC 0.564 and pixel AUC 0.512, against 0.744 and 0.719 with SIFT's scale-invariant detector. We therefore use SIFT and report Harris as a negative result.
3. **Block matching localises few copy-moves** (pixel AUC > 0.6 for 14 % of images, against 56 % for keypoints). This is presumably because many CASIA copy-moves involve geometric transformations (not measured), and block DCT features are not invariant to rotation or scale. Keypoint and block matching are compared in full in Section VII.
4. **Protocol R removes what the compression modules rely on.**
   - DCT double quantisation falls from 0.797 to 0.491 image AUC and from 0.668 to 0.543 pixel AUC.
   - JPEG ghosts fall from 0.745 to 0.602 image AUC.
   - Resampling breaks the 8×8 JPEG grid, so this is expected.

   The modules that do not depend on the grid are much less affected:
   - copy-move keypoints: 0.744 → 0.728
   - ELA: 0.696 → 0.638
   - edges: 0.610 → 0.575
5. **These localisation scores come from inconsistency within each image, so global shortcuts cannot produce them** (Section IV). Under B, the DCT and ghost modules localise pasted regions (pixel AUC 0.67 and 0.65). This is genuine tampering evidence: the pasted content has a different compression history *from the rest of the same image*. The Phase-2 concern was only that *global* compression statistics separate the two classes of images.

The choice between B and R as the main protocol is made in Section VII by a rule fixed before looking at the Phase-5 results.
