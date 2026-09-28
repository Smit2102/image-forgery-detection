# XI. Comparison with a Deep-Learning Baseline (ELA-CNN)

*Source: `results/phase8/` (`test_results.md`, `summary.json`, `selection.csv`, `test_scores_cnn_{R,B,A}.csv`, `preregistration.json`; post-hoc analysis of the saved scores in `posthoc.json` and `posthoc_size_groups.csv`). Figures: `fig_roc_cnn_vs_classical.png`, `fig_training_curves.png`, `fig_ela_examples.png`. Scripts: `scripts/make_ela.py`, `scripts/run_phase8.py`, `scripts/phase8_posthoc.py`; model code `src/forgery/cnn.py`. Run log: `results/phase7/test_runs.log`.*

## A. Why this baseline

A common approach in the CASIA v2.0 literature feeds error-level-analysis (ELA) images to a convolutional network, and reports accuracies above 90 %. We built this baseline for two reasons:
- to see whether a learned model beats the hand-crafted pipeline once the format leak is controlled
- to see whether those high numbers survive that control

The CNN uses the same splits, the same protocols and the same frozen test set as the classical pipeline.

## B. Method

**Input.** For each protocol, the ELA image is computed on the image *after* the protocol has been applied (Fig. `fig_ela_examples.png`):
- the absolute difference between the image and its JPEG re-save at quality 90, per colour channel
- scaled ×10 and clipped to 8 bits
- kept at native resolution, with no resizing, so the pixel-level compression error is preserved

**Network.** ResNet-18 with ImageNet weights, with a single-logit output.

**Training.**
- **Crops:** one random 192 × 192 crop per image per epoch, with horizontal flips.
- **Loss and optimiser:** class-weighted binary cross-entropy; AdamW (learning rate 3·10⁻⁴, weight decay 10⁻⁴).
- **Schedule:** batch 64, cosine schedule over at most 15 epochs.
- **Small images:** images smaller than a tile are zero-padded, which in ELA space means "no error".

**Image score.** The image is covered with overlapping 192 × 192 tiles at stride 96, and the tile logits are pooled by their mean or their maximum.

**Selection and refit.**
- The network was trained on the train split (8,828 images). The epoch and the pooling rule were chosen by validation AUC (1,894 images), with early stopping after 4 epochs without improvement.
- Nothing else was tuned. A second 5-fold CV, as used for the classical models, would have cost about 15 × 20 min of GPU time.
- The chosen recipe was then refit on all development images (10,722). These are the same images the classical forest was refit on, for the selected number of epochs.

| Protocol | Selected epoch | Pooling | Validation AUC |
|---|---:|---|---:|
| R | 3 | max | 0.746 (seeds 42 / 43 / 44: 0.746 / 0.762 / 0.752) |
| B | 8 | mean | 0.797 |
| A | 12 | mean | 0.994 |

The seed spread under R is small (SD 0.008), so a single training run is representative. Training and refitting took about 3.5 hours on an Apple M4 GPU (MPS backend).

**Safeguards.** Before the network saw the test split, the comparisons were **pre-registered** (`preregistration.json`), together with the SHA-1 of every model:
- **Primary:** the R CNN's test AUC; the paired difference CNN − classical under R and B; and a fixed rank-average combination of the two scores.
- **Secondary:** everything else in this section.

A preflight step checked the models, the ELA settings, the classical scores and every cached comparison result. The evaluation ran once, and its start and completion are recorded in `results/phase7/test_runs.log`. The classical comparator is the Phase 7 forest on the forensic features ("forensic, local + global"; its scores come from run 1 and nothing classical was re-run).

## C. Test results

**Table X: Test ROC-AUC [95 % CI] and paired differences (group bootstrap, 1,000 resamples)**

| | R | B | A |
|---|---:|---:|---:|
| ELA-CNN | 0.770 [0.744, 0.795] | 0.811 [0.788, 0.834] | **0.994** [0.991, 0.997] |
| Classical (forensic features, RF) | **0.805** [0.779, 0.831] | **0.920** [0.905, 0.934] | 0.981 [0.975, 0.986] |
| **CNN − classical** | **−0.036 [−0.067, −0.005]** | **−0.109 [−0.128, −0.088]** | +0.014 [0.008, 0.019] |
| CNN − (forensic + shortcut features, RF) | −0.045 [−0.074, −0.017] | −0.125 [−0.145, −0.104] | −0.005 [−0.009, −0.003] |
| CNN − shortcut features (RF) | +0.058 [0.030, 0.084] | −0.075 [−0.094, −0.055] | −0.005 [−0.008, −0.003] |
| CNN − shortcut features (SVM) | +0.017 [−0.011, 0.044] | −0.072 [−0.091, −0.053] | – |
| Rank average (CNN, classical) | **0.824** [0.803, 0.846] | 0.903 [0.886, 0.919] | 0.996 [0.994, 0.998] |
| Rank average − classical | **+0.019 [0.003, 0.035]** | −0.017 [−0.025, −0.008] | +0.015 [0.011, 0.019] |

- **Correlation:** Spearman correlation between the CNN and classical scores is 0.57 (R), 0.62 (B) and 0.78 (A).
- **Multiple testing:** four paired comparisons were pre-registered as primary (rows "CNN − classical" and "Rank average − classical", under R and B). With a Holm correction over those four, both R results stay significant: adjusted p = 0.036 each (unadjusted 0.026 and 0.018; bootstrap p-values, `posthoc.json`). The B results have p < 0.002 (no bootstrap resample crossed zero).
- **What the CIs leave out:** the variation between training runs (validation SD 0.008 over three seeds) is not in the test CIs. So the two R differences are significant on this test set but small, and not adjusted for variation between training runs.

**Findings.**
1. **Under the strict protocols, the classical pipeline beats the CNN.**
   - **Under R:** the CNN is 0.036 below the forensic forest, a small gap that survives the Holm correction.
   - **Versus the shortcut features:** it beats their random forest (+0.058). It is not significantly better than their SVM (0.753; +0.017 [−0.011, 0.044]). The CNN can also see global cues such as overall ELA level and, through max pooling, image size. So under R we cannot show that it learns more than the global shortcuts do.
   - **Under B:** the gap widens to 0.109, and the CNN falls below B's global shortcuts (0.811 vs 0.886). B keeps the within-image compression traces that the DCT, ghost and ELA modules measure explicitly, and the CNN exploits them less well.
2. **On the files as released (A), the CNN reaches AUC 0.994**, in line with the literature's high numbers. The shortcut features reach 1.000 on the same files. So a score this high does not show tampering detection: the leak alone gives it.
   - **The leak is visible in ELA itself.** In `fig_ela_examples.png` the authentic JPEG shows almost no ELA error, while the tampered TIFF shows strong error along edges and texture.
   - **Restricting to JPEG files does not remove it.** On authentic vs tampered JPEGs alone, the CNN still reaches 0.995.
   - **A plausible explanation, not tested directly:** authentic JPEGs in CASIA mostly use the standard IJG quantisation tables (89 %), tampered ones never do (Section III), and ELA against a quality-90 re-save is sensitive to the table.
   - **The explanation is incomplete.** The CNN also separates tampered JPEGs from the 126 authentic JPEGs with *non*-standard tables (AUC 0.971; `posthoc.json`), so it uses other compression-history differences as well.
   - Either way, an ELA-CNN on the released files is driven by the dataset's compression history. Protocols B and R are needed to judge it.
3. **The two approaches are complementary under R.**
   - **Correlation:** the scores are only moderately correlated (0.57).
   - **Rank average:** reaches **0.824**, +0.019 [0.003, 0.035] over the classical forest. That is the highest R AUC among detectors that use no shortcut features. It is level with the best Phase-7 model, the SVM on forensic + shortcut features (0.829; difference −0.005 [−0.024, 0.014]).
   - **Under B:** the classical forest is strong enough that averaging in the weaker CNN hurts (−0.017).
   - **Not deployable:** the combination ranks scores within the evaluation set. It uses no labels, but it cannot score a single image on its own, so the deployed detector remains the classical one.

## D. Where each approach wins

**By forgery type** (authentic images vs one type; AUC):

| | R: CNN | R: classical RF | R: CNN − classical RF | R: shortcut SVM | R: forensic + shortcut SVM | B: CNN − classical RF |
|---|---:|---:|---:|---:|---:|---:|
| Copy-move | 0.778 | 0.874 | −0.096 [−0.125, −0.069] | – | – | −0.129 [−0.152, −0.106] |
| Splicing | 0.755 | 0.682 | +0.073 [0.019, 0.128] | 0.743 | 0.752 | −0.071 [−0.101, −0.042] |

- **Copy-move: the classical pipeline's advantage.** The CNN detects copy-moves about as well as it detects splices (0.778 vs 0.755). Explicit keypoint matching is much better at it (0.874). The duplicated content is what identifies a copy-move, and the network is not built to compare distant regions of an image.
- **Splicing under R: the CNN beats the classical forest (+0.073)**, the classical pipeline's main weakness (Section IX-E). But the global shortcut features reach almost the same splicing AUC (SVM 0.743; forensic + shortcuts SVM 0.752).
  - So the CNN's edge on splicing may come from global differences in compression or size between CASIA's spliced and authentic images, not from local traces of the splice.
  - It is still what makes the rank average help under R.
  - Under B the CNN is worse than the classical forest on both types.

**By image size.** The pre-registered size check computed AUCs within area terciles, and it is **invalid**. 78 % of the test images (77 % of CASIA) are 384 × 256 in either orientation, and the code broke the tie by row order, which lists authentic images first. The terciles were therefore confounded with the label, and their AUCs are not reported (`report/KNOWN_ISSUES.md`). Instead, a post-hoc analysis (`scripts/phase8_posthoc.py`, `posthoc_size_groups.csv`) groups images by their natural size:

| Size (test images) | n | Tampered | R: CNN | R: classical | R: CNN − classical | B: CNN − classical |
|---|---:|---:|---:|---:|---:|---:|
| Smaller than 384 × 256 | 21 | 62 % | 0.702 | 0.615 | +0.087 [−0.072, 0.260] | −0.144 |
| 384 × 256, either orientation | 1,469 | 34 % | 0.770 | 0.789 | −0.019 [−0.065, 0.026] | −0.105 [−0.133, −0.080] |
| Larger than 384 × 256 | 402 | 65 % | **0.544** | 0.730 | **−0.186 [−0.232, −0.139]** | −0.217 [−0.266, −0.163] |

- **Standard-size images.** On CASIA's standard 384 × 256 images (either orientation), the R CNN is not significantly different from the classical forest (−0.019). The CI is wide, so this does not show the two are equivalent.
- **Larger images.** Most of the CNN's deficit comes from the larger images, where it is near chance (0.544). The classical forest drops less there (0.73); some of its features also depend on image size.
- **Validation shows the same drop**, from the train-only selection models: R 0.728 on standard-size vs 0.567 on larger images; B 0.792 vs 0.664.
- **Scores shift with size.** Authentic images larger than the standard size receive much higher CNN scores than standard-size authentic images: mean logit 1.35 vs −0.40 under R, 0.17 vs −1.76 under B. The network appears to have learned the ELA statistics of the dominant 384 × 256 images and treats the ELA of other images as suspicious. This is an interpretation; the image sources were not analysed.
- **Tile count plays a minor role.** Within the larger images the score correlates only weakly with the number of tiles (Spearman 0.20 under R, max pooling; −0.09 under B, mean pooling).

## E. Robustness and a second dataset

The degradations of Section X-A were applied to the test images; each was followed by the protocol and then ELA. Classical results are the frozen Phase 7b outputs.

| Condition | R: CNN | R: classical | R: rank average | B: CNN | B: classical |
|---|---:|---:|---:|---:|---:|
| clean | 0.770 | 0.805 | 0.824 | 0.811 | 0.920 |
| JPEG Q75 | 0.762 | 0.799 | 0.817 | 0.746 | 0.739 |
| resize ×0.5 | 0.685 | 0.693 | 0.720 | 0.672 | 0.698 |
| social upload | 0.733 | 0.762 | 0.788 | 0.674 | 0.721 |
| **MICC-F220** | **0.529** [0.360, 0.700] | **0.972** | 0.811 | **0.454** [0.226, 0.660] | **0.902** |

1. **Under R, the CNN loses about as much to degradation as the classical pipeline.**
   - AUC lost to JPEG Q75: 0.008 (CNN) vs 0.006 (classical).
   - Resize ×0.5: 0.085 vs 0.112.
   - Social upload: 0.037 vs 0.044.
   - After resizing, the gap between the two is not significant (−0.008 [−0.043, 0.027]).
2. **Under B, the classical pipeline loses much more,** because its B advantage rests on compression traces that any re-save overwrites (Section X-A).
   - JPEG Q75 costs it 0.180, against 0.065 for the CNN, leaving the two level (+0.007 [−0.034, 0.047]).
   - The social upload costs it 0.199, against 0.137 for the CNN.
3. **Padding is not the cause of the CNN's losses.** After downscaling, most images become smaller than one tile (95 % under R resize ×0.5; 82 % after a social upload). Scoring them without padding changes the AUC by at most 0.018.
4. **The CNN does not transfer to MICC-F220.** Its AUC is 0.53 (R) and 0.45 (B), consistent with chance, while the classical detector reaches 0.97.
   - The CIs are wide ([0.36, 0.70] and [0.23, 0.66]), because MICC-F220's forgeries come from only 11 scenes (Section X-C).
   - The failure fits the size analysis: all MICC-F220 photographs are larger than CASIA's standard size (722 × 480 to 800 × 1,070), and they come from a different dataset.
   - The classical copy-move matching finds the duplicated regions in these images regardless of their source.

## F. Summary

Given the same data, splits and leak controls, a standard ELA-CNN:
- **Scores high only on the leaking files.** It reaches the literature's CASIA v2.0 level only as released (AUC 0.994 under A). Under the controlled protocols it reaches 0.77 (R) and 0.81 (B).
- **Is below the classical pipeline under both protocols** (−0.036 under R; −0.109 under B).
  - Under R, most of the deficit comes from images larger than CASIA's standard size.
  - It is not significantly better than a shortcut-feature SVM (+0.017 [−0.011, 0.044]).
- **Is complementary to the classical forest under R.** It is better on splicing (+0.073), though shortcut features match that. A rank average of the two beats the classical forest (0.824 vs 0.805) and matches the best Phase-7 model (0.829).
- **Does not transfer to an unseen copy-move dataset** (MICC-F220: 0.53 [0.36, 0.70] vs 0.97, 11 forged scenes). This is the strongest argument in this study for the explicitly modelled classical cues.

*Limitations.*
- **Scope:** one architecture and one input representation (ELA), with no hyper-parameter search beyond the epoch and the pooling rule.
- **Selection:** a single validation split, where the classical models used nested CV.
- **Reproducibility:** GPU (MPS) training is not bit-reproducible; the seed spread on validation is small (SD 0.008).
- **Tuning effort:** the classical detector is the more carefully tuned system. A stronger network, for example one trained end-to-end on RGB with constrained convolutions, might close the gap under R, but it would also need the same leak controls.
