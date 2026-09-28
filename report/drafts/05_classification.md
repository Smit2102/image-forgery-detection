# VII. Classification

*Source: `results/phase5/classification.md`, `results.csv`, `added_value.csv`, `ablations.csv`, `copy_move_study.csv`, `breakdown_R.csv`, `coverage_{B,R}.csv`, `summary.json`. Figures: `fig_roc_{A,B,R}.png`, `fig_ablation.png`, `fig_calibration_{B,R}.png`, `fig_shap_R.png`, `fig_shap_beeswarm_R.png`. Script: `scripts/run_phase5.py`.*

## A. Set-up

- **Data:** all dev images (train + val, 10,722); the test set is untouched.
- **Cross-validation:** the five grouped folds of Section IV.
- **Hyper-parameters:** selected *inside* each training fold by grouped 3-fold CV (nested CV), maximising ROC-AUC.
  - RBF SVM with standardised features: C ∈ {1, 10, 100}, γ ∈ {0.3, 1, 3} / #features.
  - Random forest, 300 trees, balanced class weights: max features ∈ {√d, 0.3d}, min leaf ∈ {1, 4}.
- **Uncertainty:** mean ± std over the 5 folds, plus pooled out-of-fold AUC with a 95 % CI from 1,000 group-bootstrap resamples; split groups are resampled whole.
- **Feature sets:**
  - *forensic (local only):* the 70 within-image inconsistency features
  - *forensic (local + global):* 82 features
  - *shortcuts:* the 49 global features of Section IV
  - *forensic + shortcuts*

**Main-protocol rule, fixed before the results were seen.** The main protocol is R if, under R, adding the forensic features to the shortcuts raises AUC with a 95 % CI excluding 0; otherwise B under the same test; otherwise none. If no protocol qualifies, no main protocol is set, and `detect.py` then requires an explicit `--protocol`. **Outcome: R.**

## B. Results

**Table III: Out-of-fold ROC-AUC (mean ± std over 5 folds)**

| Features | B: RF | B: SVM | **R: RF** | **R: SVM** | A: RF |
|---|---:|---:|---:|---:|---:|
| Shortcuts (Section IV) | 0.876 ± 0.003 | 0.876 ± 0.010 | 0.721 ± 0.013 | 0.771 ± 0.010 | **1.000** |
| Forensic, local only | 0.902 ± 0.003 | – | **0.781 ± 0.012** | – | 0.962 |
| Forensic, local + global | 0.910 ± 0.003 | 0.892 ± 0.004 | **0.795 ± 0.008** | 0.793 ± 0.006 | 0.978 |
| Forensic + shortcuts | 0.926 ± 0.004 | 0.922 ± 0.004 | 0.810 ± 0.006 | **0.826 ± 0.008** | 0.999 |

Protocols A, B and R were all re-run after the QA fixes to the feature modules (Section V). Protocol A exists only to show the leak. The first run, before those fixes, is kept in `results/phase5/pre_qa_run/`.

Under the main protocol R, the forensic model (RF, local + global) reaches:
- **AUC 0.795** (95 % CI 0.783–0.806)
- balanced accuracy 0.705 ± 0.011
- F1 0.628 ± 0.018

**Table IV: Added value over the shortcuts (paired group bootstrap, ΔAUC [95 % CI])**

| | B: RF | R: RF | R: SVM | A: RF |
|---|---:|---:|---:|---:|
| (forensic + shortcuts) − shortcuts | +0.050 [0.044, 0.055] | **+0.088 [0.078, 0.097]** | +0.054 [0.045, 0.063] | −0.000 |
| forensic (local only) − shortcuts | +0.026 [0.018, 0.033] | **+0.059 [0.047, 0.072]** | – | −0.037 |

**Findings.**
1. **The forensic features carry real tampering evidence.** Under both honest protocols they beat every shortcut combined and add significantly on top of them. The purest test is the *local-only* features, which compare regions within one image: they beat the shortcuts by +0.059 AUC under R and +0.026 under B, both with CIs well above zero.
2. **On the dataset as released (A), the numbers are meaningless.** Shortcuts alone score AUC 1.000, and adding the forensic features changes nothing (ΔAUC −0.000). An A-protocol forensic AUC of 0.978 would look state-of-the-art, but most of it is compression history. This is the headline of Section IV again, now measured on the full pipeline.
3. **RF and SVM are statistically indistinguishable** on the forensic features under R (McNemar: 603 vs 644 discordant decisions, p = 0.26). The SVM combines forensic and shortcut features somewhat better (0.826 vs 0.810).

## C. Ablations (RF, fixed hyper-parameters; `fig_ablation.png`)

The ablations and the copy-move study use the most frequently selected hyper-parameters rather than nested selection, so they are mildly optimistic (Section VII-H).

| Module | Alone, B | Alone, R | AUC lost if removed, B | AUC lost if removed, R |
|---|---:|---:|---:|---:|
| ELA | 0.723 | 0.664 | −0.000 | **0.013** |
| Patch-histogram χ² | 0.569 | 0.566 | −0.001 | −0.001 |
| Noise level | 0.645 | 0.617 | −0.000 | 0.004 |
| JPEG ghosts | 0.758 | 0.624 | 0.010 | 0.001 |
| DCT double quantisation | **0.821** | 0.519 | **0.046** | 0.001 |
| Edge sharpness | 0.626 | 0.605 | −0.001 | 0.002 |
| Copy-move keypoints | 0.747 | **0.720** | **0.039** | **0.080** |
| Copy-move blocks | 0.600 | 0.579 | 0.000 | 0.003 |

- **Under B, the compression modules lead** (DCT 0.821, ghosts 0.758 alone). Resampling (R) removes their signal, as Section VI predicted.
- **Copy-move keypoint matching is the most robust module and the most important one under R.** It changes little under R (0.747 → 0.720), and its removal causes the largest drop under R (0.080). Under B it is second (0.039), just behind DCT (0.046).
- Under R, **ELA** is the second pillar: removing it costs 0.013.
- After the QA fix, the **noise** module scores 0.645 / 0.617 alone (B / R). Like the **histogram** and **block-matching** modules, it adds almost nothing on top of the others: removing any of the three changes AUC by 0.004 or less. Their information is redundant with ELA and the keypoint detector. We keep them for completeness and localisation, and report them as contributing little.

**Copy-move detectors compared** (copy-move vs authentic images; localisation from Section VI):

| | Image AUC, B | Image AUC, R | Pixel AUC, B | Share of images localised (pixel AUC > 0.6), B |
|---|---:|---:|---:|---:|
| Keypoints (SIFT + mirror) | 0.865 | 0.821 | 0.719 | 56 % |
| Blocks (DCT + mirror) | 0.633 | 0.603 | 0.527 | 14 % |
| Both | 0.878 | 0.839 | – | – |

Keypoint matching dominates, but block matching still adds a little (0.865 → 0.878 under B, 0.821 → 0.839 under R). Presumably it catches some copies in regions where few keypoints are found (not verified). Block matching also votes above threshold on some authentic images: QA measured 32 % of 25 authentic train images before the fixes. This is one reason it is weak on its own.

## D. Where the detector succeeds and fails (R, RF, out-of-fold)

- **By forgery type:**
  - copy-move vs authentic: AUC **0.851**
  - splicing vs authentic: AUC **0.694**
  - Copy-move leaves an explicit duplicate; splicing must be found through subtler statistical differences.
- **By tampered area** (detection rate at threshold 0.5 / AUC):
  - < 1 %: 0.525 / 0.768
  - 1–5 %: 0.530 / 0.787
  - \> 5 %: 0.596 / 0.818
  - Larger regions are easier, as expected.
  - The 16 tampered images without a valid ground-truth mask have no known area and are not in any bucket.
- **False-positive rate on authentic images:** 14.4 %.
- **By source format:** tampered images that were released as TIFF are detected more often than JPEG ones (0.606 vs 0.477), even under R. Some format-related difference may survive resampling. We flag this as a residual bias to re-check in Phase 7.

## E. Calibration and the "uncertain" verdict (`fig_calibration_{B,R}.png`)

- **Calibration (R).** The raw forest is already fairly well calibrated (ECE 0.034). Isotonic calibration, fitted inside each fold, brings ECE to **0.009** (Brier 0.176 → 0.175).
- **Accuracy vs coverage (R).** The accuracy on the images the system is willing to judge rises steadily as the uncertain band widens: 0.731 when judging every image, 0.855 at 51 % coverage, and 0.959 at 12 % coverage.
- **Chosen band, per protocol.** Rule: the smallest band that reaches 85 % accuracy while still judging at least half of the images.

  | Protocol | "Uncertain" band | Share judged | Accuracy on judged | Balanced accuracy on judged |
  |---|---|---:|---:|---:|
  | R | P(tampered) ∈ 0.5 ± 0.24 | 51 % | 0.855 | 0.810 |
  | B | P(tampered) ∈ 0.5 ± 0.04 | 96 % | 0.851 | 0.839 |

- **Caveats on the target.** The 85 % target is plain accuracy on an imbalanced dev set (59 % authentic), so it is influenced by how well the majority (authentic) class is judged. That is why balanced accuracy is also shown above (`summary.json`, key `at_band`). The band is chosen on the same out-of-fold predictions it is evaluated on (Section VII-H).
- **Trade-off.** Under the strict protocol R, the tool honestly declines to judge about 49 % of CASIA images. This is the cost of refusing the shortcuts.

## F. What the model relies on (SHAP, R; `fig_shap_R.png`)

Summed mean |SHAP| per module:

| Module | Summed mean \|SHAP\| |
|---|---:|
| Copy-move keypoints | 0.186 |
| ELA | 0.097 |
| Edges | 0.048 |
| Noise | 0.044 |
| JPEG ghosts | 0.039 |
| Copy-move blocks | 0.024 |
| Histogram | 0.015 |
| DCT | 0.006 |

The single most important feature is the keypoint inlier share. It is followed by the keypoint shift and the inlier count (log and raw). The most important non-keypoint feature is the spatial clustering (Moran's I) of texture-normalised ELA. These are within-image evidence, consistent with the ablations.

## G. Forgery type and deployed models

- A second forest, trained on tampered images only, predicts copy-move vs splicing with AUC **0.830** (R) / 0.874 (B), out-of-fold. `detect()` uses it for its "likely type" output.
- The deployed models (`models/forgery_{B,R}.joblib`, `type_{B,R}.joblib`) are trained on all dev images. Each protocol's model uses *its own* most frequently selected hyper-parameters across the outer folds (listed in `results/phase5/summary.json`), plus isotonic calibration. Each protocol also has its own uncertain band.
- `models/MAIN_PROTOCOL` = R, so `detect.py` uses protocol R by default.

## H. Limitations and sources of optimism

The nested-CV numbers of Tables III and IV select hyper-parameters inside each training fold. Several other analyses are less strict and should be read as mildly optimistic:
- **Fixed hyper-parameters.** The ablations, the copy-move study, calibration, the uncertain band, SHAP and the final models use the hyper-parameters selected *most often* across the five outer folds. That choice was made by looking at all folds, so these analyses are not fully out-of-sample.
- **Feature tuning overlaps the CV data.** The feature-module parameters were tuned in Phase 4 (Section VI) on 3,400 dev images (2,400 train + 1,000 val) that also lie inside the Phase-5 CV folds. The CV AUCs are therefore not fully independent of tuning.
- **Uncertain band.** The band is chosen on the same out-of-fold predictions it is evaluated on.
- **Deployed models are in-sample on dev images.** They are trained on all dev images, so running `detect.py` on train or val images gives in-sample outputs that look much better than reality.
- **SHAP** is computed in-sample.

Only the Phase-7 test-set evaluation is free of all of these.
