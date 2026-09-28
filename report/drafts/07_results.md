# IX. Test-Set Results and Error Analysis

*Source: `results/phase7/run_1/` (`test_results.md`, `summary.json`, `test_classification.csv`, `test_added_value.csv`, `localisation_per_image_{R,B}.csv`, `test_outcomes_R.csv`, `failure_enrichment_R.csv`; post-hoc analysis of these saved outputs in `posthoc.json` and `failure_enrichment_R_holm.csv`). Figures: `fig_roc_test_R.png`, `fig_failures_R.png`. Scripts: `scripts/run_phase7.py`, `scripts/phase7_posthoc.py`. Run log: `results/phase7/test_runs.log`.*

## A. Protocol

The test split (1,892 images: 1,123 authentic, 769 tampered; 1,092 split groups, none shared with the development set) was frozen in Phase 2. It was used **once** for the headline evaluation (run #1) and afterwards only for post-hoc evaluations with the same frozen models: the robustness and synthetic-forgery experiments of Section X. Nothing from those experiments was fed back into any model or setting, and each of them is recorded in the same run log.

**What was used, all frozen during development:**
- **Classifiers:** refit on all development images with the hyper-parameters selected most often in the Phase-5 nested CV. The refit of the main forensic random forest is identical to the deployed model's forest.
- **Deployed calibrated models:** used with their uncertain bands.
- **Localisers:** the deployed ones, with their post-processing.
- **Localisation baselines:** post-processing tuned on the validation images only.

**Safeguards:**
- **Dry runs first.** Before the test run, the full script was exercised twice in a dry-run mode (training images as "development", validation images as "test").
- **Independent review.** It was reviewed by a separate QA pass.
- **Preflight checks.** At run time a preflight confirmed that every model, feature file and cache matched the current code and configuration.
- **Audit record.** The start record in `results/phase7/test_runs.log` holds fingerprints of 36 files (all project source files and `config.yaml`) and of 12 input files (the frozen split's hash file, the models and the Phase-5/6 inputs they were built from).
  - **Completion record.** Because of a path bug, run 1 wrote its completion record, with the hashes of `summary.json` and `test_classification.csv`, into `run_1/test_runs.log`. The record was copied, unchanged and annotated, into the main log after the run, and the bug was then fixed. The fix changes the script's fingerprint for any future run.
  - **Output manifest.** A SHA-1 manifest of every output file of run 1 (`run_1/MANIFEST.sha1`) was recorded on the same day.
  - **Post-run check.** A post-run QA check recomputed the code and input fingerprints and found nothing changed since the run started.
- **Statistics.** Confidence intervals are 95 % group-bootstrap intervals (1,000 resamples of test split groups).

## B. Classification

**Table VI: Test ROC-AUC [95 % CI], with the development estimate (Section VII) for comparison**

| Features | R: RF, test | R: RF, dev | R: SVM, test | B: RF, test | B: RF, dev | A: RF, test |
|---|---:|---:|---:|---:|---:|---:|
| Shortcuts | 0.712 [0.682, 0.746] | 0.721 | 0.753 | 0.886 | 0.876 | **1.000** |
| Forensic, local only | 0.796 [0.769, 0.821] | 0.781 | – | 0.909 | 0.902 | 0.963 |
| **Forensic, local + global** | **0.805 [0.779, 0.831]** | 0.795 | 0.809 | **0.920** | 0.910 | 0.981 |
| Forensic + shortcuts | 0.815 [0.789, 0.840] | 0.810 | 0.829 | 0.936 | 0.926 | 1.000 |

**Table VII: Test added value over the shortcuts (paired group bootstrap, ΔAUC [95 % CI])**

| | R: RF | R: SVM | B: RF | A: RF |
|---|---:|---:|---:|---:|
| (forensic + shortcuts) − shortcuts | **+0.103 [0.076, 0.129]** | +0.077 [0.054, 0.099] | +0.050 [0.036, 0.063] | +0.000 |
| forensic (local + global) − shortcuts | +0.093 [0.059, 0.125] | +0.056 [0.026, 0.085] | +0.033 [0.017, 0.049] | −0.019 |
| forensic (local only) − shortcuts | **+0.083 [0.048, 0.116]** | – | +0.022 [0.003, 0.040] | −0.037 |

**Findings.**
1. **The development estimates hold on unseen data.** Every test AUC is within 0.02 of its cross-validated development estimate (within 0.015 for all random-forest rows), and every dev estimate falls inside the corresponding test CI (R forensic: dev 0.795, test 0.805 [0.779, 0.831]). There is no sign of overfitting to the development set, despite the feature tuning and model selection done on it.
2. **The central claim is confirmed on test.** Under the strict protocol R, the forensic features beat all global shortcuts combined (0.805 vs 0.712). Added on top of them, they raise AUC by **+0.103 [0.076, 0.129]**. The purely within-image features alone add +0.083 [0.048, 0.116]. Under B the gains are smaller but still significant (+0.050; local only +0.022 [0.003, 0.040]).
3. **Protocol A again shows the leak.** On the released files, shortcuts alone reach test AUC 1.000 and forensic features add nothing.
4. **RF and SVM cannot be told apart** on the forensic features (McNemar, R: 119 vs 115 discordant, p = 0.84; B: p = 0.88).

## C. The deployed detector (protocol R, calibrated, uncertain band 0.5 ± 0.24)

| | Test | Development (out-of-fold) |
|---|---:|---:|
| Share of images judged (outside the band) | 51.8 % | 51 % |
| Accuracy on judged images | 0.858 | 0.855 |
| Balanced accuracy on judged images | 0.809 | 0.810 |
| Expected calibration error (binned ECE rises on smaller samples: test 1,892, dev about 10.7k) | 0.019 | 0.009 |

Verdicts on the 1,892 test images: 235 tampered, 745 authentic, 912 uncertain.

Among the images it judged, the detector made **126 missed forgeries** and **13 false alarms** (222 correct detections, 619 correct authentic verdicts). A "tampered" verdict is right 94 % of the time (222 / 235). An "authentic" verdict is right 83 % of the time (619 / 745), because a missed forgery is the more common error. Under B (band ± 0.04), 95.7 % of images are judged, with accuracy 0.869 and balanced accuracy 0.853.

## D. Localisation (deployed localiser, frozen post-processing; 766 test tampered images with a valid mask)

| | R | B |
|---|---:|---:|
| **Pixel F1 [95 % CI]** | **0.198 [0.179, 0.216]** | **0.308 [0.284, 0.332]** |
| IoU | 0.128 | 0.210 |
| MCC | 0.163 | 0.296 |
| Pixel AUC | 0.714 | 0.829 |
| Mean of maps (val-tuned) | 0.169 | 0.208 |
| Best single module: copy-move keypoints (val-tuned) | 0.174 | 0.212 |
| Whole image | 0.135 | 0.135 |
| Validation F1 (Section VIII) | 0.185 | 0.260 |

By tampered area:

| Tampered area | F1, R | F1, B |
|---|---:|---:|
| < 1 % | 0.040 | 0.123 |
| 1–5 % | 0.148 | 0.295 |
| > 5 % | 0.333 | 0.413 |

By forgery type: copy-move 0.236 / 0.354 and splicing 0.129 / 0.226 (R / B).

**Gating by the image verdict** reduces authentic images showing a false region:
- R: from 89 % to 43 %
- B: from 73 % to 9 %

With gating, tampered F1 falls from 0.198 to 0.189 (R) and from 0.308 to 0.267 (B).

The learned fusion beats both the best single cue (copy-move keypoints, chosen on validation) and the plain average. Paired test differences in F1 [95 % CI]:
- R: +0.023 [0.009, 0.037] over keypoints, +0.029 [0.013, 0.045] over the average
- B: +0.096 [0.076, 0.114] over keypoints, +0.101 [0.080, 0.120] over the average

Under R, test localisation matches validation (0.198 vs 0.185). Under B it is *higher* than validation (0.308 vs 0.260, above the test CI); the cause was not investigated.

## E. Error analysis (protocol R; `fig_failures_R.png`, `failure_enrichment_R.csv`)

Missed forgeries (126 FN) were compared with detected ones (222 TP), and false alarms (13 FP) with correct authentic verdicts (619 TN). Images in the uncertain band are excluded from both groups. The comparison uses automatically computed image properties, with Fisher's exact test and a Holm correction over the 14 distinct tests.

| Property | Share among missed | Share among detected | Ratio | p | Holm-adjusted p |
|---|---:|---:|---:|---:|---:|
| Splicing (not copy-move) | 60 % | 3 % | **22×** | < 0.001 | < 0.001 |
| Keypoints in the lowest tercile | 49 % | 21 % | **2.4×** | < 0.001 | < 0.001 |
| Tampered area > 5 % | 33 % | 57 % | 0.57× | < 0.001 | < 0.001 |
| JPEG source (vs TIFF) | 44 % | 31 % | 1.4× | 0.015 | 0.16 (n.s.) |
| Low texture | 42 % | 34 % | 1.2× | 0.17 | n.s. |
| Flat area > 30 %, saturation > 5 %, tiny area (< 1 %) | – | – | 1.1–1.4× | ≥ 0.28 | n.s. |

**Causes of missed forgeries, in order of evidence:**
1. **Splicing.** The confident "tampered" verdicts are almost all copy-move images. Splicing leaves no internal duplicate, and after resampling its compression and noise traces are weak (splicing vs authentic AUC: 0.68 under R, 0.88 under B).
2. **Few keypoints.** The strongest module, copy-move keypoint matching, has little to work with in smooth, low-detail images.
3. **Large regions are detected more reliably.** Regions above 5 % are under-represented among misses (0.57×). Tiny regions (< 1 %) are *not* significantly over-represented. JPEG-sourced forgeries look over-represented (1.4×, p = 0.015), but this does not survive the Holm correction. That format difference is the residual effect first noted in Section VII-D, and on test it even reverses under B: TIFF-sourced forgeries are detected less often there (0.77 vs 0.89).

**False alarms.** There are only 13, and none of the image-property tags is significant after correction (low-keypoint images: 0.33×, p = 0.045, Holm p = 0.45). The copy-move features point clearly to the cause (`posthoc.json`):
- **all 13** false alarms have copy-move keypoint matches that survive RANSAC (median 13 inliers)
- **none** of the 619 correct authentic verdicts has any
- the false alarms also have many keypoints: median 1,500, the detector's cap, against 732

Most of the eight false alarms shown in the figure are **self-similar scenes**:
- colonnades and pillars
- repeated statues and ornaments
- a lattice tower
- a zebra-patterned sofa
- rows of gaming machines

These are scenes in which genuinely repeated structure looks like a copy-move, a known weakness of copy-move detection. The scene descriptions are qualitative (n = 13), but the copy-move inlier counts above are measured.

## F. Summary

On the untouched test set, the classical pipeline behaves as the development experiments predicted. Under the strict protocol it:
- detects tampering with **AUC 0.805**, clearly above what global shortcut cues achieve (0.712)
- judges about half of the images with **86 % accuracy**, and flags the rest as uncertain
- localises tampered regions with **pixel F1 0.20** (0.33 for regions above 5 % of the image)

Under B, where within-image compression evidence survives, the numbers rise to AUC 0.920 and F1 0.31. The main remaining weaknesses are:
- splicing without compression traces
- low-detail images
- self-similar scenes, which trigger the copy-move detector (all 13 test false alarms had copy-move matches, against none of the correct authentic verdicts)
