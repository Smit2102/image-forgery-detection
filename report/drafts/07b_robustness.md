# X. Robustness and Generalisation

*Source: `results/phase7b/` (`robustness.csv`, `synthetic.csv`, `external.csv`, `summary.json`). Figures: `fig_robustness.png`, `fig_synthetic_examples.png`. Script: `scripts/run_phase7b.py`. The deployed models are frozen; nothing here is tuned. These experiments re-use the test split *after* the one-time headline evaluation of Section IX, for evaluation only, with nothing fed back. Each use is recorded in `results/phase7/test_runs.log`.*

## A. Robustness to post-processing (test set, 1,892 images)

Every test image was degraded *before* the protocol was applied:
- JPEG re-compression at quality 95 … 50
- downscaling ×0.9, ×0.75, ×0.5, with the ground-truth mask resized alongside
- Gaussian blur, σ = 0.5 … 1.5
- Gaussian noise, σ = 2 … 10 grey levels
- a simulated social-media upload: ×0.8 downscale, then JPEG quality 70

We report the image-level AUC of the deployed forest and the pixel F1 of the deployed localiser (`fig_robustness.png`).

**Table VIII: Image AUC (pixel F1) under degradation**

| Condition | Protocol R | Protocol B |
|---|---:|---:|
| clean | 0.805 (0.197) | 0.920 (0.308) |
| JPEG Q85 | 0.793 (0.193) | 0.903 (0.306) |
| JPEG Q75 | 0.799 (0.193) | **0.739** (0.223) |
| JPEG Q50 | 0.781 (0.191) | 0.719 (0.214) |
| resize ×0.75 | 0.768 (0.174) | 0.747 (0.170) |
| resize ×0.5 | **0.693** (0.126) | 0.698 (0.121) |
| blur σ = 1.5 | 0.764 (0.201) | 0.754 (0.203) |
| noise σ = 10 | 0.794 (0.197) | 0.803 (0.181) |
| social upload | **0.762** (0.176) | **0.721** (0.185) |

**Findings.**
1. **Protocol R is robust to re-compression and noise.** Its AUC stays within 0.03 of the clean value down to JPEG quality 50 and noise σ = 10, and its localisation F1 hardly changes.
   - R already discards the grid-bound compression traces, so its evidence (copy-move keypoints, texture-normalised ELA, edges) does not depend on them.
   - Strong downscaling (×0.5) is its main weakness: AUC 0.69, F1 0.13. Smaller images leave fewer keypoints and blocks.
2. **Protocol B's advantage is fragile.** B survives a re-save at quality 95 or 85 (0.900, 0.903; 85 is B's own re-encode quality). At a lower quality its clean AUC (0.920) falls below protocol R's: 0.739 at Q75, 0.772 at Q65 (the decline is not monotonic) and 0.719 at Q50. Under blur, noise or downscaling, B's advantage over R largely disappears; B stays at most about 0.02 ahead (e.g. 0.820 vs 0.802 under noise σ = 2; see `robustness.csv`).
   - B's extra signal is the within-image compression history (Section VI).
   - Any further lossy processing overwrites that history.
3. **A simulated social-media upload costs about 0.04 AUC under R** (0.805 → 0.762) and 0.20 under B (0.920 → 0.721). For images that have been shared online, the strict protocol is the realistic setting. This supports choosing R as the main protocol, which Section VII made on other grounds (its pre-registered rule).

## B. Synthetic "own photo" forgeries with known masks

From the test set's authentic images we generated 480 forgeries with exact masks. Tampered regions cover a median 6 % of the image (range 0.2–22 %). There are 120 of each kind:
- **hard-edged splices:** a random region cut from another authentic image
- **seamless splices:** the same, blended in with Poisson (seamless) cloning
- **plain copy-moves:** a translated copy of a region
- **rotated and scaled copy-moves:** ±20°, ×0.85–1.15

We also kept 240 untouched authentic images, never used as hosts or donors. All images were saved losslessly and evaluated as made and after a simulated social-media upload.

**Table IX: Synthetic forgeries (AUC against the 240 authentic images [95 % CI]; share judged "tampered"; pixel F1; for copy-moves also F1 against source ∪ copy)**

| Kind | R: AUC | R: detected | R: F1 (source ∪ copy) | B: AUC | B: F1 (source ∪ copy) |
|---|---:|---:|---:|---:|---:|
| Splice, hard edge | 0.569 [0.508, 0.629] | 1 % | 0.108 | 0.570 | 0.158 |
| Splice, seamless | 0.552 [0.487, 0.609] | 2 % | 0.079 | 0.551 | 0.225 |
| Copy-move, shift | **0.885** [0.830, 0.926] | 73 % | 0.376 (**0.597**) | 0.937 | 0.497 (0.703) |
| Copy-move, rotate + scale | **0.896** [0.857, 0.934] | 63 % | 0.360 (**0.571**) | 0.908 | 0.484 (0.653) |
| After a social upload (all kinds) | 0.693 | 27 % | 0.182 | 0.687 | 0.222 |

False alarms on the untouched images: 1.2 % (R), 8.8 % (B).

**Findings.**
1. **Copy-moves are found, including rotated and scaled ones.** AUC 0.89–0.94 and F1 0.36–0.50. Mirror-aware, scale- and rotation-invariant keypoint matching does what it was designed for. `fig_synthetic_examples.png` shows that the localiser marks **both** the source and the pasted copy. The ground truth marks only the copy, so pixel F1 understates copy-move localisation. Scored against source ∪ copy, F1 rises to 0.60 / 0.57 under R and 0.70 / 0.65 under B (shift / rotate + scale).
2. **Splices between two CASIA authentic photos are essentially undetectable** (AUC 0.55–0.57; the CI for seamless splices includes 0.5). This is consistent with the test-set error analysis, where missed forgeries were mostly splices (Section IX-E). CASIA's own splices remain more detectable (AUC 0.68 under R).
   - Here the host and donor share the same kind of camera JPEG history, noise level and processing, so there is no compression, noise or sharpness inconsistency for the modules to detect.
   - CASIA's own splices are detectable to the extent that their pasted content differs in such traces, and more so under B.
   - Seamless blending removes the hard edge, and even the hard-edged splices are rarely flagged: edge sharpness alone is not enough evidence.
3. **Social-media processing weakens detection** (overall AUC 0.725 → 0.693 under R). The losses are spread over the kinds: seamless splices −0.051 (to chance, 0.501), rotated/scaled copy-moves −0.048, shifted copy-moves −0.024 and hard-edged splices −0.005.

*Mask definitions.* Hard-splice and copy-move masks are exact: no pixel outside the mask changes. Poisson blending also alters pixels around the pasted region, so a seamless splice's mask is the pasted region plus every pixel the blending changed by more than 2 grey levels. For copy-moves the mask is the pasted copy. We also score copy-move localisation against source ∪ copy, because the localiser marks both. In 112 of 240 copy-moves the copy partly overlaps and hides the source (in 34 by more than 20 %).

*Manual forgeries.* Forgeries made by hand in an image editor, and real messaging-app round trips, were not available. The synthetic set is a controlled stand-in, and adding real hand-made examples is left as optional further work.

## C. Cross-dataset test: MICC-F220

MICC-F220 (Amerini et al., IEEE TIFS 2011; downloaded for non-commercial research use) contains 110 copy-move forgeries and 110 originals, with larger images than most of CASIA. It is labelled at image level only, with no masks. The forgeries come from only **11 scenes**, each with 10 attack variants (translation, rotation, scaling and combinations), so the AUC rests on 11 forged scenes. The other 99 originals have no forged version. In the bootstrap, each scene's original and its forgeries form one group.

| | Protocol R | Protocol B |
|---|---:|---:|
| AUC [95 % CI] | **0.972 [0.933, 0.998]** | 0.902 [0.767, 0.984] |
| Share of images judged | 48 % | 89 % |
| Accuracy on judged images | 0.943 | 0.790 |
| Forgeries judged "tampered" | 86 % | 88 % |
| Originals judged "tampered" (false alarms) | 5.5 % | 28 % |
| Forgeries with copy-move keypoint matches | 97 % | 91 % |
| Originals with copy-move keypoint matches | 23 % | 32 % |

**Findings.**
1. **The CASIA-trained detector transfers to an unseen copy-move dataset without retraining**, with AUC 0.97 under R. This is driven by the keypoint copy-move module, which finds matches in 97 % of the forgeries.
2. **The false-alarm mechanism of Section IX-E appears here too.** 23 % of the originals contain copy-move keypoint matches, presumably from repeated structure (not inspected). The calibrated model still keeps false "tampered" verdicts at 5.5 % under R. It does so mostly by abstaining: of the 110 originals, 100 fall in the uncertain band, 4 are judged authentic and 6 tampered. The judged accuracy of 0.943 therefore comes almost entirely from the forgeries.
3. **B generalises worse.** Its AUC is 0.90, and the paired group-bootstrap difference R − B = +0.069 [0.010, 0.172] (`micc_paired_R_minus_B.json`). Its false-alarm share is 28 % against 5.5 % for R, although this comparison uses each protocol's own uncertain band (± 0.04 vs ± 0.24). This is consistent with B's reliance on CASIA-specific compression history.

*Not tested.* COVERAGE (hosted behind an authenticated OneDrive link), the Columbia splicing set (server refused access) and CASIA v1.0 (Kaggle account needed) could not be downloaded automatically. Of the MICC datasets, F600, the version with masks, was offline, and F2000 (4 GB) was skipped. A cross-dataset *splicing* test is therefore missing; given Section B, we expect splicing to be the weaker case.

## D. Summary

Under the strict protocol the detector:
- keeps most of its accuracy under re-compression and noise, and loses about 0.04 AUC to a social-media-style upload
- generalises to an unseen copy-move dataset (AUC 0.97)
- is reliable for copy-move forgeries, including rotated and scaled ones

Its clear limit is **splicing between photographs with the same processing history**: with no compression, noise or edge inconsistency, the classical cues used here cannot see it. Protocol B looks stronger on clean CASIA images, but that advantage vanishes after a re-save at JPEG quality 75 or below, and on an external dataset.
