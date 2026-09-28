# Known issues

## Resolved at the start of Phase 6 (2026-09-23)

| # | Issue | Fix | Evidence |
|---|---|---|---|
| 1 | DCT periodicity test: 16–48 % false positives on wide / heavy-tailed histograms | Per-histogram **permutation null** for the maximum over all candidate periods, giving a calibrated p-value; tuned level α | False-positive rate at α = 0.01: 0–3 % on the QA scenarios, and 0.4 % on real history-free images (vs 17–25 %). Detection on double-compressed images: 75 % (simulations: 56 %). Regression test added. |
| 2 | Blind step estimator returned q+1 / q+2 for large steps, and failed on saturated images | The step must be a **local maximum** of the score (beat q−1 and q+1); the noise level is relaxed from 0.5 to 1.0 if no step qualifies | Steps 20 / 24 / 30: 20 / 20 correct (vs 0 / 20). Regression test added. |
| 3 | Noise: blocks straddling a flat patch became low-noise outliers | Pixels with a perfectly flat 3×3 neighbourhood are excluded | `low_local_z_max`: 25.9 → 3.8 on the QA case. Regression test added. |
| 4 | Stale sweep rows and stale protocol-A outputs | Full re-sweep of all 8 modules and a full re-run of A / B / R | — |
| 5 | `--reuse-cv` did not check the feature fingerprint | Fingerprints are stored with the CV outputs and asserted on reuse | — |
| 6 | `window_for` cap; meta check rejected parameters written at their default | The window is capped at kmax with whole periods kept; the meta check compares fully *resolved* parameters (defaults filled in) | Tests updated. |

## Open

- The DCT periodicity test does not detect some regular double-quantisation patterns. For example, q₁/q₂ = 10/4 (a period of 2.5 bins) was detected in 0 / 30 simulations, while 9/4, 7/3 and 8/3 were detected in 30 / 30. Overall detection on simulated double compression is about 56 %.

## Phase 8 (2026-09-27)

- **The pre-registered image-size check is invalid.** `run_phase8.size_analysis` (AUCs within area terciles) broke the tie on the dominant size (78 % of test images are 384 × 256 in either orientation) by row order, which is authentic-first. The terciles were therefore confounded with the label; for example, the standard-size test images in the top tercile were 100 % tampered. The tercile AUCs in `results/phase8/summary.json` / `test_results.md` (e.g. 0.38 / 0.43) are artefacts and are not reported. The code now breaks ties at random (for any future use); this edit changed `scripts/run_phase8.py` after the test run, from SHA-1 f695477… (recorded in the preregistration and the test log) to its current version. The other post-test edit to that file only renamed a heading in the generated markdown. Section XI-D reports the post-hoc grouping by natural size instead (`scripts/phase8_posthoc.py`, on test and validation).
- The CNN's training is not bit-reproducible on the MPS backend. The seed spread under R on validation is 0.746 / 0.762 / 0.752.
- The CNN was selected on a single validation split, not by nested CV like the classical models; the refit used the same development images as the classical refit.
- The preregistration's wording "tile-count terciles" does not match the code, which used area terciles. The preregistration is an audit record and is left unchanged.
- `data/ela/meta.json` now also fingerprints the protocol code (`apply_protocol`, `resample`, JPEG encode). It was regenerated after the test run, so it no longer equals the `ela_meta` in `preregistration.json`; the old version is the one recorded there. QA recomputed clean ELA on the fly and reproduced the cached test AUC, so the PNGs are consistent with the current code.
