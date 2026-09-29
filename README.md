# Image Forgery Detection with Classical Image Processing

[![Author: Smit2102](https://img.shields.io/badge/author-Smit2102-181717?logo=github)](https://github.com/Smit2102)
[![Built with Claude](https://img.shields.io/badge/built%20with-Claude-D97757?logo=claude&logoColor=white)](https://www.anthropic.com/claude)
[![License: MIT](https://img.shields.io/badge/license-MIT-2EA44F)](LICENSE)
[![Release](https://img.shields.io/github/v/release/Smit2102/image-forgery-detection)](https://github.com/Smit2102/image-forgery-detection/releases/latest)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](requirements.txt)

Digital Image Processing (EDS 6364) — final project | Smit Patel

**Project website:** https://smit2102.github.io/image-forgery-detection/

This project detects and localises **copy-move** and **splicing** forgeries using only classical, interpretable image-processing techniques from the five course topics:
- point processing
- histogram processing
- spatial filtering
- frequency-domain filtering
- edge and corner detection

Every technique produces a map a person can inspect. A learned fusion turns them into a verdict (tampered / authentic / *uncertain*) and a suspected-region mask.

It is evaluated on **CASIA v2.0**, and the first finding is about the dataset itself: **the files as released leak their labels**. A classifier that reads only file metadata separates authentic from tampered images with **0.990 balanced accuracy**, without looking at a single pixel, because most tampered images are TIFF files and the authentic ones are JPEGs. Every result here is therefore measured under leak-controlled protocols, against what those shortcuts alone can achieve.

Full write-up:
- [`report/REPORT.md`](report/REPORT.md): readable summary, with results and a validation section documenting what went wrong and how it was fixed.
- [`report/Patel_ImageForgeryDetection_report.pdf`](report/Patel_ImageForgeryDetection_report.pdf): IEEE-format final report.

The original proposal is [`GroupJ_Patel_Gupta_proposal.pdf`](GroupJ_Patel_Gupta_proposal.pdf).

## What's in this repo

```
.
├── GroupJ_Patel_Gupta_proposal.pdf   original project proposal
├── index.html                        project website (GitHub Pages)
├── config.yaml                       every parameter (tuned values written by the sweep)
├── detect.py                         command line: python detect.py photo.jpg
├── data/
│   ├── manifest.csv                  audit of all 12,614 images (labels, formats, masks, JPEG tables)
│   ├── splits.csv                    leakage-safe train / val / test split
│   └── test_split.sha256             frozen-split hashes (the test set was fixed before any training)
├── src/forgery/                      the detector (a Python package)
│   ├── features/                     the 8 DIP modules: ela, histogram, noise, jpeg_ghost, dct_dq,
│   │                                 edges, copymove (keypoint + block)
│   ├── localize/                     fusion of the 8 evidence maps into a tampered-region mask
│   ├── eval/                         shortcut study, metrics, classification and statistics
│   ├── pipeline.py                   detect(): the single entry point used everywhere
│   └── explain.py                    per-image explanations (SHAP) and PDF reports
├── scripts/                          the pipeline, one script per step (see "Running it" below)
├── app/                              Streamlit interface: analyse, batch, dataset explorer, results
├── notebooks/
│   └── project_walkthrough.ipynb     the whole project step by step, with outputs
├── outputs/
│   ├── 00_original.png … 01b_ground_truth.png        input, protocol R, ground truth
│   ├── 02a_ela_residual_x10.png / 02b_…              (point processing)
│   ├── 03a_ela_equalised.png / 03b_… / 03c_…         (histogram processing)
│   ├── 04a_noise_residual.png / 04b_noise_sigma.png  (spatial filtering)
│   ├── 05a_dct_coefficient_histogram.png / 05b_… / 05c_jpeg_ghost.png   (frequency domain)
│   ├── 06a_canny_edges.png … 06d_copymove_blocks.png (edge & corner detection)
│   ├── 07a_fused_probability.png / 07b_detection_overlay.png            (detection)
│   ├── 08_summary_grid.png                           the whole pipeline on one image
│   ├── 09_… – 21_…                                    evaluation figures (leakage, SHAP, test ROC,
│   │                                                  errors, robustness, CNN comparison, interface)
│   └── metrics.json                  every headline number, read from the saved results
├── report/
│   ├── REPORT.md                     readable write-up
│   └── Patel_ImageForgeryDetection_report.pdf   IEEE-format final report
└── tests/                            209 automated tests
```

## Pipeline stages

| Category | Techniques | Code |
|---|---|---|
| Point processing | Error level analysis (texture-normalised); JPEG ghosts (re-save at Q50–95) | `features/ela.py`, `features/jpeg_ghost.py` |
| Histogram processing | Histogram-equalised ELA; χ² distance between patch histograms | `features/histogram.py` |
| Spatial filtering | High-pass noise residual (Immerkær), robust local noise σ with edges excluded | `features/noise.py` |
| Frequency-domain filtering | 8×8 DCT double-quantisation (permutation-tested periodicity); DCT-block copy-move matching | `features/dct_dq.py`, `features/copymove.py` |
| Edge & corner detection | Canny edge sharpness; SIFT keypoint copy-move with mirror matching and RANSAC (Harris tested, rejected) | `features/edges.py`, `features/copymove.py` |
| Fusion | Random forest (compared with an SVM), isotonic calibration and an uncertain band; gradient-boosted map fusion for localisation | `eval/classify.py`, `localize/fusion.py` |
| Optional extension | ELA + ResNet-18 CNN baseline under the same protocols | `cnn.py` |

Each module outputs an evidence map and *within-image inconsistency* features, so a region is compared with the rest of its own image, never with other images.

## Data sources

- **CASIA v2.0** (12,614 images and ground-truth masks), available on [Kaggle](https://www.kaggle.com/search?q=CASIA+2.0+image+tampering+detection). It is **not included** here: place it at `CASIA2/` with the subfolders `Au/`, `Tp/` and `CASIA 2 Groundtruth/`.
- **MICC-F220** (Amerini et al., 2011), from the LCI lab at the University of Florence (lci.micc.unifi.it), used for the cross-dataset test. It is **not included**: place it at `data/external/MICC-F220/`.

Both datasets are for non-commercial research use under their own terms.

## Running it, step by step

1. **Clone the repo and create the environment.**
   ```bash
   git clone https://github.com/Smit2102/image-forgery-detection.git
   cd image-forgery-detection
   python3 -m venv .venv
   .venv/bin/pip install -r requirements.txt      # includes `-e .` (the forgery package)
   ```

2. **Try the detector on your own photo.** The trained models are regenerated by the pipeline, so run steps 3–4 first on a fresh clone.
   ```bash
   .venv/bin/python detect.py photo.jpg            # -> results/detect/<name>/report.json, mask.png, overlay.png
   .venv/bin/streamlit run app/Home.py              # interface at http://localhost:8510
   ```

3. **Place the dataset** (see *Data sources*), then **run the pipeline in order**. Each step writes to `results/`:
   ```bash
   .venv/bin/python scripts/build_manifest.py       # 1  dataset audit
   .venv/bin/python scripts/make_splits.py          # 2  leakage-safe split (refuses to change the frozen test set)
   .venv/bin/python scripts/run_leakage_study.py    # 3  shortcut study: how far do global cues go?
   .venv/bin/python scripts/phase3_sanity.py        # 4  module sanity check
   .venv/bin/python scripts/sweep_params.py --write-config        # 5  parameter tuning on validation
   .venv/bin/python scripts/extract_features.py && .venv/bin/python scripts/run_phase5.py     # 6  classification
   .venv/bin/python scripts/extract_maps.py && .venv/bin/python scripts/run_phase6.py         # 7  localisation
   .venv/bin/python scripts/run_phase7.py --dry-run # 8  check, then ONE-TIME test evaluation (logged):
   .venv/bin/python scripts/run_phase7.py
   .venv/bin/python scripts/run_phase7b.py          # 9  robustness, synthetic forgeries, MICC-F220
   .venv/bin/python scripts/make_ela.py && .venv/bin/python scripts/run_phase8.py --stage select   # 10 CNN baseline
   .venv/bin/python scripts/make_outputs.py         # 11 collect final figures + metrics.json into outputs/
   ```

4. **Inspect the results.**
   - Quick visual overview: `outputs/08_summary_grid.png`.
   - Every number: `outputs/metrics.json`.
   - Guided tour: `notebooks/project_walkthrough.ipynb`.
   - Full write-up: `report/REPORT.md`.
   - Tests: `.venv/bin/python -m pytest` runs all 209.

## Key results at a glance

- **The leak:** file metadata alone gives **0.990** balanced accuracy on the released files. Re-encoding every image (protocol B) leaves 0.797 to the shortcuts; downscaling ×0.75 before re-encoding (protocol R) leaves 0.649.
- **Classification on the frozen test set** (1,892 images, evaluated once):
  - forensic features reach **AUC 0.805 [0.779, 0.831]** under R, against 0.712 for the shortcuts;
  - added on top of the shortcuts they gain **+0.103 [0.076, 0.129]**;
  - under protocol B they reach AUC 0.920.
- **Calibrated detector:** it judges **52 %** of images with **86 %** accuracy and marks the rest *uncertain*. A "tampered" verdict is right **94 %** of the time.
- **Localisation:** pixel **F1 0.198** under R (0.333 for regions above 5 % of the image) and 0.308 under B.
- **Robustness:** under R the AUC stays at 0.78 or above after JPEG re-compression down to quality 50. A social-media-style upload costs 0.04.
- **Unseen dataset:** the detector reaches **AUC 0.972** on MICC-F220 copy-moves, which come from only 11 forged scenes.
- **Deep-learning baseline (ELA + ResNet-18):**
  - it reaches 0.994 only on the leaking files, where shortcuts alone reach 1.000;
  - under R it trails the classical pipeline (0.770 vs 0.805);
  - on MICC-F220 its AUC is consistent with chance (0.53).
- **Weaknesses:**
  - splices between photos with the same processing history;
  - low-detail images;
  - self-similar scenes, which can mimic copy-move.

Full numbers, confidence intervals and the validation process are in [`report/REPORT.md`](report/REPORT.md).

## License

The code in this repository is under the [MIT License](LICENSE). The CASIA v2.0 and MICC-F220 datasets are **not** part of this repository or its license: they are distributed by their authors for research use under their own terms. The example images in `outputs/` are derived from CASIA v2.0 and are shown for research illustration only.

## Contributors

| | Contributor | Role |
|---|---|---|
| <img src="https://github.com/Smit2102.png" width="48" alt="Smit2102"> | **Smit Patel** ([@Smit2102](https://github.com/Smit2102)) | Author: project design, experiments, evaluation, report |
| <img src="https://img.shields.io/badge/-Claude-D97757?logo=claude&logoColor=white" alt="Claude"> | **Claude** ([Anthropic](https://www.anthropic.com/claude)), via Claude Code | AI pair programmer: code, analysis and drafting under the author's direction; credited as co-author on commits |
