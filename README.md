# Image Forgery Detection with Classical Image Processing

Interpretable detection and localisation of copy-move and splicing forgeries on **CASIA v2.0**, built from classical DIP techniques:
- ELA and JPEG ghosts
- histogram analysis
- noise maps
- DCT double quantisation
- edges and copy-move matching

It is evaluated under **format-controlled protocols**, because the dataset as released leaks its labels.

## Deliverables

| What | Where |
|---|---|
| Final report (IEEE format, PDF) | [Download PDF (Release v2.0.0)](https://github.com/Smit2102/image-forgery-detection/releases/download/v2.0.0/Patel_ImageForgeryDetection_report.pdf) · source in [report/latex/](report/latex/) · rebuild with `python scripts/build_report.py` |
| Slides (18, with speaker notes) | Online deck on claude.ai (private until shared; can be downloaded as PPTX or PDF) · slide source in [report/slides/](report/slides/) |
| Walkthrough notebook (executed, with outputs) | [notebooks/project_walkthrough.ipynb](notebooks/project_walkthrough.ipynb): dataset, leak, every DIP technique step by step, detection, all results |
| Live demo script | [report/DEMO_SCRIPT.md](report/DEMO_SCRIPT.md) |
| Interactive Forensic Tool | Streamlit HUD: `.venv/bin/streamlit run app/Home.py` |
| Command-line detector | `.venv/bin/python detect.py <image>` |
| Known issues and deviations | [report/KNOWN_ISSUES.md](report/KNOWN_ISSUES.md) |

> **Headline finding (Phase 2):**
> - File metadata alone separates authentic from tampered CASIA v2.0 images with **0.990 balanced accuracy (AUC 0.999)**, without looking at any image content.
> - Re-encoding every image once (protocol B) removes that shortcut, but global shortcuts (mainly compression-history statistics) still reach **0.797**.
> - Resampling before re-encoding (protocol R) brings all global shortcuts down to **0.649** (Phase-2 random-forest balanced accuracy). A tuned SVM on the same shortcut features still reaches AUC 0.771 under R in Phase 5; the forensic features clear that higher bar too.
>
> These are the bars a genuine forensic feature must beat. See [report/drafts/02_splits_and_leakage.md](report/drafts/02_splits_and_leakage.md).

> **Result (Phase 5, dev set, grouped nested CV, main protocol R):**
> - The 8 forensic modules reach **AUC 0.795** (random forest), against 0.721 for the shortcut features with the same model.
> - Added on top of the shortcuts, they raise AUC by **+0.088 [0.078, 0.097]**.
> - Copy-move keypoint matching and texture-normalised ELA carry most of the signal.
>
> See [report/drafts/05_classification.md](report/drafts/05_classification.md). These are dev-set numbers; the frozen test set is evaluated in Phase 7.

> **Test set (Phase 7, evaluated once, protocol R):**
> - Forensic modules: **AUC 0.805 [0.779, 0.831]**, against 0.712 for the shortcuts. They add **+0.103 [0.076, 0.129]** on top of the shortcuts.
> - The calibrated detector judges 52 % of images with 86 % accuracy.
> - Localisation reaches **pixel F1 0.198** (0.333 for regions above 5 % of the image).
>
> The development estimates held on unseen data. See [report/drafts/07_results.md](report/drafts/07_results.md).

> **Deep-learning baseline (Phase 8, test set, same splits and protocols):**
> - An ELA + ResNet-18 CNN reaches **AUC 0.994 on the files as released (A)**, where shortcut features alone reach 1.000, so that score does not show tampering detection.
> - Under protocol R it reaches **0.770**, below the classical pipeline (0.805; −0.036 [−0.067, −0.005]); under B it reaches 0.811 against 0.920.
> - The two are complementary under R: the CNN is better on splicing (+0.073, a lead the shortcut features match), and a rank average of the two reaches 0.824, level with the best Phase-7 model (0.829).
> - On MICC-F220 the CNN's AUC (0.53 [0.36, 0.70]) is consistent with chance, while the classical detector reaches 0.97.
>
> See [report/drafts/08_cnn.md](report/drafts/08_cnn.md).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt       # includes `-e .` (the forgery package)
```

Place the dataset at `CASIA2/` with the subfolders `Au/`, `Tp/` and `CASIA 2 Groundtruth/`.

## Reproduce

```bash
.venv/bin/python scripts/build_manifest.py      # Phase 1 -> data/manifest.csv, results/phase1/
.venv/bin/python scripts/make_splits.py         # Phase 2 -> data/splits.csv (+ frozen test hash)
.venv/bin/python scripts/run_leakage_study.py   # Phase 2 -> results/phase2/
.venv/bin/python scripts/phase3_sanity.py       # Phase 3 -> results/phase3/ (module maps on 20 val images)
.venv/bin/python scripts/sweep_params.py --write-config   # Phase 4 -> results/phase4/, tuned params into config.yaml
.venv/bin/python scripts/extract_features.py    # Phase 5 input -> data/features/features_{A,B,R}.csv (dev set only)
.venv/bin/python scripts/run_phase5.py          # Phase 5 -> results/phase5/, models/
.venv/bin/python scripts/extract_maps.py        # Phase 6 input -> data/maps/ (cell-level evidence, train + val)
.venv/bin/python scripts/run_phase6.py          # Phase 6 -> results/phase6/, models/localizer_{R,B}.joblib
.venv/bin/python scripts/phase6_gating.py       # Phase 6b -> results/phase6/gating.md
.venv/bin/python scripts/run_phase7.py --dry-run  # Phase 7 code check (train -> val; never touches test)
.venv/bin/python scripts/run_phase7.py          # Phase 7: ONE-TIME test evaluation -> results/phase7/run_<n>/ (logged)
.venv/bin/python scripts/phase7_posthoc.py      # Phase 7: post-hoc analysis of the saved test outputs (no re-evaluation)
.venv/bin/python scripts/run_phase7b.py         # Phase 7b: robustness, synthetic forgeries, MICC-F220 -> results/phase7b/
                                                #   (needs data/external/MICC-F220 from lci.micc.unifi.it; research use)
.venv/bin/python scripts/make_ela.py            # Phase 8 input -> data/ela/{R,B,A}/ (ELA images, all splits)
.venv/bin/python scripts/run_phase8.py --smoke  # Phase 8 code check (small subsets; never touches test)
.venv/bin/python scripts/run_phase8.py --stage select   # Phase 8: CNN training + selection on val (GPU, ~2.5 h)
.venv/bin/python scripts/run_phase8.py --stage refit    #   refit on train+val -> models/cnn_{R,B,A}.pt + pre-registration
.venv/bin/python scripts/run_phase8.py --stage test     #   ONE-TIME test evaluation of the CNN (logged)
.venv/bin/python scripts/phase8_posthoc.py      # Phase 8: post-hoc size analysis of the saved test scores
.venv/bin/streamlit run app/Home.py            # Phase 8.5: interface (run from the project folder; localhost only)
.venv/bin/python scripts/build_report.py        # Phase 9: final report PDF (needs Tectonic: brew install tectonic)
.venv/bin/python -m pytest                      # unit + data-invariant tests (incl. headless UI tests)
.venv/bin/python detect.py <image> [--protocol A|B|R]   # single image -> results/detect/<stem>/report.json
```

All parameters are in [config.yaml](config.yaml) and the random seed is fixed (42). `make_splits.py` refuses to change the frozen split (hashes in `data/test_split.sha256`) unless you pass `--refreeze`. Read the generated CSVs with `forgery.data.manifest.read_manifest()` and `forgery.data.splits.read_splits()`, which keep ids such as `00138` as strings.

Protocols: **A** as released, **C** JPEG-only, **B** re-encoded once at Q=85, **R** downscaled ×0.75 then re-encoded.

## Layout

```
config.yaml            all parameters
detect.py              CLI entry point (wraps forgery.pipeline.detect)
src/forgery/
  io.py                image/mask loading, protocols A/B/R, JPEG metadata
  pipeline.py          detect(path) -> Result  (the single entry point used everywhere)
  features/            8 modules (ELA, patch-histogram chi2, noise, JPEG ghost, DCT double quantisation,
                       edge sharpness, copy-move keypoints, copy-move blocks):
                       extract(img) -> FeatureOutput(evidence_map, features)
  data/                file-name parsing, manifest, leakage-safe splits
  eval/                shortcut study, metrics, classification + statistics
  localize/            fusion of the 8 evidence maps into a tampered-region mask
  cnn.py               Phase-8 baseline: ELA + ResNet-18 (tiles, pooling, training loop)
  explain.py           per-image SHAP explanation, plain-language text, PDF/JSON report (used by the UI)
app/                   Streamlit interface: Home.py + pages/ (analyse, batch, dataset explorer, results)
scripts/               one script per pipeline step
tests/                 pytest suite
data/                  manifest.csv, splits.csv, test_split.sha256 (cache/ is regenerable)
results/phaseN/        generated tables and figures
models/                trained classifiers (regenerable; not in git)
report/                IEEE report outline and per-section drafts
```

## Status

| Phase | Content | State |
|---|---|---|
| 0 | Environment, config, `detect()` interface, tests | ✅ |
| 1 | Manifest, mask matching, dataset statistics | ✅ |
| 2 | Grouped splits, frozen split, protocols A/B/C/R, shortcut study | ✅ |
| 3 | 8 forensic feature modules with synthetic-forgery tests | ✅ |
| 4 | Parameter sweeps (image + pixel AUC, protocols B and R) | ✅ |
| 5 | Nested-CV SVM / RF, added value over shortcuts, ablations, statistics, calibration, SHAP | ✅ |
| 6 | Localisation: learned fusion of the 8 maps, post-processing, gating; `detect.py` writes mask.png / overlay.png | ✅ |
| 7 | One-time test evaluation (run #1): classification, deployed detector, localisation, error analysis | ✅ |
| 7b | Robustness (JPEG, resize, blur, noise, social upload), synthetic forgeries, MICC-F220 cross-dataset | ✅ |
| 8 | ELA-CNN baseline (ResNet-18) vs the classical pipeline under A / B / R, pre-registered one-time test | ✅ |
| 8.5 | Streamlit interface: single-image analysis with SHAP explanation + PDF report, batch mode, dataset explorer, results dashboard | ✅ |
| 9 | Final IEEE report (LaTeX/PDF), slides, demo script, repository tidy-up | ✅ |
