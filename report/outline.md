# Report outline (IEEE conference format)

Working title: **Interpretable Classical Forgery Detection on CASIA v2.0: Exposing Format Shortcuts and Evaluating Under a Format-Controlled Protocol**

Drafts live in `report/drafts/`. The final IEEEtran report is in `report/latex/` (built into `report/Patel_ImageForgeryDetection_report.pdf` by `scripts/build_report.py`); sections III–XII are condensed from the drafts, with details in the appendices.

| § | Section | Draft file | Phase |
|---|---|---|---|
| I | Introduction | `latex/sections/01_introduction.tex` | P9 ✅ |
| II | Related work and published CASIA2 results | `latex/sections/02_related.tex` | P9 ✅ |
| III | Dataset and its pitfalls | `01_dataset.md` | P1 ✅ |
| IV | Splits, protocols and the shortcut study | `02_splits_and_leakage.md` | P2 ✅ |
| V | Method: one subsection per technique | `03_method_*.md` | P3 ✅ |
| VI | Parameter selection | `04_parameters.md` | P4 ✅ |
| VII | Classification: fusion, ablations, statistics, calibration, SHAP | `05_classification.md` | P5 ✅ |
| VIII | Localisation | `06_localization.md` | P6 ✅ |
| IX | Test results and error analysis | `07_results.md` | P7 ✅ |
| X | Robustness and cross-dataset generalisation | `07b_robustness.md` | P7b ✅ |
| XI | CNN comparison | `08_cnn.md` | P8 ✅ |
| XII | System and demo (CLI, UI) | `08b_system.md` | P8.5 ✅ |
| XIII | Limitations and conclusion | `latex/sections/13_conclusion.tex` | P9 ✅ |

Figures are generated into `results/phase*/` by the scripts and are never edited by hand.
