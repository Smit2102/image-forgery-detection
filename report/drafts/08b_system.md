# XII. System and Demonstration

*Code: `src/forgery/pipeline.py` (`detect()`), `src/forgery/explain.py` (explanations and reports), `detect.py` (command line), `app/` (Streamlit interface). Screenshots: `results/phase8_5/ui_*.png`. Tests: `tests/test_app.py`, `tests/test_explain.py`.*

## A. Architecture

All entry points call one function, `forgery.pipeline.detect(path, protocol, …)`. It carries out every step in order:
1. Load the image.
2. Apply the protocol.
3. Run the eight modules, with the parameters tuned in Phase 4.
4. Classify with the calibrated forest.
5. Estimate the forgery type.
6. Localise with the learned fusion.
7. Gate the mask by the verdict.

Before it runs, `detect()` refuses a model whose recorded feature parameters or protocol differ from the current configuration. The command line, the batch mode and the interface therefore cannot drift apart from the evaluated system.

On top of `detect()`, `forgery.explain` adds what a user needs to understand a result:
- **Per-image explanation:** exact SHAP contributions (TreeExplainer) of the 82 features to the forest's score, each translated into a plain-language description, e.g. "Copy-move: keypoint matching: matches consistent with one transform (RANSAC inliers)".
- **Verdict text:** the text states the uncertain band explicitly (P(tampered) within 0.50 ± 0.24 under R gives no verdict), together with the share of test images that fell in the band under the chosen protocol (48 % under R, 4 % under B; read from `results/phase7b/robustness.csv`).
- **Evidence sentences:** short, factual statements about the strongest evidence, including the known false-alarm cause: repeated real structure can mimic a copy-move (Section IX-E).
- **Report:** a two-page PDF (verdict, overlay, evidence chart, caveats; the eight evidence maps) and a JSON record.

**Input preparation.** `prepare_image` checks every user image before analysis:
- **Full decode:** truncated files fail here, not later in the pipeline.
- **Size and safety limits:** decompression bombs are refused, as are images outside 64–4,000 px per side.
- **Normalisation:** the image is rotated by its EXIF orientation, and 16-bit or float images are rescaled to 8 bit. The detector's loader converts every image with PIL's `convert("RGB")`. That is correct for CASIA, where all 12,614 images are 8-bit RGB without EXIF rotation, but it would turn a 16-bit greyscale image white.
- **Pixels only:** a changed image is saved losslessly as PNG. Under protocols B and R the detector works on decoded pixels, so this does not affect the analysis.
- **Same checks everywhere:** the command-line tool and both analysis pages use this function.

The module has no interface code and is unit-tested. One test checks that the SHAP contributions plus the base value reproduce the forest's probability exactly.

## B. The interface

A Streamlit application (`streamlit run app/Home.py`) with four pages:

1. **Analyze an image** (`ui_analyze.png`, `ui_evidence_maps.png`).
   - **Input:** upload a photo (JPEG, PNG, TIFF, BMP, WebP; at most 4,000 px per side) or pick an example.
   - **Output:** a colour-coded verdict with P(tampered), the likely forgery type, and the analysed image with the suspected region, the fused probability map and, for CASIA examples, the ground truth.
   - **Tabs:**
     - *Why this verdict:* evidence sentences and a SHAP bar chart.
     - *Evidence maps:* the map of each technique, with a one-sentence explanation.
     - *All features.*
     - *Download:* PDF, JSON, mask and overlay.
   - **Protocol:** chosen in the sidebar. R is the default and is explained as the recommended setting.
2. **Batch analysis.**
   - **Input:** up to 200 uploaded files, or a folder on the computer.
   - **Output:** a progress bar; a table of verdicts, probabilities, region sizes and the strongest evidence; a gallery of images judged tampered; downloads of the CSV and a ZIP of overlays.
   - **Error handling:** an unreadable file is reported in the table and does not stop the batch.
3. **Dataset explorer** (`ui_dataset_leak.png`).
   - **Browse:** CASIA v2.0 images with filters (kind, format, split), ground-truth masks and metadata: format, estimated JPEG quality, IJG tables, tampered area, host and donor id, split group.
   - **Leak tab:** shows why the released dataset leaks its labels: file format and quantisation tables by class, and the shortcut classifiers' balanced accuracy under each protocol.
4. **Results dashboard** (`ui_results_cnn.png`). The project's evaluation read directly from `results/`: leakage, classification, localisation, the test set, robustness and MICC-F220, and the CNN comparison. Nothing is recomputed.

**Keeping the evaluation honest in the interface.**
- **The explorer never lists the test split.** It shows only the 10,722 training and validation images. The results dashboard does show saved figures made from test images: the missed forgeries and false alarms of Section IX-E, and the synthetic forgeries of Section X-B, which were built from test-set authentic images. These are outputs of the one-time evaluation, not a way to browse the test set.
- **CASIA examples are labelled as training data.** The deployed models were refit on all development images, so their results on these images are optimistic.
- **Unbiased examples are offered.** MICC-F220 images serve as examples from an unseen dataset.
- **The home page states the limits:** trained on CASIA; a verdict is evidence, not proof; about half of all images fall in the uncertain band; the known weaknesses.

**Performance** (Apple M4 laptop):

| Step | Typical CASIA-sized image | Largest allowed image (4,000 × 2,667 px), R / B |
|---|---:|---:|
| Detection with all eight modules | about 0.2 s | 4.3 s / 7.1 s |
| SHAP explanation | about 0.5 s | about 0.5 s |
| PDF report | about 0.3 s | 2.0 s / 3.3 s |

Models are loaded once and cached. Uploads are stored once per content hash, so a rerun of the page, such as a protocol switch back or a download click, reuses the cached analysis and the cached PDF instead of recomputing them. This also holds for rotated or 16-bit photos, whose prepared copy is written once. Batch results are kept in the session until the input or the protocol changes. Stored uploads are user images, so they are deleted after seven days.

## C. Verification

`tests/test_app.py` runs every page headlessly with Streamlit's AppTest framework and checks that:
- no page raises an exception
- the analysis page produces a verdict for an example image
- the batch page processes a folder containing a corrupt file and reports exactly that file as an error
- a batch in which every file fails still renders, and hidden files are skipped
- batch results survive a rerun, such as a download click
- an empty filter in the explorer does not blank its other tabs

`tests/test_explain.py` covers input preparation:
- 16-bit images are rescaled, not clipped
- EXIF rotation is applied
- truncated, non-image, too-small and too-large files are refused
- ordinary files are passed through unchanged

All pages were also exercised in a real browser (headless Chrome), including a batch run and the example analysis. The screenshots in `results/phase8_5/` come from that run.

## D. Limitations

- **Local use only.** Nothing is uploaded to a server, but the folder mode reads any folder the user names. The configuration in `.streamlit/config.toml` (in the project folder and in `app/`) therefore binds the server to `localhost`, so the app is not reachable from the network whether it is started from the project folder or from `app/`.
- **Rotation matters.** An EXIF-rotated photo is analysed upright. Several modules work on fixed block grids, so the same photo rotated by 90° can receive a different score.
- **Explanations cover the forest only.** SHAP explains the forest's raw score. The calibrated probability that sets the verdict is a monotone transformation of it, so the direction of every contribution is the same, but the sizes of the contributions are not on the probability scale.
- **Region kept for uncertain images.** The suspected region is shown for uncertain images too, for inspection. This matches the command-line tool, which removes the region only for images judged authentic.
