"""Writes notebooks/project_walkthrough.ipynb (execute it with: jupyter nbconvert --to notebook --execute --inplace notebooks/project_walkthrough.ipynb)."""
import nbformat as nbf
nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Image Forgery Detection with Classical Image Processing — project walkthrough

**Smit Patel · EDS 6364 Digital Image Processing · final project**

This notebook walks through the project end to end:

1. The CASIA v2.0 dataset and why the files as released **leak their labels**
2. The **evaluation protocols** that remove the leak
3. The **eight DIP techniques**, step by step, with before/after images: point processing, histogram processing, spatial filtering, frequency domain, edge and corner detection
4. **Detection end to end**: verdict, suspected region and an explanation of the decision
5. A **summary of all results**: development CV, the one-time test set, localisation, robustness, a second dataset and the CNN comparison

The heavy computation (feature extraction, training, the test evaluation) is done by the scripts in `scripts/`. Their outputs are saved in `results/` and read here. This notebook runs the detector only on example images. **The held-out test split is never opened**: every test-set number below is read from the saved outputs of the one-time evaluation.

Full details are in the report, `report/Patel_ImageForgeryDetection_report.pdf`.
""")

code(r"""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd, cv2
import matplotlib.pyplot as plt
from IPython.display import Image, display, Markdown

from forgery.config import load_config, resolve
from forgery.data.manifest import read_manifest
from forgery.data.splits import read_splits
from forgery.io import load_rgb, apply_protocol, load_mask, protocol_mask
from forgery.features import build_modules
from forgery.pipeline import module_params
from forgery import explain as E

%matplotlib inline
warnings.filterwarnings("ignore")
pd.set_option("display.precision", 3)
plt.rcParams.update({"figure.dpi": 90, "axes.titlesize": 9})
cfg = load_config()
ROOT = resolve(".")
print("project root:", ROOT)
""")

md(r"""
## 1. The dataset

CASIA v2.0 has 12,614 images: 7,491 authentic and 5,123 tampered (copy-move or splicing). The audit in `scripts/build_manifest.py`:
- corrected 99 forgery-type labels by majority vote
- matched masks to images by the tamper id, since 142 mask names differ from their image name
- flagged 19 masks as unusable for pixel scoring

Below, only the training and validation images are used.
""")

code(r"""
df = read_manifest(cfg).merge(read_splits(cfg), on="path")
dev = df[df.split.isin(["train", "val"])]
kind = np.where(df.label == 0, "authentic", df.forgery_type)
display(pd.crosstab(kind, df.format, margins=True).rename_axis(index="class (all 12,614 images)", columns="format"))
print(f"median tampered area: {100 * df.tampered_frac.median():.1f} % of the image")
print("split sizes:", df.split.value_counts().to_dict())
display(Image(filename=str(resolve("results/phase1/fig_samples.png")), width=900))
""")

md(r"""
**Warning sign:** authentic images are never TIFF, but most tampered images are. Authentic JPEGs also mostly use the standard IJG quantisation tables, which no tampered JPEG uses.

## 2. The leak: the label can be predicted without looking at the image

A random forest trained only on *global* cues (file metadata, image size, naive ELA level, compression-history statistics) is evaluated with 5-fold grouped CV. The rows below are balanced accuracies (0.5 = chance):
""")

code(r"""
lk = pd.read_csv(resolve("results/phase2/leakage_summary.csv"))
piv = lk.pivot_table(index="feature_set", columns="protocol", values="balanced_accuracy_mean")
display(piv[["A", "C", "B85", "R85"]].rename(columns={"A": "A: as released", "C": "C: JPEG only",
                                                      "B85": "B: re-encoded Q85", "R85": "R: resampled + Q85"}))
display(Image(filename=str(resolve("results/phase2/fig_leakage.png")), width=800))
""")

md(r"""
**File metadata alone gives 0.990 balanced accuracy on the files as released (A).** Re-encoding every image (B) removes the metadata cue. Downscaling ×0.75 before re-encoding (R, the main protocol) brings all shortcuts down to 0.649. A forensic feature only counts if it beats these shortcuts under the same protocol.

## 3. Protocols in practice

Every image is analysed *after* its protocol. Here is one example under R: downscaled ×0.75, then saved as JPEG Q85.
""")

code(r"""
val = dev[dev.split == "val"]
cm_ex = val[(val.forgery_type == "copy-move") & val.mask_valid & val.tampered_frac.between(0.05, 0.25)].sort_values("path").iloc[3]
sp_ex = val[(val.forgery_type == "splicing") & val.mask_valid & val.tampered_frac.between(0.05, 0.25)].sort_values("path").iloc[3]
print("copy-move example:", cm_ex.path, "| splice example:", sp_ex.path)

raw = load_rgb(resolve(cm_ex.path))
img = apply_protocol(raw, "R", cfg)
truth = protocol_mask(load_mask(resolve(cm_ex.mask_path)), "R", cfg)
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
ax[0].imshow(raw); ax[0].set_title(f"as released {raw.shape[1]}x{raw.shape[0]} ({cm_ex.format})")
ax[1].imshow(img); ax[1].set_title(f"protocol R {img.shape[1]}x{img.shape[0]} (JPEG Q85)")
ax[2].imshow(E.overlay(img, truth, color=(0, 200, 0))); ax[2].set_title("ground-truth tampered region (green)")
for a in ax: a.axis("off")
plt.tight_layout(); plt.show()
""")

md(r"""
## 4. The DIP techniques, step by step

`build_modules` creates the eight forensic modules with the parameters tuned in Phase 4 (from `config.yaml`). Each module returns an **evidence map** (bright = suspicious) and scalar **features**. The steps below show the intermediate images for each course topic.

### 4.1 Point processing — Error Level Analysis (ELA)
ELA re-saves the image as JPEG and takes the per-pixel absolute difference: |I − JPEG_q(I)|. A region with a different compression history re-compresses differently. Busy texture also produces error, so the module divides the block ELA by the block's texture.
""")

code(r"""
mods = {m.name: m for m in build_modules(params=module_params(cfg, "R"))}
ela = mods["ela"].maps(img)
fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))
ax[0].imshow(img); ax[0].set_title("input (protocol R)")
ax[1].imshow(np.clip(ela["residual"] * 10, 0, 255), cmap="gray"); ax[1].set_title("ELA residual x10 (point operation)")
ax[2].imshow(ela["ela_block"], cmap="magma"); ax[2].set_title("mean ELA per 8x8 block")
ax[3].imshow(np.where(ela["valid"], ela["norm_block"], np.nan), cmap="magma"); ax[3].set_title("texture-normalised ELA (flat blocks masked)")
for a in ax: a.axis("off")
plt.tight_layout(); plt.show()
""")

md(r"""
### 4.2 Histogram processing — equalisation and patch-histogram χ²
The raw ELA residual is low-contrast, so it is **histogram-equalised**. Each patch's histogram is then compared with the image's typical patch histogram using the χ² distance.
""")

code(r"""
hm = mods["histogram"].maps(img)
resid = ela["residual"]
fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))
ax[0].hist(resid.ravel(), bins=50, color="#4C78A8"); ax[0].set_title("histogram of the ELA residual (before)"); ax[0].set_yscale("log")
ax[1].imshow(hm["equalized"], cmap="gray"); ax[1].set_title("after histogram equalisation")
ax[2].hist(hm["equalized"].ravel(), bins=50, color="#E45756"); ax[2].set_title("histogram after equalisation")
ax[3].imshow(np.where(hm["valid"], hm["distance"], np.nan), cmap="magma"); ax[3].set_title("chi-square distance per patch")
for a in (ax[1], ax[3]): a.axis("off")
plt.tight_layout(); plt.show()
""")

md(r"""
### 4.3 Spatial filtering — local noise level
A high-pass kernel (Immerkær's noise estimator) removes image content and leaves noise. The robust noise σ is estimated per block over non-edge pixels. A spliced region from another camera often has a different noise level.
""")

code(r"""
nm = mods["noise"].maps(img)
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
ax[0].imshow(np.clip(np.abs(nm["residual"]), 0, 20), cmap="gray"); ax[0].set_title("high-pass residual |r|")
ax[1].imshow(nm["usable"], cmap="gray"); ax[1].set_title("usable pixels (edges, clipped, flat removed)")
ax[2].imshow(nm["log_sigma"], cmap="magma"); ax[2].set_title("log noise sigma per block")
for a in ax: a.axis("off")
plt.tight_layout(); plt.show()
""")

md(r"""
### 4.4 Frequency domain — DCT coefficients and JPEG ghosts
JPEG quantises each 8×8 block's DCT coefficients, so a coefficient's histogram is concentrated on multiples of the quantisation step. Double compression leaves a periodic pattern in these histograms, and regions pasted in after the first compression lack it. The **DCT double-quantisation** module tests for this periodicity. The **JPEG-ghost** module re-saves the image at many qualities and looks for blocks whose error curve differs from the rest.
""")

code(r"""
Y = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)[..., 0].astype(np.float32) - 128
h, w = (Y.shape[0] // 8) * 8, (Y.shape[1] // 8) * 8
coef = np.array([cv2.dct(Y[i:i + 8, j:j + 8])[0, 1] for i in range(0, h, 8) for j in range(0, w, 8)])
out = {n: mods[n].extract(img).evidence_map for n in ("dct_dq", "jpeg_ghost")}
fig, ax = plt.subplots(1, 3, figsize=(14, 3.4))
ax[0].hist(coef[np.abs(coef) <= 60], bins=121, range=(-60.5, 60.5), color="#4C78A8"); ax[0].set_title("DCT coefficient (0,1) over all 8x8 blocks (|c| <= 60)")
ax[1].imshow(out["dct_dq"], cmap="magma", vmin=0, vmax=1); ax[1].set_title("DCT double-quantisation evidence")
ax[2].imshow(out["jpeg_ghost"], cmap="magma", vmin=0, vmax=1); ax[2].set_title("JPEG-ghost evidence")
for a in ax[1:]: a.axis("off")
plt.tight_layout(); plt.show()
""")

md(r"""
### 4.5 Edge and corner detection — edge sharpness and copy-move matching
**Canny** finds edges, and the module measures how sharp each edge is relative to its local contrast. A pasted object can have unnaturally hard or feathered boundaries.

For copy-move, **SIFT keypoints** (Harris corners were tested and rejected in Phase 4) are matched against the image itself *and its mirror image*. The matches are then verified with **RANSAC**. The block matcher (frequency domain) compares DCT descriptors of overlapping blocks and votes on their shift.
""")

code(r"""
em = mods["edges"].maps(img)
cmk = mods["cm_keypoint"].extract(img)
cmb = mods["cm_block"].extract(img)
fig, ax = plt.subplots(1, 4, figsize=(16, 3.4))
ax[0].imshow(em["edges"], cmap="gray"); ax[0].set_title("Canny edges")
ax[1].imshow(em["block_sharpness"], cmap="magma"); ax[1].set_title("edge sharpness per block")
ax[2].imshow(cmk.evidence_map, cmap="magma", vmin=0, vmax=1)
ax[2].set_title(f"keypoint copy-move: {int(cmk.features['local_n_inliers'])} RANSAC inliers")
ax[3].imshow(cmb.evidence_map, cmap="magma", vmin=0, vmax=1)
ax[3].set_title(f"block copy-move: {int(cmb.features['local_max_votes'])} votes for top shift")
for a in ax: a.axis("off")
plt.tight_layout(); plt.show()
""")

md(r"""
## 5. Detection end to end

`forgery.pipeline.detect()` (used by `detect.py`, the batch mode and the Streamlit app) runs the protocol and the eight modules. It then:
- classifies with the **calibrated random forest**, abstaining inside an *uncertain* band;
- estimates the forgery type;
- **fuses the eight maps** into a suspected-region mask.

`forgery.explain` adds a per-image **SHAP** explanation of the decision.

The first two examples are validation images, which the deployed models were trained on, so their results are optimistic. The third comes from **MICC-F220**, an unseen copy-move dataset (downloaded separately; skipped if absent).
""")

code(r"""
M = E.load_models("R")
examples = [("CASIA copy-move (validation)", resolve(cm_ex.path), resolve(cm_ex.mask_path)),
            ("CASIA splicing (validation)", resolve(sp_ex.path), resolve(sp_ex.mask_path))]
micc = resolve("data/external/MICC-F220/CRW_4853tamp1.jpg")
if micc.exists():
    examples.append(("MICC-F220 copy-move (unseen dataset)", micc, None))

for title, path, mpath in examples:
    res = E.analyse(path, M, cfg)
    im = E.analysed_image(path, "R", cfg)
    display(Markdown(f"**{title}** — " + E.verdict_text(res, M)))
    n = 4 if mpath is not None else 3
    fig, ax = plt.subplots(1, n, figsize=(4 * n, 3.3))
    ax[0].imshow(im); ax[0].set_title("analysed image")
    ax[1].imshow(E.overlay(im, res.mask)); ax[1].set_title("suspected region (red)")
    ax[2].imshow(res.maps["fused"], cmap="magma", vmin=0, vmax=1); ax[2].set_title("fused tampering probability")
    if mpath is not None:
        ax[3].imshow(E.overlay(im, protocol_mask(load_mask(mpath), "R", cfg), color=(0, 200, 0))); ax[3].set_title("ground truth (green)")
    for a in ax: a.axis("off")
    plt.tight_layout(); plt.show()
    display(E.explain(M, res.features, top=5)[["description", "value", "contribution"]])
""")

md(r"""
## 6. Results

All numbers below are read from the saved outputs of the evaluation scripts.

### 6.1 Development set (grouped nested cross-validation)
Out-of-fold ROC-AUC by feature set, protocol and model:
""")

code(r"""
r5 = pd.read_csv(resolve("results/phase5/results.csv"))
display(r5.pivot_table(index="feature_set", columns=["protocol", "model"], values="roc_auc_mean"))
av = pd.read_csv(resolve("results/phase5/added_value.csv"))
display(av[av.protocol == "R"][["model", "comparison", "delta_auc", "ci_low", "ci_high"]])
""")

md(r"""
### 6.2 The frozen test set (1,892 images, evaluated once)
Image-level metrics (AUC, and accuracy / precision / recall / F1 at threshold 0.5):
""")

code(r"""
t = pd.read_csv(resolve("results/phase7/run_1/test_classification.csv"))
display(t[t.feature_set.isin(["shortcuts", "forensic_all"])][["protocol", "feature_set", "model", "auc", "auc_ci_low",
         "auc_ci_high", "accuracy", "precision", "recall", "f1", "balanced_accuracy"]])
o = pd.read_csv(resolve("results/phase7/run_1/test_outcomes_R.csv"))
display(Markdown("**Deployed calibrated detector (protocol R), verdicts vs truth (confusion matrix):**"))
display(pd.crosstab(o.label.map({0: "authentic", 1: "tampered"}), o.verdict).rename_axis(index="true class", columns="verdict"))
display(Image(filename=str(resolve("results/phase7/run_1/fig_roc_test_R.png")), width=420))
""")

md(r"""
### 6.3 Localisation on the test set (pixel F1 and IoU)
""")

code(r"""
for P in ("R", "B"):
    per = pd.read_csv(resolve(f"results/phase7/run_1/localisation_per_image_{P}.csv"))
    print(f"protocol {P}: pixel F1 {per.f1.mean():.4f}, IoU {per.iou.mean():.4f} over {len(per)} tampered images")
""")

md(r"""
### 6.4 Robustness, synthetic forgeries and a second dataset
""")

code(r"""
rb = pd.read_csv(resolve("results/phase7b/robustness.csv"))
display(rb.pivot_table(index="condition", columns="protocol", values="auc", sort=False))
ex = pd.read_csv(resolve("results/phase7b/external.csv"))
display(ex[["protocol", "dataset", "auc", "auc_ci", "coverage", "accuracy_judged"]])
display(Image(filename=str(resolve("results/phase7b/fig_robustness.png")), width=950))
""")

md(r"""
### 6.5 Comparison with a deep-learning baseline (ELA + ResNet-18)
""")

code(r"""
s8 = json.loads(resolve("results/phase8/summary.json").read_text())
rows = []
for P, r in s8["protocols"].items():
    rows.append({"protocol": P, "CNN AUC": r["auc"]["cnn"]["auc"], "classical AUC": r["auc"]["classical"]["auc"],
                 "CNN - classical": r["cnn_minus_classical"]["delta_auc"],
                 "CI": f"[{r['cnn_minus_classical']['ci_low']:.3f}, {r['cnn_minus_classical']['ci_high']:.3f}]",
                 "MICC-F220 CNN": r.get("micc_f220", {}).get("cnn", {}).get("auc")})
display(pd.DataFrame(rows))
display(Image(filename=str(resolve("results/phase8/fig_roc_cnn_vs_classical.png")), width=950))
""")

md(r"""
**Reading the CNN result:**
- On the files as released (A), the CNN reaches AUC 0.994, but the shortcut features alone reach 1.000 there.
- Under the controlled protocols it trails the classical pipeline.
- On MICC-F220 its AUC is consistent with chance.

## 7. Reproduce everything

```bash
python scripts/build_manifest.py        # dataset audit
python scripts/make_splits.py           # leakage-safe frozen split
python scripts/run_leakage_study.py     # shortcut study
python scripts/sweep_params.py --write-config
python scripts/extract_features.py && python scripts/run_phase5.py      # classification
python scripts/extract_maps.py && python scripts/run_phase6.py          # localisation
python scripts/run_phase7.py            # ONE-TIME test evaluation (logged)
python scripts/run_phase7b.py           # robustness, synthetic, MICC-F220
python scripts/make_ela.py && python scripts/run_phase8.py --stage select|refit|test   # CNN baseline
streamlit run app/Home.py               # interactive tool
python scripts/build_report.py          # final report PDF
```
""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "dip-forgery", "display_name": "Python (DIP forgery)", "language": "python"}
nbf.write(nb, str(__import__("forgery.config", fromlist=["resolve"]).resolve("notebooks/project_walkthrough.ipynb")))
print("cells", len(C))
