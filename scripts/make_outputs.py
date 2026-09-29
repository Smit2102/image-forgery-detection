"""Builds outputs/: the project's final figures and numbers, in one small folder.

  * Stage images 00–08: every DIP technique, step by step, on one validation copy-move image under protocol R
    (the same example as notebooks/project_walkthrough.ipynb).
  * Evaluation figures 09–21, copied from results/ (made by the pipeline scripts, never edited by hand).
  * metrics.json: every headline number (dataset, leakage, development CV, the one-time test evaluation,
    localisation, robustness, MICC-F220, the CNN comparison), read from the saved results.

The interface's results dashboard, the notebook and the website read outputs/ only, so they work from a fresh
clone without the full results/ tree. Run after the pipeline:  python scripts/make_outputs.py
"""

from __future__ import annotations

import json
import re
import shutil

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from forgery import explain as E  # noqa: E402
from forgery.config import load_config, resolve  # noqa: E402
from forgery.data.manifest import read_manifest  # noqa: E402
from forgery.data.splits import read_splits  # noqa: E402
from forgery.features import build_modules  # noqa: E402
from forgery.io import apply_protocol, load_mask, load_rgb, protocol_mask  # noqa: E402
from forgery.pipeline import module_params  # noqa: E402

OUT = resolve("outputs")
RES = resolve("results")

COPIED = {   # outputs name -> generated figure
    "09_dataset_samples.png": "phase1/fig_samples.png",
    "10_leakage_shortcuts.png": "phase2/fig_leakage.png",
    "11_shap_importance.png": "phase5/fig_shap_R.png",
    "12_module_ablation.png": "phase5/fig_ablation.png",
    "13_calibration.png": "phase5/fig_calibration_R.png",
    "14_localisation_examples.png": "phase6/fig_examples_R.png",
    "15_test_roc.png": "phase7/run_1/fig_roc_test_R.png",
    "16_error_analysis.png": "phase7/run_1/fig_failures_R.png",
    "17_robustness.png": "phase7b/fig_robustness.png",
    "18_synthetic_forgeries.png": "phase7b/fig_synthetic_examples.png",
    "19_cnn_vs_classical_roc.png": "phase8/fig_roc_cnn_vs_classical.png",
    "20_ela_inputs_format_leak.png": "phase8/fig_ela_examples.png",
    "21_interface.png": "phase8_5/ui_analyze.png",
}


def _save(name: str, arr, cmap=None, vmin=None, vmax=None):
    """One image per file, at the analysed resolution (no axes), like a pipeline stage output."""
    path = OUT / name
    if arr.ndim == 3:
        plt.imsave(path, arr)
    else:
        plt.imsave(path, np.nan_to_num(arr, nan=np.nanmin(arr) if np.isfinite(arr).any() else 0),
                   cmap=cmap or "gray", vmin=vmin, vmax=vmax)
    return path


def _upsample(block_map: np.ndarray, shape) -> np.ndarray:
    return cv2.resize(np.nan_to_num(block_map.astype(np.float32)), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def example(cfg):
    df = read_manifest(cfg).merge(read_splits(cfg), on="path")
    val = df[df.split == "val"]
    return val[(val.forgery_type == "copy-move") & val.mask_valid
               & val.tampered_frac.between(0.05, 0.25)].sort_values("path").iloc[3]


def stage_images(cfg) -> dict:
    ex = example(cfg)
    raw = load_rgb(resolve(ex.path))
    img = apply_protocol(raw, "R", cfg)
    truth = protocol_mask(load_mask(resolve(ex.mask_path)), "R", cfg)
    mods = {m.name: m for m in build_modules(params=module_params(cfg, "R"))}
    s = img.shape[:2]
    _save("00_original.png", raw)
    _save("01a_protocol_R.png", img)
    _save("01b_ground_truth.png", E.overlay(img, truth, color=(0, 200, 0)))
    ela = mods["ela"].maps(img)                                                    # point processing
    _save("02a_ela_residual_x10.png", np.clip(ela["residual"] * 10, 0, 255), vmin=0, vmax=255)
    _save("02b_ela_texture_normalised.png", _upsample(np.where(ela["valid"], ela["norm_block"], 0), s), "magma")
    hm = mods["histogram"].maps(img)                                               # histogram processing
    _save("03a_ela_equalised.png", hm["equalized"], vmin=0, vmax=255)
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].hist(ela["residual"].ravel(), bins=50, color="#4C78A8"); ax[0].set_yscale("log")
    ax[0].set_title("ELA residual: before equalisation", fontsize=9)
    ax[1].hist(hm["equalized"].ravel(), bins=50, color="#E45756")
    ax[1].set_title("after histogram equalisation", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "03b_histogram_before_after.png", dpi=110); plt.close(fig)
    _save("03c_chi2_distance.png", cv2.resize(np.where(hm["valid"], hm["distance"], 0).astype(np.float32),
                                               (s[1], s[0]), interpolation=cv2.INTER_NEAREST), "magma")
    nm = mods["noise"].maps(img)                                                   # spatial filtering
    _save("04a_noise_residual.png", np.clip(np.abs(nm["residual"]), 0, 20), vmin=0, vmax=20)
    _save("04b_noise_sigma.png", _upsample(nm["log_sigma"], s), "magma")
    Y = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)[..., 0].astype(np.float32) - 128   # frequency domain
    h, w = (Y.shape[0] // 8) * 8, (Y.shape[1] // 8) * 8
    coef = np.array([cv2.dct(Y[i:i + 8, j:j + 8])[0, 1] for i in range(0, h, 8) for j in range(0, w, 8)])
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(coef[np.abs(coef) <= 60], bins=121, range=(-60.5, 60.5), color="#4C78A8")
    ax.set_title("DCT coefficient (0,1) over all 8x8 blocks: quantisation spikes", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "05a_dct_coefficient_histogram.png", dpi=110); plt.close(fig)
    outs = {n: mods[n].extract(img) for n in ("dct_dq", "jpeg_ghost", "cm_keypoint", "cm_block")}
    _save("05b_dct_double_quantisation.png", outs["dct_dq"].evidence_map, "magma", 0, 1)
    _save("05c_jpeg_ghost.png", outs["jpeg_ghost"].evidence_map, "magma", 0, 1)
    em = mods["edges"].maps(img)                                                   # edge & corner detection
    _save("06a_canny_edges.png", em["edges"].astype(np.uint8) * 255, vmin=0, vmax=255)
    _save("06b_edge_sharpness.png", _upsample(em["block_sharpness"], s), "magma")
    _save("06c_copymove_keypoints.png", outs["cm_keypoint"].evidence_map, "magma", 0, 1)
    _save("06d_copymove_blocks.png", outs["cm_block"].evidence_map, "magma", 0, 1)
    M = E.load_models("R")                                                         # detection
    res = E.analyse(resolve(ex.path), M, cfg)
    _save("07a_fused_probability.png", res.maps["fused"], "magma", 0, 1)
    _save("07b_detection_overlay.png", E.overlay(img, res.mask))
    panels = [("as released", raw, None), ("protocol R", img, None), ("ground truth", E.overlay(img, truth, color=(0, 200, 0)), None),
              ("ELA (point)", np.clip(ela["residual"] * 10, 0, 255), "gray"),
              ("equalised ELA (histogram)", hm["equalized"], "gray"),
              ("noise sigma (spatial)", _upsample(nm["log_sigma"], s), "magma"),
              ("DCT double quantisation (frequency)", outs["dct_dq"].evidence_map, "magma"),
              ("JPEG ghost (frequency)", outs["jpeg_ghost"].evidence_map, "magma"),
              ("Canny edges (edge)", em["edges"], "gray"),
              ("keypoint copy-move (corner)", outs["cm_keypoint"].evidence_map, "magma"),
              ("fused probability", res.maps["fused"], "magma"),
              (f"detection: {res.verdict} (P={res.confidence:.2f})", E.overlay(img, res.mask), None)]
    fig, axes = plt.subplots(3, 4, figsize=(16, 8.4))
    for ax, (t, a, cm) in zip(axes.ravel(), panels):
        ax.imshow(a, cmap=cm); ax.set_title(t, fontsize=10); ax.axis("off")
    fig.suptitle("The pipeline on one image (validation copy-move, protocol R)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(OUT / "08_summary_grid.png", dpi=90); plt.close(fig)
    return {"image": ex.path, "protocol": "R", "verdict": res.verdict, "p_tampered": round(res.confidence, 4),
            "likely_type": res.likely_type, "region_pct": round(100 * float(res.mask.mean()), 2),
            "copy_move_inliers": int(outs["cm_keypoint"].features["local_n_inliers"]),
            "note": "validation image: the deployed models were refit on all development images, so this is in-sample"}


def _md_json(md: str, label: str, section: str) -> dict:
    part = md.split(section, 1)[1]
    return json.loads(re.search(rf"- {label}: (\{{.*\}})", part).group(1))


def metrics(cfg) -> dict:
    m = {}
    df = read_manifest(cfg).merge(read_splits(cfg), on="path")
    kind = np.where(df.label == 0, "authentic", df.forgery_type)
    m["dataset"] = {"images": len(df), "authentic": int((df.label == 0).sum()), "tampered": int(df.label.sum()),
                    "format_by_class": pd.crosstab(kind, df.format).to_dict(orient="index"),
                    "median_tampered_area_pct": round(100 * float(df.tampered_frac.median()), 2),
                    "split_sizes": df.split.value_counts().to_dict(),
                    "valid_masks": int(df.mask_valid.fillna(False).astype(bool).sum())}
    lk = pd.read_csv(RES / "phase2/leakage_summary.csv")
    piv = lk.pivot_table(index="feature_set", columns="protocol", values="balanced_accuracy_mean")
    m["leakage_balanced_accuracy"] = piv[["A", "C", "B85", "R85"]].round(4).to_dict(orient="index")
    r5 = pd.read_csv(RES / "phase5/results.csv")
    m["dev_cv_auc"] = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in
                       r5[["protocol", "feature_set", "model", "roc_auc_mean", "roc_auc_std"]].to_dict(orient="records")]
    av = pd.read_csv(RES / "phase5/added_value.csv").round(4)
    m["dev_added_value"] = av.to_dict(orient="records")
    t = pd.read_csv(RES / "phase7/run_1/test_classification.csv")
    m["test_classification"] = t[["protocol", "feature_set", "model", "auc", "auc_ci_low", "auc_ci_high", "accuracy",
                                  "precision", "recall", "f1", "balanced_accuracy", "ba_ci_low", "ba_ci_high"]
                                 ].round(4).to_dict(orient="records")
    m["test_added_value"] = pd.read_csv(RES / "phase7/run_1/test_added_value.csv").round(4).to_dict(orient="records")
    md = (RES / "phase7/run_1/test_results.md").read_text()
    o = pd.read_csv(RES / "phase7/run_1/test_outcomes_R.csv")
    m["test_deployed"] = {}
    m["test_localisation"] = {}
    for P in ("R", "B"):
        dep = _md_json(md, "Deployed model", f"## Protocol {P}: deployed model")
        m["test_deployed"][P] = dep
        loc = _md_json(md, "Localisation", f"## Protocol {P}: deployed model")
        per = pd.read_csv(RES / f"phase7/run_1/localisation_per_image_{P}.csv")
        loc["f1_by_area"] = per.groupby("area_bucket").f1.mean().round(4).to_dict()
        loc["f1_by_type"] = per.groupby("forgery_type").f1.mean().round(4).to_dict()
        m["test_localisation"][P] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in loc.items()}
    m["test_deployed"]["R"]["confusion_matrix"] = pd.crosstab(o.label.map({0: "authentic", 1: "tampered"}),
                                                              o.verdict).to_dict(orient="index")
    rb = pd.read_csv(RES / "phase7b/robustness.csv")
    m["robustness"] = rb[["protocol", "condition", "auc", "loc_f1", "coverage", "accuracy_judged"]].round(4).to_dict(orient="records")
    sy = pd.read_csv(RES / "phase7b/synthetic.csv")
    m["synthetic"] = sy[["protocol", "version", "kind", "auc", "loc_f1", "loc_f1_source_and_copy"]].round(4).to_dict(orient="records")
    ex = pd.read_csv(RES / "phase7b/external.csv")
    m["micc_f220"] = ex[["protocol", "auc", "auc_ci", "coverage", "accuracy_judged", "detected_tampered",
                         "false_alarm_authentic"]].round(4).to_dict(orient="records")
    s8 = json.loads((RES / "phase8/summary.json").read_text())
    m["cnn"] = {P: {"cnn_auc": r["auc"]["cnn"], "classical_auc": r["auc"]["classical"],
                    "rank_average_auc": r["auc"]["combination"], "cnn_minus_classical": r["cnn_minus_classical"],
                    "micc_f220": r.get("micc_f220", {}).get("cnn")} for P, r in s8["protocols"].items()}
    ph = json.loads((RES / "phase8/posthoc.json").read_text())
    m["cnn_size_groups_test"] = [{k: r[k] for k in ("protocol", "size_group", "n", "cnn_auc", "classical_auc",
                                                    "cnn_minus_classical")}
                                 for r in ph["size_groups"] if r["split"] == "test"]
    return m


def main():
    cfg = load_config()
    OUT.mkdir(exist_ok=True)
    for name, src in COPIED.items():
        shutil.copy2(RES / src, OUT / name)
    ex = stage_images(cfg)
    m = metrics(cfg)
    m["example_image"] = ex
    (OUT / "metrics.json").write_text(json.dumps(m, indent=2, default=str))
    print(f"wrote {len(list(OUT.glob('*.png')))} figures and metrics.json to {OUT}")


if __name__ == "__main__":
    main()
