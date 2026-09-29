from __future__ import annotations

import pandas as pd
import streamlit as st

from common import metrics, output_file, page_setup

page_setup("Results dashboard", "📊")

st.markdown("""
<div style="margin-bottom: 1.5rem;">
    <div style="display:flex; align-items:center; gap: 8px;">
        <span style="font-size: 1.8rem;">📊</span>
        <h1 style="font-size: 2.2rem; font-weight: 800; letter-spacing: -0.8px; margin: 0;
                   background: linear-gradient(135deg, #FFFFFF 20%, #94A3B8 100%);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            Empirical Results & Benchmark Intelligence
        </h1>
    </div>
    <p style="font-size: 0.95rem; color: #94A3B8; margin: 4px 0 0 0;">
        Immutable audit trail generated across 7 phases: leakage analysis, nested CV, fusion ablation, held-out evaluation, and deep learning comparison.
    </p>
</div>
""", unsafe_allow_html=True)


M = metrics()
if not M:
    st.error("outputs/metrics.json not found: run `python scripts/make_outputs.py` after the pipeline.")
    st.stop()


def table(key: str) -> pd.DataFrame:
    return pd.DataFrame(M.get(key, []))


def fig(name: str, caption: str = ""):
    f = output_file(name)
    if f.exists():
        st.image(str(f), caption=caption, width="stretch")
    else:
        st.info(f"outputs/{name} not found (run scripts/make_outputs.py).")


def fmt_ci(v, lo, hi) -> str:
    return f"{v:.3f} [{lo:.3f}, {hi:.3f}]"


tabs = st.tabs([
    "🛡️ 1 · Leakage Study",
    "📈 2 · Dev Classification",
    "🎯 3 · Localisation",
    "🏆 4 · Test Set (Held-Out)",
    "🌪️ 5 · Robustness & MICC",
    "🤖 6 · CNN Comparison"
])

with tabs[0]:
    st.markdown("""
    <div style="background: rgba(19, 28, 49, 0.4); padding: 14px 18px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.06); margin-bottom: 14px;">
        <strong style="color: #F8FAFC;">Dataset Leakage Hypothesis & Prevention</strong>
        <div style="color: #94A3B8; font-size: 0.88rem; margin-top: 4px;">
            Can labels be predicted without inspecting semantic content? Evaluating shortcut classifiers (EXIF metadata, dimensions, naive ELA, compression statistics) across protocols.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.dataframe(pd.DataFrame(M["leakage_balanced_accuracy"]).T.round(3).rename_axis("feature_set"), width="stretch")
    st.caption("Protocol Definitions · A: As released · C: JPEG only · B85: Re-saved JPEG Q85 · R85: Resampled x0.75 + JPEG Q85 (Strict Standard).")
    fig("10_leakage_shortcuts.png")

with tabs[1]:
    st.markdown("""
    <div style="background: rgba(19, 28, 49, 0.4); padding: 14px 18px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.06); margin-bottom: 14px;">
        <strong style="color: #F8FAFC;">Grouped Nested Cross-Validation (Dev Set)</strong>
        <div style="color: #94A3B8; font-size: 0.88rem; margin-top: 4px;">
            Verifying whether the 8 classical forensic modules add statistical value over shortcut baseline features (Group-bootstrap paired ΔAUC).
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    r = table("dev_cv_auc")
    st.dataframe(r.pivot_table(index=["feature_set"], columns=["protocol", "model"], values="roc_auc_mean")
                 .round(3), width="stretch")
    av = table("dev_added_value")
    av["ΔAUC [95 % CI]"] = [fmt_ci(a, b, c) for a, b, c in zip(av.delta_auc, av.ci_low, av.ci_high)]
    st.dataframe(av[["protocol", "model", "comparison", "ΔAUC [95 % CI]"]], hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        fig("13_calibration.png", "Isotonic Calibration & Calibrated Uncertain Band")
    with c2:
        fig("11_shap_importance.png", "Global SHAP Module Feature Importance")
        fig("12_module_ablation.png", "Leave-One-Module-Out Ablation Study")

with tabs[2]:
    st.markdown("""
    <div style="background: rgba(19, 28, 49, 0.4); padding: 14px 18px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.06); margin-bottom: 14px;">
        <strong style="color: #F8FAFC;">Pixel-Level Localization on Validation Split</strong>
        <div style="color: #94A3B8; font-size: 0.88rem; margin-top: 4px;">
            Learned logistic fusion combining 8 spatial evidence maps vs individual heuristic baselines.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    rows = []
    for P, L in M["test_localisation"].items():
        rows.append({"protocol": P, "pixel F1": L["f1"], "F1 95% CI": f"[{L['f1_ci'][0]:.3f}, {L['f1_ci'][1]:.3f}]",
                     "IoU": L["iou"], "MCC": L["mcc"], "pixel AUC": L["pixel_auc"],
                     **{f"F1 area {k}": v for k, v in L["f1_by_area"].items()},
                     **{f"F1 {k}": v for k, v in L["f1_by_type"].items()}})
    st.markdown("**Test-set localisation** (766 tampered test images with a valid mask; deployed localiser)")
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")
    fig("14_localisation_examples.png", "Localization Examples on validation images (Protocol R)")

with tabs[3]:
    st.markdown("""
    <div style="background: rgba(19, 28, 49, 0.4); padding: 14px 18px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.06); margin-bottom: 14px;">
        <strong style="color: #F8FAFC;">One-Shot Evaluation on 1,892 Held-Out Test Images</strong>
        <div style="color: #94A3B8; font-size: 0.88rem; margin-top: 4px;">
            Classification AUC with 95% group-bootstrap confidence intervals on strictly isolated test images.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    t = table("test_classification")
    t["AUC [95 % CI]"] = [fmt_ci(a, b, c) for a, b, c in zip(t.auc, t.auc_ci_low, t.auc_ci_high)]
    st.dataframe(t[["protocol", "feature_set", "model", "AUC [95 % CI]", "accuracy", "precision", "recall", "f1",
                    "balanced_accuracy"]].round(3), hide_index=True, width="stretch")
    st.caption("Accuracy, precision, recall and F1 at threshold 0.5 of the uncalibrated model.")
    cm = pd.DataFrame(M["test_deployed"]["R"]["confusion_matrix"]).T
    st.markdown("**Deployed calibrated detector (protocol R): verdicts vs truth**")
    st.dataframe(cm.rename_axis("true class"), width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        fig("15_test_roc.png", "Final Test ROC (Protocol R)")
    with c2:
        fig("16_error_analysis.png", "Failure Modes: Missed Forgeries & False Alarms")

with tabs[4]:
    st.markdown("""
    <div style="background: rgba(19, 28, 49, 0.4); padding: 14px 18px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.06); margin-bottom: 14px;">
        <strong style="color: #F8FAFC;">Adversarial Perturbations & Cross-Domain Generalization</strong>
        <div style="color: #94A3B8; font-size: 0.88rem; margin-top: 4px;">
            Evaluation under simulated real-world degradation (blur, downscaling, compression) and cross-dataset testing on MICC-F220.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    rb = table("robustness")
    st.dataframe(rb.pivot_table(index="condition", columns="protocol", values=["auc", "loc_f1"], sort=False)
                 .round(3), width="stretch")
    fig("17_robustness.png")
    ex = table("micc_f220")
    st.markdown("**Unseen Benchmark: MICC-F220 (110 forgeries from 11 scenes, 110 originals)**")
    st.dataframe(ex[["protocol", "auc", "auc_ci", "coverage", "accuracy_judged", "false_alarm_authentic"]]
                 .round(3), hide_index=True, width="stretch")
    syn = table("synthetic")
    with st.expander("Synthetic Forgeries with Ground Truth"):
        st.dataframe(syn[["protocol", "version", "kind", "auc", "loc_f1"]].round(3), hide_index=True,
                     width="stretch")
        fig("18_synthetic_forgeries.png")

with tabs[5]:
    st.markdown("""
    <div style="background: rgba(19, 28, 49, 0.4); padding: 14px 18px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.06); margin-bottom: 14px;">
        <strong style="color: #F8FAFC;">Classical Signal Processing vs. Deep Learning (ResNet-18)</strong>
        <div style="color: #94A3B8; font-size: 0.88rem; margin-top: 4px;">
            Trained and tested on identical splits. Shows how deep nets exploit file format leakage under raw protocols, but fail to generalize to cross-domain sets compared to classical pipelines.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    rows = []
    for P, r in M["cnn"].items():
        d = r["cnn_minus_classical"]
        rows.append({"protocol": P, "CNN": fmt_ci(r["cnn_auc"]["auc"], *r["cnn_auc"]["ci"]),
                     "classical": fmt_ci(r["classical_auc"]["auc"], *r["classical_auc"]["ci"]),
                     "CNN − classical": fmt_ci(d["delta_auc"], d["ci_low"], d["ci_high"]),
                     "rank average": f"{r['rank_average_auc']['auc']:.3f}",
                     "MICC-F220 CNN": f"{r['micc_f220']['auc']:.3f}" if r.get("micc_f220") else "-"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    fig("19_cnn_vs_classical_roc.png", "Test ROC: Classical vs. Deep Learning")
    c1, _ = st.columns([2, 1])
    with c1:
        fig("20_ela_inputs_format_leak.png", "ELA Input Comparison Across Dataset Releases")
    with st.expander("Subgroup Stratification by Image Dimensions (post-hoc)"):
        st.dataframe(table("cnn_size_groups_test").round(3), hide_index=True, width="stretch")
