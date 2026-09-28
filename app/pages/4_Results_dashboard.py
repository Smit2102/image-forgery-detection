from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from common import page_setup, result_file

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


def csv(rel: str) -> pd.DataFrame | None:
    f = result_file(rel)
    return pd.read_csv(f) if f.exists() else None


def fig(rel: str, caption: str = ""):
    f = result_file(rel)
    if f.exists():
        st.image(str(f), caption=caption, width="stretch")
    else:
        st.info(f"Artifact {rel} not found in results directory.")


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
    
    lk = csv("results/phase2/leakage_summary.csv")
    if lk is not None:
        piv = lk.pivot_table(index="feature_set", columns="protocol", values="balanced_accuracy_mean").round(3)
        main = [c for c in ("A", "C", "B85", "R85") if c in piv]
        st.dataframe(piv[main], width="stretch")
        st.caption("Protocol Definitions · A: As released · C: JPEG only · B85: Re-saved JPEG Q85 · R85: Resampled x0.75 + JPEG Q85 (Strict Standard).")
        with st.expander("Explore all protocol variants (quality sensitivity, subsets)"):
            st.dataframe(piv, width="stretch")
    fig("results/phase2/fig_leakage.png")

with tabs[1]:
    st.markdown("""
    <div style="background: rgba(19, 28, 49, 0.4); padding: 14px 18px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.06); margin-bottom: 14px;">
        <strong style="color: #F8FAFC;">Grouped Nested Cross-Validation (Dev Set)</strong>
        <div style="color: #94A3B8; font-size: 0.88rem; margin-top: 4px;">
            Verifying whether the 8 classical forensic modules add statistical value over shortcut baseline features (Group-bootstrap paired ΔAUC).
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    r = csv("results/phase5/results.csv")
    if r is not None:
        st.dataframe(r.pivot_table(index=["feature_set"], columns=["protocol", "model"], values="roc_auc_mean")
                     .round(3), width="stretch")
    av = csv("results/phase5/added_value.csv")
    if av is not None:
        av["ΔAUC [95 % CI]"] = [fmt_ci(a, b, c) for a, b, c in zip(av.delta_auc, av.ci_low, av.ci_high)]
        st.dataframe(av[["protocol", "model", "comparison", "ΔAUC [95 % CI]"]], hide_index=True,
                     width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        fig("results/phase5/fig_roc_R.png", "Out-of-fold ROC (Strict Protocol R)")
        fig("results/phase5/fig_calibration_R.png", "Isotonic Calibration & Calibrated Uncertain Band")
    with c2:
        fig("results/phase5/fig_shap_R.png", "Global SHAP Module Feature Importance")
        fig("results/phase5/fig_ablation.png", "Leave-One-Module-Out Ablation Study")

with tabs[2]:
    st.markdown("""
    <div style="background: rgba(19, 28, 49, 0.4); padding: 14px 18px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.06); margin-bottom: 14px;">
        <strong style="color: #F8FAFC;">Pixel-Level Localization on Validation Split</strong>
        <div style="color: #94A3B8; font-size: 0.88rem; margin-top: 4px;">
            Learned logistic fusion combining 8 spatial evidence maps vs individual heuristic baselines.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    for P in ("R", "B"):
        m = csv(f"results/phase6/methods_{P}.csv")
        if m is not None:
            st.markdown(f"**Protocol {P} Performance**")
            st.dataframe(m[["method", "f1", "iou", "mcc", "pixel_auc", "authentic_any_region"]].round(3),
                         hide_index=True, width="stretch")
    fig("results/phase6/fig_examples_R.png", "Localization Examples (Protocol R)")

with tabs[3]:
    st.markdown("""
    <div style="background: rgba(19, 28, 49, 0.4); padding: 14px 18px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.06); margin-bottom: 14px;">
        <strong style="color: #F8FAFC;">One-Shot Evaluation on 1,892 Held-Out Test Images</strong>
        <div style="color: #94A3B8; font-size: 0.88rem; margin-top: 4px;">
            Classification AUC with 95% group-bootstrap confidence intervals on strictly isolated test images.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    t = csv("results/phase7/run_1/test_classification.csv")
    if t is not None:
        t["AUC [95 % CI]"] = [fmt_ci(a, b, c) for a, b, c in zip(t.auc, t.auc_ci_low, t.auc_ci_high)]
        st.dataframe(t[["protocol", "feature_set", "model", "AUC [95 % CI]", "balanced_accuracy"]].round(3),
                     hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        fig("results/phase7/run_1/fig_roc_test_R.png", "Final Test ROC (Protocol R)")
    with c2:
        fig("results/phase7/run_1/fig_failures_R.png", "Failure Modes: Missed Forgeries & False Alarms")
    f = result_file("results/phase7/run_1/test_results.md")
    if f.exists():
        with st.expander("Full Verified Test Report (Markdown Document)"):
            st.markdown(f.read_text())

with tabs[4]:
    st.markdown("""
    <div style="background: rgba(19, 28, 49, 0.4); padding: 14px 18px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.06); margin-bottom: 14px;">
        <strong style="color: #F8FAFC;">Adversarial Perturbations & Cross-Domain Generalization</strong>
        <div style="color: #94A3B8; font-size: 0.88rem; margin-top: 4px;">
            Evaluation under simulated real-world degradation (blur, downscaling, compression) and cross-dataset testing on MICC-F220.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    rb = csv("results/phase7b/robustness.csv")
    if rb is not None:
        st.dataframe(rb.pivot_table(index="condition", columns="protocol", values=["auc", "loc_f1"], sort=False)
                     .round(3), width="stretch")
    fig("results/phase7b/fig_robustness.png")
    ex = csv("results/phase7b/external.csv")
    if ex is not None:
        st.markdown("**Unseen Benchmark: MICC-F220 (110 forgeries, 110 originals)**")
        st.dataframe(ex[["protocol", "auc", "auc_ci", "coverage", "accuracy_judged", "false_alarm_authentic"]]
                     .round(3), hide_index=True, width="stretch")
    syn = csv("results/phase7b/synthetic.csv")
    if syn is not None:
        with st.expander("Synthetic Forgeries with Ground Truth"):
            st.dataframe(syn[["protocol", "version", "kind", "auc", "loc_f1"]].round(3), hide_index=True,
                         width="stretch")
            fig("results/phase7b/fig_synthetic_examples.png")

with tabs[5]:
    st.markdown("""
    <div style="background: rgba(19, 28, 49, 0.4); padding: 14px 18px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.06); margin-bottom: 14px;">
        <strong style="color: #F8FAFC;">Classical Signal Processing vs. Deep Learning (ResNet-18)</strong>
        <div style="color: #94A3B8; font-size: 0.88rem; margin-top: 4px;">
            Trained and tested on identical splits. Shows how deep nets exploit file format leakage under raw protocols, but fail to generalize to cross-domain sets compared to classical pipelines.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    s = result_file("results/phase8/summary.json")
    if s.exists():
        d = json.loads(s.read_text())
        rows = []
        for P, r in d["protocols"].items():
            a = r["auc"]
            rows.append({"protocol": P, "CNN": fmt_ci(a["cnn"]["auc"], *a["cnn"]["ci"]),
                         "classical": fmt_ci(a["classical"]["auc"], *a["classical"]["ci"]),
                         "CNN − classical": fmt_ci(r["cnn_minus_classical"]["delta_auc"],
                                                   r["cnn_minus_classical"]["ci_low"],
                                                   r["cnn_minus_classical"]["ci_high"]),
                         "rank average": f"{a['combination']['auc']:.3f}",
                         "MICC-F220 CNN / classical": (f"{r['micc_f220']['cnn']['auc']:.3f} / "
                                                       f"{r['micc_f220']['classical_auc']:.3f}")
                         if "micc_f220" in r else "-"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    fig("results/phase8/fig_roc_cnn_vs_classical.png", "Test ROC: Classical vs. Deep Learning")
    c1, _ = st.columns([2, 1])
    with c1:
        fig("results/phase8/fig_ela_examples.png", "ELA Input Comparison Across Dataset Releases")
    sg = csv("results/phase8/posthoc_size_groups.csv")
    if sg is not None:
        with st.expander("Subgroup Stratification by Image Dimensions"):
            st.dataframe(sg[sg.split == "test"][["protocol", "size_group", "n", "cnn_auc", "classical_auc",
                                                 "cnn_minus_classical"]].round(3),
                         hide_index=True, width="stretch")
