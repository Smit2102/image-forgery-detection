"""Streamlit entry point:  .venv/bin/streamlit run app/Home.py

Running this file with plain Python (e.g. the editor's Run button) relaunches it with `streamlit run`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

try:
    import streamlit as st
    from streamlit import runtime
except ImportError:
    venv = Path(__file__).resolve().parents[1] / ".venv" / "bin" / "streamlit"
    sys.exit(f"Streamlit is not installed for {sys.executable}.\n"
             f"Start the app with the project's environment instead:\n    {venv} run {Path(__file__).resolve()}")

if __name__ == "__main__" and not runtime.exists():
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", __file__, *sys.argv[1:]]))

from common import metrics, page_setup, render_kpi

page_setup("Home", "🏠")


def headline() -> dict:
    """Headline numbers of the one-time test evaluation (protocol R), from outputs/metrics.json."""
    m, out = metrics(), {}
    try:
        out["auc"] = next(r["auc"] for r in m["test_classification"]
                          if r["protocol"] == "R" and r["feature_set"] == "forensic_all" and r["model"] == "rf")
        out["acc"] = m["test_deployed"]["R"]["accuracy_judged"]
        out["cov"] = m["test_deployed"]["R"]["coverage"]
        out["f1"] = m["test_localisation"]["R"]["f1"]
        out["micc"] = next(r["auc"] for r in m["micc_f220"] if r["protocol"] == "R")
    except Exception:
        pass
    return out


# Hero Header Section
st.markdown("""
<div style="margin-bottom: 2rem; padding: 2rem; background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
            border-radius: 20px; border: 1px solid rgba(255, 255, 255, 0.08); box-shadow: 0 20px 40px -15px rgba(0,0,0,0.5);">
    <div style="display: flex; gap: 8px; margin-bottom: 12px; flex-wrap: wrap;">
        <span style="background: rgba(99, 102, 241, 0.2); color: #818CF8; border: 1px solid rgba(99, 102, 241, 0.4);
                     padding: 4px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600;">
            🔬 FORENSIC VISION SUITE
        </span>
        <span style="background: rgba(16, 185, 129, 0.2); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.4);
                     padding: 4px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600;">
            STRICT PROTOCOL R
        </span>
        <span style="background: rgba(14, 165, 233, 0.2); color: #38BDF8; border: 1px solid rgba(14, 165, 233, 0.4);
                     padding: 4px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600;">
            8 INTERPRETABLE DETECTORS
        </span>
    </div>
    <h1 style="font-size: 2.6rem; font-weight: 800; letter-spacing: -1px; margin: 0 0 10px 0;
               background: linear-gradient(135deg, #FFFFFF 20%, #94A3B8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        Image Forgery Detection & Forensic Intelligence
    </h1>
    <p style="font-size: 1.05rem; color: #94A3B8; max-width: 900px; margin: 0; line-height: 1.6;">
        Detect and localise <strong style="color:#F1F5F9">copy-move</strong> and <strong style="color:#F1F5F9">splicing</strong>
        manipulations using classical signal processing: Error Level Analysis (ELA), DCT double quantization,
        noise residuals, JPEG ghosts, edge gradients, and keypoint/block matching.
    </p>
</div>
""", unsafe_allow_html=True)

# Performance KPIs
h = headline()
show = lambda k, f: f.format(h[k]) if k in h else "n/a"

c1, c2, c3, c4 = st.columns(4)
with c1:
    render_kpi("🎯", "Test AUC (Strict Protocol R)", show("auc", "{:.3f}"), "Held-out Test",
               "1,892 held-out CASIA v2.0 images, evaluated once without shortcut leakage.")
with c2:
    cov_str = show('cov', '{:.0%}')
    render_kpi("🛡️", "Accuracy on Judged", show("acc", "{:.0%}"), f"{cov_str} coverage",
               f"{cov_str} of test images judged; unconfident samples safely held in uncertain band.")
with c3:
    render_kpi("🌐", "Cross-Dataset AUC", show("micc", "{:.2f}"), "MICC-F220",
               "Evaluated on completely unseen MICC-F220 copy-move dataset.")
with c4:
    render_kpi("🔬", "Localization Pixel F1", show("f1", "{:.2f}"), "Protocol R",
               "Mean pixel F1 across tampered images; ~0.33 on regions larger than 5% area.")

st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

# Interactive Application Modules Grid
st.markdown("""
<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
    <h3 style="margin:0;font-size:1.35rem;font-weight:700;color:#F8FAFC">Forensic Workspaces</h3>
    <span style="font-size:0.8rem;color:#64748B">Select a workspace below to inspect or run detection</span>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

with col1:
    st.markdown("""
    <div class="nav-card">
        <div style="font-size: 1.8rem; margin-bottom: 8px;">🔍</div>
        <div style="font-size: 1.15rem; font-weight: 700; color: #F1F5F9; margin-bottom: 6px;">
            Analyze an Image
        </div>
        <div style="font-size: 0.88rem; color: #94A3B8; line-height: 1.5; margin-bottom: 16px;">
            Upload any photo or pick benchmark samples. Inspect the final verdict, suspected region mask,
            8 forensic maps, and SHAP decision explanations.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/1_Analyze_an_image.py", label="Launch Single Analyzer →", icon="🔎")

with col2:
    st.markdown("""
    <div class="nav-card">
        <div style="font-size: 1.8rem; margin-bottom: 8px;">⚡</div>
        <div style="font-size: 1.15rem; font-weight: 700; color: #F1F5F9; margin-bottom: 6px;">
            Batch Analysis
        </div>
        <div style="font-size: 0.88rem; color: #94A3B8; line-height: 1.5; margin-bottom: 16px;">
            Process high-volume image folders or multi-file uploads at scale. Stream results in real-time
            and export forensic verdict reports as CSV.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/2_Batch_analysis.py", label="Open Batch Processor →", icon="⚡")

with col3:
    st.markdown("""
    <div class="nav-card">
        <div style="font-size: 1.8rem; margin-bottom: 8px;">🗂️</div>
        <div style="font-size: 1.15rem; font-weight: 700; color: #F1F5F9; margin-bottom: 6px;">
            Dataset Explorer
        </div>
        <div style="font-size: 0.88rem; color: #94A3B8; line-height: 1.5; margin-bottom: 16px;">
            Explore the CASIA v2.0 benchmark repository with ground-truth masks. Inspect empirical proof
            of dataset shortcut leakage and mitigation.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/3_Dataset_explorer.py", label="Explore Dataset Archive →", icon="🗂️")

with col4:
    st.markdown("""
    <div class="nav-card">
        <div style="font-size: 1.8rem; margin-bottom: 8px;">📊</div>
        <div style="font-size: 1.15rem; font-weight: 700; color: #F1F5F9; margin-bottom: 6px;">
            Results Dashboard
        </div>
        <div style="font-size: 0.88rem; color: #94A3B8; line-height: 1.5; margin-bottom: 16px;">
            Examine full empirical evaluations: leakage study, nested cross-validation, test curves,
            feature ablations, and modern CNN comparisons.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/4_Results_dashboard.py", label="View Benchmark Dashboard →", icon="📊")

st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

# Forensic Guidance & Methodology Callout
st.markdown("""
<div style="background: rgba(19, 28, 49, 0.7); border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px; padding: 22px; margin-top: 1rem;">
    <div style="display:flex; align-items:center; gap: 8px; margin-bottom: 12px;">
        <span style="font-size: 1.1rem;">⚖️</span>
        <span style="font-weight: 700; font-size: 1.05rem; color: #F1F5F9;">Forensic Operating Methodology & Boundaries</span>
    </div>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; font-size: 0.88rem; color: #94A3B8; line-height: 1.5;">
        <div style="background: rgba(15, 23, 42, 0.5); padding: 14px; border-radius: 10px; border-left: 3px solid #6366F1;">
            <strong style="color: #F8FAFC;">Evidence, Not Absolute Proof:</strong>
            Decisions are based on multi-cue signal inconsistencies. Images falling in the calibrated uncertain band (~50%) are intentionally withheld from judgment to protect against false accusations.
        </div>
        <div style="background: rgba(15, 23, 42, 0.5); padding: 14px; border-radius: 10px; border-left: 3px solid #06B6D4;">
            <strong style="color: #F8FAFC;">Strict Protocol R Resampling:</strong>
            Images are standardized (downscaled x0.75 & Q85 recompressed) to eliminate compression shortcuts so only genuine spatial content evidence is evaluated.
        </div>
        <div style="background: rgba(15, 23, 42, 0.5); padding: 14px; border-radius: 10px; border-left: 3px solid #F59E0B;">
            <strong style="color: #F8FAFC;">Natural Scene Repetitions:</strong>
            Architectural patterns, repeated windows, or low-texture sky regions can sometimes trigger false copy-move keypoint clusters. Inspect SHAP explanations for verification.
        </div>
        <div style="background: rgba(15, 23, 42, 0.5); padding: 14px; border-radius: 10px; border-left: 3px solid #10B981;">
            <strong style="color: #F8FAFC;">Unseen Generalization:</strong>
            Trained strictly on CASIA v2.0 development splits; verified for cross-domain stability against the untouched MICC-F220 copy-move dataset.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
st.caption("Built by Smit Patel · DIP Course Project · CLI execution: `python detect.py <image>`")
