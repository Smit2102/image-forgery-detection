from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from common import (IMAGE_TYPES, config, dev_manifest, heat, models, page_setup, protocol_picker, save_upload,
                    to_png_bytes, verdict_badge)
from forgery import explain as E
from forgery.config import resolve
from forgery.io import load_mask, protocol_mask

page_setup("Analyze an image", "🔎")

# Header banner
st.markdown("""
<div style="margin-bottom: 1.5rem;">
    <div style="display:flex; align-items:center; gap: 8px;">
        <span style="font-size: 1.8rem;">🔎</span>
        <h1 style="font-size: 2.2rem; font-weight: 800; letter-spacing: -0.8px; margin: 0;
                   background: linear-gradient(135deg, #FFFFFF 20%, #94A3B8 100%);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            Forensic Image Analyzer
        </h1>
    </div>
    <p style="font-size: 0.95rem; color: #94A3B8; margin: 4px 0 0 0;">
        Multi-modal forensic scrutiny: upload or pick an image to extract 8 forensic evidence maps, detect localized tampering, and review calibrated confidence.
    </p>
</div>
""", unsafe_allow_html=True)

protocol = protocol_picker()
M = models(protocol)
cfg = config()


def examples() -> dict[str, dict]:
    """A few fixed examples: unseen MICC-F220 images (if downloaded) and CASIA validation images."""
    out = {}
    micc = resolve("data/external/MICC-F220")
    if (micc / "groundtruthDB_220.txt").exists():
        gt = pd.read_csv(micc / "groundtruthDB_220.txt", sep=r"\s+", header=None, names=["file", "label"])
        scene = gt.file.str.extract(r"^(.+?)_?(?:tamp\d+|scale)\.[A-Za-z]+$", expand=False).fillna(gt.file)
        forged = sorted(scene[gt.label == 1].unique())[:3]
        for sc in forged:                      # one forgery and the original of three forged scenes
            for lab, kind in ((1, "copy-move forgery"), (0, "original")):
                f = gt[(scene == sc) & (gt.label == lab)].file.sort_values()
                if len(f):
                    out[f"MICC-F220 · {f.iloc[0]} ({kind}, unseen dataset)"] = {"path": micc / f.iloc[0], "mask": None}
    df = dev_manifest()
    val = df[df.split == "val"]
    picks = [("copy-move", val[(val.forgery_type == "copy-move") & val.mask_valid]),
             ("splicing", val[(val.forgery_type == "splicing") & val.mask_valid]),
             ("authentic", val[val.label == 0])]
    for kind, sub in picks:
        for _, r in sub.sort_values("path").iloc[[0, 7]].iterrows():
            out[f"CASIA · {Path(r.path).name} ({kind}, training data)"] = {
                "path": resolve(r.path), "mask": resolve(r.mask_path) if isinstance(r.mask_path, str) and r.mask_valid
                else None}
    return out


# Source selector in styled container
st.markdown("""
<div style="background: rgba(19, 28, 49, 0.5); padding: 14px 18px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.06); margin-bottom: 1rem;">
    <span style="font-size:0.75rem; text-transform:uppercase; letter-spacing:1px; color:#818CF8; font-weight:700;">
        Input Selection
    </span>
</div>
""", unsafe_allow_html=True)

src = st.radio("Image source", ["Upload", "Example"], horizontal=True, key="source", label_visibility="collapsed")
path, truth_path, name, notes = None, None, None, []
if src == "Upload":
    up = st.file_uploader("Upload an image to inspect", type=IMAGE_TYPES)
    if up is not None:
        (path, notes), name = save_upload(up), up.name
else:
    ex = examples()
    choice = st.selectbox("Select Benchmark Example", list(ex))
    path, truth_path, name = ex[choice]["path"], ex[choice]["mask"], Path(ex[choice]["path"]).name
    if "training data" in choice:
        st.info("ℹ️ This CASIA image was part of the training dataset. Results here are optimistic; use your own photos or the MICC-F220 examples for unbiased evaluation.")

if path is None:
    st.stop()
for n in notes:
    st.info(f"Pre-processing notice: Image was {n} before forensic extraction.")


@st.cache_data(show_spinner="Running 8 forensic pipelines, learned fusion & explainability ...", max_entries=32)
def run(path_str: str, protocol: str, mtime: float):
    m = models(protocol)
    res = E.analyse(path_str, m, config())
    return res, E.explain(m, res.features, top=12), E.analysed_image(path_str, protocol, config())


try:
    res, contrib, img = run(str(path), protocol, Path(path).stat().st_mtime)
except Exception as exc:
    st.error(f"Analysis pipeline error for {name}: {type(exc).__name__}: {exc}")
    st.stop()


@st.cache_data(show_spinner="Generating high-resolution forensic PDF report ...", max_entries=16)
def pdf_bytes(path_str: str, protocol: str, mtime: float, name: str) -> bytes:
    r, c, im = run(path_str, protocol, mtime)
    return E.pdf_report(r, im, models(protocol), c, name)


st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

left, right = st.columns([1, 2], gap="medium")
with left:
    verdict_badge(res.verdict, res.confidence)
    
    # Verdict summary & details
    st.markdown(f"""
    <div style="background: rgba(19, 28, 49, 0.6); border: 1px solid rgba(255,255,255,0.06); border-radius: 14px; padding: 16px 18px; margin-bottom: 12px;">
        <div style="font-size: 0.92rem; color: #E2E8F0; line-height: 1.5; margin-bottom: 12px;">
            {E.verdict_text(res, M)}
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 0.8rem;">
            <div style="background: rgba(15, 23, 42, 0.6); padding: 8px 10px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.04);">
                <div style="color: #64748B; font-size: 0.7rem; text-transform: uppercase;">Manipulation Type</div>
                <div style="color: #F8FAFC; font-weight: 700; margin-top: 2px;">{res.likely_type or 'None / Inconclusive'}</div>
            </div>
            <div style="background: rgba(15, 23, 42, 0.6); padding: 8px 10px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.04);">
                <div style="color: #64748B; font-size: 0.7rem; text-transform: uppercase;">Suspected Region</div>
                <div style="color: #F8FAFC; font-weight: 700; margin-top: 2px;">{f"{100 * res.mask.mean():.1f} %" if res.mask is not None else '0 %'}</div>
            </div>
            <div style="background: rgba(15, 23, 42, 0.6); padding: 8px 10px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.04);">
                <div style="color: #64748B; font-size: 0.7rem; text-transform: uppercase;">Resolution</div>
                <div style="color: #F8FAFC; font-weight: 600; margin-top: 2px;">{res.shape[1]}×{res.shape[0]} px</div>
            </div>
            <div style="background: rgba(15, 23, 42, 0.6); padding: 8px 10px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.04);">
                <div style="color: #64748B; font-size: 0.7rem; text-transform: uppercase;">Processing Time</div>
                <div style="color: #F8FAFC; font-weight: 600; margin-top: 2px;">{sum(res.timings_ms.values()) / 1000:.2f} s</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with right:
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 8px;">
        <span style="font-size: 0.75rem; letter-spacing: 1px; text-transform: uppercase; color: #818CF8; font-weight: 700;">
            Visual Localization Viewports
        </span>
        <span style="font-size: 0.75rem; color: #64748B;">Protocol {protocol}</span>
    </div>
    """.format(protocol=protocol), unsafe_allow_html=True)
    
    cols = st.columns(3 if truth_path is None else 4)
    cols[0].image(img, caption="1 · Preprocessed Input", width="stretch")
    cols[1].image(E.overlay(img, res.mask), caption="2 · Suspected Region (Red)", width="stretch")
    if "fused" in res.maps:
        cols[2].image(heat(res.maps["fused"]), caption="3 · Fused Tampering Probability", width="stretch")
    if truth_path is not None:
        t = protocol_mask(load_mask(truth_path), protocol, cfg)
        if t.shape == img.shape[:2]:
            cols[3].image(E.overlay(img, t, color=(0, 200, 0)), caption="4 · Ground Truth (Green)",
                          width="stretch")

st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

tab_why, tab_maps, tab_feat, tab_dl = st.tabs(["💡 Why This Verdict", "🗺️ Evidence Maps (8 Modules)", "📋 All Extracted Features", "📥 Forensic Export & Reports"])

with tab_why:
    st.markdown("""
    <div style="background: rgba(19, 28, 49, 0.4); padding: 14px 18px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.06); margin-bottom: 14px;">
        <span style="font-weight: 600; color: #F8FAFC; font-size: 0.95rem;">Key Evidence Drivers</span>
    </div>
    """, unsafe_allow_html=True)
    
    for line in E.evidence_text(res, contrib):
        st.markdown(f"- {line}")
        
    chart = (alt.Chart(contrib.assign(direction=np.where(contrib.contribution > 0, "towards tampered",
                                                          "towards authentic")))
             .mark_bar(cornerRadius=4)
             .encode(x=alt.X("contribution:Q", title="SHAP Feature Contribution to P(tampered)"),
                     y=alt.Y("description:N", sort=None, title=None, axis=alt.Axis(labelLimit=420)),
                     color=alt.Color("direction:N", scale=alt.Scale(domain=["towards tampered", "towards authentic"],
                                                                    range=["#EF4444", "#3B82F6"]), title=None),
                     tooltip=["feature", alt.Tooltip("value:Q", format=".4g"),
                              alt.Tooltip("contribution:Q", format="+.4f")])
             .properties(height=28 * len(contrib)))
    st.altair_chart(chart, width="stretch")
    st.caption("SHAP decomposes the random forest's decision score into interpretable forensic contributions. "
               "The verdict applies isotonic probability calibration.")

with tab_maps:
    st.caption("All 8 forensic modules use the normalized [0, 1] Magma spectrum (Dark/Black = zero signal, Bright Gold/Yellow = peak tampering evidence).")
    names = [n for n in E.MODULES if n in res.maps and n != "fused"]
    for i in range(0, len(names), 4):
        cols = st.columns(4)
        for c, n in zip(cols, names[i:i + 4]):
            c.image(heat(res.maps[n]), width="stretch")
            c.markdown(f"**{E.MODULES[n][0]}**")
            c.caption(E.MODULES[n][1])

with tab_feat:
    f = pd.DataFrame({"feature": list(res.features), "value": list(res.features.values())})
    f["description"] = [E.describe_feature(n) for n in f.feature]
    st.dataframe(f, width="stretch", hide_index=True, height=420)

with tab_dl:
    stem = Path(name).stem
    st.markdown("""
    <div style="background: rgba(19, 28, 49, 0.4); padding: 14px 18px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.06); margin-bottom: 16px;">
        <span style="font-weight: 600; color: #F8FAFC;">Export Forensic Artifacts</span>
        <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 4px;">Download formal verification reports and high-resolution binary mask overlays.</div>
    </div>
    """, unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.download_button("📄 Forensic PDF Report", pdf_bytes(str(path), protocol, Path(path).stat().st_mtime, name),
                       f"{stem}_report.pdf", "application/pdf", on_click="ignore")
    c2.download_button("📊 JSON Metadata", json.dumps(E.report_dict(res, M, contrib, name), indent=2),
                       f"{stem}_report.json", "application/json", on_click="ignore")
    if res.mask is not None:
        c3.download_button("🎭 Binary Mask (PNG)", to_png_bytes((res.mask * 255).astype(np.uint8)), f"{stem}_mask.png",
                           "image/png", on_click="ignore")
        c4.download_button("🖼️ Overlay Composite (PNG)", to_png_bytes(E.overlay(img, res.mask)), f"{stem}_overlay.png",
                           "image/png", on_click="ignore")
