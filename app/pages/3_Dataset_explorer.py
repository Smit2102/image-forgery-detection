from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from common import dev_manifest, metrics, page_setup
from forgery import explain as E
from forgery.config import resolve
from forgery.io import load_mask, load_rgb

page_setup("Dataset explorer", "🗃️")
st.title("Dataset explorer: CASIA v2.0")
st.caption("Training and validation images only (10,722). The 1,892 held-out test images are not shown.")
df = dev_manifest()
df["kind"] = np.where(df.label == 0, "authentic", df.forgery_type)

tab_browse, tab_leak, tab_stats = st.tabs(["Browse images", "Why the released dataset leaks", "Statistics"])

with tab_browse:
    c1, c2, c3, c4 = st.columns(4)
    kinds = c1.multiselect("Kind", ["authentic", "copy-move", "splicing"], default=["copy-move", "splicing"])
    fmts = c2.multiselect("File format", sorted(df.format.unique()), default=sorted(df.format.unique()))
    splits = c3.multiselect("Split", ["train", "val"], default=["train", "val"])
    only_mask = c4.checkbox("Only images with a valid mask", value=True)
    sub = df[df.kind.isin(kinds) & df.format.isin(fmts) & df.split.isin(splits)]
    if only_mask:
        sub = sub[(sub.label == 0) | sub.mask_valid.fillna(False).astype(bool)]
    st.caption(f"{len(sub):,} images match.")
    if sub.empty:
        st.info("No images match these filters.")
    else:
        sub = sub.sort_values("path").reset_index(drop=True)
        if "browse_i" not in st.session_state:
            st.session_state.browse_i = 0
        b1, b2, b3 = st.columns([1, 1, 6])
        if b1.button("◀ Previous"):
            st.session_state.browse_i -= 1
        if b2.button("Random ▶"):
            st.session_state.browse_i = int(np.random.default_rng().integers(len(sub)))
        i = st.session_state.browse_i % len(sub)
        r = sub.iloc[i]
        img = load_rgb(resolve(r.path))
        cols = st.columns([2, 2, 1.3])
        cols[0].image(img, caption=r.path, width="stretch")
        if r.label == 1 and isinstance(r.mask_path, str) and bool(r.mask_valid):
            m = load_mask(resolve(r.mask_path))
            if m.shape == img.shape[:2]:
                cols[1].image(E.overlay(img, m, color=(0, 200, 0)), caption="ground-truth tampered region (green)",
                              width="stretch")
        elif r.label == 1:
            cols[1].warning("No valid ground-truth mask for this image.")
        is_jpeg = r.format == "JPEG" and pd.notna(r.jpeg_quality) and r.jpeg_quality > 0
        info = {"Kind": r.kind, "Split": r.split, "File format": r.format, "Size": f"{r.width} x {r.height} px",
                "JPEG quality (estimated)": int(r.jpeg_quality) if is_jpeg else "n/a",
                "Standard IJG tables": ("yes" if bool(r.qtable_standard) else "no") if is_jpeg else "n/a",
                "Tampered area": f"{100 * r.tampered_frac:.1f} %" if pd.notna(r.tampered_frac) else "n/a",
                "Host image id": r.host_id, "Donor image id": r.donor_id if isinstance(r.donor_id, str) else "n/a",
                "Split group": r.split_group}
        cols[2].markdown("\n".join(f"**{k}**: {v}  " for k, v in info.items()))
        st.caption("Analyse this image on the *Analyze an image* page (choose Example, or upload the file). "
                   "It is a training image, so the detector's result on it is optimistic.")

with tab_leak:
    st.markdown(
        "CASIA v2.0 *as released* can be classified almost perfectly **without looking at the image content**. "
        "Most tampered images were saved as uncompressed TIFF, while authentic images are JPEGs, mostly with "
        "the standard IJG quantisation tables. Any detector trained on the raw files can learn this difference "
        "instead of tampering. This project therefore evaluates under protocols that remove it.")
    fmt = (df.groupby(["kind", "format"]).size().rename("images").reset_index())
    st.altair_chart(alt.Chart(fmt).mark_bar().encode(
        x=alt.X("images:Q", stack="normalize", title="share of images"), y=alt.Y("kind:N", title=None, axis=alt.Axis(labelOverlap=False)),
        color=alt.Color("format:N"), tooltip=["kind", "format", "images"]).properties(
        height=alt.Step(34), title="File format by kind"), width="stretch")
    j = df[df.format == "JPEG"].copy()
    j["tables"] = np.where(j.qtable_standard.astype(bool), "standard IJG", "non-standard")
    tab = j.groupby(["kind", "tables"]).size().rename("images").reset_index()
    st.altair_chart(alt.Chart(tab).mark_bar().encode(
        x=alt.X("images:Q", stack="normalize", title="share of JPEG images"), y=alt.Y("kind:N", title=None, axis=alt.Axis(labelOverlap=False)),
        color=alt.Color("tables:N", scale=alt.Scale(range=["#999999", "#4C78A8"])),
        tooltip=["kind", "tables", "images"]).properties(height=alt.Step(34), title="JPEG quantisation tables by kind"),
        width="stretch")
    lk = metrics().get("leakage_balanced_accuracy")
    if lk:
        st.markdown("**Shortcut classifiers** (no forensic features; 5-fold grouped CV on the development set):")
        st.dataframe(pd.DataFrame(lk).T.round(3).rename_axis("feature_set"), width="stretch")
        st.caption("Balanced accuracy (0.5 = chance). A: files as released; C: JPEG files only; "
                   "B85: every image re-saved as JPEG Q85; R85: downscaled x0.75, then re-saved.")

with tab_stats:
    c1, c2 = st.columns(2)
    counts = df.groupby(["split", "kind"]).size().rename("images").reset_index()
    c1.altair_chart(alt.Chart(counts).mark_bar().encode(
        x="split:N", y="images:Q", color="kind:N", xOffset="kind:N", tooltip=["split", "kind", "images"]).properties(
        title="Images per split"), width="stretch")
    t = df[(df.label == 1) & df.tampered_frac.notna()]
    c2.altair_chart(alt.Chart(t.assign(pct=100 * t.tampered_frac)).mark_bar().encode(
        x=alt.X("pct:Q", bin=alt.Bin(maxbins=40), title="tampered area (% of image)"), y=alt.Y("count():Q"),
        color="kind:N").properties(title="Size of the tampered region"), width="stretch")
    sizes = df.assign(size=df.width.astype(str) + " x " + df.height.astype(str))
    top = sizes["size"].value_counts().head(8).rename_axis("size").reset_index(name="images")
    st.markdown("**Most common image sizes**: most of CASIA is 384 x 256 (either orientation).")
    st.dataframe(top, hide_index=True)
