"""Shared helpers for the Streamlit pages (cached models and data, uploads, small widgets)."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from forgery import explain as E
from forgery.config import load_config, resolve
from forgery.data.manifest import read_manifest
from forgery.data.splits import read_splits

IMAGE_TYPES = ["jpg", "jpeg", "png", "tif", "tiff", "bmp", "webp"]
MAX_SIDE = E.MAX_SIDE               # larger images are refused (analysis time and memory)
UPLOAD_DIR = Path(os.environ.get("FORGERY_UI_UPLOAD_DIR") or Path(tempfile.gettempdir()) / "forgery_ui_uploads")
KEEP_DAYS = 7


def _clean_old_uploads():
    """Uploads are user images: remove those older than KEEP_DAYS (runs once per server process)."""
    import shutil
    import time
    if not UPLOAD_DIR.is_dir():
        return
    cutoff = time.time() - KEEP_DAYS * 86400
    for d in UPLOAD_DIR.iterdir():
        try:
            if d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass


if "_forgery_cleaned" not in globals():
    _clean_old_uploads()
    _forgery_cleaned = True

VERDICT_CONFIG = {
    "tampered": {
        "label": "TAMPERED DETECTED",
        "icon": "⚠️",
        "color": "#EF4444",
        "bg_gradient": "linear-gradient(135deg, rgba(239, 68, 68, 0.18) 0%, rgba(153, 27, 27, 0.28) 100%)",
        "border": "rgba(239, 68, 68, 0.5)",
        "glow": "0 8px 32px rgba(239, 68, 68, 0.25)",
        "pulse_class": "hud-pulse-red",
        "desc": "Significant digital manipulation identified across forensic feature maps."
    },
    "authentic": {
        "label": "AUTHENTIC / UNMODIFIED",
        "icon": "🛡️",
        "color": "#10B981",
        "bg_gradient": "linear-gradient(135deg, rgba(16, 185, 129, 0.18) 0%, rgba(5, 150, 105, 0.28) 100%)",
        "border": "rgba(16, 185, 129, 0.5)",
        "glow": "0 8px 32px rgba(16, 185, 129, 0.25)",
        "pulse_class": "hud-pulse-green",
        "desc": "Natural compression history and edge characteristics; no splice or clone detected."
    },
    "uncertain": {
        "label": "UNCERTAIN BAND",
        "icon": "⚖️",
        "color": "#F59E0B",
        "bg_gradient": "linear-gradient(135deg, rgba(245, 158, 11, 0.18) 0%, rgba(180, 83, 9, 0.28) 100%)",
        "border": "rgba(245, 158, 11, 0.5)",
        "glow": "0 8px 32px rgba(245, 158, 11, 0.25)",
        "pulse_class": "hud-pulse-amber",
        "desc": "Evidence falls within calibrated threshold band; not judged to prevent false alarms."
    },
    None: {
        "label": "NO VERDICT",
        "icon": "⚪",
        "color": "#94A3B8",
        "bg_gradient": "linear-gradient(135deg, rgba(100, 116, 139, 0.15) 0%, rgba(71, 85, 105, 0.2) 100%)",
        "border": "rgba(148, 163, 184, 0.3)",
        "glow": "none",
        "pulse_class": "",
        "desc": "Awaiting image analysis."
    }
}

PROTOCOL_HELP = {
    "R": "Strict (recommended): downscaled x0.75 and re-saved as JPEG Q85 before analysis. "
         "Removes format and compression artifacts so only content evidence counts. "
         "Holds up best against social-media processing.",
    "B": "Re-encode: re-saved once as JPEG Q85. Keeps intra-image compression traces. "
         "Stronger on clean CASIA images, but sensitive to previous compression or resizing.",
}

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* Background ambient glow */
.stApp {
    background: radial-gradient(circle at 10% 10%, rgba(99, 102, 241, 0.08) 0%, transparent 45%),
                radial-gradient(circle at 90% 90%, rgba(14, 165, 233, 0.06) 0%, transparent 45%),
                #0A0E1A !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #0E1322 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
}

[data-testid="stSidebarNav"] {
    padding-top: 1rem;
}

[data-testid="stSidebarNav"] a {
    border-radius: 8px !important;
    margin: 3px 0 !important;
    transition: all 0.2s ease !important;
}

[data-testid="stSidebarNav"] a:hover {
    background: rgba(99, 102, 241, 0.12) !important;
    color: #818CF8 !important;
}

/* Tab navigation styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: rgba(19, 28, 49, 0.6);
    padding: 6px;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.08);
}

.stTabs [data-baseweb="tab"] {
    height: 38px;
    border-radius: 8px;
    color: #94A3B8;
    font-weight: 500;
    font-size: 0.9rem;
    border: none !important;
    padding: 6px 16px;
    transition: all 0.2s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #F8FAFC;
    background-color: rgba(255, 255, 255, 0.06);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%) !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.4) !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
}

/* Buttons */
.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    border: 1px solid rgba(255, 255, 255, 0.12);
    transition: all 0.2s ease;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
}

.stButton > button:hover {
    transform: translateY(-2px);
    border-color: #818CF8;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.35);
}

.stDownloadButton > button {
    border-radius: 10px;
    font-weight: 600;
    background: rgba(30, 41, 59, 0.8) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    transition: all 0.2s ease;
}

.stDownloadButton > button:hover {
    background: rgba(99, 102, 241, 0.2) !important;
    border-color: #818CF8 !important;
    transform: translateY(-2px);
}

/* Metric card container */
.kpi-card {
    background: rgba(19, 28, 49, 0.7);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 20px 22px;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.kpi-card:hover {
    transform: translateY(-3px);
    border-color: rgba(99, 102, 241, 0.45);
    box-shadow: 0 12px 28px -5px rgba(99, 102, 241, 0.25);
}

.kpi-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #6366F1, #06B6D4);
    opacity: 0.8;
}

/* Feature cards */
.nav-card {
    background: linear-gradient(135deg, rgba(23, 33, 56, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 24px;
    transition: all 0.25s ease;
    height: 100%;
}

.nav-card:hover {
    border-color: rgba(129, 140, 248, 0.5);
    transform: translateY(-3px);
    box-shadow: 0 16px 32px -8px rgba(0, 0, 0, 0.5), 0 0 24px -4px rgba(99, 102, 241, 0.25);
}

/* Radar pulse animations for HUD */
@keyframes hud-pulse-red {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
    70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(239, 68, 68, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
}

@keyframes hud-pulse-green {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
    70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

@keyframes hud-pulse-amber {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.7); }
    70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(245, 158, 11, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(245, 158, 11, 0); }
}

.hud-pulse-red {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #EF4444;
    animation: hud-pulse-red 2s infinite;
}

.hud-pulse-green {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #10B981;
    animation: hud-pulse-green 2s infinite;
}

.hud-pulse-amber {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #F59E0B;
    animation: hud-pulse-amber 2s infinite;
}

/* Image preview containers */
[data-testid="stImage"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
}

/* Dataframe containers */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.08);
}
</style>
"""


def page_setup(title: str, icon: str = "🔎"):
    st.set_page_config(page_title=f"{title} · Forgery detection", page_icon=icon, layout="wide")
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    
    # Custom top sidebar brand header
    st.sidebar.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding:6px 0 16px 0;border-bottom:1px solid rgba(255,255,255,0.08);margin-bottom:12px">
        <div style="font-size:1.6rem;background:linear-gradient(135deg,#6366F1,#06B6D4);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:800;letter-spacing:-0.5px">
            FORENSIQ
        </div>
        <span style="font-size:0.65rem;background:rgba(99,102,241,0.2);color:#A5B4FC;padding:2px 7px;border-radius:20px;border:1px solid rgba(99,102,241,0.4);font-weight:600">
            DIP v2.0
        </span>
    </div>
    """, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading models ...")
def models(protocol: str) -> E.Models:
    return E.load_models(protocol)


@st.cache_resource
def config() -> dict:
    return load_config()


@st.cache_data(show_spinner="Loading the dataset index ...")
def dev_manifest() -> pd.DataFrame:
    """Train + validation images only: the test split stays unseen, also in the interface."""
    df = read_manifest().merge(read_splits(), on="path")
    return df[df.split.isin(["train", "val"])].reset_index(drop=True)


def protocol_picker(key: str = "protocol") -> str:
    options = E.available_protocols()
    if not options:
        st.error("No trained models found in models/. Run scripts/run_phase5.py and scripts/run_phase6.py first.")
        st.stop()
    main = E.main_protocol()
    p = st.sidebar.radio("Analysis protocol", options, index=options.index(main) if main in options else 0,
                         format_func=lambda x: {"R": "R · Strict (Recommended)", "B": "B · Re-encode"}.get(x, x),
                         key=key)
    st.sidebar.caption(PROTOCOL_HELP.get(p, ""))
    return p


def save_upload(upload) -> tuple[Path | None, list[str]]:
    """Store an upload once per content (so reruns reuse it and its cached result) and prepare it for the
    detector (full decode, size limits, EXIF rotation, 16-bit). Shows an error and returns None if unusable."""
    data = upload.getvalue()
    digest = hashlib.sha1(data).hexdigest()[:16]
    folder = UPLOAD_DIR / digest
    src = folder / ("upload" + (Path(upload.name).suffix.lower() or ".img"))
    if not src.exists():
        folder.mkdir(parents=True, exist_ok=True)
        src.write_bytes(data)
    try:
        path, notes = E.prepare_image(src, folder)
    except E.ImageError as exc:
        st.error(f"{upload.name}: {exc}")
        return None, []
    return path, notes


def verdict_badge(verdict: str | None, p: float | None):
    """Futuristic forensic inspection HUD verdict card with animated radar pulse and confidence meter."""
    cfg = VERDICT_CONFIG.get(verdict, VERDICT_CONFIG[None])
    pct_val = 0 if p is None else int(round(p * 100))
    p_str = "N/A" if p is None else f"{pct_val}%"
    
    pulse_dot = f'<span class="{cfg["pulse_class"]}" style="margin-right:8px"></span>' if cfg["pulse_class"] else ''
    
    progress_bar = ""
    if p is not None:
        progress_bar = (
            f'<div style="margin-top:14px;background:rgba(0,0,0,0.4);border-radius:10px;height:8px;overflow:hidden;border:1px solid rgba(255,255,255,0.06)">'
            f'<div style="width:{pct_val}%;height:100%;background:{cfg["color"]};box-shadow:0 0 10px {cfg["color"]};border-radius:10px;transition:width 0.8s ease"></div>'
            f'</div>'
            f'<div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#94A3B8;margin-top:6px;font-family:\'JetBrains Mono\',monospace">'
            f'<span>P(tampered): {p:.3f}</span>'
            f'<span>Confidence: {p_str}</span>'
            f'</div>'
        )

    html = (
        f'<div style="background:{cfg["bg_gradient"]};border:1px solid {cfg["border"]};box-shadow:{cfg["glow"]};border-radius:16px;padding:20px 22px;backdrop-filter:blur(10px);margin-bottom:14px">'
        f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px">'
        f'<span style="font-size:0.72rem;letter-spacing:1.2px;text-transform:uppercase;color:#94A3B8;font-weight:700">FORENSIC VERDICT</span>'
        f'<span style="font-size:0.8rem;background:rgba(255,255,255,0.08);padding:2px 8px;border-radius:12px;color:#CBD5E1;border:1px solid rgba(255,255,255,0.05)">{cfg["icon"]} System Judgement</span>'
        f'</div>'
        f'<div style="display:flex;align-items:center;font-size:1.45rem;font-weight:800;color:{cfg["color"]};letter-spacing:0.5px">{pulse_dot} {cfg["label"]}</div>'
        f'<div style="font-size:0.85rem;color:#CBD5E1;margin-top:4px;line-height:1.4">{cfg["desc"]}</div>'
        f'{progress_bar}'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_kpi(icon: str, title: str, value: str, badge: str, help_text: str = ""):
    """Render a premium glassmorphic KPI metric card."""
    import textwrap
    help_attr = f'title="{help_text}"' if help_text else ""
    html = textwrap.dedent(f"""
    <div class="kpi-card" {help_attr}>
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px">
            <span style="font-size:1.3rem">{icon}</span>
            <span style="font-size:0.7rem;font-weight:600;background:rgba(99,102,241,0.15);color:#A5B4FC;padding:3px 8px;border-radius:20px;border:1px solid rgba(99,102,241,0.3)">
                {badge}
            </span>
        </div>
        <div style="font-size:2.2rem;font-weight:800;color:#F8FAFC;letter-spacing:-0.5px;line-height:1.1">
            {value}
        </div>
        <div style="font-size:0.85rem;color:#94A3B8;font-weight:500;margin-top:6px">
            {title}
        </div>
    </div>
    """).strip()
    st.markdown(html, unsafe_allow_html=True)


def to_png_bytes(arr: np.ndarray) -> bytes:
    import io
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def heat(m: np.ndarray) -> np.ndarray:
    """A [0, 1] map as an RGB heat image (magma) on a fixed 0-1 scale, so maps are comparable."""
    import matplotlib
    rgba = matplotlib.colormaps["magma"](np.clip(np.nan_to_num(m), 0, 1))
    return (rgba[..., :3] * 255).astype(np.uint8)


def result_file(rel: str) -> Path:
    return resolve(rel)
