from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

from common import IMAGE_TYPES, UPLOAD_DIR, config, models, page_setup, protocol_picker, to_png_bytes
from forgery import explain as E

page_setup("Batch analysis", "🗂️")
st.title("Batch analysis")
protocol = protocol_picker()
M = models(protocol)
MAX_FILES = 200
COLUMNS = ["image", "verdict", "p_tampered", "likely_type", "region_pct", "copy_move_inliers", "strongest_evidence",
           "note", "error"]

st.markdown(f"Analyse up to {MAX_FILES} images at once, either uploaded or from a folder on this computer. "
            "A typical photo takes under a second; the largest allowed images (4,000 px) take up to about 7 s.")
src = st.radio("Source", ["Upload files", "Folder on this computer"], horizontal=True, key="source")

items: list[tuple[str, Path | None, bytes | None]] = []        # (display name, path on disk, uploaded bytes)
if src == "Upload files":
    ups = st.file_uploader("Images", type=IMAGE_TYPES, accept_multiple_files=True)
    if ups and len(ups) > MAX_FILES:
        st.warning(f"Only the first {MAX_FILES} files are analysed.")
    items = [(up.name, None, up.getvalue()) for up in (ups or [])[:MAX_FILES]]
else:
    folder = st.text_input("Folder path", placeholder="/Users/me/Pictures/to_check")
    if folder:
        d = Path(folder).expanduser()
        if not d.is_dir():
            st.error("This is not a folder on this computer.")
        else:
            files = sorted(f for f in d.iterdir() if f.is_file() and not f.name.startswith(".")
                           and f.suffix.lower().lstrip(".") in IMAGE_TYPES)
            if not files:
                st.warning("No image files (" + ", ".join(IMAGE_TYPES) + ") in this folder.")
            elif len(files) > MAX_FILES:
                st.warning(f"{len(files)} images found; only the first {MAX_FILES} are analysed.")
            items = [(f.name, f, None) for f in files[:MAX_FILES]]
            if files:
                st.caption(f"{len(items)} images found.")


def _digest(b: bytes | str) -> str:
    return hashlib.sha1(b if isinstance(b, bytes) else b.encode()).hexdigest()[:16]


key = (protocol, tuple((n, str(p), None if b is None else _digest(b)) for n, p, b in items))
if items and st.button(f"Analyse {len(items)} image(s)", type="primary"):
    rows, overlays = [], {}
    bar = st.progress(0.0, text="Analysing ...")
    for i, (name, path, data) in enumerate(items):
        row = dict.fromkeys(COLUMNS)
        row["image"] = name
        try:
            work = UPLOAD_DIR / (_digest(data) if data is not None else _digest(str(path)))   # created only if needed
            if data is not None:                      # uploaded: stored once per content
                work.mkdir(parents=True, exist_ok=True)
                path = work / ("upload" + (Path(name).suffix.lower() or ".img"))
                if not path.exists():
                    path.write_bytes(data)
            prepared, notes = E.prepare_image(path, work)   # same checks as the single-image page
            res = E.analyse(prepared, M, config())
            top = E.explain(M, res.features, top=1).iloc[0]
            row.update({"verdict": res.verdict, "p_tampered": res.confidence, "likely_type": res.likely_type,
                        "region_pct": None if res.mask is None else 100 * float(res.mask.mean()),
                        "copy_move_inliers": int(res.features.get("cm_keypoint.local_n_inliers", 0)),
                        "strongest_evidence": top.description, "note": "; ".join(notes) or None})
            if res.mask is not None and res.mask.any():
                overlays[i] = E.overlay(E.analysed_image(prepared, protocol, config()), res.mask)
        except E.ImageError as exc:
            row["error"] = str(exc)
        except Exception as exc:                  # one bad file must not stop the batch
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        bar.progress((i + 1) / len(items), text=f"{i + 1} / {len(items)}: {name}")
    bar.empty()
    st.session_state["batch"] = {"key": key, "df": pd.DataFrame(rows, columns=COLUMNS), "overlays": overlays}

batch = st.session_state.get("batch")
if not batch or batch["key"] != key:            # results stay until the input or the protocol changes
    st.stop()

df, overlays = batch["df"], batch["overlays"]
counts = df.verdict.value_counts()
c = st.columns(4)
for col, v in zip(c, ["tampered", "uncertain", "authentic"]):
    col.metric(v.capitalize(), int(counts.get(v, 0)))
c[3].metric("Errors", int(df.error.notna().sum()))
st.dataframe(df.style.map(lambda v: {"tampered": "color:#C62828;font-weight:600", "authentic": "color:#2E7D32",
                                     "uncertain": "color:#B26A00"}.get(v, ""), subset=["verdict"])
             .format({"p_tampered": "{:.3f}", "region_pct": "{:.1f}", "copy_move_inliers": "{:.0f}"}, na_rep="–"),
             width="stretch", hide_index=True)

d1, d2 = st.columns(2)
d1.download_button("Download results (CSV)", df.to_csv(index=False), f"batch_results_{protocol}.csv", "text/csv",
                   on_click="ignore")
if overlays:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for i, arr in overlays.items():               # row number in the name: duplicate file names stay apart
            z.writestr(f"{i + 1:03d}_{Path(df.image[i]).stem}_overlay.png", to_png_bytes(arr))
    d2.download_button(f"Download {len(overlays)} overlays (ZIP)", buf.getvalue(), "overlays.zip", "application/zip",
                       on_click="ignore")

flagged = df[df.verdict == "tampered"].sort_values("p_tampered", ascending=False)
if len(flagged):
    st.subheader("Images judged tampered")
    for k in range(0, min(len(flagged), 12), 4):
        cols = st.columns(4)
        for col, (i, r) in zip(cols, flagged.iloc[k:k + 4].iterrows()):
            if i in overlays:
                col.image(overlays[i], caption=f"{r.image} · P = {r.p_tampered:.2f}", width="stretch")
            else:
                col.markdown(f"{r.image} · P = {r.p_tampered:.2f} (no region marked)")
lo, hi = E.band_edges(M)
st.caption(f"Verdicts use protocol {protocol}; 'uncertain' means the calibrated probability fell between {lo:.2f} "
           f"and {hi:.2f}, where the detector does not give a verdict. A suspected region is kept for tampered and "
           "uncertain images (for inspection) and removed for authentic ones.")
