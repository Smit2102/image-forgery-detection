"""Everything the user interface needs on top of pipeline.detect(): models, overlays, per-image explanations
(SHAP on the deployed forest), plain-language text and a PDF report. Kept free of Streamlit so it is testable.
"""

from __future__ import annotations

import io
import textwrap
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from .config import ROOT, load_config  # noqa: E402
from .io import apply_protocol, load_rgb  # noqa: E402
from .pipeline import Result, detect  # noqa: E402

MODELS = ROOT / "models"

MODULES = {   # name -> (title, what its evidence map shows)
    "ela": ("Error level analysis (ELA)",
            "How strongly each area changes when the image is re-saved as JPEG, normalised for texture. "
            "A pasted region with a different compression history stands out."),
    "histogram": ("Local histogram consistency",
                  "The ELA error is histogram-equalised; each patch's histogram of it is compared with the image's "
                  "typical patch histogram (chi-square distance). Patches whose error distribution differs stand out."),
    "noise": ("Noise level",
              "Local sensor-noise estimate (edges excluded). Regions pasted from another photo often carry "
              "a different noise level."),
    "jpeg_ghost": ("JPEG ghosts",
                   "Re-saving at many qualities: a region compressed earlier at a different quality shows a "
                   "'ghost' minimum at that quality."),
    "dct_dq": ("DCT double quantisation",
               "Periodic patterns in the DCT coefficient histograms that appear when a block was JPEG-compressed "
               "twice; blocks without the pattern are marked."),
    "edges": ("Edge sharpness",
              "Sharpness of object boundaries; unnaturally hard or soft edges can reveal a pasted object."),
    "cm_keypoint": ("Copy-move: keypoint matching",
                    "SIFT keypoints matched to other keypoints in the same image (also mirrored), kept only if "
                    "a consistent geometric transform explains them (RANSAC)."),
    "cm_block": ("Copy-move: block matching",
                 "Overlapping blocks compared by their DCT coefficients; many blocks sharing one shift "
                 "indicate a duplicated region."),
    "fused": ("Fused localisation",
              "The learned fusion of all eight evidence maps: probability that each 8 x 8 cell is tampered."),
}

_SUFFIX = {
    "local_z_max": "strongest local outlier (z-score)",
    "local_z_p99": "99th-percentile local z-score",
    "local_frac_z3": "share of blocks with z > 3",
    "local_cluster_z25": "largest connected cluster of outlying blocks",
    "local_moran": "spatial clustering of the evidence (Moran's I)",
    "local_top5_z": "mean z-score of the top 5 % of blocks",
    "local_n_matches": "matched keypoint pairs",
    "local_n_inliers": "matches consistent with one transform (RANSAC inliers)",
    "local_inlier_frac": "share of matches that are inliers",
    "local_log_inliers": "log number of inliers",
    "local_hull_frac": "area covered by the matched keypoints",
    "local_shift": "distance between source and copy",
    "local_frac_mirrored": "share of mirrored matches",
    "global_n_keypoints": "number of keypoints in the image",
    "local_max_votes": "blocks voting for the most common shift",
    "local_log_votes": "log of the top shift's votes",
    "local_mirrored_votes": "votes for a mirrored copy",
    "local_n_shifts": "number of shifts with enough votes",
    "local_area_frac": "area of the matched blocks",
    "global_frac_textured_blocks": "share of textured blocks",
    "global_mean": "mean ELA level of the whole image",
    "global_p95": "95th-percentile ELA level",
    "global_mean_distance": "mean histogram distance",
    "global_log_sigma": "overall noise level (log)",
    "local_frac_usable_blocks": "share of blocks usable for noise estimation",
    "local_frac_argmin_off": "share of blocks whose ghost minimum differs from the image's",
    "global_mode_quality": "most common ghost quality",
    "global_mean_deviation": "mean ghost deviation",
    "local_frac_blocks_positive": "share of blocks with a double-quantisation pattern",
    "global_frac_periodic_freqs": "share of DCT frequencies with a periodic histogram",
    "global_mean_step": "mean estimated quantisation step",
    "local_frac_edge_blocks": "share of blocks with edges",
    "global_edge_density": "edge density",
    "global_median_sharpness": "median edge sharpness",
}
_STATS = {"local_z_max", "local_z_p99", "local_frac_z3", "local_cluster_z25", "local_moran", "local_top5_z"}
_VARIANT = {"raw": "raw", "norm": "texture-normalised", "low": "low-noise side", "sharp": "sharp side",
            "soft": "soft side"}


def describe_feature(name: str) -> str:
    """'ela.norm_local_z_max' -> 'Error level analysis (ELA): strongest local outlier (z-score), texture-normalised'."""
    module, feat = name.split(".", 1)
    title = MODULES.get(module, (module,))[0]
    variant = ""
    head = feat.split("_", 1)[0]
    if head in _VARIANT and feat.split("_", 1)[1] in _SUFFIX:
        variant, feat = _VARIANT[head], feat.split("_", 1)[1]
    text = _SUFFIX.get(feat, feat.replace("_", " "))
    if module == "noise" and not variant and feat in _STATS:     # un-prefixed noise statistics: high-noise side
        variant = "high-noise side"
    return f"{title}: {text}" + (f" ({variant})" if variant else "")


@dataclass
class Models:
    protocol: str
    forgery: object | None
    type: object | None
    localizer: object | None

    @property
    def band(self) -> float | None:
        return (getattr(self.forgery, "meta", {}) or {}).get("uncertain_half_width")

    @property
    def forest(self):
        """The random forest inside the calibrated model (explained by SHAP)."""
        return self.forgery.estimator.calibrated_classifiers_[0].estimator


def main_protocol() -> str:
    f = MODELS / "MAIN_PROTOCOL"
    return f.read_text().strip() if f.exists() else "R"


def available_protocols() -> list[str]:
    return [p for p in ("R", "B") if (MODELS / f"forgery_{p}.joblib").exists()]


def load_models(protocol: str) -> Models:
    def _load(kind):
        f = MODELS / f"{kind}_{protocol}.joblib"
        return joblib.load(f) if f.exists() else None
    return Models(protocol, _load("forgery"), _load("type"), _load("localizer"))


class ImageError(ValueError):
    """An input the detector cannot analyse; the message is meant for the user."""


MIN_SIDE, MAX_SIDE = 64, 4000


def prepare_image(path: str | Path, out_dir: str | Path) -> tuple[Path, list[str]]:
    """Check an arbitrary user image and, if needed, normalise it for the detector.

    The detector decodes images with PIL's convert("RGB"), which is right for CASIA (all 8-bit RGB, no EXIF
    rotation) but wrong for some user files. Here the image is fully decoded (catching truncated files and
    decompression bombs), rotated by its EXIF orientation, and 16-bit / float images are rescaled to 8 bit.
    If anything changed, the result is saved losslessly as PNG in out_dir; under protocols B and R the
    detector works on decoded pixels, so this does not change the analysis. Returns (path to analyse, notes).
    """
    from PIL import Image, ImageOps
    path, notes = Path(path), []
    try:
        with Image.open(path) as im:
            w, h = im.size                              # header only: check the size before decoding
            if max(w, h) > MAX_SIDE:
                raise ImageError(f"{w} x {h} px is larger than the {MAX_SIDE} px limit; please downscale it first.")
            if min(w, h) < MIN_SIDE:
                raise ImageError(f"{w} x {h} px is too small to analyse (minimum {MIN_SIDE} px).")
            im.load()                                  # full decode: truncated files fail here
            out, changed = im, False
            if im.getexif().get(274, 1) not in (1, None):
                out, changed = ImageOps.exif_transpose(im), True
                notes.append("rotated according to its EXIF orientation")
            if out.mode in ("I;16", "I;16B", "I;16L", "I;16N", "I", "F"):
                a = np.asarray(out, dtype=np.float64)
                lo, hi = float(a.min()), float(a.max())
                a = np.zeros_like(a) if hi <= lo else (a - lo) / (hi - lo) * 255.0
                out, changed = Image.fromarray(np.rint(a).astype(np.uint8), "L"), True
                notes.append("rescaled from 16 bit (or higher) to 8 bit")
            if not changed:
                return path, notes
            dst = Path(out_dir) / (path.stem + "_prepared.png")
            if not dst.exists():                       # same content -> same file (keeps caches keyed on it)
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp = dst.with_suffix(".tmp.png")
                out.convert("RGB").save(tmp)
                tmp.replace(dst)
            return dst, notes
    except ImageError:
        raise
    except Image.DecompressionBombError:
        raise ImageError("the image is too large to decode safely (decompression-bomb protection).") from None
    except Image.UnidentifiedImageError:
        raise ImageError("not an image file, or a format that cannot be read.") from None
    except Exception as exc:                             # truncated, corrupt, ...
        msg = str(exc).replace(str(path), path.name)
        raise ImageError(f"not a readable image ({type(exc).__name__}: {msg}).") from None


def protocol_stats(protocol: str) -> dict:
    """Test-set coverage and accuracy of the deployed detector for a protocol (Phase 7b, clean condition)."""
    f = ROOT / "results/phase7b/robustness.csv"
    try:
        rb = pd.read_csv(f)
        r = rb[(rb.protocol == protocol) & (rb.condition == "clean")].iloc[0]
        return {"coverage": float(r.coverage), "accuracy_judged": float(r.accuracy_judged), "auc": float(r.auc)}
    except Exception:
        return {}


def band_edges(models: "Models", cfg: dict | None = None) -> tuple[float, float]:
    """The uncertain band used by pipeline.verdict_from_probability (model band, else config.yaml)."""
    if models.band is not None:
        return 0.5 - models.band, 0.5 + models.band
    cfg = cfg or load_config()
    return cfg["verdict"]["uncertain_low"], cfg["verdict"]["uncertain_high"]


def analyse(path: str | Path, models: Models, cfg: dict | None = None) -> Result:
    return detect(path, protocol=models.protocol, cfg=cfg or load_config(), model=models.forgery,
                  type_model=models.type, localizer=models.localizer)


def analysed_image(path: str | Path, protocol: str, cfg: dict | None = None) -> np.ndarray:
    """The image as the detector saw it (after the protocol: re-encoded, and resampled under R)."""
    return apply_protocol(load_rgb(path), protocol, cfg or load_config())


def overlay(img: np.ndarray, mask: np.ndarray | None, color=(255, 0, 0), alpha: float = 0.55) -> np.ndarray:
    out = img.copy()
    if mask is not None and mask.any():
        out[mask] = ((1 - alpha) * out[mask] + alpha * np.array(color)).astype(np.uint8)
    return out


def explain(models: Models, features: dict[str, float], top: int = 10) -> pd.DataFrame:
    """SHAP contributions of each feature to this image's forest P(tampered), largest |contribution| first."""
    import shap
    names = models.forgery.feature_names_
    x = np.array([[features[n] for n in names]], float)
    sv = shap.TreeExplainer(models.forest).shap_values(x)
    sv = np.asarray(sv)
    if sv.ndim == 3:            # (n, features, classes) in recent shap versions
        sv = sv[0, :, 1]
    elif sv.ndim == 2 and sv.shape[0] == 2:
        sv = sv[1]
    sv = np.ravel(sv)
    df = pd.DataFrame({"feature": names, "value": x[0], "contribution": sv,
                       "module": [n.split(".", 1)[0] for n in names],
                       "description": [describe_feature(n) for n in names]})
    return df.reindex(df.contribution.abs().sort_values(ascending=False).index).head(top).reset_index(drop=True)


def verdict_text(res: Result, models: "Models") -> str:
    if res.verdict is None:
        return "No verdict: no trained model is available for this protocol."
    p = res.confidence
    lo, hi = band_edges(models)
    if res.verdict == "tampered":
        s = f"The detector judges this image **tampered** (P(tampered) = {p:.3f}, at or above {hi:.2f})."
        if res.likely_type:
            s += f" The evidence looks most like **{res.likely_type}**."
    elif res.verdict == "authentic":
        s = f"The detector judges this image **authentic** (P(tampered) = {p:.3f}, at or below {lo:.2f})."
    else:
        s = (f"The detector is **uncertain** (P(tampered) = {p:.3f}, between {lo:.2f} and {hi:.2f}). "
             "On the development data, accuracy inside this band was too low to give a verdict.")
        st_ = protocol_stats(res.protocol)
        if st_:
            s += f" Under protocol {res.protocol}, {1 - st_['coverage']:.0%} of test images fell in this band."
    return s


def evidence_text(res: Result, contrib: pd.DataFrame) -> list[str]:
    """Short, factual sentences about the strongest evidence for and against."""
    lines = []
    f = res.features
    n_in = f.get("cm_keypoint.local_n_inliers", 0)
    if n_in > 0:
        lines.append(f"Copy-move keypoint matching found {int(n_in)} keypoint pairs consistent with one geometric "
                     "transform: part of the image may be duplicated. Repeated real structures (windows, "
                     "columns, patterns) can cause this too.")
    for _, r in contrib.head(3).iterrows():
        direction = "towards **tampered**" if r.contribution > 0 else "towards **authentic**"
        lines.append(f"{r.description} = {r.value:.3g} pushed the score {direction} "
                     f"({r.contribution:+.3f}).")
    if res.mask is not None and res.mask.any():
        lines.append(f"The suspected region covers {100 * res.mask.mean():.1f} % of the image.")
    return lines


def module_maps_figure(res: Result, img: np.ndarray, ncols: int = 3, figsize=None):
    names = [n for n in list(MODULES) if n in res.maps]
    rows = int(np.ceil((len(names) + 1) / ncols))
    fig, axes = plt.subplots(rows, ncols, figsize=figsize or (4 * ncols, 3.1 * rows))
    axes = np.ravel(axes)
    axes[0].imshow(img)
    axes[0].set_title("analysed image", fontsize=9)
    for ax, n in zip(axes[1:], names):
        ax.imshow(res.maps[n], cmap="magma", vmin=0, vmax=1)          # all maps are in [0, 1]
        ax.set_title(MODULES[n][0], fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    return fig


def pdf_report(res: Result, img: np.ndarray, models: Models, contrib: pd.DataFrame, name: str) -> bytes:
    """A 2-page PDF: verdict, overlay and explanation; then all evidence maps."""
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))            # A4 portrait
        fig.text(0.06, 0.96, "Image forgery analysis", fontsize=16, weight="bold")
        fig.text(0.06, 0.935, f"{name}  ·  protocol {res.protocol}  ·  analysed at {res.shape[1]} x {res.shape[0]} px",
                 fontsize=9, color="#555")
        text = "\n".join(textwrap.wrap(verdict_text(res, models).replace("**", ""), 95))
        fig.text(0.06, 0.915, text, fontsize=9.5, va="top")
        ax1, ax2 = fig.add_axes([0.06, 0.58, 0.42, 0.28]), fig.add_axes([0.52, 0.58, 0.42, 0.28])
        ax1.imshow(img)
        ax1.set_title("analysed image", fontsize=9)
        ax2.imshow(overlay(img, res.mask))
        ax2.set_title("suspected region (red)" if res.mask is not None and res.mask.any() else "no region marked",
                      fontsize=9)
        for a in (ax1, ax2):
            a.axis("off")
        ax3 = fig.add_axes([0.45, 0.22, 0.49, 0.30])
        c = contrib.iloc[::-1]
        ax3.barh(range(len(c)), c.contribution, color=np.where(c.contribution > 0, "#E45756", "#4C78A8"))
        ax3.set_yticks(range(len(c)), ["\n".join(textwrap.wrap(d, 52)) for d in c.description], fontsize=6.5)
        ax3.axvline(0, color="k", lw=0.6)
        ax3.set_xlabel("SHAP contribution to the forest's P(tampered)", fontsize=8)
        ax3.set_title("Strongest evidence (red: towards tampered, blue: towards authentic)", fontsize=9)
        ax3.tick_params(axis="x", labelsize=7)
        st_ = protocol_stats(res.protocol)
        perf = (f"on its held-out test set under protocol {res.protocol} it judged {st_['coverage']:.0%} of images, "
                f"with {st_['accuracy_judged']:.0%} accuracy on those. ") if st_ else ""
        notes = ("How to read this report: the detector combines 8 classical forensic techniques and was trained on "
                 f"CASIA v2.0 under a leak-controlled protocol; {perf}A verdict is evidence, not proof. Repeated "
                 "real structures can mimic copy-move, and splices between photos with the same processing history "
                 "are hard to detect. Evidence maps are on a fixed 0-1 scale.")
        fig.text(0.06, 0.1, "\n".join(textwrap.wrap(notes, 125)), fontsize=7.5, color="#333", va="top")
        pdf.savefig(fig)
        plt.close(fig)
        fig = module_maps_figure(res, img, figsize=(8.27, 11.69))                  # A4 portrait, like page 1
        fig.suptitle("Evidence maps of the individual techniques (fixed 0-1 scale; brighter = more suspicious)",
                     fontsize=10)
        fig.tight_layout(rect=(0, 0.02, 1, 0.96))
        pdf.savefig(fig)
        plt.close(fig)
    return buf.getvalue()


def report_dict(res: Result, models: Models, contrib: pd.DataFrame, name: str) -> dict:
    d = res.to_dict()
    d.update({"image": name, "uncertain_half_width": models.band,
              "top_evidence": contrib[["feature", "value", "contribution"]].to_dict(orient="records")})
    return d
