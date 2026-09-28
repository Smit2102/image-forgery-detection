"""The single entry point used by the CLI, notebooks, evaluation scripts and the UI."""

from __future__ import annotations

import copy
import hashlib
import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .features import FeatureModule, build_modules
from .io import apply_protocol, load_rgb


@dataclass
class Result:
    path: str
    protocol: str
    shape: tuple[int, int]                  # analysed geometry (after the protocol, e.g. resampled under R)
    original_shape: tuple[int, int] | None = None
    features: dict[str, float] = field(default_factory=dict)
    maps: dict[str, np.ndarray] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    verdict: str | None = None          # "tampered" | "authentic" | "uncertain" | None (no model)
    confidence: float | None = None     # P(tampered)
    likely_type: str | None = None      # "copy-move" | "splicing" | None (Phase 5/6)
    mask: np.ndarray | None = None      # localisation mask (Phase 6)

    def to_dict(self) -> dict:
        return {
            "image": self.path,
            "protocol": self.protocol,
            "shape": list(self.shape),
            "original_shape": None if self.original_shape is None else list(self.original_shape),
            "verdict": self.verdict,
            "confidence": self.confidence,
            "likely_type": self.likely_type,
            "region_area_pct": None if self.mask is None else round(100 * float(self.mask.mean()), 2),
            "features": {k: float(v) for k, v in self.features.items()},
            "timings_ms": {k: round(v, 1) for k, v in self.timings_ms.items()},
        }

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), **kw)


def verdict_from_probability(p: float, cfg: dict, half_width: float | None = None) -> str:
    """tampered / authentic / uncertain; the band comes from the model (Phase 5) or config.yaml."""
    if half_width is not None:
        lo, hi = 0.5 - half_width, 0.5 + half_width
    else:
        lo, hi = cfg["verdict"]["uncertain_low"], cfg["verdict"]["uncertain_high"]
    if p >= hi:
        return "tampered"
    if p <= lo:
        return "authentic"
    return "uncertain"


def feature_code_sha1() -> str:
    """Hash of the feature-module and I/O source (same definition as scripts/extract_features.py)."""
    src = Path(__file__).resolve().parent
    h = hashlib.sha1()
    for f in sorted([*(src / "features").glob("*.py"), src / "io.py"]):
        h.update(f.read_bytes())
    return h.hexdigest()


def module_params(cfg: dict, protocol: str) -> dict:
    """Feature-module parameters for a protocol: config.yaml `features`, plus the last JPEG quality
    for the DCT module under protocols that end with a known re-encode (B, R)."""
    params = copy.deepcopy(cfg["features"])
    if protocol in ("B", "R"):
        params.setdefault("dct_dq", {})["last_quality"] = cfg["protocols"][protocol]["jpeg_quality"]
    return params


def resolved_params(cfg: dict, protocol: str) -> dict:
    """Every module's full parameter set (explicit config values plus constructor defaults), so that
    writing a parameter out at its default value does not count as a change."""
    return {m.name: dict(m.params) for m in build_modules(params=module_params(cfg, protocol))}


def _ordered_row(model, features: dict):
    names = getattr(model, "feature_names_", None)
    sk_names = getattr(model, "feature_names_in_", None)
    if names is None and sk_names is None:
        raise AttributeError("model needs `feature_names_` or `feature_names_in_` to order its inputs")
    names = list(names if names is not None else sk_names)
    missing = [n for n in names if n not in features]
    if missing:
        raise KeyError(f"model expects features not produced by the modules: {missing[:5]}")
    row = [[features[n] for n in names]]
    # a DataFrame only for estimators fitted with column names, else sklearn warns
    return pd.DataFrame(row, columns=names) if sk_names is not None else np.asarray(row)


def detect(path: str | Path, protocol: str = "A", modules: list[FeatureModule] | None = None,
           model=None, cfg: dict | None = None, type_model=None, localizer=None) -> Result:
    """Run every feature module on one image and (if a model is given) classify it.

    `model` must expose `predict_proba` and fix the feature order through `feature_names_`
    (our wrapper) or scikit-learn's `feature_names_in_` (estimator fitted on a DataFrame).
    `type_model` (optional) gives P(copy-move) for images judged tampered.
    `localizer` (optional, Phase 6) fuses the evidence maps into `Result.mask` and `Result.maps["fused"]`.
    With a model, the mask is kept only when the verdict is not "authentic" (authentic images would
    otherwise often show a small spurious region).
    """
    cfg = cfg or load_config()
    if modules is None:        # the tuned (Phase-4) parameters the models were trained with
        modules = build_modules(params=module_params(cfg, protocol))
    for m in (model, type_model, localizer):
        meta = getattr(m, "meta", None) or {}
        if "feature_params" in meta and meta["feature_params"] != resolved_params(cfg, protocol):
            raise ValueError("model was trained with different feature parameters than config.yaml; "
                             "re-run scripts/extract_features.py and scripts/run_phase5.py")
        if "protocol" in meta and meta["protocol"] != protocol:
            raise ValueError(f"model was trained for protocol {meta['protocol']}, not {protocol}")
        if "feature_code_sha1" in meta and meta["feature_code_sha1"] != feature_code_sha1():
            warnings.warn("feature-module code changed since this model was trained; "
                          "re-run scripts/extract_features.py and scripts/run_phase5.py", stacklevel=2)

    t0 = time.perf_counter()
    raw = load_rgb(path)
    img = apply_protocol(raw, protocol, cfg)
    res = Result(path=str(path), protocol=protocol, shape=img.shape[:2], original_shape=raw.shape[:2])
    res.timings_ms["load"] = 1000 * (time.perf_counter() - t0)

    for m in modules:
        t0 = time.perf_counter()
        out = m.extract(img)
        out.validate(res.shape, m.name)
        res.timings_ms[m.name] = 1000 * (time.perf_counter() - t0)
        if out.evidence_map is not None:
            res.maps[m.name] = out.evidence_map
        for k, v in out.features.items():
            res.features[f"{m.name}.{k}"] = float(v)

    if localizer is not None:
        t0 = time.perf_counter()
        prob, mask = localizer.predict(res.maps, res.shape)
        res.timings_ms["localizer"] = 1000 * (time.perf_counter() - t0)
        res.maps["fused"] = np.clip(prob, 0, 1).astype(np.float32)
        res.mask = mask
    if model is not None:
        res.confidence = float(model.predict_proba(_ordered_row(model, res.features))[0, 1])
        band = getattr(model, "meta", {}).get("uncertain_half_width")
        res.verdict = verdict_from_probability(res.confidence, cfg, band)
        if type_model is not None and res.verdict == "tampered":
            p_cm = float(type_model.predict_proba(_ordered_row(type_model, res.features))[0, 1])
            res.likely_type = "copy-move" if p_cm >= 0.5 else "splicing"
        if res.verdict == "authentic" and res.mask is not None:     # gate: no region for authentic
            res.mask = np.zeros_like(res.mask)
            res.maps["fused"] = np.zeros_like(res.maps["fused"])
    return res
