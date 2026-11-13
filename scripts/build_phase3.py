"""Phase 3 course submission: assembles build/phase3/ (self-contained, runs from its own folder).

  python scripts/build_phase3.py            # assemble the folder (code, models, sample images, supporting data, cache)

The notebook (GroupJ_Patel_Gupta_Phase3.ipynb) is written by scripts/build_phase3_notebook.py.

What goes into the folder
  src/forgery/   the detector package (copy; the CNN baseline module is left out)
  config.yaml    tuned parameters
  models/        the three protocol-R models (classifier, forgery-type, localiser)
  sample_images/ 14 CASIA v2.0 validation images (stratified random sample + the 2 Phase-2 examples) with their
                 ground-truth masks, and 22 MICC-F220 images (unseen dataset): see sample_images/README.txt
  supporting/    saved tables and figures from the project's experiments (read by the notebook)
  precomputed/   cached detector outputs for the sample images (used only if the models cannot be loaded)
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from forgery.config import load_config, resolve
from forgery.data.manifest import read_manifest
from forgery.data.splits import read_splits
from forgery.io import load_mask

DEST = resolve("build/phase3")
SEED = 42
PHASE2 = {"cm_mirror": "Tp_S_CND_S_B_cha00088_cha00088_10179.tif",       # the examples used in the Phase 2 notebook
          "sp_cat": "Tp_D_CRD_S_O_ani10111_ani10103_10635.jpg"}

SUPPORTING = {   # destination name -> source in the project
    "metrics.json": "outputs/metrics.json",
    "sweep_records.csv": "results/phase4/sweep_records.csv",
    "chosen_params.json": "results/phase4/chosen_params.json",
    "phase5_results.csv": "results/phase5/results.csv",
    "phase5_added_value.csv": "results/phase5/added_value.csv",
    "phase5_ablations.csv": "results/phase5/ablations.csv",
    "phase5_coverage_R.csv": "results/phase5/coverage_R.csv",
    "phase6_methods_R.csv": "results/phase6/methods_R.csv",
    "test_runs.log": "results/phase7/test_runs.log",                 # fingerprints recorded at the one-time test evaluation
    "phase7_summary.json": "results/phase7/run_1/summary.json",
    "fig_roc_test_R.png": "outputs/15_test_roc.png",
    "fig_error_analysis.png": "outputs/16_error_analysis.png",
    "fig_robustness.png": "outputs/17_robustness.png",
    "fig_cnn_vs_classical_roc.png": "outputs/19_cnn_vs_classical_roc.png",
    "fig_ela_format_leak.png": "outputs/20_ela_inputs_format_leak.png",
    "fig_interface.png": "outputs/21_interface.png",
}

README = """PHASE 3 SUBMISSION - Group J (Smit Patel, Ananay Gupta) - EDS 6364 Digital Image Processing
Image Forgery Detection Using Classical Image Processing Techniques

FILES TO SUBMIT:  GroupJ_Patel_Gupta_Phase3.ipynb  and  GroupJ_Patel_Gupta_Phase3.pdf
(this folder holds everything the notebook needs; it is also provided as GroupJ_Patel_Gupta_Phase3_supporting.zip)

HOW TO RUN (no file paths to edit)
  1. Python 3.12:   python3 -m venv .venv && source .venv/bin/activate
                    pip install -r requirements.txt jupyter
  2. Start Jupyter FROM THIS FOLDER:   jupyter notebook GroupJ_Patel_Gupta_Phase3.ipynb
  3. Run all cells (about 30 seconds on a laptop CPU).  All paths are relative to this folder.
  If the saved models cannot be loaded (different scikit-learn / NumPy version), the notebook says so and
  uses the cached detector outputs in precomputed/ instead; the numbered stages still run live.

FOLDER CONTENTS
  GroupJ_Patel_Gupta_Phase3.ipynb / .pdf   the submission
  src/forgery/       the implementation (the notebook also shows each stage in compact form)
  config.yaml        tuned parameters (Phase 2 / Phase 3 experiments)
  models/            trained protocol-R models (random forest, forgery-type model, localiser)
  sample_images/     14 CASIA v2.0 images with masks + 22 MICC-F220 images (see sample_images/README.txt)
  supporting/        saved results of the full experiments (tables, figures) read by the notebook
  precomputed/       cached outputs for the sample images (fallback only)

DATASETS (not included in full)
  CASIA v2.0  Image Tampering Detection Evaluation Database (Dong, Wang, Tan 2013), 12,614 images + masks.
              Kaggle: search "CASIA 2.0 Image Tampering Detection Dataset". Needed only to re-run the full experiments
              (the project repository https://github.com/Smit2102/image-forgery-detection has the scripts).
  MICC-F220   Amerini et al. 2011, http://lci.micc.unifi.it/labd/2015/01/copy-move-forgery-detection-and-localization/
Both are for research use under their authors' terms; the few images included here are for reproducing this notebook.

Project repository: https://github.com/Smit2102/image-forgery-detection
Project website:    https://smit2102.github.io/image-forgery-detection/
"""

SAMPLES_README = """Sample images used by the Phase 3 notebook.

casia/  14 images from the CASIA v2.0 *validation* split (Dong, Wang & Tan, Institute of Automation, Chinese
        Academy of Sciences, 2013; obtained from Kaggle). 12 are a stratified random sample (seed 42: 4 authentic,
        4 copy-move, 4 splicing); 2 are the examples used in the Phase 2 notebook. *_mask.png is the ground-truth
        tampered region. The deployed models were trained on train+validation images, so results on these
        images are optimistic; they are for demonstration.
micc/   22 images from MICC-F220 (Amerini et al., IEEE TIFS 2011), a dataset the models have never seen: one forged
        variant of each of the 11 forged scenes plus 11 randomly chosen originals (seed 42).
samples.csv  index (name, file, group, kind, label, ...).
Both datasets are distributed by their authors for research use.
"""

REQUIREMENTS = """numpy==2.5.3
pandas==3.0.6
scipy==1.18.1
scikit-learn==1.9.1
opencv-python==5.0.0.93
pillow==12.3.0
matplotlib==3.11.2
joblib==1.6.0
PyYAML==6.0.3
shap==0.52.0        # optional: only for the per-image explanation figure
"""


def select_casia(cfg):
    df = read_manifest(cfg).merge(read_splits(cfg), on="path")
    val = df[df.split == "val"]
    ok = val[(val.label == 0) | val.mask_valid.fillna(False).astype(bool)]
    rows = []
    for kind, tag in (("authentic", "au"), ("copy-move", "cm"), ("splicing", "sp")):
        s = ok[ok.label == 0] if kind == "authentic" else ok[ok.forgery_type == kind]
        s = s.sample(4, random_state=SEED).reset_index(drop=True)
        for i, r in s.iterrows():
            rows.append((f"{tag}{i + 1}", r))
    for name, fname in PHASE2.items():
        rows.append((name, df[df.path.str.endswith(fname)].iloc[0]))
    return rows


def main():
    cfg = load_config()
    if DEST.exists():
        shutil.rmtree(DEST)
    (DEST / "sample_images/casia").mkdir(parents=True)
    (DEST / "sample_images/micc").mkdir(parents=True)
    (DEST / "supporting").mkdir()
    (DEST / "models").mkdir()

    # --- code, parameters, models
    shutil.copytree(resolve("src/forgery"), DEST / "src/forgery",
                    ignore=shutil.ignore_patterns("__pycache__", "cnn.py"))
    shutil.copy2(resolve("config.yaml"), DEST / "config.yaml")
    for m in ("forgery_R", "type_R", "localizer_R"):
        shutil.copy2(resolve(f"models/{m}.joblib"), DEST / "models" / f"{m}.joblib")
    (DEST / "requirements.txt").write_text(REQUIREMENTS)
    (DEST / "README.txt").write_text(README)
    (DEST / "sample_images/README.txt").write_text(SAMPLES_README)
    for dst, src in SUPPORTING.items():
        shutil.copy2(resolve(src), DEST / "supporting" / dst)

    # --- sample images: CASIA
    index = []
    for name, r in select_casia(cfg):
        src = resolve(r.path)
        shutil.copy2(src, DEST / "sample_images/casia" / src.name)
        mask_file = ""
        if r.label == 1:
            mask_file = f"{src.stem}_mask.png"
            Image.fromarray((load_mask(resolve(r.mask_path)) * 255).astype(np.uint8)).save(
                DEST / "sample_images/casia" / mask_file)
        index.append({"name": name, "group": "casia", "file": f"casia/{src.name}", "mask": mask_file and f"casia/{mask_file}",
                      "kind": "authentic" if r.label == 0 else r.forgery_type, "label": int(r.label),
                      "format": r.format, "width": int(r.width), "height": int(r.height),
                      "tampered_pct": round(100 * float(r.tampered_frac), 2) if r.label else 0.0,
                      "note": "Phase 2 example" if name in PHASE2 else "stratified random sample (seed 42)"})

    # --- sample images: MICC-F220 (unseen dataset)
    root = resolve("data/external/MICC-F220")
    gt = pd.read_csv(root / "groundtruthDB_220.txt", sep=r"\s+", header=None, names=["file", "label"])
    gt["scene"] = gt.file.str.extract(r"^(.+?)_?(?:tamp\d+|scale)\.[A-Za-z]+$", expand=False).fillna(gt.file)
    forged = gt[gt.label == 1].sort_values("file").groupby("scene").head(1)
    orig = gt[(gt.label == 0) & ~gt.scene.isin(forged.scene)].sample(11, random_state=SEED)
    for tag, sub, label in (("mf", forged, 1), ("mo", orig, 0)):
        for i, r in enumerate(sub.itertuples(), 1):
            shutil.copy2(root / r.file, DEST / "sample_images/micc" / r.file)
            w, h = Image.open(root / r.file).size
            index.append({"name": f"{tag}{i}", "group": "micc", "file": f"micc/{r.file}", "mask": "",
                          "kind": "copy-move" if label else "authentic", "label": label, "format": "JPEG",
                          "width": w, "height": h, "tampered_pct": np.nan,
                          "note": "unseen dataset: forged variant" if label else "unseen dataset: original"})
    pd.DataFrame(index).to_csv(DEST / "sample_images/samples.csv", index=False)

    # --- cached detector outputs (fallback), produced by the folder's OWN copy of the code
    subprocess.run([sys.executable, "-c", CACHE_SCRIPT], cwd=DEST, check=True,
                   env={**__import__("os").environ, "PYTHONPATH": str(DEST / "src")})
    size = sum(f.stat().st_size for f in DEST.rglob("*") if f.is_file()) / 1e6
    print(f"assembled {DEST} ({size:.1f} MB, {len(index)} sample images)")


CACHE_SCRIPT = r'''
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import forgery
assert Path(forgery.__file__).resolve().is_relative_to(Path.cwd().resolve()), forgery.__file__   # the folder's own code
from forgery import explain as E
from forgery.config import load_config
cfg = load_config(); M = E.load_models("R")
S = pd.read_csv("sample_images/samples.csv")
Path("precomputed").mkdir(exist_ok=True)
out, keep_maps = {}, {"cm_mirror", "sp_cat"}
for _, r in S.iterrows():
    res = E.analyse(Path("sample_images") / r.file, M, cfg)
    entry = {"verdict": res.verdict, "confidence": res.confidence, "likely_type": res.likely_type,
             "features": res.features, "shape": list(res.shape)}
    arrays = {"mask": res.mask.astype(bool), "fused": res.maps["fused"].astype(np.float16)}
    if r["name"] in keep_maps:
        arrays.update({k: v.astype(np.float16) for k, v in res.maps.items() if k != "fused"})
        c = E.explain(M, res.features, top=12)
        entry["contrib"] = c[["feature", "value", "contribution", "description"]].to_dict(orient="records")
    np.savez_compressed(f"precomputed/{r['name']}.npz", **arrays)
    out[r["name"]] = entry
Path("precomputed/results.json").write_text(json.dumps(out))
print("cached", len(out), "results")
'''

if __name__ == "__main__":
    main()
