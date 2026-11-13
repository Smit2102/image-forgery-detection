"""Phase 3 course submission: packs the notebook folder build/phase3/ and the project into ONE zip.

    python scripts/package_phase3.py            # -> submissions/phase3_submission/GroupJ_Phase3.zip

The course asks for GroupX_Phase3.ipynb, GroupX_Phase3.pdf and the supporting image files; the instructor also
wants everything needed to start the project. So the zip holds, in one folder:
  - GroupJ_Phase3.ipynb, GroupJ_Phase3.pdf   (GroupJ_Patel_Gupta_Phase3.* renamed; the notebook and PDF do not refer to their names)
  - sample_images/                           the supporting image files (CASIA v2.0 and MICC-F220 samples + masks)
  - the notebook's own files                 supporting/ (saved tables and figures), precomputed/ (fallback outputs)
  - the project                              src/, scripts/, tests/, app/, detect.py, config.yaml, pyproject.toml, requirements,
                                             data/ (manifest, splits, frozen-test hashes, features), models/, results/
Run after scripts/export_phase3.py.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

from forgery.config import resolve

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_reproduction_zip import REQUIREMENTS, SCRIPTS      # the lean pinned environment and the step scripts

FOLDER = resolve("build/phase3")
OLD, NEW = "GroupJ_Patel_Gupta_Phase3", "GroupJ_Phase3"
ZIP = resolve(f"submissions/phase3_submission/{NEW}.zip")      # the only file in the submission folder
NOTEBOOK_PARTS = ["GroupJ_Patel_Gupta_Phase3.ipynb", "GroupJ_Patel_Gupta_Phase3.pdf", "sample_images", "supporting", "precomputed"]
MODELS = [f"{m}_{p}.joblib" for p in ("R", "B") for m in ("forgery", "type", "localizer")] + ["MAIN_PROTOCOL"]
SKIP_DIRS = {"__pycache__", ".pytest_cache", "pre_qa_run", "detect", "phase8_smoke"}
SKIP_NAMES = {".DS_Store"}
PROJECT_SCRIPTS = SCRIPTS + ["run_phase7.py", "run_phase7b.py", "phase7_posthoc.py", "make_ela.py", "run_phase8.py",
                             "phase8_posthoc.py", "make_outputs.py"]

README = """PHASE 3 SUBMISSION - Group J (Smit Patel 2479966, Ananay Gupta 2618966) - EDS 6364 Digital Image Processing
Image Forgery Detection Using Classical Image Processing Techniques

WHAT THE COURSE ASKED FOR                                    WHERE IT IS
  GroupJ_Phase3.ipynb   Jupyter notebook                      GroupJ_Phase3.ipynb
  GroupJ_Phase3.pdf     PDF export of the notebook            GroupJ_Phase3.pdf
  Supporting image files                                      sample_images/   14 CASIA v2.0 images with ground-truth masks and
                                                              22 MICC-F220 images (see sample_images/README.txt)
                                                              plus supporting/ (saved result tables and figures the
                                                              notebook reads) and precomputed/ (fallback outputs)
  Access to the datasets                                      see "DATASETS" below (the full datasets are not included)

EVERYTHING ELSE IN THIS FOLDER IS THE PROJECT ITSELF, so the project can be started and the experiments re-run:
  src/forgery/      the Python package (8 image-processing modules, fusion, evaluation, pipeline)
  config.yaml       all parameters (the values chosen in the Phase 2/3 experiments)
  data/             manifest.csv (all 12,614 images), splits.csv (leakage-safe train/validation/test split),
                    test_split.sha256 (hashes of the frozen test split), features/ (pre-computed features)
  models/           trained models for protocol R (main) and protocol B; MAIN_PROTOCOL names the main protocol
  results/          saved output of every experiment step
  scripts/          one script per experiment step, and check_dataset.py
  tests/            automated tests (python -m pytest)
  app/, detect.py   Streamlit interface and command-line detector (optional)

HOW TO RUN THE NOTEBOOK (no file paths to edit; about 30 seconds on a laptop CPU)
  1. Python 3.12:   python3 -m venv .venv && source .venv/bin/activate        (Windows: .venv\\Scripts\\activate)
                    pip install -r requirements.txt
                    pip install -e .                  (installs the project package `forgery`; the notebook itself does not need it,
                                                       the scripts in scripts/ and the tests do)
  2. Register the environment as a Jupyter kernel, so the notebook uses the packages just installed:
                    python -m ipykernel install --user --name forgery-phase3 --display-name "Python 3 (forgery Phase 3)"
  3. Start Jupyter FROM THIS FOLDER:   jupyter notebook GroupJ_Phase3.ipynb      and choose that kernel; Run All.
     Non-interactively:   jupyter nbconvert --to notebook --execute GroupJ_Phase3.ipynb --output executed.ipynb \\
                              --ExecutePreprocessor.kernel_name=forgery-phase3
  The notebook needs only the files in this folder (not the full datasets). Keep scikit-learn at 1.9.1 (the saved
  models are pickles). If the models cannot be loaded, the notebook says so and uses the cached outputs in precomputed/.

DATASETS (not included in full)
  CASIA v2.0  Image Tampering Detection Evaluation Database (Dong, Wang & Tan 2013), 12,614 images and masks.
              Kaggle: search "CASIA 2.0 Image Tampering Detection Dataset". Place it in CASIA2/ with the sub-folders
                  CASIA2/Au/ (7,491)   CASIA2/Tp/ (5,123)   CASIA2/CASIA 2 Groundtruth/ (5,123 masks)
              and check with:  python scripts/check_dataset.py      (prints DATASET OK)
  MICC-F220   Amerini et al. 2011, http://lci.micc.unifi.it/labd/2015/01/copy-move-forgery-detection-and-localization/
              Place it in data/external/MICC-F220/  (only needed for the cross-dataset test, scripts/run_phase7b.py)
  Both are for research use under their authors' terms; the few sample images included are for reproducing the notebook.

RE-RUNNING THE EXPERIMENTS (optional; needs CASIA2/, takes hours; see "Running it, step by step" in the repository README)
  python scripts/build_manifest.py -> make_splits.py -> run_leakage_study.py -> phase3_sanity.py ->
  sweep_params.py --write-config -> extract_features.py -> run_phase5.py -> extract_maps.py -> run_phase6.py ->
  run_phase7.py (the ONE-TIME test evaluation; it refuses to run twice unless forced) -> run_phase7b.py
  The CNN baseline (scripts/make_ela.py, run_phase8.py) additionally needs PyTorch (requirements-full.txt); its
  weights are not included because they are large (134 MB) and regenerate from the scripts.

Project repository: https://github.com/Smit2102/image-forgery-detection
Project website:    https://smit2102.github.io/image-forgery-detection/
"""


def collect() -> dict[Path, Path]:
    """archive path (relative to the zip's folder) -> source file"""
    out: dict[Path, Path] = {}

    def add(src: Path, arc: Path) -> None:
        if src.is_file() and src.name not in SKIP_NAMES and src.suffix != ".pyc":
            out[arc] = src

    def add_tree(src: Path, arc: Path) -> None:
        for f in sorted(src.rglob("*")):
            if f.is_file() and not (set(f.relative_to(src).parts) & SKIP_DIRS):
                add(f, arc / f.relative_to(src))

    for part in NOTEBOOK_PARTS:        # the notebook, the PDF and its supporting files, under the course's file names
        src = FOLDER / part
        (add_tree if src.is_dir() else add)(src, Path(part.replace(OLD, NEW)))
    add_tree(resolve("src"), Path("src"))
    add_tree(resolve("tests"), Path("tests"))
    add_tree(resolve("app"), Path("app"))
    add_tree(resolve(".streamlit"), Path(".streamlit"))
    for s in PROJECT_SCRIPTS:
        add(resolve(f"scripts/{s}"), Path("scripts") / s)
    for f in ("config.yaml", "pyproject.toml", "detect.py", "LICENSE"):
        add(resolve(f), Path(f))
    add(resolve("requirements.txt"), Path("requirements-full.txt"))
    for f in ("manifest.csv", "splits.csv", "test_split.sha256"):
        add(resolve(f"data/{f}"), Path("data") / f)
    add_tree(resolve("data/features"), Path("data/features"))
    for m in MODELS:
        add(resolve(f"models/{m}"), Path("models") / m)
    for d in sorted(resolve("results").iterdir()):
        if d.is_dir() and d.name not in SKIP_DIRS:
            add_tree(d, Path("results") / d.name)
    add_tree(resolve("outputs"), Path("outputs"))
    return out


def main() -> None:
    files = collect()
    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for arc, src in sorted(files.items()):
            z.write(src, Path(NEW) / arc)
        z.writestr(f"{NEW}/README.txt", README)
        z.writestr(f"{NEW}/requirements.txt", REQUIREMENTS)
        z.writestr(f"{NEW}/CASIA2/PUT_DATASET_HERE.txt",
                   "Place the CASIA v2.0 folders here: Au/, Tp/ and 'CASIA 2 Groundtruth/'. See README.txt, DATASETS.\n")
        n = len(z.namelist())
    print(f"{ZIP}  {ZIP.stat().st_size / 1e6:.1f} MB, {n} files")


if __name__ == "__main__":
    main()
