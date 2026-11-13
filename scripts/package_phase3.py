"""Phase 3 course submission: packs the notebook folder build/phase3/ and the project into ONE zip.

    python scripts/package_phase3.py            # -> submissions/phase3_submission/GroupJ_Phase3.zip

The zip has two folders, so the instructor can find the submission material and, if she wants, set up the project:
  GroupJ_Phase3/Submission/   what the course asks for: GroupJ_Phase3.ipynb, GroupJ_Phase3.pdf, supporting_images/
                              (GroupJ_Patel_Gupta_Phase3.* renamed; the notebook and PDF do not refer to their names)
  GroupJ_Phase3/Project/      everything needed to run the notebook and the project: src/, config.yaml, models/,
                              supporting/ (saved tables and figures), precomputed/ (fallback outputs), data/, results/,
                              scripts/, tests/, app/, detect.py, pyproject.toml, requirements
The notebook finds ../Project by itself, so it runs from Submission/ with no paths to edit.
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
SUB, PROJ = Path("Submission"), Path("Project")
SUBMISSION_PARTS = {f"{OLD}.ipynb": SUB / f"{NEW}.ipynb", f"{OLD}.pdf": SUB / f"{NEW}.pdf",
                    "sample_images": SUB / "supporting_images"}
NOTEBOOK_DATA = ["supporting", "precomputed"]                  # read by the notebook, kept with the project
MODELS = [f"{m}_{p}.joblib" for p in ("R", "B") for m in ("forgery", "type", "localizer")] + ["MAIN_PROTOCOL"]
SKIP_DIRS = {"__pycache__", ".pytest_cache", "pre_qa_run", "detect", "phase8_smoke"}
SKIP_NAMES = {".DS_Store"}
PROJECT_SCRIPTS = SCRIPTS + ["run_phase7.py", "run_phase7b.py", "phase7_posthoc.py", "make_ela.py", "run_phase8.py",
                             "phase8_posthoc.py", "make_outputs.py"]

README = """PHASE 3 SUBMISSION - Group J (Smit Patel 2479966, Ananay Gupta 2618966) - EDS 6364 Digital Image Processing
Image Forgery Detection Using Classical Image Processing Techniques

This zip has two folders:

  Submission/   THE SUBMISSION MATERIAL (what the course asked for)
                  GroupJ_Phase3.ipynb     Jupyter notebook
                  GroupJ_Phase3.pdf       PDF export of the notebook
                  supporting_images/      the images used by the notebook: 14 CASIA v2.0 images with their
                                          ground-truth masks and 22 MICC-F220 images (see supporting_images/README.txt)

  Project/      THE PROJECT SET-UP (optional: only needed to run the notebook or the experiments yourself)
                  src/forgery/            the Python package (8 image-processing modules, fusion, evaluation, pipeline)
                  config.yaml             all parameters (the values chosen in the Phase 2/3 experiments)
                  models/                 trained models for protocol R (main) and protocol B
                  supporting/             saved result tables and figures that the notebook reads
                  precomputed/            cached detector outputs (used only if the models cannot be loaded)
                  data/                   manifest.csv (all 12,614 images), splits.csv (leakage-safe split),
                                          test_split.sha256 (frozen test split), features/ (pre-computed features)
                  results/, outputs/      saved output of every experiment step
                  scripts/, tests/        one script per experiment step (+ check_dataset.py); automated tests
                  app/, detect.py         Streamlit interface and command-line detector (optional)
                  requirements.txt, pyproject.toml

HOW TO RUN THE NOTEBOOK (no file paths to edit; about 30 seconds on a laptop CPU)
  Keep the two folders side by side: the notebook in Submission/ finds ../Project by itself.
  1. In a terminal, from the Project folder (Python 3.12):
         cd Project
         python3 -m venv .venv && source .venv/bin/activate        (Windows: .venv\\Scripts\\activate)
         pip install -r requirements.txt
         pip install -e .                  (installs the package `forgery`; needed by the scripts and tests)
  2. Register the environment as a Jupyter kernel, so the notebook uses the packages just installed:
         python -m ipykernel install --user --name forgery-phase3 --display-name "Python 3 (forgery Phase 3)"
  3. Open the notebook, choose that kernel and Run All:
         jupyter notebook ../Submission/GroupJ_Phase3.ipynb
     Non-interactively:
         jupyter nbconvert --to notebook --execute ../Submission/GroupJ_Phase3.ipynb --output executed.ipynb \\
             --ExecutePreprocessor.kernel_name=forgery-phase3
  The notebook needs only these two folders, not the full datasets. Keep scikit-learn at 1.9.1 (the saved models
  are pickles). If the models cannot be loaded, the notebook says so and uses the cached outputs in Project/precomputed/.

DATASETS (not included in full)
  CASIA v2.0  Image Tampering Detection Evaluation Database (Dong, Wang & Tan 2013), 12,614 images and masks.
              Kaggle: search "CASIA 2.0 Image Tampering Detection Dataset". Place it in Project/CASIA2/ with
                  CASIA2/Au/ (7,491)   CASIA2/Tp/ (5,123)   CASIA2/CASIA 2 Groundtruth/ (5,123 masks)
              and check with:  python scripts/check_dataset.py      (prints DATASET OK)
  MICC-F220   Amerini et al. 2011, http://lci.micc.unifi.it/labd/2015/01/copy-move-forgery-detection-and-localization/
              Place it in Project/data/external/MICC-F220/  (only for the cross-dataset test, scripts/run_phase7b.py)
  Both are for research use under their authors' terms; the few sample images included are for reproducing the notebook.

RE-RUNNING THE EXPERIMENTS (optional; needs Project/CASIA2/, takes hours; run from Project/)
  python scripts/build_manifest.py -> make_splits.py -> run_leakage_study.py -> phase3_sanity.py ->
  sweep_params.py --write-config -> extract_features.py -> run_phase5.py -> extract_maps.py -> run_phase6.py ->
  run_phase7.py (the ONE-TIME test evaluation; it refuses to run twice unless forced) -> run_phase7b.py
  The CNN baseline (scripts/make_ela.py, run_phase8.py) additionally needs PyTorch (requirements-full.txt); its
  weights are not included because they are large (134 MB) and regenerate from the scripts.

Project repository: https://github.com/Smit2102/image-forgery-detection
Project website:    https://smit2102.github.io/image-forgery-detection/
"""


def collect() -> dict[Path, Path]:
    """archive path (relative to the zip's top folder) -> source file"""
    out: dict[Path, Path] = {}

    def add(src: Path, arc: Path) -> None:
        if src.is_file() and src.name not in SKIP_NAMES and src.suffix != ".pyc":
            out[arc] = src

    def add_tree(src: Path, arc: Path) -> None:
        for f in sorted(src.rglob("*")):
            if f.is_file() and not (set(f.relative_to(src).parts) & SKIP_DIRS):
                add(f, arc / f.relative_to(src))

    for part, arc in SUBMISSION_PARTS.items():           # the course deliverables, under the course's file names
        src = FOLDER / part
        (add_tree if src.is_dir() else add)(src, arc)
    for part in NOTEBOOK_DATA:
        add_tree(FOLDER / part, PROJ / part)
    add_tree(resolve("src"), PROJ / "src")
    add_tree(resolve("tests"), PROJ / "tests")
    add_tree(resolve("app"), PROJ / "app")
    add_tree(resolve(".streamlit"), PROJ / ".streamlit")
    for s in PROJECT_SCRIPTS:
        add(resolve(f"scripts/{s}"), PROJ / "scripts" / s)
    for f in ("config.yaml", "pyproject.toml", "detect.py", "LICENSE"):
        add(resolve(f), PROJ / f)
    add(resolve("requirements.txt"), PROJ / "requirements-full.txt")
    for f in ("manifest.csv", "splits.csv", "test_split.sha256"):
        add(resolve(f"data/{f}"), PROJ / "data" / f)
    add_tree(resolve("data/features"), PROJ / "data/features")
    for m in MODELS:
        add(resolve(f"models/{m}"), PROJ / "models" / m)
    for d in sorted(resolve("results").iterdir()):
        if d.is_dir() and d.name not in SKIP_DIRS:
            add_tree(d, PROJ / "results" / d.name)
    add_tree(resolve("outputs"), PROJ / "outputs")
    return out


def main() -> None:
    files = collect()
    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for arc, src in sorted(files.items()):
            z.write(src, Path(NEW) / arc)
        z.writestr(f"{NEW}/README.txt", README)
        z.writestr(f"{NEW}/{PROJ}/requirements.txt", REQUIREMENTS)
        z.writestr(f"{NEW}/{PROJ}/CASIA2/PUT_DATASET_HERE.txt",
                   "Place the CASIA v2.0 folders here: Au/, Tp/ and 'CASIA 2 Groundtruth/'. See ../../README.txt, DATASETS.\n")
        n = len(z.namelist())
    print(f"{ZIP}  {ZIP.stat().st_size / 1e6:.1f} MB, {n} files")


if __name__ == "__main__":
    main()
