"""Builds the Phase 2 reproduction package: everything needed to re-run the Phase 2 notebook with the CASIA v2.0 dataset.

    python scripts/build_reproduction_zip.py   # -> submissions/phase2_submission/GroupJ_Patel_Gupta_Phase2_reproduction.zip

The package holds the `forgery` source code, configuration, dataset manifest and split files, saved models,
experiment results, tests and the unchanged Phase 2 notebook (only its kernel name is made portable).
The CASIA v2.0 images themselves are not included; README_REPRODUCE.md says where to put them.
"""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path

import nbformat

from forgery.config import resolve

NAME = "GroupJ_Patel_Gupta_Phase2_reproduction"
ZIP = resolve(f"submissions/phase2_submission/{NAME}.zip")
SRC_NB = resolve("build/phase2/GroupJ_Patel_Gupta_Phase2.ipynb")
SRC_PDF = resolve("build/phase2/GroupJ_Patel_Gupta_Phase2.pdf")

# Phase 2 scope only: the later work (one-time test-set evaluation, robustness, CNN baseline, interface) is left out
SKIP = shutil.ignore_patterns("__pycache__", ".DS_Store", "*.pyc", "*.egg-info", ".pytest_cache")
SKIP_SRC = shutil.ignore_patterns("__pycache__", ".DS_Store", "*.pyc", "*.egg-info", "cnn.py")
SKIP_TESTS = shutil.ignore_patterns("__pycache__", ".DS_Store", ".pytest_cache", "test_cnn.py", "test_app.py")
SKIP_FEATURES = shutil.ignore_patterns(".DS_Store", "*_test.*", "*_dry.*")       # test-set / dry-run caches (later phase)
SKIP_RESULTS = shutil.ignore_patterns("__pycache__", ".DS_Store", "pre_qa_run")
RESULTS = ["phase1", "phase2", "phase3", "phase4", "phase5", "phase6"]
MODELS = [f"{m}_{p}.joblib" for p in ("R", "B") for m in ("forgery", "type", "localizer")] + ["MAIN_PROTOCOL"]
SCRIPTS = ["check_dataset.py", "build_manifest.py", "make_splits.py", "run_leakage_study.py", "phase3_sanity.py",
           "sweep_params.py", "extract_features.py", "run_phase5.py", "extract_maps.py", "run_phase6.py", "phase6_gating.py"]

REQUIREMENTS = """# Python 3.12 (3.11+ works for the code, but the saved models were written with the versions below)
numpy==2.5.3
pandas==3.0.6
scipy==1.18.1
scikit-learn==1.9.1       # the saved models are pickles: use this exact version to load them
scikit-image==0.26.0
opencv-python==5.0.0.93
pillow==12.3.0
matplotlib==3.11.2
joblib==1.6.0
PyYAML==6.0.3
tabulate==0.10.0
shap==0.52.0              # feature explanations (scripts/run_phase5.py)
pytest==9.1.1
# to run the notebook
jupyter
ipykernel
nbconvert
"""

README = """PHASE 2 - REPRODUCTION PACKAGE
Group J (Smit Patel, Ananay Gupta) - EDS 6364 Digital Image Processing
Image Forgery Detection Using Classical Image Processing Techniques

This package lets you run the Phase 2 notebook (GroupJ_Patel_Gupta_Phase2.ipynb) and re-create its results.
The notebook imports our own Python package `forgery`; it is in src/forgery/ and installs in one command (step 2).
The only thing NOT included is the CASIA v2.0 image dataset itself (3.3 GB, 12,614 images).

1. DATASET
   Download "CASIA 2.0 Image Tampering Detection Dataset" (Kaggle; Dong, Wang & Tan 2013) and place it so that
   this folder contains

        CASIA2/Au/                       7,491 authentic images (jpg, bmp)
        CASIA2/Tp/                       5,123 tampered images (jpg, tif)
        CASIA2/CASIA 2 Groundtruth/      5,123 ground-truth masks (*_gt.png)

   (the folder is named CASIA2 in the project; unzip so these three folders are directly inside it - or make
   CASIA2 a link to wherever you keep the dataset). Then check it:

        python scripts/check_dataset.py          # prints DATASET OK when every image in the manifest is found

   data/manifest.csv lists all 12,614 images with their paths relative to this folder.

2. ENVIRONMENT (Python 3.12, about 1 GB)
        python3 -m venv .venv
        source .venv/bin/activate                # Windows: .venv\\Scripts\\activate
        pip install -r requirements.txt
        pip install -e .                         # installs the `forgery` package from src/
   (Instead of `pip install -e .`, setting PYTHONPATH=src before starting Jupyter also works.)
   Keep scikit-learn at 1.9.1: the saved models are pickles written with that version.

3. RUN THE NOTEBOOK  (start Jupyter from THIS folder, so the relative paths resolve)
   Register this environment as a Jupyter kernel, so the notebook runs with the packages just installed
   (a "Python 3" kernel registered earlier on your machine may point to a different Python):
        python -m ipykernel install --user --name forgery-phase2 --display-name "Python 3 (forgery Phase 2)"
   then either open it and choose Kernel -> "Python 3 (forgery Phase 2)" -> Run All (about 30 seconds):
        jupyter notebook GroupJ_Patel_Gupta_Phase2.ipynb
   or run it non-interactively:
        jupyter nbconvert --to notebook --execute GroupJ_Patel_Gupta_Phase2.ipynb --output executed.ipynb \\
            --ExecutePreprocessor.kernel_name=forgery-phase2
   GroupJ_Patel_Gupta_Phase2.pdf is the submitted, already-executed version for comparison. We checked this
   package on a clean environment: every printed table and number, and every figure, came out identical to it.

4. WHAT THE NOTEBOOK READS
   data/manifest.csv, data/splits.csv       dataset audit and the leakage-safe train / validation / test split
   config.yaml                              all parameters (tuned values written by scripts/sweep_params.py)
   models/*_R.joblib                        trained protocol-R models (classifier, forgery type, localiser)
   results/phase4, phase5, phase6           saved experiment tables (parameter sweep, ablations, added value, methods)
   CASIA2/...                               the images shown in the demonstrations

5. RE-CREATING THE RESULTS FROM SCRATCH (optional; needs the dataset, takes hours)
   Each script reads the previous step's output and writes to results/ or models/. Run from this folder:
        python scripts/build_manifest.py         1  audit the dataset -> data/manifest.csv
        python scripts/make_splits.py            2  split (refuses to change the frozen test set: data/test_split.sha256)
        python scripts/run_leakage_study.py      3  shortcut / leakage study            -> results/phase2
        python scripts/phase3_sanity.py          4  module sanity check                 -> results/phase3
        python scripts/sweep_params.py --write-config    5  parameter tuning            -> results/phase4, config.yaml
        python scripts/extract_features.py       6a features of every image (data/features is included, so this can be skipped)
        python scripts/run_phase5.py             6b classification, ablations, models   -> results/phase5, models/
        python scripts/extract_maps.py && python scripts/run_phase6.py    7  localisation -> results/phase6
   All Phase 2 results use the training and validation images only; the test split stays frozen (not used in Phase 2).
   Retraining gives the same conclusions, but tiny numerical differences can occur between library versions.

6. TESTS
        python -m pytest                         # data-dependent tests run only when the dataset is in place

FOLDER CONTENTS
   GroupJ_Patel_Gupta_Phase2.ipynb / .pdf   the Phase 2 submission
   src/forgery/                             the package: features/ (the 8 image-processing modules), localize/, eval/, data/,
                                            pipeline.py (detect()), explain.py, io.py, config.py
   config.yaml, pyproject.toml              parameters and package definition
   data/                                    manifest.csv, splits.csv, test_split.sha256, features/ (pre-computed train/validation features)
   models/                                  protocol R (main) and protocol B models; MAIN_PROTOCOL names the main protocol
   results/phase1 ... phase6                saved outputs of every experiment step (dataset audit, leakage study,
                                            module checks, parameter sweep, classification, localisation)
   scripts/                                 one script per experiment step + check_dataset.py
   tests/                                   automated tests

Project repository: https://github.com/Smit2102/image-forgery-detection
Contact: Smit Patel (smitpatel5115@gmail.com)
"""


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / NAME
        root.mkdir()

        # --- the submission itself (kernel name made portable: the original names a kernel that only exists on our machine)
        nb = nbformat.read(SRC_NB, 4)
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nbformat.write(nb, root / SRC_NB.name)
        shutil.copy2(SRC_PDF, root / SRC_PDF.name)

        # --- code and configuration
        shutil.copytree(resolve("src"), root / "src", ignore=SKIP_SRC)
        shutil.copytree(resolve("tests"), root / "tests", ignore=SKIP_TESTS)
        (root / "scripts").mkdir()
        for f in SCRIPTS:
            shutil.copy2(resolve(f"scripts/{f}"), root / "scripts" / f)
        for f in ("config.yaml", "pyproject.toml", "LICENSE"):
            shutil.copy2(resolve(f), root / f)
        (root / "requirements.txt").write_text(REQUIREMENTS)
        (root / "README_REPRODUCE.md").write_text(README)

        # --- dataset manifest, splits, features
        (root / "data").mkdir()
        for f in ("manifest.csv", "splits.csv", "test_split.sha256"):
            shutil.copy2(resolve(f"data/{f}"), root / "data" / f)
        shutil.copytree(resolve("data/features"), root / "data/features", ignore=SKIP_FEATURES)

        # --- saved models and results
        (root / "models").mkdir()
        for m in MODELS:
            shutil.copy2(resolve(f"models/{m}"), root / "models" / m)
        for r in RESULTS:
            shutil.copytree(resolve(f"results/{r}"), root / "results" / r, ignore=SKIP_RESULTS)
        (root / "CASIA2").mkdir()
        (root / "CASIA2/PUT_DATASET_HERE.txt").write_text(
            "Place the CASIA v2.0 folders here: Au/, Tp/ and 'CASIA 2 Groundtruth/'. See README_REPRODUCE.md, step 1.\n")

        if ZIP.exists():
            ZIP.unlink()
        with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(root.rglob("*")):
                if f.is_file():
                    z.write(f, Path(NAME) / f.relative_to(root))
        n = sum(1 for f in root.rglob("*") if f.is_file())
    print(f"{ZIP}  {ZIP.stat().st_size / 1e6:.1f} MB, {n} files")


if __name__ == "__main__":
    main()
