"""Headless smoke tests of the Streamlit pages (streamlit.testing.AppTest): every page renders without an
exception, and the analysis page produces a verdict on an example image."""

from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
APP = Path(__file__).resolve().parents[1] / "app"
MODELS = Path(__file__).resolve().parents[1] / "models"
needs_models = pytest.mark.skipif(not (MODELS / "forgery_R.joblib").exists(), reason="trained models missing")


@pytest.fixture(autouse=True)
def _upload_dir(tmp_path, monkeypatch):
    """Keep the UI's upload store out of the real system temp folder during tests."""
    import sys
    monkeypatch.setenv("FORGERY_UI_UPLOAD_DIR", str(tmp_path / "uploads"))
    sys.modules.pop("common", None)               # re-import app/common.py so it picks up the folder


def _run(page: str, timeout: int = 120):
    import sys
    sys.path.insert(0, str(APP))                  # pages import app/common.py
    at = st_testing.AppTest.from_file(str(APP / page), default_timeout=timeout)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_home():
    at = _run("Home.py")
    assert "Image Forgery Detection" in at.title[0].value


def test_results_dashboard():
    at = _run("pages/4_Results_dashboard.py")
    assert len(at.tabs) == 6


@needs_models
def test_dataset_explorer():
    at = _run("pages/3_Dataset_explorer.py")
    assert any("test images are not shown" in c.value for c in at.caption)


@needs_models
def test_analyze_example_gives_verdict():
    at = _run("pages/1_Analyze_an_image.py")
    at.radio(key="source").set_value("Example").run()
    assert not at.exception, [e.value for e in at.exception]
    html = " ".join(m.value for m in at.markdown)
    assert any(v in html for v in ("TAMPERED", "AUTHENTIC", "UNCERTAIN"))
    assert len(at.tabs) == 4


@needs_models
def test_batch_page_renders():
    _run("pages/2_Batch_analysis.py")


@needs_models
def test_batch_folder_run_handles_bad_files(tmp_path):
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(1)
    for i in range(2):
        Image.fromarray(rng.integers(0, 255, (200, 260, 3), dtype=np.uint8)).save(tmp_path / f"img{i}.png")
    (tmp_path / "broken.jpg").write_text("not an image")
    at = _run("pages/2_Batch_analysis.py")
    at.radio(key="source").set_value("Folder on this computer").run()
    at.text_input[0].input(str(tmp_path)).run()
    at.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    df = at.dataframe[0].value
    assert len(df) == 3 and df.error.notna().sum() == 1
    assert set(df.verdict.dropna()) <= {"tampered", "authentic", "uncertain"}


@needs_models
def test_batch_all_files_bad_and_results_persist(tmp_path):
    (tmp_path / "a.jpg").write_text("not an image")
    (tmp_path / "b.png").write_bytes(b"\x89PNG broken")
    (tmp_path / "._c.jpg").write_text("macOS metadata file, must be skipped")
    at = _run("pages/2_Batch_analysis.py")
    at.radio(key="source").set_value("Folder on this computer").run()
    at.text_input[0].input(str(tmp_path)).run()
    at.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    df = at.dataframe[0].value
    assert list(df.image) == ["a.jpg", "b.png"] and df.error.notna().all()
    at.run()                                        # any rerun (e.g. a download click) keeps the results
    assert not at.exception and len(at.dataframe) == 1


@needs_models
def test_explorer_empty_filter_keeps_other_tabs():
    at = _run("pages/3_Dataset_explorer.py")
    at.multiselect[0].set_value([]).run()
    assert not at.exception
    assert any("without looking at the image content" in m.value for m in at.markdown)
