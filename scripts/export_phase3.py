"""Phase 3 submission: sets a portable kernel in the executed notebook and exports the PDF.

  1. python scripts/build_phase3.py            (assemble the folder)
  2. python scripts/build_phase3_notebook.py   (write the notebook)
  3. jupyter nbconvert --to notebook --execute --inplace build/phase3/GroupJ_Patel_Gupta_Phase3.ipynb \
         --ExecutePreprocessor.kernel_name=<kernel of the project environment>
  4. python scripts/export_phase3.py           (this script)
  5. python scripts/package_phase3.py          (the single submission zip, GroupJ_Phase3.zip)
"""

from __future__ import annotations

import subprocess
import sys

import nbformat
from playwright.sync_api import sync_playwright

from forgery.config import resolve

FOLDER = resolve("build/phase3")
NB = FOLDER / "GroupJ_Patel_Gupta_Phase3.ipynb"
PDF = FOLDER / "GroupJ_Patel_Gupta_Phase3.pdf"
HTML = resolve("build/_phase3_tmp.html")

CSS = """
body { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
.jp-OutputArea-output img { max-width: 100% !important; height: auto !important; break-inside: avoid; page-break-inside: avoid; }
.jp-RenderedHTMLCommon table, .jp-OutputArea-output table { break-inside: avoid; font-size: 9pt !important; }
h1, h2, h3 { break-after: avoid; page-break-after: avoid; }
.jp-RenderedHTMLCommon pre { break-inside: avoid; }
.jp-InputPrompt, .jp-OutputPrompt { display: none !important; }
.jp-Notebook { padding: 0 !important; }
pre, code { white-space: pre-wrap !important; word-break: break-word; font-size: 8pt !important; }
"""


def main():
    nb = nbformat.read(NB, 4)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}   # portable
    nbformat.write(nb, NB)

    subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "html", str(NB), "--output", str(HTML)], check=True)
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True)
        pg = b.new_page()
        pg.goto("file://" + str(HTML))
        pg.wait_for_load_state("networkidle")
        pg.add_style_tag(content=CSS)
        pg.wait_for_timeout(2000)
        pg.pdf(path=str(PDF), format="A4", print_background=True,
               margin={"top": "14mm", "bottom": "14mm", "left": "11mm", "right": "11mm"}, display_header_footer=True,
               header_template="<span></span>",
               footer_template='<div style="font-size:8px;width:100%;text-align:center;color:#888">Group J (Smit Patel, Ananay Gupta)'
                               ' — Phase 3 — page <span class="pageNumber"></span> / <span class="totalPages"></span></div>')
        b.close()
    HTML.unlink()

    import pymupdf
    print(f"notebook {NB.stat().st_size / 1e6:.1f} MB | pdf {PDF.stat().st_size / 1e6:.1f} MB, {len(pymupdf.open(PDF))} pages")


if __name__ == "__main__":
    main()
