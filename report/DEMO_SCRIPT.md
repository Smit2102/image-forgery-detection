# Demo script (about 5 minutes)

For the live demonstration after the slides. Everything runs locally on the laptop, with no internet needed.

## Before the session (5 minutes)

1. Open a terminal in the project folder and start the interface:
   ```bash
   .venv/bin/streamlit run app/Home.py
   ```
   It opens at http://localhost:8510 (the project pins this port in `.streamlit/config.toml`, because 8501 is often taken by other Streamlit apps). Keep the terminal open.
2. Warm it up once. Open **Analyze an image** → **Example** → the first MICC-F220 example, so the models are loaded and cached before the audience watches.
3. Keep a folder with 4–6 images ready for batch mode, for example a few of your own phone photos plus two MICC-F220 files from `data/external/MICC-F220/`.
4. **Backup if the laptop fails:** the screenshots in `results/phase8_5/` and the slides.

## 1. Home page (30 s)

- "The detector uses eight classical techniques from this course."
- Point at the four numbers:
  - **test AUC 0.805** under the strict protocol
  - **86 % accuracy** on the images it judges
  - **0.97** on a dataset it never saw
  - **pixel F1 0.20**
- Read one line of "Read the results carefully": a verdict is evidence, not proof.

## 2. Analyze an image: a real copy-move (2 min)

1. **Analyze an image** → **Example** → *MICC-F220 · …tamp… (copy-move forgery, unseen dataset)*.
2. The verdict badge reads **TAMPERED**, P ≈ 0.94, likely type **copy-move**. Say:
   - "Protocol R first shrinks the image and re-saves it as JPEG, so file-format tricks cannot help."
   - "The red overlay marks the suspected region, and the heat map is the fused probability."
3. Tab **Why this verdict**:
   - "Keypoint matching found pairs of points that one geometric transform explains, which means part of the image is duplicated."
   - Point at the SHAP bars: red pushes towards tampered, blue towards authentic.
4. Tab **Evidence maps**:
   - "One map per technique: ELA, histograms, noise, JPEG ghosts, DCT, edges and the two copy-move matchers."
   - "They share one 0–1 scale, so you can compare them."
5. Tab **Download** → **PDF report**. Open it briefly: a two-page report ready to attach to a case.

## 3. An authentic image (30 s)

- **Example** → the matching *original* of the same scene.
- It usually comes out **UNCERTAIN** or **AUTHENTIC**. Say: "The detector abstains when it isn't sure. About half of all images fall in this band, and that is how it reaches 86 % accuracy on the rest."
- Optional: switch the sidebar to **protocol B** and show that the band is much narrower.

## 4. The dataset leak (1 min)

- **Dataset explorer** → tab **Why the released dataset leaks**. Say:
  - "Authentic images are JPEGs, mostly with standard tables. Most tampered images are TIFF files."
  - "A classifier that reads only the file metadata gets 99 % balanced accuracy, without looking at a single pixel. That is why we evaluate under protocol R."
- Tab **Browse images** → click **Random** once to show an image with its ground-truth mask.

## 5. Batch mode (30 s)

- **Batch analysis** → **Folder on this computer** → paste the prepared folder → **Analyse**.
- Show the verdict table and the gallery. The CSV download keeps the table.

## 6. Results dashboard (30 s, if time allows)

- Tab **6 · CNN comparison**: "A standard ELA-CNN scores 0.994 on the raw files, but there the shortcut features alone reach 1.000, so that score alone does not show tampering detection. Under protocol R it is below our classical pipeline, and on the unseen dataset its AUC is consistent with chance."

## Likely questions

| Question | Short answer |
|---|---|
| Why not just use deep learning? | We tested it (Section XI). The ELA-CNN scored 0.994 only on the leaking files, where shortcuts alone reach 1.000; under R it trails the classical pipeline, and on MICC-F220 it was consistent with chance. The classical cues are interpretable and did transfer. |
| Why is localisation F1 only 0.20? | Tampered regions are small (median 3 % of the image). F1 is 0.33 for regions above 5 %. The ground truth marks only the pasted copy; on synthetic copy-moves we showed that the localiser marks both source and copy. |
| What does it miss? | Splices between photos with the same processing history, low-detail images, and small regions. False alarms come from genuinely repeated structure. |
| Was the test set reused? | It was evaluated once for the headline results. Every later use is logged in `results/phase7/test_runs.log`, and nothing was tuned on it. |
| Can I run it on my photo? | Yes: upload it (JPEG, PNG, TIFF, BMP or WebP, up to 4,000 px), or use `python detect.py photo.jpg`. |
