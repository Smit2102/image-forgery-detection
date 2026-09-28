# Style guide for the LaTeX sections (IEEEtran conference, two columns)

Audience: a Digital Image Processing course instructor and graders (master's level). Formal, concise, third person
or "we" is fine; no marketing language. The author is Smit Patel alone — never mention any other person.

## Source of truth
- Each section is a condensed version of its Markdown draft in `report/drafts/`. The drafts have been QA-checked:
  **copy every number exactly as the draft states it** (same rounding, same CIs). Never add a number, claim,
  comparison or interpretation that is not in the draft. If you shorten, drop detail — do not paraphrase numbers.
- Where the draft says "presumably", "hypothesis", "not tested", keep that hedging.
- Do not describe code or file names in the main text except where essential (script names can go in the
  appendix section you write). No Markdown in LaTeX.

## LaTeX conventions
- File: `report/latex/sections/<NN>_<name>.tex`, starting with `\section{Title}\label{sec:<key>}`.
  Section keys (use exactly these for \ref): intro, related, dataset, leakage, method, params, classification,
  localisation, test, robustness, cnn, system, conclusion. Cross-references to other sections: only
  `Section~\ref{sec:key}` (never to other authors' subsections). Within your own section you may add
  `\subsection{...}\label{sec:key-short}`.
- Tables: `table` (single column) or `table*` (full width) with `\caption{...}\label{tab:...}` ABOVE the tabular,
  booktabs rules (`\toprule \midrule \bottomrule`), `\footnotesize` or `\scriptsize` inside, no vertical rules.
  Keep single-column tables ≤ 3.4 in wide (few columns, short headers). Use `\ci{lo}{hi}` for CIs, e.g.
  `0.805\ci{0.779}{0.831}` renders "0.805 [0.779, 0.831]". Protocol names: plain R, B, A.
- Figures: `figure` (single column, `width=\columnwidth`) or `figure*` (full width, `width=\textwidth` or less),
  `\includegraphics[...]{fig_xxx.png}` (files live in `report/latex/figures/`; only use the names listed below),
  caption BELOW, `\label{fig:...}`. Open each figure image you use (Read tool) to write an accurate caption.
  Every figure and table must be referenced in the text (`Fig.~\ref{}`, `Table~\ref{}`).
- Math: `$\ldots$`; percent as `\%`; en-dash ranges `--`; minus sign `$-$0.036`; `$\times$`; `$\pm$`; `$\chi^2$`.
  Escape `_`, `&`, `%`, `#`. No Unicode math symbols (write `$\geq$`, `$\sigma$`, `$\Delta$AUC`).
- Citations: `\cite{key}` using only keys that exist in `report/latex/refs.bib` (read it). Do not invent references.
- Bullet lists: use sparingly (`itemize` with `\setlength\itemsep{0pt}` if needed); prefer compact prose.
- Appendix material: if a table or detail is valuable but does not fit your word budget, put it in
  `report/latex/sections/appendix_<key>.tex` as `\section{Title}\label{app:<key>}` (the appendix is one-column),
  and refer to it as `Appendix~\ref{app:<key>}`.

## Figures available (report/latex/figures/)
fig_dataset_samples, fig_dataset_format, fig_tampered_area, fig_leakage, fig_modules_R, fig_ela_before_after,
fig_dct_histogram, fig_copymove_matches, fig_sweeps, fig_roc_dev_R, fig_shap_R, fig_ablation, fig_calibration_R,
fig_loc_methods_R, fig_loc_examples_R, fig_roc_test_R, fig_failures_R, fig_robustness, fig_synthetic_examples,
fig_cnn_roc, fig_ela_examples, fig_cnn_training, fig_ui_analyze, fig_ui_evidence_maps (all .png)

## Check your work
Every section file already exists as a one-line placeholder (so all `\ref{sec:key}` resolve); overwrite only
your own files. Other writers work in parallel, so never compile in `report/latex` itself: copy it to your
scratch dir (`cp -R report/latex <scratch>/latex_<you>`), copy your files there, and run
`tectonic --keep-logs main.tex` in the copy (run it twice if references show as ??). Fix every LaTeX error,
undefined reference/citation and Overfull \hbox warning that comes from your files
(`grep -n "Warning\|Error\|Overfull" main.log`). Do not edit other sections, main.tex, refs.bib or STYLE.md.
Report back: files written, word count of each, figures/tables used, anything in the draft you had to drop.
