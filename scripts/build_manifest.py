"""Phase 1: build data/manifest.csv, dataset statistics and report figures.

Usage:  python scripts/build_manifest.py
"""

from __future__ import annotations

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forgery.config import load_config, resolve
from forgery.data.manifest import build_manifest
from forgery.io import load_mask, load_rgb

OUT = resolve("results/phase1")
TYPE_ORDER = ["none", "copy-move", "splicing"]
TYPE_LABEL = {"none": "Authentic", "copy-move": "Copy-move", "splicing": "Splicing"}


def summarize(df: pd.DataFrame) -> dict:
    tp = df[df.label == 1]
    s = {
        "n_images": len(df),
        "n_authentic": int((df.label == 0).sum()),
        "n_tampered": len(tp),
        "by_type": df.forgery_type.value_counts().to_dict(),
        "format_by_class": pd.crosstab(df.forgery_type, df.ext).to_dict(orient="index"),
        "category_by_class": pd.crosstab(df.category, df.label_name).to_dict(orient="index"),
        "mask_status": tp.mask_status.value_counts().to_dict(),
        "n_mask_valid": int(df.mask_valid.sum()),
        "tampered_frac_percentiles": {str(q): round(float(tp.loc[tp.mask_valid, "tampered_frac"].quantile(q / 100)), 4)
                                      for q in (10, 25, 50, 75, 90)},
        "area_bucket": tp.area_bucket.value_counts().to_dict(),
        "area_bucket_by_type": pd.crosstab(tp.forgery_type, tp.area_bucket).to_dict(orient="index"),
        "mask_components_median": float(tp.loc[tp.mask_valid, "mask_components"].median()),
        "jpeg_quality_by_class": {
            name: df[(df.format == "JPEG") & (df.label == lab)].jpeg_quality.value_counts().head(6).to_dict()
            for name, lab in (("authentic", 0), ("tampered", 1))},
        "standard_ijg_table_share_of_jpegs": {
            name: round(float(df[(df.format == "JPEG") & (df.label == lab)].qtable_standard.mean()), 3)
            for name, lab in (("authentic", 0), ("tampered", 1))},
        "top_sizes_by_class": {
            name: (df[df.label == lab].width.astype(str) + "x" + df[df.label == lab].height.astype(str))
            .value_counts().head(5).to_dict()
            for name, lab in (("authentic", 0), ("tampered", 1))},
        "share_384x256_or_256x384": {
            name: round(float(df[df.label == lab].apply(
                lambda r: {r.width, r.height} == {384, 256}, axis=1).mean()), 3)
            for name, lab in (("authentic", 0), ("tampered", 1))},
        "n_groups": int(df.group_id.nunique()),
        "max_group_size": int(df.group_id.value_counts().max()),
        "by_type_from_name_letter": df.forgery_type_name.value_counts().to_dict(),
        "n_type_votes_disagree": int((~df.forgery_type_votes_agree).sum()),
        "n_type_changed_by_majority_vote": int((df.forgery_type != df.forgery_type_name).sum()),
        "type_change_detail": {f"{a} -> {b}": int(n) for (a, b), n in
                               tp.loc[tp.forgery_type != tp.forgery_type_name]
                               .groupby(["forgery_type_name", "forgery_type"]).size().items()},
        "mask_name_relation": tp.mask_name_relation.value_counts().to_dict(),
        "n_mask_names_with_placeholder_id": int(tp.mask_has_placeholder_id.sum()),
        "mask_size_problems": [
            {"image": r.path.rsplit("/", 1)[-1], "status": r.mask_status,
             "image_wh": [int(r.width), int(r.height)], "mask_wh": [int(r.mask_width), int(r.mask_height)]}
            for r in tp[tp.mask_status.isin(["transposed", "size_mismatch"])].itertuples()],
        "background_census": background_census(tp),
        "n_exact_duplicates": int((df.dup_count > 1).sum()),
        "n_cross_label_duplicates": int(df.dup_cross_label.sum()),
        "host_diff_recall_median": round(float(tp.host_diff_recall.median()), 3),
        "host_diff_precision_median": round(float(tp.host_diff_precision.median()), 3),
        "n_host_diff_checked": int(tp.host_diff_recall.notna().sum()),
        "n_low_mask_agreement": int((tp.host_diff_recall < 0.2).sum()),
    }
    return s


def background_census(tp: pd.DataFrame) -> dict:
    """Which named id supplies most pixels (grey-level similarity > 0.5 to the tampered image)?"""
    out = {}
    for t in ("splicing", "copy-move"):
        d = tp[(tp.forgery_type_name == t) & (tp.host_id != tp.donor_id)]
        hs, ds = d.host_similarity, d.donor_similarity
        first = (hs > 0.5) & ~(ds > hs)
        second = d.donor_dominant
        neither = d[~first & ~second]
        out[f"{t}_name_with_distinct_ids"] = {
            "n": len(d), "first_id_dominant": int(first.sum()), "second_id_dominant": int(second.sum()),
            "neither": len(neither), "neither_host_size_differs": int(neither.host_similarity.isna().sum()),
            "neither_no_image_matches_half": int(neither.host_similarity.notna().sum())}
    dom = tp[tp.donor_dominant]
    out["donor_dominant_total"] = len(dom)
    out["donor_dominant_large_paste"] = int((dom.mask_reference != "donor").sum())
    out["donor_dominant_ids_swapped"] = int((dom.mask_reference == "donor").sum())
    out["mask_reference_counts"] = tp.mask_reference.value_counts().to_dict()
    out["donor_dominant_median_tampered_frac"] = round(float(tp.loc[tp.donor_dominant, "tampered_frac"].median()), 3)
    return out


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


def fig_class_format(df):
    ct = pd.crosstab(df.forgery_type, df.ext).reindex(TYPE_ORDER).fillna(0)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bottom = np.zeros(len(ct))
    colors = {"jpg": "#4C78A8", "tif": "#F58518", "bmp": "#54A24B"}
    for ext in ["jpg", "tif", "bmp"]:
        if ext in ct:
            ax.bar([TYPE_LABEL[t] for t in ct.index], ct[ext], bottom=bottom, label=ext.upper(),
                   color=colors[ext])
            bottom += ct[ext].values
    for i, total in enumerate(bottom):
        ax.text(i, total + 80, f"{int(total):,}", ha="center", fontsize=9)
    ax.set_ylabel("images")
    ax.set_title("CASIA v2.0: class and file format")
    ax.legend(title="format", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_class_format.png", dpi=160)
    plt.close(fig)


def fig_jpeg_quality(df):
    j = df[df.format == "JPEG"]
    bins = np.arange(j.jpeg_quality.min() - 0.5, 101.5, 1)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for lab, name, c in ((0, "Authentic", "#4C78A8"), (1, "Tampered (JPEG only)", "#E45756")):
        q = j[j.label == lab].jpeg_quality
        ax.hist(q, bins=bins, alpha=0.6, label=f"{name} (n={len(q):,})", color=c, density=True)
    ax.set_xlabel("closest IJG quality factor")
    ax.set_xlim(59.5, 100.5)
    n_low = int((j.jpeg_quality < 60).sum())
    if n_low:
        ax.text(60, ax.get_ylim()[1] * 0.95, f"{n_low} images below 60 not shown", fontsize=7, color="grey")
    ax.set_ylabel("share of images")
    ax.set_title("Compression settings differ between classes")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_jpeg_quality.png", dpi=160)
    plt.close(fig)


def fig_area(df):
    tp = df[df.mask_valid]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    bins = np.logspace(-4, 0, 41)
    for t, c in (("copy-move", "#72B7B2"), ("splicing", "#B279A2")):
        ax.hist(tp[tp.forgery_type == t].tampered_frac.clip(1e-4, 1), bins=bins, alpha=0.65,
                label=TYPE_LABEL[t], color=c)
    for x in (0.01, 0.05):
        ax.axvline(x, color="grey", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("tampered area (fraction of image, log scale)")
    ax.set_ylabel("images")
    ax.set_title("Tampered regions are small (dashed: 1% and 5% bucket edges)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_tampered_area.png", dpi=160)
    plt.close(fig)


def fig_samples(df, seed):
    rng = np.random.default_rng(seed)
    picks = []
    for t in ("copy-move", "splicing"):
        pool = df[(df.forgery_type == t) & df.mask_valid & (df.tampered_frac > 0.02)]
        picks += [pool.iloc[i] for i in rng.choice(len(pool), 3, replace=False)]
    fig, axes = plt.subplots(2, 6, figsize=(15, 4.6))
    for col, row in enumerate(picks):
        img = load_rgb(resolve(row.path))
        mask = load_mask(resolve(row.mask_path))
        overlay = img.copy()
        overlay[mask] = (0.45 * overlay[mask] + 0.55 * np.array([255, 0, 0])).astype(np.uint8)
        axes[0, col].imshow(img)
        axes[0, col].set_title(f"{TYPE_LABEL[row.forgery_type]} · {row.ext.upper()}", fontsize=9)
        axes[1, col].imshow(overlay)
        axes[1, col].set_title(f"mask: {100 * row.tampered_frac:.1f}% of pixels", fontsize=9)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("Tampered samples (top) and ground-truth masks overlaid in red (bottom)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_samples.png", dpi=130)
    plt.close(fig)


def write_markdown(s: dict, elapsed: float):
    fb = s["format_by_class"]
    fmt = lambda t, e: int(fb.get(t, {}).get(e, 0))
    lines = [
        "# Phase 1: dataset statistics", "",
        f"Generated by `scripts/build_manifest.py` in {elapsed:.0f}s.", "",
        "| Class | Images | JPEG | TIFF | BMP |", "|---|---:|---:|---:|---:|",
    ]
    for t in TYPE_ORDER:
        n = s["by_type"].get(t, 0)
        lines.append(f"| {TYPE_LABEL[t]} | {n:,} | {fmt(t, 'jpg'):,} | {fmt(t, 'tif'):,} | {fmt(t, 'bmp'):,} |")
    lines += [
        f"| **Total** | **{s['n_images']:,}** | | | |", "",
        "## Masks", "",
        f"- Status of tampered-image masks: {s['mask_status']}",
        f"- Masks valid for localisation scoring: {s['n_mask_valid']:,} / {s['n_tampered']:,}",
        f"- Tampered-area percentiles (fraction of image): {s['tampered_frac_percentiles']}",
        f"- Area buckets: {s['area_bucket']}",
        f"- Median connected components per mask: {s['mask_components_median']:.0f}",
        f"- Mask-name vs image-name: {s['mask_name_relation']} "
        f"({s['n_mask_names_with_placeholder_id']} mask names contain the placeholder id xxx00000); "
        "all matched on the unique trailing tamper id",
        f"- Masks with size problems: " + "; ".join(
            f"{m['image']} image {m['image_wh'][0]}x{m['image_wh'][1]} vs mask {m['mask_wh'][0]}x{m['mask_wh'][1]}"
            f" ({m['status']})" for m in s["mask_size_problems"]),
        f"- Background-difference check ({s['n_host_diff_checked']:,} masks; reference = the named image that "
        "matches > 50% of the unmasked area, normally the host): "
        f"median recall {s['host_diff_recall_median']}, median precision {s['host_diff_precision_median']}; "
        f"{s['n_low_mask_agreement']} masks with recall < 0.2 (flagged, not removed)",
        "", "## Compression and size", "",
        f"- Closest IJG quality factor, top values: {s['jpeg_quality_by_class']}",
        f"- Share of JPEGs using a standard IJG (libjpeg) quantisation table: "
        f"{s['standard_ijg_table_share_of_jpegs']}",
        f"- Most common sizes: {s['top_sizes_by_class']}",
        f"- Share of images that are 384x256 or 256x384: {s['share_384x256_or_256x384']}",
        "", "## Grouping and label hygiene", "",
        f"- Groups (host image id): {s['n_groups']:,}; largest group: {s['max_group_size']} images",
        f"- Forgery type from the image-name letter: {s['by_type_from_name_letter']}; the evidence votes "
        f"(image letter, mask letter, host==donor) disagree for {s['n_type_votes_disagree']} images; "
        f"majority vote changes {s['n_type_changed_by_majority_vote']}: {s['type_change_detail']}",
        f"- Which named id supplies most pixels (tampered names with two different ids): {s['background_census']}",
        f"- Images with an exact pixel duplicate: {s['n_exact_duplicates']}; "
        f"duplicates spanning both labels: {s['n_cross_label_duplicates']}",
    ]
    (OUT / "dataset_stats.md").write_text("\n".join(lines) + "\n")


def main():
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = build_manifest(cfg)
    out = resolve(cfg["paths"]["manifest"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    elapsed = time.time() - t0

    s = summarize(df)
    (OUT / "dataset_stats.json").write_text(json.dumps(_jsonable(s), indent=2))
    write_markdown(s, elapsed)
    fig_class_format(df)
    fig_jpeg_quality(df)
    fig_area(df)
    fig_samples(df, cfg["seed"])
    print(f"manifest: {len(df):,} rows -> {out}  ({elapsed:.0f}s)")
    print((OUT / "dataset_stats.md").read_text())


if __name__ == "__main__":
    main()
