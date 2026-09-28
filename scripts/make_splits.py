"""Phase 2a: leakage-safe train/val/test split + grouped CV folds, and freeze the test set.

Usage:  python scripts/make_splits.py            (refuses to change an already-frozen split)
        python scripts/make_splits.py --refreeze (deliberately re-create the frozen split)
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from forgery.config import load_config, resolve
from forgery.data.manifest import read_manifest
from forgery.data.splits import donor_merge_component, frozen_test_hash, full_split_hash, make_splits, stratum

OUT = resolve("results/phase2")


def composition(df: pd.DataFrame) -> pd.DataFrame:
    ct = pd.crosstab(stratum(df).rename("stratum"), df["split"]).reindex(columns=["train", "val", "test"])
    ct.loc["TOTAL"] = ct.sum()
    pct = ct.div(ct.sum(axis=1), axis=0).mul(100).round(1)
    return ct.astype(str) + " (" + pct.astype(str) + "%)"


def check_integrity(df: pd.DataFrame) -> None:
    assert df.path.is_unique
    assert (df.groupby("split_group")["split"].nunique() == 1).all(), "a split group spans several splits"
    dev = df[df.split != "test"]
    assert (dev.cv_fold >= 0).all() and (df.loc[df.split == "test", "cv_fold"] == -1).all()
    assert (dev.groupby("split_group")["cv_fold"].nunique() == 1).all(), "a split group spans several CV folds"
    assert (df.groupby("group_id")["split"].nunique() == 1).all(), "a host and its tampered images are split"
    assert (df.groupby("pixel_md5")["split"].nunique() == 1).all(), "exact duplicates are split"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refreeze", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    manifest = read_manifest(cfg)
    splits, info = make_splits(manifest, cfg)
    df = manifest.merge(splits, on="path", validate="one_to_one")
    check_integrity(df)                      # nothing is written unless the split is sound

    test_digest, full_digest = frozen_test_hash(splits), full_split_hash(splits)
    hash_file = resolve(cfg["paths"]["test_hash"])
    if hash_file.exists() and not args.refreeze:
        frozen = {tok[1]: tok[0] for tok in (line.split() for line in hash_file.read_text().splitlines())
                  if len(tok) >= 2}
        missing = [k for k in ("test", "full") if k not in frozen]
        if missing:
            raise SystemExit(f"{hash_file.name} lacks the {missing} hash(es) (old format?); "
                             "re-run with --refreeze to rewrite it deliberately")
        if frozen["test"] != test_digest or frozen["full"] != full_digest:
            raise SystemExit("split would change relative to the frozen hashes in "
                             f"{hash_file.name}; re-run with --refreeze only if that is intended")
    # write the CSV atomically first, the hashes last, so a failure never leaves them inconsistent
    out_csv = resolve(cfg["paths"]["splits"])
    tmp = out_csv.with_suffix(".csv.tmp")
    splits.to_csv(tmp, index=False)
    tmp.replace(out_csv)
    hash_file.write_text(f"{test_digest}  test\n{full_digest}  full\n")

    # Donor content is not covered by host grouping (except donor-dominant images); measure it.
    split_of = dict(zip(df.image_id, df.split))
    tp = df[(df.label == 1) & (df.host_id != df.donor_id)]
    donor_split = tp.donor_id.map(split_of)
    cross = tp[donor_split.notna() & (donor_split != tp.split)]
    merged_max = donor_merge_component(manifest, splits["split_group"])

    share = pd.crosstab(stratum(df), df["split"], normalize="index").mul(100)
    target = {k: 100 * cfg["splits"][k] for k in ("train", "val", "test")}
    dev_pp = pd.DataFrame({k: (share[k] - v).abs() for k, v in target.items()}).max(axis=1)
    small = dev_pp.index.str.endswith("_bmp")
    comp = composition(df)
    dev = df[df.split != "test"]
    cv = pd.crosstab(stratum(dev).rename("stratum"), dev["cv_fold"])
    summary = {**info, "test_sha256": test_digest, "full_split_sha256": full_digest,
               "n": df.split.value_counts().to_dict(),
               "n_cv_fold": dev.cv_fold.value_counts().sort_index().to_dict(),
               "tampered_with_donor_in_other_split": len(cross),
               "tampered_with_donor_in_other_split_by_type": cross.forgery_type.value_counts().to_dict(),
               "donor_cross_tampered_frac_median": round(float(cross.tampered_frac.median()), 4),
               "donor_cross_tampered_frac_p90": round(float(cross.tampered_frac.quantile(0.9)), 4),
               "largest_group_if_donors_were_merged": merged_max,
               "largest_group_if_donors_were_merged_share": round(merged_max / len(df), 3),
               "max_stratum_deviation_pp": round(float(dev_pp[~small].max()), 2),
               "max_stratum_deviation_pp_bmp": round(float(dev_pp[small].max()), 2)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "split_summary.json").write_text(json.dumps(summary, indent=2, default=int))
    md = ["# Phase 2: splits", "",
          f"- Split groups: {info['n_split_groups']:,} from {info['n_host_groups']:,} host ids "
          f"({info['merges_exact_duplicate']} merges from exact duplicates, {info['merges_near_duplicate']} from "
          f"{info['near_duplicate_pairs']} verified near-duplicate pairs, {info['merges_donor_dominant']} from "
          f"donor-dominant images); largest group {info['largest_split_group']} images",
          "- Every split group, host group and exact-duplicate set lies in one split and one CV fold (asserted "
          "before anything is written).",
          f"- Largest deviation from the 70/15/15 target: {summary['max_stratum_deviation_pp']} percentage points "
          f"(54-image BMP stratum: {summary['max_stratum_deviation_pp_bmp']}).",
          f"- Tampered images whose donor sits in a different split: {len(cross)} "
          f"({summary['tampered_with_donor_in_other_split_by_type']}); their pasted region covers a median "
          f"{100 * summary['donor_cross_tampered_frac_median']:.1f}% of the image (90th percentile "
          f"{100 * summary['donor_cross_tampered_frac_p90']:.1f}%). Reported, not prevented: merging every donor "
          f"would create one group of {merged_max:,} images ({100 * summary['largest_group_if_donors_were_merged_share']:.1f}%).",
          f"- Frozen hashes: test `{test_digest}`, full split `{full_digest}`", "",
          "## Composition (forgery type x format)", "", comp.to_markdown(), "",
          "## Dev-set CV folds (train + val)", "", cv.to_markdown(), ""]
    (OUT / "split_summary.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
