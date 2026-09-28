"""Phase 7 post-hoc analysis: reads the saved outputs of a test run only (never re-evaluates the test set).

  * SHA-1 manifest of every file in results/phase7/run_<n>/ (dated), for the audit trail;
  * paired localisation differences (deployed fusion vs mean of maps / best single module) with
    group-bootstrap CIs, from the saved per-image scores;
  * copy-move evidence behind the false alarms (saved test features);
  * Holm correction of the failure-enrichment Fisher tests.
Usage:  python scripts/phase7_posthoc.py [--run 1]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime

import numpy as np
import pandas as pd

from forgery.config import load_config, resolve


def paired_ci(a: np.ndarray, b: np.ndarray, groups: np.ndarray, seed: int, n: int = 1000):
    d = a - b
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    sums, counts = np.bincount(inv, weights=d), np.bincount(inv)
    draws = rng.integers(0, len(uniq), size=(n, len(uniq)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int, default=1)
    args = ap.parse_args()
    cfg = load_config()
    run_dir = resolve(f"results/phase7/run_{args.run}")
    out = {"created": datetime.now().isoformat(timespec="seconds"), "run": args.run}

    files = sorted(p for p in run_dir.iterdir() if p.is_file() and p.name not in ("MANIFEST.sha1", "posthoc.json"))
    manifest = [f"{hashlib.sha1(p.read_bytes()).hexdigest()}  {p.name}" for p in files]
    (run_dir / "MANIFEST.sha1").write_text(f"# SHA-1 of run_{args.run} outputs, recorded {out['created']}\n"
                                           + "\n".join(manifest) + "\n")

    out["localisation_paired"] = {}
    for P in ("R", "B"):
        per = pd.read_csv(run_dir / f"localisation_per_image_{P}.csv")
        g = per.split_group.to_numpy()
        single = [c for c in per.columns if c.startswith("single:") and c.endswith("_f1")][0]
        out["localisation_paired"][P] = {
            "fusion_minus_mean_of_maps": paired_ci(per.f1.to_numpy(), per["mean_f1"].to_numpy(), g, cfg["seed"]),
            f"fusion_minus_{single[:-3]}": paired_ci(per.f1.to_numpy(), per[single].to_numpy(), g, cfg["seed"])}

    outcomes = pd.read_csv(run_dir / "test_outcomes_R.csv", dtype={"path": str})
    feats = pd.read_csv(resolve("data/features/features_R_test.csv"), dtype={"path": str})
    o = outcomes.merge(feats[["path", "cm_keypoint.local_n_inliers", "cm_keypoint.global_n_keypoints"]], on="path")
    out["false_alarm_copymove_evidence"] = {
        k: {"median_inliers": float(o[o.outcome == k]["cm_keypoint.local_n_inliers"].median()),
            "median_keypoints": float(o[o.outcome == k]["cm_keypoint.global_n_keypoints"].median()),
            "share_with_inliers": float((o[o.outcome == k]["cm_keypoint.local_n_inliers"] > 0).mean()),
            "n": int((o.outcome == k).sum())} for k in ("FP", "TN")}

    enr = pd.read_csv(run_dir / "failure_enrichment_R.csv")
    enr = enr[enr.tag != "TIFF source"].copy()                  # same test as "JPEG source"
    enr["holm_p"] = holm(enr.fisher_p.to_numpy())
    enr.to_csv(run_dir / "failure_enrichment_R_holm.csv", index=False)
    out["enrichment_holm"] = enr[["comparison", "tag", "ratio", "fisher_p", "holm_p"]].round(4).to_dict(orient="records")
    (run_dir / "posthoc.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2)[:3000])


if __name__ == "__main__":
    main()
