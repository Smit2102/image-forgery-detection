"""Invariants of the generated manifest and splits (skipped until the scripts have run)."""

import pandas as pd
import pytest

from forgery.config import resolve
from forgery.data.splits import frozen_test_hash, full_split_hash

pytestmark = pytest.mark.data


def test_manifest_counts(manifest):
    assert len(manifest) == 12614
    assert (manifest.label == 0).sum() == 7491
    assert manifest.forgery_type_name.value_counts().to_dict() == {"none": 7491, "copy-move": 3274,
                                                                   "splicing": 1849}
    assert set(manifest.forgery_type) == {"none", "copy-move", "splicing"}
    assert manifest.path.is_unique


def test_ids_keep_leading_zeros(manifest):
    tp = manifest[manifest.label == 1]
    assert tp.tamper_id.str.fullmatch(r"\d{5}").all()
    assert (tp.path.str.rsplit("_", n=1).str[-1].str.split(".").str[0] == tp.tamper_id).all()
    assert manifest.dhash.str.fullmatch(r"[0-9a-f]{16}").all()


def test_corrected_type_is_majority_of_evidence(manifest):
    tp = manifest[manifest.label == 1]
    votes = tp[["forgery_type_name", "forgery_type_mask", "forgery_type_ids"]]
    agree = (votes.eq(tp.forgery_type, axis=0)).sum(axis=1)
    assert (agree >= 2).all()


def test_every_tampered_image_has_a_mask(manifest):
    tp = manifest[manifest.label == 1]
    assert tp.mask_path.notna().all()
    assert tp.mask_path.is_unique
    assert set(tp.mask_status) <= {"ok", "transposed", "size_mismatch", "full", "empty"}
    assert (manifest.mask_valid == ((manifest.label == 1) & (manifest.mask_status == "ok"))).all()
    # the mask's trailing id matches the image's tamper id
    assert (tp.mask_path.str.extract(r"_(\d{5})_+gt\d*\.png$")[0] == tp.tamper_id).all()


def test_valid_masks_are_reasonable(manifest):
    v = manifest[manifest.mask_valid]
    assert v.tampered_frac.between(0, 0.95).all() and (v.tampered_frac > 0).all()
    assert v.area_bucket.notna().all()


def test_authentic_rows_have_no_mask(manifest):
    au = manifest[manifest.label == 0]
    assert au.mask_path.isna().all() and not au.mask_valid.any()


def test_splits_cover_manifest(manifest, splits):
    assert set(splits.path) == set(manifest.path) and splits.path.is_unique
    frac = splits.split.value_counts(normalize=True)
    for name, target in (("train", 0.70), ("val", 0.15), ("test", 0.15)):
        assert abs(frac[name] - target) < 0.01


def test_groups_do_not_cross_splits_or_folds(splits):
    assert (splits.groupby("split_group").split.nunique() == 1).all()
    dev = splits[splits.split != "test"]
    assert (dev.cv_fold.between(0, 4)).all()
    assert (dev.groupby("split_group").cv_fold.nunique() == 1).all()
    assert (splits.loc[splits.split == "test", "cv_fold"] == -1).all()


def test_host_duplicates_and_dominant_donors_share_a_split(manifest, splits):
    df = manifest.merge(splits, on="path")
    assert (df.groupby("group_id").split.nunique() == 1).all()
    assert (df.groupby("pixel_md5").split.nunique() == 1).all()
    split_of = dict(zip(df.image_id, df.split))
    dom = df[df.donor_dominant]
    assert len(dom) > 0 and (dom.donor_id.map(split_of) == dom.split).all()


def test_split_is_frozen(cfg, splits):
    frozen = {tok[1]: tok[0] for tok in (l.split() for l in resolve(cfg["paths"]["test_hash"]).read_text()
                                          .splitlines()) if len(tok) >= 2}
    assert frozen_test_hash(splits) == frozen["test"]
    assert full_split_hash(splits) == frozen["full"]


def test_stratification_balanced(manifest, splits):
    df = manifest.merge(splits, on="path")
    share = pd.crosstab(df.forgery_type + "_" + df.ext, df.split, normalize="index")
    assert (share["test"] - 0.15).abs().max() < 0.02


def test_detect_reproduces_training_features(cfg, manifest, splits):
    """detect() must compute exactly the features the Phase-5 models were trained on."""
    import importlib.util
    import json
    feats, meta = resolve("data/features/features_R.csv"), resolve("data/features/features_R.json")
    if not feats.exists() or not meta.exists():
        pytest.skip("features not extracted")
    spec = importlib.util.spec_from_file_location("extract_features", resolve("scripts/extract_features.py"))
    ef = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ef)
    stored = json.loads(meta.read_text())
    current = ef.fingerprint(cfg, "R", [])
    if stored["code_sha1"] != current["code_sha1"] or stored["features"] != current["features"]:
        pytest.skip("features_R.csv is stale (module code or parameters changed); re-run extract_features.py")
    from forgery.pipeline import detect
    table = pd.read_csv(feats, dtype={"path": str}).set_index("path")
    val = splits[splits.split == "val"].path.iloc[:2]
    for path in val:
        res = detect(resolve(path), protocol="R", cfg=cfg)
        row = table.loc[path]
        for k, v in res.features.items():
            assert v == pytest.approx(row[k], rel=1e-6, abs=1e-9), (path, k)
