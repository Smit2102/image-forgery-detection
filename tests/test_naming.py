import pytest

from forgery.data.naming import mask_tamper_id, normalize_id, parse_authentic, parse_tampered


def test_normalize_pads_dropped_digit():
    assert normalize_id("pla0006") == "pla00006"
    assert normalize_id("ani00018") == "ani00018"
    with pytest.raises(ValueError):
        normalize_id("xx123")


def test_parse_authentic():
    info = parse_authentic("Au_ani_00018.jpg")
    assert info == {"stem": "Au_ani_00018", "category": "ani", "image_id": "ani00018"}


def test_parse_tampered_types_and_ids():
    s = parse_tampered("Tp_S_NRN_S_N_arc00013_arc00013_00243.tif")
    d = parse_tampered("Tp_D_CND_M_N_ani00018_sec00096_00138.tif")
    assert s["forgery_type"] == "copy-move" and d["forgery_type"] == "splicing"
    assert (d["host_id"], d["donor_id"], d["tamper_id"]) == ("ani00018", "sec00096", "00138")
    assert parse_tampered("Tp_S_CNN_S_N_cha0003_cha00003_00323.tif")["host_id"] == "cha00003"
    with pytest.raises(ValueError):
        parse_tampered("Au_ani_00018.jpg")


@pytest.mark.parametrize("name,tid", [
    ("Tp_D_CND_M_N_ani00018_sec00096_00138_gt.png", "00138"),
    ("Tp_D_CNN_M_N_arc00086_xxx00000_00306_gt.png", "00306"),   # placeholder id
    ("Tp_D_NRD_S_N_cha10002_cha10001_20094_gt3.png", "20094"),  # odd suffix
])
def test_mask_tamper_id(name, tid):
    assert mask_tamper_id(name) == tid


def test_mask_tamper_id_rejects_non_mask():
    with pytest.raises(ValueError):
        mask_tamper_id("Tp_D_CND_M_N_ani00018_sec00096_00138.png")


@pytest.mark.parametrize("mask,rel", [
    ("Tp_D_CND_M_N_ani00018_sec00096_00138_gt.png", "identical"),
    ("Tp_S_CND_M_N_ani00018_sec00096_00138_gt.png", "letter_only"),
    ("Tp_D_CND_M_N_ani00018_xxx00000_00138_gt.png", "ids"),
    ("Tp_D_CND_M_N_ani00018_sec00096_00138_gt3.png", "odd_suffix"),
])
def test_mask_name_relation(mask, rel):
    from forgery.data.manifest import mask_name_relation
    assert mask_name_relation("Tp_D_CND_M_N_ani00018_sec00096_00138", mask) == rel
