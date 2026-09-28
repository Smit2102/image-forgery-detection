"""Parsing of CASIA v2.0 file names.

Authentic:  Au_<cat>_<nnnnn>.<ext>               e.g. Au_ani_00018.jpg  -> id "ani00018"
Tampered:   Tp_<S|D>_<f1>_<f2>_<f3>_<host>_<donor>_<tid>.<ext>
            S = copy-move (same image), D = splicing (different images).
            The first id is the host/background image (verified by pixel
            comparison against the authentic set), the second the donor.
            A few ids have a dropped digit ("pla0006") and are zero-padded here.
Masks:      <tampered stem>_gt.png, but 142 mask names differ from their image
            name (97 only in the S/D letter, 44 in the host/donor ids - 10 of
            them the placeholder "xxx00000" - and one is named "..._gt3.png");
            masks are therefore matched on the trailing tamper id, which is unique.
"""

from __future__ import annotations

import re
from pathlib import Path

_ID = re.compile(r"^([a-z]{3})(\d{1,5})$")
_MASK = re.compile(r"^(?P<body>.+?)_gt\d*$")


def normalize_id(token: str) -> str:
    m = _ID.match(token)
    if not m:
        raise ValueError(f"not a CASIA image id: {token!r}")
    return m.group(1) + m.group(2).zfill(5)


def parse_authentic(name: str) -> dict:
    stem = Path(name).stem
    parts = stem.split("_")
    if len(parts) != 3 or parts[0] != "Au":
        raise ValueError(f"not an authentic CASIA name: {name!r}")
    return {"stem": stem, "category": parts[1], "image_id": normalize_id(parts[1] + parts[2])}


def parse_tampered(name: str) -> dict:
    stem = Path(name).stem
    parts = stem.split("_")
    if len(parts) != 8 or parts[0] != "Tp" or parts[1] not in ("S", "D"):
        raise ValueError(f"not a tampered CASIA name: {name!r}")
    host, donor = normalize_id(parts[5]), normalize_id(parts[6])
    return {
        "stem": stem,
        "forgery_type": "copy-move" if parts[1] == "S" else "splicing",
        "flags": "_".join(parts[2:5]),
        "host_id": host,
        "donor_id": donor,
        "category": host[:3],
        "tamper_id": parts[7],
    }


def mask_tamper_id(mask_name: str) -> str:
    """Trailing tamper id of a mask file name (robust to placeholder ids / extra underscores)."""
    m = _MASK.match(Path(mask_name).stem)
    if not m:
        raise ValueError(f"not a CASIA mask name: {mask_name!r}")
    return [t for t in m.group("body").split("_") if t][-1]
