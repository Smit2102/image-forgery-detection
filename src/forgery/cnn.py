"""Phase 8 deep-learning baseline: ResNet-18 (ImageNet weights) fine-tuned on ELA images.

The network never sees a resized image: it trains on random CROP x CROP crops of the native-resolution ELA
image and scores a whole image by covering it with overlapping tiles (stride STRIDE) and pooling the tile
logits (mean or max; the pooling is chosen on validation). Images smaller than a tile are zero-padded,
which in ELA space means "no compression error".
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

CROP, STRIDE = 192, 96
ELA_QUALITY, ELA_SCALE = 90, 10
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255
AGGREGATORS = ("mean", "max")


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def ensure_certificates():
    """python.org builds ship without a CA bundle; torchvision's weight download needs one."""
    if "SSL_CERT_FILE" not in os.environ:
        import certifi
        os.environ["SSL_CERT_FILE"] = certifi.where()


def build_model(pretrained: bool = True) -> nn.Module:
    from torchvision.models import ResNet18_Weights, resnet18
    if pretrained:
        ensure_certificates()
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m


def ela_image(x: np.ndarray) -> np.ndarray:
    """ELA = |x - JPEG_q90(x)| per RGB channel, x10, clipped to uint8 (x: the image after its protocol)."""
    from .io import jpeg_roundtrip
    diff = np.abs(x.astype(np.int16) - jpeg_roundtrip(x, ELA_QUALITY).astype(np.int16))
    return np.clip(diff * ELA_SCALE, 0, 255).astype(np.uint8)


def load_ela(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def pad_to_crop(x: np.ndarray) -> np.ndarray:
    h, w = x.shape[:2]
    if h >= CROP and w >= CROP:
        return x
    return np.pad(x, ((0, max(0, CROP - h)), (0, max(0, CROP - w)), (0, 0)))


def random_crop(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = pad_to_crop(x)
    y0 = rng.integers(0, x.shape[0] - CROP + 1)
    x0 = rng.integers(0, x.shape[1] - CROP + 1)
    c = x[y0:y0 + CROP, x0:x0 + CROP]
    return c[:, ::-1] if rng.random() < 0.5 else c


def _starts(n: int, size: int = CROP) -> list[int]:
    s = list(range(0, n - size + 1, STRIDE))
    return s if s[-1] == n - size else s + [n - size]


def tiles(x: np.ndarray, pad: bool = True) -> np.ndarray:
    """Overlapping CROP x CROP tiles covering the whole (padded) image, including its right/bottom edge.

    pad=False (analysis variant only): a side shorter than CROP is not zero-padded; the tiles then span
    that side in full (the network's global average pooling accepts any size >= 32).
    """
    if pad:
        x = pad_to_crop(x)
    th, tw = min(CROP, x.shape[0]), min(CROP, x.shape[1])
    return np.stack([x[y:y + th, c:c + tw] for y in _starts(x.shape[0], th) for c in _starts(x.shape[1], tw)])


def n_tiles(shape: tuple[int, ...]) -> int:
    return len(_starts(max(shape[0], CROP))) * len(_starts(max(shape[1], CROP)))


def is_padded(shape: tuple[int, ...]) -> bool:
    return shape[0] < CROP or shape[1] < CROP


def to_input(batch: np.ndarray, dev: torch.device) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(batch)).to(dev).permute(0, 3, 1, 2).float()
    return (t - _MEAN.to(dev)) / _STD.to(dev)


@torch.no_grad()
def predict(model: nn.Module, images: list[np.ndarray], dev: torch.device, batch: int = 256,
            pad: bool = True) -> dict[str, np.ndarray]:
    """Image scores (pooled tile logits) for every aggregator (pad: see tiles())."""
    model.eval()
    owner, logits, buf, buf_owner = [], [], [], []

    def flush():
        logits.append(model(to_input(np.stack(buf), dev)).squeeze(1).float().cpu().numpy())
        owner.extend(buf_owner)
        buf.clear()
        buf_owner.clear()

    for i, x in enumerate(images):
        for t in tiles(x, pad):
            if buf and t.shape != buf[0].shape:              # unpadded tiles differ in size: new batch
                flush()
            buf.append(t)
            buf_owner.append(i)
            if len(buf) == batch:
                flush()
    if buf:
        flush()
    z, owner = np.concatenate(logits), np.asarray(owner)
    counts = np.bincount(owner, minlength=len(images))
    out = {"mean": np.bincount(owner, weights=z, minlength=len(images)) / counts,
           "max": np.full(len(images), -np.inf)}
    np.maximum.at(out["max"], owner, z)
    return out


def train_epoch(model: nn.Module, opt, sched, images: list[np.ndarray], labels: np.ndarray,
                rng: np.random.Generator, dev: torch.device, batch: int, pos_weight: float) -> float:
    """One pass over the training images (one random crop each); returns the mean loss."""
    model.train()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=dev))
    order = rng.permutation(len(images))
    total = 0.0
    for k in range(0, len(order), batch):
        idx = order[k:k + batch]
        x = to_input(np.stack([random_crop(images[i], rng) for i in idx]), dev)
        y = torch.from_numpy(labels[idx].astype(np.float32)).to(dev)
        loss = loss_fn(model(x).squeeze(1), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        total += float(loss.detach()) * len(idx)
    return total / len(order)


def code_sha1() -> str:
    return hashlib.sha1(Path(__file__).read_bytes()).hexdigest()
