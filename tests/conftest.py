import numpy as np
import pytest

from forgery.config import load_config, resolve
from forgery.data.manifest import read_manifest
from forgery.data.splits import read_splits


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture
def rgb():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(64, 96, 3), dtype=np.uint8)


def _need(path):
    if not path.exists():
        pytest.skip(f"{path} not generated yet")
    return path


@pytest.fixture(scope="session")
def manifest(cfg):
    _need(resolve(cfg["paths"]["manifest"]))
    return read_manifest(cfg)


@pytest.fixture(scope="session")
def splits(cfg):
    _need(resolve(cfg["paths"]["splits"]))
    return read_splits(cfg)
