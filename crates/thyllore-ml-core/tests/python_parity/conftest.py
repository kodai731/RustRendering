import json
import os
import struct

import numpy as np
import pytest

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def f32_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def array_from_bits_2d(bits_2d) -> np.ndarray:
    return np.array(
        [[f32_from_bits(b) for b in row] for row in bits_2d],
        dtype=np.float32,
    )


def array_from_bits_1d(bits_1d) -> np.ndarray:
    return np.array([f32_from_bits(b) for b in bits_1d], dtype=np.float32)


def load_fixture(name: str) -> dict:
    with open(os.path.join(FIXTURE_DIR, name)) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def skeleton_fixture():
    return load_fixture("skeleton_fixture.json")


@pytest.fixture(scope="session")
def context_fixture():
    return load_fixture("context_fixture.json")


@pytest.fixture(scope="session")
def window_fixture():
    return load_fixture("window_fixture.json")


@pytest.fixture(scope="session")
def query_fixture():
    return load_fixture("query_fixture.json")


@pytest.fixture(scope="session")
def sampling_fixture():
    return load_fixture("sampling_fixture.json")
