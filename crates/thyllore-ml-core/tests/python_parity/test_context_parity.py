import numpy as np

import thyllore_ml_core as tml

from conftest import array_from_bits_1d, array_from_bits_2d, f32_from_bits


def test_flatten_context_matches_rust(context_fixture):
    keyframes = array_from_bits_2d(context_fixture["keyframes"])
    max_keyframes = context_fixture["max_keyframes"]
    clip_duration = f32_from_bits(context_fixture["clip_duration"])

    py_flat, py_mean, py_std = tml.flatten_context(
        keyframes, max_keyframes, clip_duration
    )

    rust_flat = array_from_bits_1d(context_fixture["flat"])
    rust_mean = f32_from_bits(context_fixture["mean"])
    rust_std = f32_from_bits(context_fixture["std"])

    assert py_flat.shape == rust_flat.shape
    assert np.array_equal(
        py_flat.view(np.uint32),
        rust_flat.view(np.uint32),
    ), "Python and Rust flatten_context outputs differ at bit level"

    assert np.float32(py_mean).tobytes() == np.float32(rust_mean).tobytes()
    assert np.float32(py_std).tobytes() == np.float32(rust_std).tobytes()
