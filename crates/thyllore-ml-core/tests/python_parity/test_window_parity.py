import numpy as np

import thyllore_ml_core as tml

from conftest import array_from_bits_1d, f32_from_bits


def test_sample_window_matches_rust(window_fixture):
    times = array_from_bits_1d(window_fixture["times"])
    values = array_from_bits_1d(window_fixture["values"])
    t_start = f32_from_bits(window_fixture["t_start"])
    t_end = f32_from_bits(window_fixture["t_end"])
    curve_mean = f32_from_bits(window_fixture["curve_mean"])
    curve_std = f32_from_bits(window_fixture["curve_std"])

    py_window = tml.sample_window(times, values, t_start, t_end, curve_mean, curve_std)
    rust_window = array_from_bits_1d(window_fixture["window"])

    assert py_window.shape == rust_window.shape
    assert np.array_equal(
        py_window.view(np.uint32),
        rust_window.view(np.uint32),
    ), "Python and Rust sample_window outputs differ at bit level"
