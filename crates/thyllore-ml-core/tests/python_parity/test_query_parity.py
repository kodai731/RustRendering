import numpy as np

import thyllore_ml_core as tml

from conftest import array_from_bits_1d, f32_from_bits


def test_generate_query_times_with_future(query_fixture):
    times = array_from_bits_1d(query_fixture["times_with_future"])
    current_time = f32_from_bits(query_fixture["current_time_with_future"])
    clip_duration = f32_from_bits(query_fixture["clip_duration"])
    max_steps = query_fixture["max_steps"]

    py_qt = tml.generate_query_times(times, current_time, clip_duration, max_steps)
    rust_qt = array_from_bits_1d(query_fixture["query_times_with_future"])

    assert py_qt.shape == rust_qt.shape
    assert np.array_equal(py_qt.view(np.uint32), rust_qt.view(np.uint32))


def test_generate_query_times_no_future(query_fixture):
    times = array_from_bits_1d(query_fixture["times_no_future"])
    current_time = f32_from_bits(query_fixture["current_time_no_future"])
    clip_duration = f32_from_bits(query_fixture["clip_duration"])
    max_steps = query_fixture["max_steps"]

    py_qt = tml.generate_query_times(times, current_time, clip_duration, max_steps)
    rust_qt = array_from_bits_1d(query_fixture["query_times_no_future"])

    assert py_qt.shape == rust_qt.shape
    assert np.array_equal(py_qt.view(np.uint32), rust_qt.view(np.uint32))
