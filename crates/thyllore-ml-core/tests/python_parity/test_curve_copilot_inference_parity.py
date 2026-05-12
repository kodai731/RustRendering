"""Verifies that Rust's curve_copilot inference produces the same outputs
as raw onnxruntime in Python, given identical inputs.

If outputs differ: Rust input encoding bug (tensor shape, dtype, name, ordering).
If outputs match: inaccurate predictions are a model quality issue, not a Rust bug.

Run from Rust first to generate the fixture:
    THYLLORE_SHARED_DATA_DIR=/path/to/SharedData \\
      cargo test -p thyllore-ml-core --test curve_copilot_inference_fixture \\
      -- --ignored --nocapture

Then run the Python comparison:
    THYLLORE_SHARED_DATA_DIR=/path/to/SharedData \\
      pytest crates/thyllore-ml-core/tests/python_parity/test_curve_copilot_inference_parity.py
"""

import json
import os

import numpy as np
import pytest

from conftest import (
    FIXTURE_DIR,
    array_from_bits_1d,
    array_from_bits_2d,
    f32_from_bits,
)

FIXTURE_PATH = os.path.join(FIXTURE_DIR, "curve_copilot_inference_fixture.json")


def _load_fixture():
    with open(FIXTURE_PATH) as f:
        return json.load(f)


def _resolve_onnx_path(onnx_filename: str) -> str:
    shared = os.environ.get("THYLLORE_SHARED_DATA_DIR")
    if not shared:
        pytest.skip("THYLLORE_SHARED_DATA_DIR not set")
    return os.path.join(shared, "exports", onnx_filename)


@pytest.mark.skipif(
    not os.path.exists(FIXTURE_PATH),
    reason=(
        "fixture missing — run "
        "`cargo test -p thyllore-ml-core --test curve_copilot_inference_fixture "
        "-- --ignored` first"
    ),
)
def test_curve_copilot_inference_matches_raw_onnxruntime():
    onnxruntime = pytest.importorskip("onnxruntime")

    fixture = _load_fixture()
    onnx_path = _resolve_onnx_path(fixture["onnx_filename"])
    if not os.path.exists(onnx_path):
        pytest.skip(f"ONNX model not found at {onnx_path}")

    context_flat = array_from_bits_1d(fixture["context_flat"])
    context = context_flat.reshape(1, 8, 6)

    property_type = np.array(
        [int(fixture["property_type_id"])], dtype=np.int64
    )
    topology = array_from_bits_1d(fixture["topology_features"]).reshape(1, -1)
    tokens = np.array(fixture["bone_name_tokens"], dtype=np.int64).reshape(1, -1)
    query_times = array_from_bits_1d(fixture["query_times"]).reshape(1, -1)
    curve_window = array_from_bits_1d(fixture["curve_window"]).reshape(1, -1)

    session = onnxruntime.InferenceSession(
        onnx_path, providers=["CPUExecutionProvider"]
    )

    outputs = session.run(
        None,
        {
            "context_keyframes": context,
            "property_type": property_type,
            "topology_features": topology,
            "bone_name_tokens": tokens,
            "query_times": query_times,
            "curve_window": curve_window,
        },
    )

    py_predictions = np.asarray(outputs[0], dtype=np.float32).reshape(-1, 5)
    py_confidences_raw = np.asarray(outputs[1], dtype=np.float32).reshape(-1)
    py_confidences = np.clip(py_confidences_raw, 0.0, 1.0)

    rust_predictions = array_from_bits_2d(fixture["rust_predictions"])
    rust_confidences = array_from_bits_1d(fixture["rust_confidences"])

    num_steps = rust_predictions.shape[0]
    py_predictions = py_predictions[:num_steps]
    py_confidences = py_confidences[:num_steps]

    assert (
        py_predictions.shape == rust_predictions.shape
    ), f"shape mismatch: py {py_predictions.shape} vs rust {rust_predictions.shape}"

    np.testing.assert_allclose(
        py_predictions,
        rust_predictions,
        atol=1e-5,
        rtol=1e-4,
        err_msg=(
            "Predictions differ between Rust path and raw onnxruntime — "
            "indicates Rust input encoding bug (check tensor names / shapes / dtypes "
            "in build_curve_copilot_tensors)"
        ),
    )

    np.testing.assert_allclose(
        py_confidences,
        rust_confidences,
        atol=1e-5,
        rtol=1e-4,
        err_msg="Confidences differ between Rust and Python paths",
    )

    print("\nParity OK:")
    print(f"  num_steps    = {num_steps}")
    print(f"  ONNX file    = {fixture['onnx_filename']}")
    print(f"  context_mean = {f32_from_bits(fixture['context_mean']):.4f}")
    print(f"  context_std  = {f32_from_bits(fixture['context_std']):.4f}")
    print("  step  rust_value     conf    py_value")
    for i in range(num_steps):
        print(
            f"  {i + 1:2d}    "
            f"{rust_predictions[i, 0]:+.4f}      "
            f"{rust_confidences[i]:.3f}   "
            f"{py_predictions[i, 0]:+.4f}"
        )
