"""Wheel `build_feedback_record` must match the legacy Python `_finalize`.

The `curve_copilot_feedback/v1` record construction moved from
``telemetry/records.py`` into the ``thyllore_ml_core`` wheel (Rust SSoT).
This test freezes the legacy Python output and asserts the wheel produces an
identical record for the same inputs. The wheel is loaded from
``blender_addon/wheels/`` (run ``scripts/collect_wheels.sh`` first).
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pytest

WHEELS_DIR = Path(__file__).resolve().parents[1] / "wheels"


@pytest.fixture(scope="module")
def tml(tmp_path_factory):
    wheels = sorted(WHEELS_DIR.glob("thyllore_ml_core-*.whl"))
    if not wheels:
        pytest.skip("thyllore_ml_core wheel not built (run scripts/collect_wheels.sh)")

    extract_dir = tmp_path_factory.mktemp("tml_wheel")
    with zipfile.ZipFile(wheels[-1]) as archive:
        archive.extractall(extract_dir)

    sys.path.insert(0, str(extract_dir))
    try:
        import thyllore_ml_core

        yield thyllore_ml_core
    finally:
        sys.path.remove(str(extract_dir))


ENTRY = {
    "channel_kind": "location",
    "array_index": 1,
    "scene_fps": 24.0,
    "deploy_fps": 60.0,
    "frame_step": 0.4,
    "origin_value": 2.5,
    "context": [2.5, 3.25, 1.75],
    "prediction_values": [4.0, 2.0],
    "model_hash": "abc123",
    "signal": "ignored",
}
TS = 1752200000


def _quantize(value: float) -> float:
    return round(value * 1e4) / 1e4


def _legacy_finalize(entry: dict, ground_truth: list[float] | None, ts: int) -> dict:
    origin_value = entry["origin_value"]
    context = [value - origin_value for value in entry["context"]]
    scale = max((abs(value) for value in context), default=0.0)
    if scale < 1e-9:
        scale = 1.0
    return {
        "schema": "curve_copilot_feedback/v1",
        "model_hash": entry["model_hash"],
        "channel": {"kind": entry["channel_kind"], "array_index": entry["array_index"]},
        "fps": {
            "scene": entry["scene_fps"],
            "deploy": entry["deploy_fps"],
            "frame_step": entry["frame_step"],
        },
        "context": [_quantize(value / scale) for value in context],
        "prediction": [
            _quantize((value - origin_value) / scale) for value in entry["prediction_values"]
        ],
        "ground_truth": (
            None if ground_truth is None else [_quantize(value / scale) for value in ground_truth]
        ),
        "signal": entry["signal"],
        "record_id": entry.get("record_id", ""),
        "revision": entry.get("revision", 0),
        "ts": ts - ts % 86400,
    }


def _wheel_record(tml, entry: dict, ground_truth: list[float] | None, ts: int) -> dict:
    return tml.build_feedback_record(
        model_hash=entry["model_hash"],
        channel_kind=entry["channel_kind"],
        array_index=entry["array_index"],
        scene_fps=entry["scene_fps"],
        deploy_fps=entry["deploy_fps"],
        frame_step=entry["frame_step"],
        origin_value=entry["origin_value"],
        context=entry["context"],
        prediction=entry["prediction_values"],
        ground_truth=ground_truth,
        signal=entry["signal"],
        record_id=entry.get("record_id", ""),
        revision=entry.get("revision", 0),
        ts=ts,
    )


def test_record_with_ground_truth_matches_legacy(tml):
    ground_truth = [0.5, -0.25]
    assert _wheel_record(tml, ENTRY, ground_truth, TS) == _legacy_finalize(
        ENTRY, ground_truth, TS
    )


def test_record_without_ground_truth_matches_legacy(tml):
    assert _wheel_record(tml, ENTRY, None, TS) == _legacy_finalize(ENTRY, None, TS)


def test_record_cleared_signal_matches_legacy(tml):
    entry = {**ENTRY, "signal": "cleared"}
    assert _wheel_record(tml, entry, None, TS) == _legacy_finalize(entry, None, TS)


def test_record_id_and_revision_round_trip(tml):
    entry = {**ENTRY, "record_id": "11111111-2222-3333-4444-555555555555", "revision": 3}
    record = _wheel_record(tml, entry, None, TS)
    assert record == _legacy_finalize(entry, None, TS)
    assert record["record_id"] == entry["record_id"]
    assert record["revision"] == 3


def test_ts_defaults_to_now_at_day_granularity(tml):
    import time

    record = _wheel_record(tml, ENTRY, None, None)
    assert record["ts"] % 86400 == 0
    assert 0 <= int(time.time()) - record["ts"] < 86400 + 60
