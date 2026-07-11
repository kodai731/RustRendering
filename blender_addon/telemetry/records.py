"""Two-stage collection of Curve Copilot correction pairs (mode B only).

Stage 1 (prediction time): ``record_prediction`` buffers the wheel input, the
predicted ghost and where the ground truth must be sampled later. Stage 2
(save time): ``complete_and_flush`` samples the now-edited FCurves at the same
future frames, anonymizes the pair and hands the batch to ``sender`` on a
background thread. Failed sends are kept in a local outbox and retried on the
next flush.

Record construction (schema ``curve_copilot_feedback/v0``) is delegated to the
``thyllore_ml_core`` wheel, the schema's single source of truth; this module
keeps only the bpy-dependent parts (FCurve sampling, outbox storage).

Anonymization: records never contain object names, file paths or bone names;
curve values are stored relative to ``origin_value``.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import bpy

from . import sender

PENDING_LIMIT = 256

_pending: list[dict] = []
_flush_lock = threading.Lock()


def record_prediction(
    *,
    object_name: str,
    data_path: str,
    array_index: int,
    channel_kind: str,
    scene_fps: float,
    deploy_fps: float,
    frame_step: float,
    origin_value: float,
    context: list[float],
    prediction_frames: list[float],
    prediction_values: list[float],
    model_hash: str,
) -> None:
    if len(_pending) >= PENDING_LIMIT:
        del _pending[0]
    _pending.append(
        {
            "object_name": object_name,
            "data_path": data_path,
            "array_index": array_index,
            "channel_kind": channel_kind,
            "scene_fps": scene_fps,
            "deploy_fps": deploy_fps,
            "frame_step": frame_step,
            "origin_value": origin_value,
            "context": context,
            "prediction_frames": prediction_frames,
            "prediction_values": prediction_values,
            "model_hash": model_hash,
            "signal": "ignored",
        }
    )


def mark_all_cleared() -> None:
    for entry in _pending:
        entry["signal"] = "cleared"


def pending_count() -> int:
    return len(_pending)


def _sample_ground_truth(entry: dict) -> list[float] | None:
    from ..operators.curve_copilot import _action_fcurves

    obj = bpy.data.objects.get(entry["object_name"])
    if obj is None:
        return None
    for fcurve in _action_fcurves(obj):
        if (
            fcurve.data_path == entry["data_path"]
            and fcurve.array_index == entry["array_index"]
        ):
            origin_value = entry["origin_value"]
            return [
                float(fcurve.evaluate(frame)) - origin_value
                for frame in entry["prediction_frames"]
            ]
    return None


def _finalize(entry: dict, ground_truth: list[float] | None) -> dict:
    import thyllore_ml_core as tml

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
    )


def _outbox_path() -> Path:
    config_dir = Path(bpy.utils.user_resource("CONFIG", create=True))
    return config_dir / "thyllore_curve_copilot_outbox.jsonl"


def _load_outbox() -> list[dict]:
    try:
        lines = _outbox_path().read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]
    except (OSError, ValueError):
        return []


def _store_outbox(records: list[dict]) -> None:
    try:
        _outbox_path().write_text(
            "\n".join(json.dumps(record) for record in records), encoding="utf-8"
        )
    except OSError:
        pass


def _clear_outbox() -> None:
    try:
        _outbox_path().unlink(missing_ok=True)
    except OSError:
        pass


def _send_outbox(records: list[dict]) -> None:
    with _flush_lock:
        if sender.send_feedback_batch(records):
            _clear_outbox()
        else:
            _store_outbox(records)


def complete_and_flush(prefs) -> None:
    """Complete pending pairs against the current FCurves and send them.

    Must run on the main thread (reads bpy data); only the network send moves
    to a background thread. When sending is not allowed the completed records
    stay in the outbox.
    """
    global _pending
    completed = [_finalize(entry, _sample_ground_truth(entry)) for entry in _pending]
    _pending = []

    records = _load_outbox() + completed
    if not records:
        return
    _store_outbox(records)
    if not sender.should_send(prefs):
        return
    threading.Thread(target=_send_outbox, args=(records,), daemon=True).start()


def request_token_refresh() -> None:
    """Empty-batch handshake so ctx64 unlocks right after opt-in."""
    threading.Thread(
        target=sender.send_feedback_batch, args=([],), daemon=True
    ).start()
