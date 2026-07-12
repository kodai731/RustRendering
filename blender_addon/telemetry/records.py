"""Two-stage collection of Curve Copilot correction pairs (mode B only).

Stage 1 (prediction time): ``record_prediction`` buffers the wheel input, the
predicted ghost and where the ground truth must be sampled later. Stage 2
(save time): ``complete_and_flush`` samples the now-edited FCurves at the same
future frames, anonymizes the pair and hands the batch to ``sender`` on a
background thread. Failed sends are kept in a local outbox and retried on the
next flush.

Record construction (schema ``curve_copilot_feedback/v1``) is delegated to the
``thyllore_ml_core`` wheel, the schema's single source of truth; this module
keeps only the bpy-dependent parts (FCurve sampling, outbox storage).

Anonymization: records never contain object names, file paths or bone names.
The wheel stores curve values origin-relative, amplitude-normalized (scale not
transmitted) and quantized, with day-granularity timestamps; batches are
shuffled before sending so records cannot be re-correlated into runs or bones
and the original animation cannot be reconstructed. ``record_id`` only links
the revisions of one fragment's ground truth, nothing across fragments.
"""
from __future__ import annotations

import json
import random
import threading
import uuid
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
    representation: str = "direct",
) -> None:
    channel_key = (object_name, data_path, array_index, representation)
    _pending[:] = [entry for entry in _pending if _channel_key(entry) != channel_key]
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
            "representation": representation,
            "record_id": str(uuid.uuid4()),
            "revision": 0,
        }
    )


def _channel_key(entry: dict) -> tuple:
    return (
        entry["object_name"],
        entry["data_path"],
        entry["array_index"],
        entry.get("representation", "direct"),
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
    if entry.get("representation") == "quaternion_euler":
        return _sample_quaternion_euler_ground_truth(obj, entry)
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


def _sample_quaternion_euler_ground_truth(obj, entry: dict) -> list[float] | None:
    """Re-samples the edited quaternion siblings as a continuous Euler curve.

    Mirrors the prediction-time conversion in ``_forecast_quaternion_ghost``:
    the context frames seed the Euler continuity, then the prediction frames
    are evaluated on the same continuous curve. ``array_index`` of a
    quaternion_euler entry is the Euler axis (0..2).
    """
    import thyllore_ml_core as tml
    from mathutils import Quaternion

    from ..operators.curve_copilot import _action_fcurves

    siblings = {}
    for fcurve in _action_fcurves(obj):
        if fcurve.data_path == entry["data_path"] and 0 <= fcurve.array_index <= 3:
            siblings[fcurve.array_index] = fcurve
    if len(siblings) != 4:
        return None

    frame_step = entry["frame_step"]
    prediction_frames = entry["prediction_frames"]
    context_offsets, future_offsets = tml.forecast_sample_offsets()
    origin_frame = prediction_frames[0] - future_offsets[0] * frame_step
    context_frames = [origin_frame + offset * frame_step for offset in context_offsets]

    axis = entry["array_index"]
    origin_value = entry["origin_value"]
    previous = None
    ground_truth = []
    for index, frame in enumerate(context_frames + list(prediction_frames)):
        quat = Quaternion(
            (
                siblings[0].evaluate(frame),
                siblings[1].evaluate(frame),
                siblings[2].evaluate(frame),
                siblings[3].evaluate(frame),
            )
        )
        euler = quat.to_euler("XYZ") if previous is None else quat.to_euler("XYZ", previous)
        previous = euler
        if index >= len(context_frames):
            ground_truth.append(float(euler[axis]) - origin_value)
    return ground_truth


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
        record_id=entry.get("record_id", ""),
        revision=entry.get("revision", 0),
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

    Entries whose ground truth could be sampled are RETAINED with a bumped
    revision: later saves within the prediction window re-send the same
    record_id with the updated ground truth (training keeps the highest
    revision). An entry leaves the pending list when its channel gets a new
    prediction, its curves disappear, or the pending limit evicts it.
    """
    global _pending
    completed = []
    retained = []
    for entry in _pending:
        ground_truth = _sample_ground_truth(entry)
        completed.append(_finalize(entry, ground_truth))
        if ground_truth is not None:
            entry["revision"] += 1
            retained.append(entry)
    _pending = retained

    records = _load_outbox() + completed
    if not records:
        return
    random.shuffle(records)
    _store_outbox(records)
    if not sender.should_send(prefs):
        return
    threading.Thread(target=_send_outbox, args=(records,), daemon=True).start()


def request_token_refresh() -> None:
    """Empty-batch handshake so ctx64 is granted right after opt-in."""
    threading.Thread(
        target=sender.send_feedback_batch, args=([],), daemon=True
    ).start()
