"""Curve Copilot (Tier B) - in-process ONNX inference for FCurve suggestions.

Imports ``thyllore_ml_core`` (the L3 wheel) directly and uses the wheel's
public, ABI-marked surface only. Helper functions are private; the module
deliberately ships no shared facade so the L3 boundary stays single-hop.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import bpy
from bpy.types import Operator

# L3 boundary: import the wheel directly. Wheel API stability is enforced by
# the L2 trait in crates/thyllore-ml-api/ via `__abi_marker__` plus the
# `capabilities()` discovery used in `poll`.
try:
    import thyllore_ml_core as tml  # type: ignore

    _TML_AVAILABLE = True
except ImportError:
    tml = None  # type: ignore
    _TML_AVAILABLE = False


# Property-type encoding shared with the model. Phase 6 will move this into
# ml-api once the encoding is finalized.
_PROPERTY_TYPE_LOC_X = 0
_PROPERTY_TYPE_LOC_Y = 1
_PROPERTY_TYPE_LOC_Z = 2

_DATA_PATH_TO_PROPERTY_TYPE: dict[Tuple[str, int], int] = {
    ("location", 0): _PROPERTY_TYPE_LOC_X,
    ("location", 1): _PROPERTY_TYPE_LOC_Y,
    ("location", 2): _PROPERTY_TYPE_LOC_Z,
}

_CONTEXT_KEYFRAME_COUNT = 8
_CONTEXT_FEATURE_DIM = 6
_BONE_NAME_TOKEN_LENGTH = 32
_TOPOLOGY_FEATURE_DIM = 6
_CURVE_WINDOW_SIZE = 64
_CONFIDENCE_THRESHOLD = 0.3
_BONE_CONTEXT_N_MAX = 32
_BONE_CONTEXT_REST_POSITION_DIM = 3


class THYLLORE_OT_CurveCopilot(Operator):
    bl_idname = "thyllore.curve_copilot"
    bl_label = "Curve Copilot (ONNX)"
    bl_description = "Suggest next keyframes using AI (in-process inference)"
    bl_options = {"REGISTER", "UNDO"}

    num_suggestions: bpy.props.IntProperty(  # type: ignore[valid-type]
        name="Suggestions",
        default=4,
        min=1,
        max=16,
    )

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if obj is None or obj.type != "ARMATURE":
            return False
        # Capability discovery against the L3 wheel. Old/stripped wheels gray
        # the button out instead of crashing on call.
        if not _TML_AVAILABLE:
            return False
        return "curve_copilot" in tml.capabilities()

    def execute(self, context):
        if not _TML_AVAILABLE:
            self.report({"ERROR"}, "thyllore_ml_core wheel is not loaded")
            return {"CANCELLED"}

        from .. import preferences as prefs_module

        prefs = prefs_module.get_preferences()
        if not prefs.enable_curve_copilot:
            self.report({"ERROR"}, "Curve Copilot is disabled in Preferences")
            return {"CANCELLED"}

        armature = context.active_object
        fcurve = _find_active_fcurve(armature)
        if fcurve is None or len(fcurve.keyframe_points) < _CONTEXT_KEYFRAME_COUNT:
            self.report(
                {"ERROR"},
                f"Select an FCurve with at least {_CONTEXT_KEYFRAME_COUNT} keyframes",
            )
            return {"CANCELLED"}

        try:
            model_path = _resolve_model_path(prefs)
        except FileNotFoundError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}

        try:
            preds, conf = _run_curve_copilot(
                armature=armature,
                fcurve=fcurve,
                model_path=model_path,
                num_suggestions=self.num_suggestions,
            )
        except Exception as e:  # noqa: BLE001
            self.report({"ERROR"}, f"Curve Copilot failed: {e}")
            return {"CANCELLED"}

        applied = _apply_predictions_to_fcurve(fcurve, preds, conf)
        self.report({"INFO"}, f"Inserted {applied} keyframes")
        return {"FINISHED"}


def _find_active_fcurve(armature: bpy.types.Object):
    if armature.animation_data is None or armature.animation_data.action is None:
        return None
    fcurves = armature.animation_data.action.fcurves
    if not fcurves:
        return None
    return getattr(fcurves, "active", None) or fcurves[0]


def _resolve_model_path(prefs) -> str:
    if prefs.curve_copilot_model_path:
        path = Path(prefs.curve_copilot_model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        return str(path)

    addon_dir = Path(__file__).resolve().parents[1]
    bundled = addon_dir / "models" / "curve_copilot.onnx"
    if not bundled.exists():
        raise FileNotFoundError(
            "Bundled curve_copilot.onnx not found. "
            "Set Curve Copilot Model Path in Preferences, or rebuild the addon ZIP."
        )
    return str(bundled)


def _run_curve_copilot(
    armature: bpy.types.Object,
    fcurve,
    model_path: str,
    num_suggestions: int,
):
    """Run the full Tier B pipeline for the active FCurve.

    All helpers are private functions in this module. There is no shared
    Python facade between the operator and the wheel -- the wheel's typed
    methods (run_curve_copilot, PySession) plus capabilities() are the
    contract. Keep this orchestration in one place.
    """
    import numpy as np  # local import: numpy is large

    skel = _snapshot_armature(armature)
    bone_idx, prop_type = _decode_fcurve_target(fcurve, armature)
    if bone_idx < 0:
        raise ValueError("FCurve does not target a bone in this armature")

    topology = tml.compute_topology(skel)
    if bone_idx >= topology.shape[0]:
        raise ValueError(f"Bone index {bone_idx} out of range")
    bone_topology = topology[bone_idx].astype(np.float32)

    bone_tokens_array = tml.tokenize_bone_names(skel)
    bone_tokens = bone_tokens_array[bone_idx].astype(np.int64)

    context, current_time, clip_duration = _flatten_context_from_fcurve(fcurve)
    sampled_window = _build_sampled_window(fcurve)

    session = tml.PySession.from_onnx_path(model_path)
    model_max_steps = session.max_steps() or num_suggestions
    query_times = _build_query_times(
        fcurve, current_time, clip_duration, model_max_steps
    )

    bone_context_keyframes = np.zeros(
        _BONE_CONTEXT_N_MAX * _CONTEXT_KEYFRAME_COUNT * _CONTEXT_FEATURE_DIM,
        dtype=np.float32,
    )
    bone_context_topology = np.zeros(
        _BONE_CONTEXT_N_MAX * _TOPOLOGY_FEATURE_DIM,
        dtype=np.float32,
    )
    bone_context_rest_positions = np.zeros(
        _BONE_CONTEXT_N_MAX * _BONE_CONTEXT_REST_POSITION_DIM,
        dtype=np.float32,
    )
    bone_context_mask = np.zeros(_BONE_CONTEXT_N_MAX, dtype=bool)

    preds_2d, conf_1d = session.run_curve_copilot(
        context.flatten().astype(np.float32),
        prop_type,
        bone_topology.flatten().astype(np.float32),
        bone_tokens.flatten().astype(np.int64),
        query_times.astype(np.float32),
        sampled_window.astype(np.float32),
        bone_context_keyframes,
        bone_context_topology,
        bone_context_rest_positions,
        bone_context_mask,
    )

    return preds_2d, conf_1d


def _snapshot_armature(armature):
    """bpy.Armature -> tml.PySkeleton."""
    import numpy as np

    bones = list(armature.data.bones)
    bone_names = [b.name for b in bones]

    name_to_index = {b.name: i for i, b in enumerate(bones)}
    parent_indices = np.array(
        [
            name_to_index[b.parent.name] if b.parent and b.parent.name in name_to_index else -1
            for b in bones
        ],
        dtype=np.int32,
    )

    local_matrices = np.zeros((len(bones), 4, 4), dtype=np.float32)
    for i, b in enumerate(bones):
        m = b.matrix_local
        for r in range(4):
            for c in range(4):
                local_matrices[i, r, c] = m[r][c]

    return tml.PySkeleton.from_flat(bone_names, parent_indices, local_matrices)


def _decode_fcurve_target(fcurve, armature) -> Tuple[int, int]:
    """Map an FCurve to (bone_index, property_type_id)."""
    data_path = fcurve.data_path
    array_index = fcurve.array_index

    bone_name: Optional[str] = None
    sub_path: Optional[str] = None

    if data_path.startswith('pose.bones["'):
        end = data_path.find('"]', 12)
        if end > 12:
            bone_name = data_path[12:end]
            sub_path = data_path[end + 3 :]
    if not bone_name or not sub_path:
        return -1, 0

    prop_type = _DATA_PATH_TO_PROPERTY_TYPE.get((sub_path, array_index))
    if prop_type is None:
        return -1, 0

    bones = list(armature.data.bones)
    for i, b in enumerate(bones):
        if b.name == bone_name:
            return i, prop_type
    return -1, prop_type


def _flatten_context_from_fcurve(fcurve):
    """Build the 8-keyframe context tensor + (current_time, clip_duration)."""
    import numpy as np

    keyframes = list(fcurve.keyframe_points)[-_CONTEXT_KEYFRAME_COUNT:]
    while len(keyframes) < _CONTEXT_KEYFRAME_COUNT:
        keyframes.insert(0, keyframes[0])

    rows = []
    for kp in keyframes:
        rows.append(
            [
                kp.co.x,
                kp.co.y,
                kp.handle_left.x,
                kp.handle_left.y,
                kp.handle_right.x,
                kp.handle_right.y,
            ]
        )
    context = np.array(rows, dtype=np.float32)
    current_time = float(keyframes[-1].co.x)
    clip_duration = float(keyframes[-1].co.x - keyframes[0].co.x) or 1.0
    return context, current_time, clip_duration


def _build_sampled_window(fcurve):
    import numpy as np

    keyframes = list(fcurve.keyframe_points)[-_CONTEXT_KEYFRAME_COUNT:]
    t_min = float(keyframes[0].co.x)
    t_max = float(keyframes[-1].co.x)
    if t_max <= t_min:
        t_max = t_min + 1.0
    sample_times = np.linspace(t_min, t_max, _CURVE_WINDOW_SIZE)
    return np.array(
        [fcurve.evaluate(float(t)) for t in sample_times], dtype=np.float32
    )


def _build_query_times(fcurve, current_time: float, clip_duration: float, count: int):
    import numpy as np

    keyframes = list(fcurve.keyframe_points)[-_CONTEXT_KEYFRAME_COUNT:]
    if len(keyframes) >= 2:
        t_min = float(keyframes[0].co.x)
        t_max = float(keyframes[-1].co.x)
        dt_avg = (t_max - t_min) / max(1, _CONTEXT_KEYFRAME_COUNT - 1)
    else:
        dt_avg = max(1.0, clip_duration / max(1, _CONTEXT_KEYFRAME_COUNT))
    return np.array(
        [current_time + dt_avg * (i + 1) for i in range(count)],
        dtype=np.float32,
    )


def _apply_predictions_to_fcurve(fcurve, preds, conf) -> int:
    """Insert predicted keyframes into fcurve. Returns the count actually inserted."""
    inserted = 0
    n = len(preds)
    for i in range(n):
        if conf[i] < _CONFIDENCE_THRESHOLD:
            continue
        row = preds[i]
        # Wheel returns rows shaped (5,): [value, in_x, in_y, out_x, out_y].
        # Time is index-aligned with query_times, recovered below.
        t = float(_query_time_for_index(fcurve, i, n))
        v = float(row[0])
        ilx = float(row[1])
        ily = float(row[2])
        orx = float(row[3])
        ory = float(row[4])

        kp = fcurve.keyframe_points.insert(t, v)
        kp.handle_left = (ilx, ily)
        kp.handle_right = (orx, ory)
        kp.interpolation = "BEZIER"
        inserted += 1

    fcurve.update()
    return inserted


def _query_time_for_index(fcurve, i: int, count: int) -> float:
    """Recover the query time for prediction index ``i``.

    Mirrors the spacing used in :func:`_build_query_times` so the indices line
    up. Phase 5 will replace this with the wheel returning ``time`` directly.
    """
    keyframes = list(fcurve.keyframe_points)[-_CONTEXT_KEYFRAME_COUNT:]
    if len(keyframes) < 2:
        return 0.0
    t_min = float(keyframes[0].co.x)
    t_max = float(keyframes[-1].co.x)
    dt_avg = (t_max - t_min) / max(1, _CONTEXT_KEYFRAME_COUNT - 1)
    return t_max + dt_avg * (i + 1)
