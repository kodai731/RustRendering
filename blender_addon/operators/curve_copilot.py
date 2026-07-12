"""Curve Copilot (Tier B) - in-process ONNX forecast preview for FCurves.

Caller layer: all numeric work (window offsets, origin resolution, ONNX run,
continuity, ghost polyline) lives in the Rust wheel ``thyllore_ml_core`` and is
the single source of truth shared with the engine. This module only does the
Blender-specific work — extract samples from a bpy FCurve, call the wheel, and
hand the result to the GPU ghost overlay.

Curve Copilot is a *preview* feature: it shows the predicted curve as a
non-destructive ghost in the Graph Editor and never edits the real FCurve.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import bpy
from bpy.types import Operator

from .. import _debuglog, _ghost_overlay

try:
    import thyllore_ml_core as tml  # type: ignore

    _TML_AVAILABLE = True
except ImportError:
    tml = None  # type: ignore
    _TML_AVAILABLE = False

try:
    from .. import telemetry
except ImportError:
    telemetry = None


@dataclass
class _ForecastRun:
    """Per-execute settings shared by the per-curve forecast helpers."""

    session: object
    object_name: str
    scene_fps: float
    deploy_fps: float
    frame_step: float
    full_token: str | None
    record_feedback: bool
    model_hash: str


_MODEL_HASH_CACHE: dict[str, tuple[float, str]] = {}


def _resolve_full_token() -> str | None:
    from .. import _full_token

    return _full_token.resolve_full_token()


def _model_hash(model_path: str) -> str:
    path = Path(model_path)
    mtime = path.stat().st_mtime
    cached = _MODEL_HASH_CACHE.get(model_path)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    _MODEL_HASH_CACHE[model_path] = (mtime, digest)
    return digest


def _channel_kind(data_path: str) -> str:
    return data_path.rsplit(".", 1)[-1]


class THYLLORE_OT_CurveCopilot(Operator):
    bl_idname = "thyllore.curve_copilot"
    bl_label = "Curve Copilot (ONNX)"
    bl_description = "Toggle an AI ghost-curve forecast of the enabled channels (press again to clear)"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        if not _TML_AVAILABLE:
            return False
        if context.active_object is None:
            return False
        return "curve_forecast" in tml.capabilities()

    def execute(self, context):
        if not _TML_AVAILABLE:
            self.report({"ERROR"}, "thyllore_ml_core wheel is not loaded")
            return {"CANCELLED"}

        # Shift+C toggles: draw -> clear -> draw -> clear. When a preview is on
        # screen, this press erases it so the animator can drop unwanted curves;
        # the next press redraws from the current channel selection.
        if _ghost_overlay.has_ghost():
            if telemetry is not None:
                telemetry.mark_all_cleared()
            _ghost_overlay.clear_ghost()
            self.report({"INFO"}, "Forecast preview cleared")
            return {"FINISHED"}

        from .. import preferences as prefs_module

        prefs = prefs_module.get_preferences()
        if not prefs.enable_curve_copilot:
            self.report({"ERROR"}, "Curve Copilot is disabled in Preferences")
            return {"CANCELLED"}

        fcurves = _forecast_fcurves(context)
        if not fcurves:
            self.report(
                {"ERROR"},
                "Select one or more curves in the Graph Editor (with >=2 keyframes)",
            )
            return {"CANCELLED"}

        try:
            model_path = _resolve_model_path(prefs)
        except FileNotFoundError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}

        logger = _debuglog.get_logger()
        scene_fps = _scene_fps(context.scene)
        deploy_fps = tml.deploy_fps()
        frame_step = scene_fps / deploy_fps
        playhead = float(context.scene.frame_current_final)
        if logger is not None:
            logger.info(
                "curve_copilot run: object=%r selected_fcurves=%d playhead=%.4f "
                "scene_fps=%.4f deploy_fps=%.4f frame_step=%.4f model=%s",
                context.active_object.name,
                len(fcurves),
                playhead,
                scene_fps,
                deploy_fps,
                frame_step,
                model_path,
            )

        record_feedback = telemetry is not None and telemetry.should_send(prefs)
        all_fcurves = _action_fcurves(context.active_object)
        try:
            session = tml.PyV2CurveCopilotSession.from_onnx_path(model_path)
            run = _ForecastRun(
                session=session,
                object_name=context.active_object.name,
                scene_fps=scene_fps,
                deploy_fps=deploy_fps,
                frame_step=frame_step,
                full_token=_resolve_full_token(),
                record_feedback=record_feedback,
                model_hash=_model_hash(model_path) if record_feedback else "",
            )
            ghosts = [
                ghost
                for index, fcurve in enumerate(fcurves)
                if (ghost := _forecast_ghost_for_fcurve(
                    fcurve, all_fcurves, run, playhead, index, logger
                )) is not None
            ]
        except Exception as e:  # noqa: BLE001
            if logger is not None:
                logger.exception("curve_copilot inference failed")
            self.report({"ERROR"}, f"Curve Copilot failed: {e}")
            return {"CANCELLED"}

        if not ghosts:
            self.report({"ERROR"}, "Move the playhead onto or after a keyframe")
            return {"CANCELLED"}

        _ghost_overlay.set_ghosts(ghosts)
        self.report(
            {"INFO"},
            f"Forecast preview: {len(ghosts)} curve(s) (ghost only, no keyframes inserted)",
        )
        return {"FINISHED"}


class THYLLORE_OT_CurveCopilotClear(Operator):
    bl_idname = "thyllore.curve_copilot_clear"
    bl_label = "Clear Forecast Preview"
    bl_description = "Remove the Curve Copilot ghost curve"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return _ghost_overlay.has_ghost()

    def execute(self, context):
        if telemetry is not None:
            telemetry.mark_all_cleared()
        _ghost_overlay.clear_ghost()
        return {"FINISHED"}


def _action_fcurves(obj: bpy.types.Object):
    """Enumerate the active action's FCurves across Blender action APIs.

    Blender <= 4.3 exposes a flat ``action.fcurves``; 4.4+ slotted actions keep
    them in the channelbag bound to the object's action slot.
    """
    anim = obj.animation_data
    if anim is None or anim.action is None:
        return []
    action = anim.action

    legacy = getattr(action, "fcurves", None)
    if legacy is not None:
        return list(legacy)

    slot = getattr(anim, "action_slot", None)
    fcurves = []
    for layer in action.layers:
        for strip in layer.strips:
            channelbag = strip.channelbag(slot) if slot is not None else None
            if channelbag is None and strip.channelbags:
                channelbag = strip.channelbags[0]
            if channelbag is not None:
                fcurves.extend(channelbag.fcurves)
    return fcurves


def _forecast_fcurves(context):
    """The curves Blender reports as selected in the Graph Editor.

    `context.selected_editable_fcurves` is Blender's own list of the curves the
    user has selected (it already respects visibility / lock / editability), so
    forecasting these matches exactly what is highlighted on screen — one ghost
    per selected curve.

    Headless fallback: with no Graph Editor area (e.g. the operator smoke run
    under ``--background``), that context member is empty, so fall back to the
    active object's selected channels.
    """
    selected = getattr(context, "selected_editable_fcurves", None) or []
    fcurves = [fc for fc in selected if len(fc.keyframe_points) >= 2]
    if fcurves:
        return fcurves

    obj = context.active_object
    if obj is None:
        return []
    return [
        fc
        for fc in _action_fcurves(obj)
        if getattr(fc, "select", False) and len(fc.keyframe_points) >= 2
    ]


def _resolve_model_path(prefs) -> str:
    if prefs.curve_copilot_model_path:
        path = Path(prefs.curve_copilot_model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        return str(path)

    # SSOT: prefer the same SharedData model the engine resolves, so the addon
    # and the desktop app always run the identical ONNX.
    shared = tml.resolve_curve_copilot_model_path()
    if shared:
        return shared

    addon_dir = Path(__file__).resolve().parents[1]
    bundled = addon_dir / "models" / "curve_copilot.onnx"
    if not bundled.exists():
        raise FileNotFoundError(
            "Bundled curve_copilot.onnx not found. "
            "Set Curve Copilot Model Path in Preferences, or rebuild the addon ZIP."
        )
    return str(bundled)


def _scene_fps(scene) -> float:
    fps_base = scene.render.fps_base or 1.0
    return float(scene.render.fps) / float(fps_base)


def _forecast_ghost_for_fcurve(fcurve, all_fcurves, run: _ForecastRun, playhead: float, index: int, logger):
    """Forecast one selected FCurve and return its ghost polyline (or None).

    Samples are read at the model's deploy rate (every ``frame_step =
    scene_fps / deploy_fps`` Blender frames) so the input matches the engine and
    the training (60 fps) regardless of scene fps. The model is trained on Euler
    radians, so a ``rotation_quaternion`` channel is forecast in Euler space and
    converted back; Euler / location / scale channels are forecast directly.
    """
    origin_frame = tml.resolve_origin_frame(
        [float(kp.co.x) for kp in fcurve.keyframe_points], playhead
    )
    if origin_frame is None:
        if logger is not None:
            logger.info(
                "  skip %s[%d]: no keyframe at/before playhead",
                fcurve.data_path,
                fcurve.array_index,
            )
        return None

    if fcurve.data_path.endswith("rotation_quaternion"):
        ghost = _forecast_quaternion_ghost(fcurve, all_fcurves, run, origin_frame)
        representation = "quaternion->euler"
    else:
        ghost = _forecast_direct_ghost(fcurve, run, origin_frame)
        representation = "direct"

    if logger is not None:
        color = _ghost_overlay.color_for_index(index)
        logger.info(
            "curve[%d] %s[%d] repr=%s keyframes=%d origin_frame=%.4f "
            "color=(%.2f, %.2f, %.2f) predicted_frames=%d",
            index,
            fcurve.data_path,
            fcurve.array_index,
            representation,
            len(fcurve.keyframe_points),
            origin_frame,
            color[0],
            color[1],
            color[2],
            len(ghost) - 1,
        )
        for i, (frame, value) in enumerate(ghost):
            logger.info("    ghost[%d] frame=%.4f value=%.6f", i, frame, value)
    return ghost


def _forecast_direct_ghost(fcurve, run: _ForecastRun, origin_frame: float):
    context_offsets, _future_offsets = tml.forecast_sample_offsets()
    context = [
        fcurve.evaluate(origin_frame + offset * run.frame_step) for offset in context_offsets
    ]
    origin_value = float(fcurve.evaluate(origin_frame))
    ghost = run.session.build_forecast_preview(
        context, run.deploy_fps, float(origin_frame), origin_value, run.frame_step,
        run.full_token,
    )

    if run.record_feedback and telemetry is not None:
        telemetry.record_prediction(
            object_name=run.object_name,
            data_path=fcurve.data_path,
            array_index=fcurve.array_index,
            channel_kind=_channel_kind(fcurve.data_path),
            scene_fps=run.scene_fps,
            deploy_fps=run.deploy_fps,
            frame_step=run.frame_step,
            origin_value=origin_value,
            context=[float(value) for value in context],
            prediction_frames=[float(frame) for frame, _ in ghost[1:]],
            prediction_values=[float(value) for _, value in ghost[1:]],
            model_hash=run.model_hash,
        )
    return ghost


def _quaternion_siblings(all_fcurves, fcurve):
    """The 4 quaternion component FCurves (W,X,Y,Z) sharing this data_path."""
    found = {}
    for candidate in all_fcurves:
        if candidate.data_path == fcurve.data_path and 0 <= candidate.array_index <= 3:
            found[candidate.array_index] = candidate
    return found if len(found) == 4 else None


def _forecast_quaternion_ghost(fcurve, all_fcurves, run: _ForecastRun, origin_frame):
    """Forecast a quaternion channel in Euler space (the trained representation).

    Samples the bone's full quaternion, converts to a continuous Euler curve
    (the model's training representation), forecasts each Euler axis with the
    Rust session, then converts the predicted Euler back to a quaternion and
    extracts the selected component for the ghost.

    Feedback recording is skipped here: the ground truth would have to be
    re-sampled in Euler space from all four sibling curves, which the two-stage
    collector does not support yet.
    """
    from mathutils import Euler, Quaternion

    siblings = _quaternion_siblings(all_fcurves, fcurve)
    if siblings is None:
        return _forecast_direct_ghost(fcurve, run, origin_frame)

    selected = fcurve.array_index
    context_offsets, _future_offsets = tml.forecast_sample_offsets()
    sample_frames = [origin_frame + offset * run.frame_step for offset in context_offsets]

    eulers = []
    previous = None
    for frame in sample_frames:
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
        eulers.append((euler.x, euler.y, euler.z))

    axis_ghosts = []
    for axis in range(3):
        axis_ghosts.append(
            run.session.build_forecast_preview(
                [e[axis] for e in eulers],
                run.deploy_fps,
                float(origin_frame),
                eulers[-1][axis],
                run.frame_step,
                run.full_token,
            )
        )

    points = [(float(origin_frame), float(fcurve.evaluate(origin_frame)))]
    for j in range(1, len(axis_ghosts[0])):
        frame = axis_ghosts[0][j][0]
        euler = Euler(
            (axis_ghosts[0][j][1], axis_ghosts[1][j][1], axis_ghosts[2][j][1]), "XYZ"
        )
        points.append((frame, euler.to_quaternion()[selected]))
    return points
