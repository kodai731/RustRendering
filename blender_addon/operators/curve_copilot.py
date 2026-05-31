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

        all_fcurves = _action_fcurves(context.active_object)
        try:
            session = tml.PyV1CurveCopilotSession.from_onnx_path(model_path)
            ghosts = [
                ghost
                for index, fcurve in enumerate(fcurves)
                if (ghost := _forecast_ghost_for_fcurve(
                    fcurve, all_fcurves, session, playhead, deploy_fps, frame_step, index, logger
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


def _forecast_ghost_for_fcurve(
    fcurve, all_fcurves, session, playhead: float, deploy_fps: float, frame_step: float,
    index: int, logger,
):
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
        ghost = _forecast_quaternion_ghost(
            fcurve, all_fcurves, session, origin_frame, deploy_fps, frame_step
        )
        representation = "quaternion->euler"
    else:
        ghost = _forecast_direct_ghost(fcurve, session, origin_frame, deploy_fps, frame_step)
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


def _forecast_direct_ghost(fcurve, session, origin_frame: float, deploy_fps: float, frame_step: float):
    context_offsets, future_offsets = tml.forecast_sample_offsets()
    context = [fcurve.evaluate(origin_frame + offset * frame_step) for offset in context_offsets]
    future = [fcurve.evaluate(origin_frame + offset * frame_step) for offset in future_offsets]
    reveal_mask = [False] * len(future_offsets)
    origin_value = float(fcurve.evaluate(origin_frame))
    return session.build_forecast_preview(
        context, future, reveal_mask, deploy_fps, float(origin_frame), origin_value, frame_step
    )


def _quaternion_siblings(all_fcurves, fcurve):
    """The 4 quaternion component FCurves (W,X,Y,Z) sharing this data_path."""
    found = {}
    for candidate in all_fcurves:
        if candidate.data_path == fcurve.data_path and 0 <= candidate.array_index <= 3:
            found[candidate.array_index] = candidate
    return found if len(found) == 4 else None


def _forecast_quaternion_ghost(fcurve, all_fcurves, session, origin_frame, deploy_fps, frame_step):
    """Forecast a quaternion channel in Euler space (the trained representation).

    Samples the bone's full quaternion, converts to a continuous Euler curve
    (the model's training representation), forecasts each Euler axis with the
    Rust session, then converts the predicted Euler back to a quaternion and
    extracts the selected component for the ghost.
    """
    from mathutils import Euler, Quaternion

    siblings = _quaternion_siblings(all_fcurves, fcurve)
    if siblings is None:
        return _forecast_direct_ghost(fcurve, session, origin_frame, deploy_fps, frame_step)

    selected = fcurve.array_index
    context_offsets, future_offsets = tml.forecast_sample_offsets()
    sample_frames = [origin_frame + offset * frame_step for offset in context_offsets]
    sample_frames += [origin_frame + offset * frame_step for offset in future_offsets]

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

    n_context = len(context_offsets)
    context_eulers = eulers[:n_context]
    future_eulers = eulers[n_context:]
    reveal_mask = [False] * len(future_offsets)

    axis_ghosts = []
    for axis in range(3):
        axis_ghosts.append(
            session.build_forecast_preview(
                [e[axis] for e in context_eulers],
                [e[axis] for e in future_eulers],
                reveal_mask,
                deploy_fps,
                float(origin_frame),
                context_eulers[-1][axis],
                frame_step,
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

    return ghost
