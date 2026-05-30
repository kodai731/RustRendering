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
    bl_description = "Preview an AI forecast of the active FCurve as a ghost curve"
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

        from .. import preferences as prefs_module

        prefs = prefs_module.get_preferences()
        if not prefs.enable_curve_copilot:
            self.report({"ERROR"}, "Curve Copilot is disabled in Preferences")
            return {"CANCELLED"}

        fcurve = _find_active_fcurve(context.active_object)
        if fcurve is None or len(fcurve.keyframe_points) < 2:
            self.report({"ERROR"}, "Select an animated object with an FCurve")
            return {"CANCELLED"}

        keyframe_times = [float(kp.co.x) for kp in fcurve.keyframe_points]
        playhead = float(context.scene.frame_current_final)
        origin_frame = tml.resolve_origin_frame(keyframe_times, playhead)
        if origin_frame is None:
            self.report({"ERROR"}, "Move the playhead onto or after a keyframe")
            return {"CANCELLED"}

        try:
            model_path = _resolve_model_path(prefs)
        except FileNotFoundError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}

        logger = _debuglog.get_logger()
        fps = _scene_fps(context.scene)
        if logger is not None:
            logger.info(
                "curve_copilot run: object=%r fcurve=%s[%d] keyframes=%d "
                "origin_frame=%.4f playhead=%.4f fps=%.4f model=%s",
                context.active_object.name,
                fcurve.data_path,
                fcurve.array_index,
                len(fcurve.keyframe_points),
                origin_frame,
                playhead,
                fps,
                model_path,
            )

        try:
            ghost_points = _forecast_ghost(fcurve, model_path, origin_frame, fps)
        except Exception as e:  # noqa: BLE001
            if logger is not None:
                logger.exception("curve_copilot inference failed")
            self.report({"ERROR"}, f"Curve Copilot failed: {e}")
            return {"CANCELLED"}

        _ghost_overlay.set_ghost(ghost_points)
        _log_ghost_points(logger, ghost_points)
        self.report(
            {"INFO"},
            f"Forecast preview: {len(ghost_points) - 1} frames "
            "(ghost only, no keyframes inserted)",
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


def _find_active_fcurve(obj: bpy.types.Object):
    fcurves = _action_fcurves(obj)
    if not fcurves:
        return None
    for fcurve in fcurves:
        if getattr(fcurve, "select", False):
            return fcurve
    return fcurves[0]


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


def _forecast_ghost(fcurve, model_path: str, origin_frame: float, fps: float):
    """Sample the FCurve at the wheel-defined offsets and call the Rust forecast.

    The only Python-side work is sampling ``fcurve.evaluate`` (bpy interface);
    every numeric step happens in the wheel and is returned as the ghost polyline.
    """
    context_offsets, future_offsets = tml.forecast_sample_offsets()
    context = [fcurve.evaluate(origin_frame + offset) for offset in context_offsets]
    future = [fcurve.evaluate(origin_frame + offset) for offset in future_offsets]
    reveal_mask = [False] * len(future_offsets)
    origin_value = float(fcurve.evaluate(origin_frame))

    session = tml.PyRawFutureSession.from_onnx_path(model_path)
    return session.build_forecast_preview(
        context, future, reveal_mask, fps, float(origin_frame), origin_value
    )


def _log_ghost_points(logger, ghost_points) -> None:
    if logger is None:
        return
    for i, (frame, value) in enumerate(ghost_points):
        logger.info("  ghost[%d] frame=%.4f value=%.6f", i, frame, value)
    logger.info("forecast: ghost preview built (%d predicted frames)", len(ghost_points) - 1)
