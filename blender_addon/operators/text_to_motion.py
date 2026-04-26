"""Text to Motion (Tier A) - prompt + Armature -> light_t2m gRPC -> bpy.Action."""
from __future__ import annotations

from typing import Optional, Tuple

import bpy
from bpy.types import Operator

from ..grpc_client import AnimationCurve, MotionClient, MotionInput
from ..modal import AsyncDispatcher, is_headless
from ._grpc_helpers import build_server_config_from_preferences, grpc_error_message


# Mapping from MotionResult.AnimationCurve.property_type strings to (data_path,
# array_index). Phase 4 covers location/rotation/scale; Phase 6 will extend to
# custom properties and shape keys.
_PROPERTY_TYPE_TO_BPY_DATA_PATH: dict[str, Tuple[str, int]] = {
    "TRANSLATION_X": ("location", 0),
    "TRANSLATION_Y": ("location", 1),
    "TRANSLATION_Z": ("location", 2),
    "ROTATION_QUATERNION_W": ("rotation_quaternion", 0),
    "ROTATION_QUATERNION_X": ("rotation_quaternion", 1),
    "ROTATION_QUATERNION_Y": ("rotation_quaternion", 2),
    "ROTATION_QUATERNION_Z": ("rotation_quaternion", 3),
    "SCALE_X": ("scale", 0),
    "SCALE_Y": ("scale", 1),
    "SCALE_Z": ("scale", 2),
}


class THYLLORE_OT_TextToMotion(Operator):
    bl_idname = "thyllore.text_to_motion"
    bl_label = "Text to Motion"
    bl_description = "Generate motion curves from a text prompt and apply them to the active armature"
    bl_options = {"REGISTER", "UNDO"}

    prompt: bpy.props.StringProperty(  # type: ignore[valid-type]
        name="Prompt",
        default="walking forward",
    )
    duration_seconds: bpy.props.FloatProperty(  # type: ignore[valid-type]
        name="Duration (s)",
        default=3.0,
        min=0.5,
        max=30.0,
    )
    target_fps: bpy.props.IntProperty(  # type: ignore[valid-type]
        name="Target FPS",
        default=30,
        min=12,
        max=120,
    )
    skeleton_type: bpy.props.EnumProperty(  # type: ignore[valid-type]
        name="Skeleton",
        items=[
            ("SMPL_22", "SMPL 22", "SMPL 22-bone skeleton"),
            ("SMPL_24", "SMPL 24", "SMPL 24-bone skeleton"),
        ],
        default="SMPL_22",
    )

    _dispatcher: Optional[AsyncDispatcher] = None
    _timer = None
    _target_armature_name: str = ""

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "ARMATURE"

    def invoke(self, context, event):
        if is_headless() or event is None or context.window is None:
            return self._execute_synchronous(context)
        return context.window_manager.invoke_props_dialog(self, width=400)

    def execute(self, context):
        return self._start_modal(context)

    def _start_modal(self, context):
        armature = context.active_object
        if armature is None or armature.type != "ARMATURE":
            self.report({"ERROR"}, "Select an armature object")
            return {"CANCELLED"}

        config = build_server_config_from_preferences()
        client = MotionClient(config)
        motion_input = MotionInput(
            prompt=self.prompt,
            duration_seconds=self.duration_seconds,
            target_fps=self.target_fps,
            skeleton_type=self.skeleton_type,
        )

        self._target_armature_name = armature.name
        self._dispatcher = AsyncDispatcher()
        self._dispatcher.start(client.generate_motion, motion_input)

        self._timer = context.window_manager.event_timer_add(0.1, window=context.window)
        context.window_manager.modal_handler_add(self)
        context.workspace.status_text_set(f"Text to Motion: '{self.prompt[:40]}'...")
        context.window.cursor_modal_set("WAIT")
        return {"RUNNING_MODAL"}

    def _execute_synchronous(self, context):
        armature = context.active_object
        if armature is None or armature.type != "ARMATURE":
            self.report({"ERROR"}, "Select an armature object")
            return {"CANCELLED"}

        config = build_server_config_from_preferences()
        client = MotionClient(config)
        motion_input = MotionInput(
            prompt=self.prompt,
            duration_seconds=self.duration_seconds,
            target_fps=self.target_fps,
            skeleton_type=self.skeleton_type,
        )
        try:
            result = client.generate_motion(motion_input)
        except BaseException as e:  # noqa: BLE001
            self.report({"ERROR"}, grpc_error_message(e))
            return {"CANCELLED"}
        try:
            _apply_motion_to_armature(
                armature,
                result.curves,
                action_name=f"{armature.name}_GenMotion",
            )
        except Exception as e:  # noqa: BLE001
            self.report({"ERROR"}, f"Failed to apply motion: {e}")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Text to Motion: {len(result.curves)} curves")
        return {"FINISHED"}

    def modal(self, context, event):
        if event.type == "ESC":
            self._cleanup(context)
            self.report({"INFO"}, "Text to Motion cancelled")
            return {"CANCELLED"}

        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        polled = self._dispatcher.poll() if self._dispatcher else None
        if polled is None:
            return {"PASS_THROUGH"}

        kind, payload = polled
        self._cleanup(context)

        if kind == "err":
            self.report({"ERROR"}, grpc_error_message(payload))
            return {"CANCELLED"}

        armature = bpy.data.objects.get(self._target_armature_name)
        if armature is None or armature.type != "ARMATURE":
            self.report({"ERROR"}, "Target armature was removed during inference")
            return {"CANCELLED"}

        try:
            _apply_motion_to_armature(
                armature,
                payload.curves,
                action_name=f"{armature.name}_GenMotion",
            )
        except Exception as e:  # noqa: BLE001
            self.report({"ERROR"}, f"Failed to apply motion: {e}")
            return {"CANCELLED"}

        self.report({"INFO"}, f"Text to Motion: {len(payload.curves)} curves applied")
        return {"FINISHED"}

    def cancel(self, context):
        if self._dispatcher is not None:
            self._dispatcher.cancel()
        self._cleanup(context)

    def _cleanup(self, context):
        if self._dispatcher is not None:
            self._dispatcher.cancel()
            self._dispatcher = None
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        if context.window is not None:
            context.window.cursor_modal_restore()
        if context.workspace is not None:
            context.workspace.status_text_set(None)


def _apply_motion_to_armature(
    armature: bpy.types.Object,
    curves: tuple[AnimationCurve, ...],
    action_name: str,
) -> None:
    if armature.animation_data is None:
        armature.animation_data_create()

    action = bpy.data.actions.new(name=action_name)
    armature.animation_data.action = action

    fps = bpy.context.scene.render.fps or 30
    skipped: list[str] = []

    for curve in curves:
        mapping = _PROPERTY_TYPE_TO_BPY_DATA_PATH.get(curve.property_type)
        if mapping is None:
            skipped.append(f"{curve.bone_name}.{curve.property_type}")
            continue

        sub_path, array_index = mapping
        data_path = f'pose.bones["{curve.bone_name}"].{sub_path}'
        try:
            fcurve = action.fcurves.new(data_path=data_path, index=array_index)
        except RuntimeError:
            skipped.append(f"{curve.bone_name}.{sub_path}[{array_index}]")
            continue

        for kf in curve.keyframes:
            point = fcurve.keyframe_points.insert(
                frame=kf.time * fps, value=kf.value
            )
            point.handle_left = (
                kf.time * fps + kf.tangent_in_dt * fps,
                kf.value + kf.tangent_in_dv,
            )
            point.handle_right = (
                kf.time * fps + kf.tangent_out_dt * fps,
                kf.value + kf.tangent_out_dv,
            )
            point.interpolation = "BEZIER"

        fcurve.update()

    if skipped:
        print(f"[Thyllore] Skipped unsupported curves: {skipped}")
