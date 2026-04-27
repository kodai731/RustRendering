"""Auto Rig (Tier A) - selected mesh -> GLB -> UniRig gRPC -> Armature import.

Uses the Modal pattern from ``modal/dispatcher.py`` so the gRPC call (which
takes 5-30 seconds for UniRig) does not block the Blender UI.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import bpy
from bpy.types import Operator

from ..grpc_client import AutoRigClient, AutoRigInput
from ..modal import AsyncDispatcher, is_headless
from ._grpc_helpers import (
    build_server_config_from_preferences,
    grpc_error_message,
    write_temp_glb,
)


class THYLLORE_OT_AutoRig(Operator):
    bl_idname = "thyllore.auto_rig"
    bl_label = "Generate Rig (UniRig)"
    bl_description = "Send the selected mesh to the AutoRigging service and import the rigged GLB"
    bl_options = {"REGISTER", "UNDO"}

    num_sample_points: bpy.props.IntProperty(  # type: ignore[valid-type]
        name="Sample Points",
        default=65536,
        min=1024,
        max=131072,
        description="Number of surface samples sent to UniRig",
    )

    _dispatcher: Optional[AsyncDispatcher] = None
    _timer = None
    _source_mesh_name: str = ""

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH"

    def invoke(self, context, event):
        if is_headless() or event is None or context.window is None:
            return self._execute_synchronous(context)
        return self._start_modal(context)

    def _start_modal(self, context):
        mesh_obj = context.active_object
        if mesh_obj is None or mesh_obj.type != "MESH":
            self.report({"ERROR"}, "Select a mesh object")
            return {"CANCELLED"}

        try:
            glb_bytes = _export_object_to_glb(mesh_obj)
        except Exception as e:  # noqa: BLE001
            self.report({"ERROR"}, f"GLB export failed: {e}")
            return {"CANCELLED"}

        config = build_server_config_from_preferences()
        client = AutoRigClient(config)
        rig_input = AutoRigInput(
            glb_data=glb_bytes,
            num_sample_points=self.num_sample_points,
        )

        self._source_mesh_name = mesh_obj.name
        self._dispatcher = AsyncDispatcher()
        self._dispatcher.start(client.generate_rig, rig_input)

        self._timer = context.window_manager.event_timer_add(0.1, window=context.window)
        context.window_manager.modal_handler_add(self)
        context.workspace.status_text_set("Auto Rig: contacting server...")
        context.window.cursor_modal_set("WAIT")
        return {"RUNNING_MODAL"}

    def _execute_synchronous(self, context):
        """Single-shot path used by headless tests."""
        mesh_obj = context.active_object
        if mesh_obj is None or mesh_obj.type != "MESH":
            self.report({"ERROR"}, "Select a mesh object")
            return {"CANCELLED"}

        try:
            glb_bytes = _export_object_to_glb(mesh_obj)
        except Exception as e:  # noqa: BLE001
            self.report({"ERROR"}, f"GLB export failed: {e}")
            return {"CANCELLED"}

        config = build_server_config_from_preferences()
        client = AutoRigClient(config)
        rig_input = AutoRigInput(
            glb_data=glb_bytes,
            num_sample_points=self.num_sample_points,
        )

        try:
            result = client.generate_rig(rig_input)
        except BaseException as e:  # noqa: BLE001
            self.report({"ERROR"}, grpc_error_message(e))
            return {"CANCELLED"}

        try:
            _import_rigged_glb(result.rigged_glb_data, source_mesh_name=mesh_obj.name)
        except Exception as e:  # noqa: BLE001
            self.report({"ERROR"}, f"Failed to import rigged GLB: {e}")
            return {"CANCELLED"}

        self.report({"INFO"}, f"Auto Rig completed: {result.bone_count} bones")
        return {"FINISHED"}

    def modal(self, context, event):
        if event.type == "ESC":
            self._cleanup(context)
            self.report({"INFO"}, "Auto Rig cancelled")
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

        try:
            _import_rigged_glb(
                payload.rigged_glb_data,
                source_mesh_name=self._source_mesh_name,
            )
        except Exception as e:  # noqa: BLE001
            self.report({"ERROR"}, f"Failed to import rigged GLB: {e}")
            return {"CANCELLED"}

        self.report({"INFO"}, f"Auto Rig completed: {payload.bone_count} bones")
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


def _export_object_to_glb(obj: bpy.types.Object) -> bytes:
    """Export a single mesh object to in-memory GLB bytes."""
    if obj.data is None or len(obj.data.vertices) == 0:
        raise ValueError(f"Mesh '{obj.name}' has no vertices")

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
        tmp_path = tmp.name

    prev_selected = list(bpy.context.selected_objects)
    prev_active = bpy.context.view_layer.objects.active

    try:
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        bpy.ops.export_scene.gltf(
            filepath=tmp_path,
            export_format="GLB",
            use_selection=True,
            export_apply=True,
            export_yup=True,
        )

        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        bpy.ops.object.select_all(action="DESELECT")
        for o in prev_selected:
            try:
                o.select_set(True)
            except (RuntimeError, ReferenceError):
                pass
        try:
            bpy.context.view_layer.objects.active = prev_active
        except (RuntimeError, ReferenceError):
            pass
        Path(tmp_path).unlink(missing_ok=True)


def _import_rigged_glb(glb_bytes: bytes, source_mesh_name: str) -> None:
    """Import a rigged GLB and rename the new armature for traceability."""
    if not glb_bytes:
        raise ValueError("Server returned empty rigged GLB")

    glb_path = write_temp_glb(glb_bytes, prefix="thyllore_rigged_")
    try:
        prev_active = bpy.context.view_layer.objects.active
        bpy.ops.import_scene.gltf(filepath=str(glb_path))

        new_armatures = [
            o for o in bpy.context.selected_objects if o.type == "ARMATURE"
        ]
        if new_armatures:
            new_arm = new_armatures[0]
            new_arm.name = f"{source_mesh_name}_Armature"
            print(
                f"[Thyllore] Imported armature: {new_arm.name} "
                f"({len(new_arm.data.bones)} bones)"
            )

        try:
            bpy.context.view_layer.objects.active = prev_active
        except (RuntimeError, ReferenceError):
            pass
    finally:
        glb_path.unlink(missing_ok=True)
