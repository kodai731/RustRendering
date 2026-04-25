"""Text to Mesh (Tier A) - prompt -> TRELLIS/Hunyuan3D/Era3D gRPC -> Mesh import."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import bpy
from bpy.types import Operator

from ..grpc_client import MeshClient, MeshInput
from ..modal import AsyncDispatcher
from ._grpc_helpers import (
    build_server_config_from_preferences,
    grpc_error_message,
    write_temp_glb,
)


class THYLLORE_OT_TextToMesh(Operator):
    bl_idname = "thyllore.text_to_mesh"
    bl_label = "Text to Mesh"
    bl_description = "Generate a mesh from a text prompt via the MeshGeneration service"
    bl_options = {"REGISTER", "UNDO"}

    prompt: bpy.props.StringProperty(  # type: ignore[valid-type]
        name="Prompt",
        default="a low-poly character",
    )
    target_faces: bpy.props.IntProperty(  # type: ignore[valid-type]
        name="Target Faces",
        default=30000,
        min=1000,
        max=200000,
    )
    seed: bpy.props.IntProperty(  # type: ignore[valid-type]
        name="Seed",
        default=42,
    )
    model: bpy.props.EnumProperty(  # type: ignore[valid-type]
        name="Model",
        items=[
            ("TRELLIS", "TRELLIS", "Microsoft TRELLIS"),
            ("HUNYUAN3D", "Hunyuan3D", "Tencent Hunyuan3D"),
            ("ERA3D", "Era3D", "Era3D pipeline"),
        ],
        default="TRELLIS",
    )

    _dispatcher: Optional[AsyncDispatcher] = None
    _timer = None
    _resolved_prompt: str = ""

    def invoke(self, context, event):
        if event is None or context.window is None:
            return self._execute_synchronous(context)
        return context.window_manager.invoke_props_dialog(self, width=400)

    def execute(self, context):
        return self._start_modal(context)

    def _start_modal(self, context):
        config = build_server_config_from_preferences()
        client = MeshClient(config)
        mesh_input = MeshInput(
            prompt=self.prompt,
            target_faces=self.target_faces,
            seed=self.seed,
            model_type=self.model,
        )

        self._resolved_prompt = self.prompt
        self._dispatcher = AsyncDispatcher()
        self._dispatcher.start(client.generate_mesh, mesh_input)

        self._timer = context.window_manager.event_timer_add(0.1, window=context.window)
        context.window_manager.modal_handler_add(self)
        context.workspace.status_text_set(f"Text to Mesh: '{self.prompt[:40]}'...")
        context.window.cursor_modal_set("WAIT")
        return {"RUNNING_MODAL"}

    def _execute_synchronous(self, context):
        config = build_server_config_from_preferences()
        client = MeshClient(config)
        mesh_input = MeshInput(
            prompt=self.prompt,
            target_faces=self.target_faces,
            seed=self.seed,
            model_type=self.model,
        )
        try:
            result = client.generate_mesh(mesh_input)
        except BaseException as e:  # noqa: BLE001
            self.report({"ERROR"}, grpc_error_message(e))
            return {"CANCELLED"}
        try:
            _import_generated_mesh(result.glb_data, self.prompt)
        except Exception as e:  # noqa: BLE001
            self.report({"ERROR"}, f"Failed to import generated mesh: {e}")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Text to Mesh: {result.face_count} faces")
        return {"FINISHED"}

    def modal(self, context, event):
        if event.type == "ESC":
            self._cleanup(context)
            self.report({"INFO"}, "Text to Mesh cancelled")
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
            _import_generated_mesh(payload.glb_data, self._resolved_prompt)
        except Exception as e:  # noqa: BLE001
            self.report({"ERROR"}, f"Failed to import generated mesh: {e}")
            return {"CANCELLED"}

        self.report({"INFO"}, f"Text to Mesh: {payload.face_count} faces")
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


def _import_generated_mesh(glb_bytes: bytes, prompt: str) -> None:
    if not glb_bytes:
        raise ValueError("Server returned empty mesh GLB")

    glb_path = write_temp_glb(glb_bytes, prefix="thyllore_mesh_")
    try:
        bpy.ops.import_scene.gltf(filepath=str(glb_path))
        new_meshes = [
            o for o in bpy.context.selected_objects if o.type == "MESH"
        ]
        prompt_label = prompt[:30].strip() or "Generated"
        for mesh_obj in new_meshes:
            mesh_obj.location = (0.0, 0.0, 0.0)
            mesh_obj.name = f"Generated_{prompt_label}"
    finally:
        Path(glb_path).unlink(missing_ok=True)
