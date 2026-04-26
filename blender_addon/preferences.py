"""Addon preferences UI and accessor.

Stores the gRPC server endpoint, license key, device id, and Tier B
(Curve Copilot) configuration. Read by Operators via :func:`get_preferences`.
"""
from __future__ import annotations

import uuid

import bpy
from bpy.props import BoolProperty, FloatProperty, IntProperty, StringProperty
from bpy.types import AddonPreferences, Operator

# Resolve the addon's package name so AddonPreferences.bl_idname matches the
# top-level extension id (legacy install: "blender_addon"; extension install:
# "thyllore_animation"). __package__ is the parent package.
ADDON_PACKAGE = __package__ or "blender_addon"


class ThylloreAnimationPreferences(AddonPreferences):
    bl_idname = ADDON_PACKAGE

    server_host: StringProperty(  # type: ignore[valid-type]
        name="Server Host",
        default="127.0.0.1",
        description="gRPC server hostname or IP",
    )
    server_port: IntProperty(  # type: ignore[valid-type]
        name="Server Port",
        default=50051,
        min=1,
        max=65535,
    )
    use_tls: BoolProperty(  # type: ignore[valid-type]
        name="Use TLS",
        default=False,
        description="Enable TLS for gRPC connection (required for Thyllore Cloud)",
    )
    deadline_seconds: FloatProperty(  # type: ignore[valid-type]
        name="Deadline (s)",
        default=120.0,
        min=1.0,
        max=600.0,
    )

    license_key: StringProperty(  # type: ignore[valid-type]
        name="License Key",
        default="",
        subtype="PASSWORD",
        description="License token issued by Thyllore Cloud (Polar.sh)",
    )
    device_id: StringProperty(  # type: ignore[valid-type]
        name="Device ID",
        default="",
        description="Unique device identifier (auto-generated on first install)",
    )

    enable_curve_copilot: BoolProperty(  # type: ignore[valid-type]
        name="Enable Curve Copilot",
        default=True,
        description="Enable in-process ONNX inference for FCurve suggestions (Tier B)",
    )
    curve_copilot_model_path: StringProperty(  # type: ignore[valid-type]
        name="Curve Copilot Model Path",
        default="",
        subtype="FILE_PATH",
        description="Path to curve_copilot.onnx (leave empty to use bundled model)",
    )

    def draw(self, context):
        layout = self.layout

        box = layout.box()
        box.label(text="Server (Tier A - gRPC)", icon="URL")
        box.prop(self, "server_host")
        box.prop(self, "server_port")
        box.prop(self, "use_tls")
        box.prop(self, "deadline_seconds")

        box = layout.box()
        box.label(text="License", icon="LOCKED")
        box.prop(self, "license_key")
        row = box.row()
        row.label(text=f"Device: {self.device_id or '(not set)'}")
        row.operator("thyllore.regenerate_device_id", text="Regenerate")

        box = layout.box()
        box.label(text="Curve Copilot (Tier B - PyO3)", icon="FCURVE")
        box.prop(self, "enable_curve_copilot")
        box.prop(self, "curve_copilot_model_path")


class THYLLORE_OT_RegenerateDeviceID(Operator):
    bl_idname = "thyllore.regenerate_device_id"
    bl_label = "Regenerate Device ID"
    bl_description = "Generate a new device identifier (forces re-authentication)"
    bl_options = {"INTERNAL"}

    def execute(self, context):
        prefs = get_preferences()
        prefs.device_id = str(uuid.uuid4())
        self.report({"INFO"}, f"Device ID regenerated: {prefs.device_id}")
        return {"FINISHED"}


_CLASSES = (ThylloreAnimationPreferences, THYLLORE_OT_RegenerateDeviceID)


def register() -> None:
    for cls in _CLASSES:
        bpy.utils.register_class(cls)

    prefs = get_preferences()
    if not prefs.device_id:
        prefs.device_id = str(uuid.uuid4())


def unregister() -> None:
    for cls in reversed(_CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass


def get_preferences() -> ThylloreAnimationPreferences:
    """Return the addon's AddonPreferences instance.

    Operators use this to read server/license configuration without binding
    to any specific Blender layout.
    """
    return bpy.context.preferences.addons[ADDON_PACKAGE].preferences
