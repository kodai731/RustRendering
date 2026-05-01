"""N-Panel UI for the Thyllore Animation addon.

Lives in the View3D > Sidebar (N-Panel) under category "Thyllore Animation".
Four sub-panels (one per operator) plus a Placeholder shown when license
verification or ABI check fails.
"""
from __future__ import annotations

import bpy
from bpy.types import Panel

_CATEGORY = "Thyllore Animation"


# ---------------------------------------------------------------------------
# Main sub-panels (registered when license + ABI checks pass).
# ---------------------------------------------------------------------------


class THYLLORE_PT_AutoRig(Panel):
    bl_label = "Auto Rig"
    bl_idname = "THYLLORE_PT_auto_rig"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = _CATEGORY

    def draw(self, context):
        layout = self.layout
        obj = context.active_object

        if obj is None or obj.type != "MESH":
            layout.label(text="Select a mesh", icon="ERROR")
            return

        layout.label(text=f"Mesh: {obj.name}")
        layout.label(text=f"Vertices: {len(obj.data.vertices)}")
        layout.operator(
            "thyllore.auto_rig",
            text="Generate Rig (UniRig)",
            icon="ARMATURE_DATA",
        )


class THYLLORE_PT_TextToMesh(Panel):
    bl_label = "Text to Mesh"
    bl_idname = "THYLLORE_PT_text_to_mesh"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = _CATEGORY

    def draw(self, context):
        layout = self.layout
        layout.operator("thyllore.text_to_mesh", icon="MESH_MONKEY")


class THYLLORE_PT_TextToMotion(Panel):
    bl_label = "Text to Motion"
    bl_idname = "THYLLORE_PT_text_to_motion"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = _CATEGORY

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        if obj is None or obj.type != "ARMATURE":
            layout.label(text="Select an armature", icon="ERROR")
            return
        layout.operator("thyllore.text_to_motion", icon="POSE_HLT")


class THYLLORE_PT_CurveCopilot(Panel):
    bl_label = "Curve Copilot"
    bl_idname = "THYLLORE_PT_curve_copilot"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = _CATEGORY

    def draw(self, context):
        layout = self.layout
        if context.active_object is None or context.active_object.type != "ARMATURE":
            layout.label(text="Select an armature with FCurves", icon="ERROR")
            return
        layout.operator("thyllore.curve_copilot", icon="FCURVE")


# ---------------------------------------------------------------------------
# Placeholder sub-panel (shown only when license or ABI verification fails).
# ---------------------------------------------------------------------------


class THYLLORE_PT_PlaceholderLicenseFailure(Panel):
    bl_label = "Activation Required"
    bl_idname = "THYLLORE_PT_placeholder"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = _CATEGORY

    _error_message: str = "License or ABI verification failed"

    def draw(self, context):
        layout = self.layout
        layout.label(text="Thyllore Animation is not activated", icon="LOCKED")
        layout.label(text=self._error_message)
        op = layout.operator(
            "preferences.addon_show",
            text="Open Preferences",
        )
        # __package__ here is "blender_addon.panels"; the addon module name
        # (one level up) is what AddonPreferences.bl_idname uses.
        op.module = (__package__ or "blender_addon.panels").rsplit(".", 1)[0]


_TIER_A_PANELS = (
    THYLLORE_PT_AutoRig,
    THYLLORE_PT_TextToMesh,
)


def _resolve_main_panels() -> tuple[type[Panel], ...]:
    """Return the panel classes registered for the current Variant.

    Each panel is only registered when its backing operator was successfully
    imported. Phase 5.5 lite Variant ships Curve Copilot only until the
    text_to_motion PyO3 rewrite lands, after which both Tier B panels become
    available regardless of Tier A inclusion.
    """
    from .. import operators

    panels: list[type[Panel]] = []
    if operators.has_tier_a():
        panels.extend(_TIER_A_PANELS)
    if operators.has_text_to_motion():
        panels.append(THYLLORE_PT_TextToMotion)
    panels.append(THYLLORE_PT_CurveCopilot)
    return tuple(panels)


_MAIN_PANELS: tuple[type[Panel], ...] = ()


def register() -> None:
    global _MAIN_PANELS
    _MAIN_PANELS = _resolve_main_panels()
    for cls in _MAIN_PANELS:
        bpy.utils.register_class(cls)


def register_placeholder(error_message: str) -> None:
    THYLLORE_PT_PlaceholderLicenseFailure._error_message = error_message
    bpy.utils.register_class(THYLLORE_PT_PlaceholderLicenseFailure)


def unregister() -> None:
    classes = (THYLLORE_PT_PlaceholderLicenseFailure, *reversed(_MAIN_PANELS))
    for cls in classes:
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass


