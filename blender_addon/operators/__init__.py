"""Operator registration collation."""
from __future__ import annotations

import bpy

from . import auto_rig, curve_copilot, text_to_mesh, text_to_motion

_OPERATORS = (
    auto_rig.THYLLORE_OT_AutoRig,
    text_to_mesh.THYLLORE_OT_TextToMesh,
    text_to_motion.THYLLORE_OT_TextToMotion,
    curve_copilot.THYLLORE_OT_CurveCopilot,
)


def register() -> None:
    for cls in _OPERATORS:
        bpy.utils.register_class(cls)


def unregister() -> None:
    for cls in reversed(_OPERATORS):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
