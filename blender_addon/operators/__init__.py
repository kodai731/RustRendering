"""Operator registration collation.

- ``curve_copilot`` (PyO3) is the only operator shipped in the distribution
  ZIP. It depends only on ``thyllore_ml_core`` and never imports gRPC.
- ``text_to_motion``, ``auto_rig`` and ``text_to_mesh`` are unshipped
  gRPC-backed operators, excluded from the ZIP by
  ``scripts/build_blender_addon.sh/.ps1``.

Each operator import is guarded independently with ``ImportError``, so a
build that omits any subset still registers the survivors.
"""
from __future__ import annotations

import bpy

from .. import _ghost_overlay
from . import curve_copilot

_OPERATORS: list[type] = [
    curve_copilot.THYLLORE_OT_CurveCopilot,
    curve_copilot.THYLLORE_OT_CurveCopilotClear,
]
_HAS_TEXT_TO_MOTION = False
_HAS_TIER_A = False

try:
    from . import text_to_motion

    _OPERATORS.append(text_to_motion.THYLLORE_OT_TextToMotion)
    _HAS_TEXT_TO_MOTION = True
except ImportError:
    pass

try:
    from . import auto_rig, text_to_mesh

    _OPERATORS.insert(0, text_to_mesh.THYLLORE_OT_TextToMesh)
    _OPERATORS.insert(0, auto_rig.THYLLORE_OT_AutoRig)
    _HAS_TIER_A = True
except ImportError:
    pass


def has_tier_a() -> bool:
    """Whether Tier A operators (auto_rig / text_to_mesh) were bundled."""
    return _HAS_TIER_A


def has_text_to_motion() -> bool:
    """Whether the text_to_motion operator was bundled."""
    return _HAS_TEXT_TO_MOTION


# Shift+C in animation curve editors, matching the desktop engine's curve
# editor trigger. The 3D Viewport is intentionally excluded: Shift+C there is
# Blender's "Center Cursor and Frame All".
_CURVE_COPILOT_KEYMAPS = (
    ("Graph Editor", "GRAPH_EDITOR"),
    ("Dope Sheet", "DOPESHEET_EDITOR"),
)
_addon_keymaps: list = []


def _register_keymaps() -> None:
    keyconfig = bpy.context.window_manager.keyconfigs.addon
    if keyconfig is None:
        return
    for keymap_name, space_type in _CURVE_COPILOT_KEYMAPS:
        keymap = keyconfig.keymaps.new(name=keymap_name, space_type=space_type)
        keymap_item = keymap.keymap_items.new(
            "thyllore.curve_copilot", type="C", value="PRESS", shift=True
        )
        _addon_keymaps.append((keymap, keymap_item))


def _unregister_keymaps() -> None:
    for keymap, keymap_item in _addon_keymaps:
        try:
            keymap.keymap_items.remove(keymap_item)
        except (RuntimeError, ReferenceError):
            pass
    _addon_keymaps.clear()


def register() -> None:
    for cls in _OPERATORS:
        bpy.utils.register_class(cls)
    _register_keymaps()
    _ghost_overlay.register()


def unregister() -> None:
    _ghost_overlay.unregister()
    _unregister_keymaps()
    for cls in reversed(_OPERATORS):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
