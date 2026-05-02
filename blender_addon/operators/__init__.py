"""Operator registration collation.

Phase 5.5 MVP variant gating:

- ``curve_copilot`` (Tier B PyO3) is the only operator guaranteed to be
  present in every Variant. It depends only on ``thyllore_ml_core`` and never
  imports gRPC.
- ``text_to_motion`` is currently the gRPC-backed Phase 4 implementation. It
  will be rewritten as a Tier B PyO3 operator once the ``light_t2m`` ONNX
  ships (see phase5_5_onnx/Phase5_5_TextToMotionMlCore.md). Until then it
  shares the Tier A fate -- excluded from the ``-Variant lite`` ZIP because
  ``grpc_client`` is not bundled.
- ``auto_rig`` and ``text_to_mesh`` (Tier A) are always excluded from the
  ``-Variant lite`` ZIP and stay only in the full Variant.

Each operator import is guarded independently with ``ImportError``, so a
build that omits any subset still registers the survivors.
"""
from __future__ import annotations

import bpy

from . import curve_copilot

_OPERATORS: list[type] = [curve_copilot.THYLLORE_OT_CurveCopilot]
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


def register() -> None:
    for cls in _OPERATORS:
        bpy.utils.register_class(cls)


def unregister() -> None:
    for cls in reversed(_OPERATORS):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
