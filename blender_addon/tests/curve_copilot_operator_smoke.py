"""curve_copilot Operator headless smoke.

Boots Blender 4.2 LTS in --background mode with the addon enabled, builds a
minimal armature + FCurve programmatically, then drives
``bpy.ops.thyllore.curve_copilot`` synchronously through the Operator path.
Asserts that the operator returns ``FINISHED`` (or a documented graceful
``CANCELLED`` for a missing wheel) and writes a status file the Rust shim
verifies.

The script is invoked with::

    blender --background --factory-startup --python <this script> -- \\
        --result <result.json> --onnx <absolute onnx path>

``--onnx`` is the deterministic curve_copilot.onnx copied into the parity
fixture root from ``${SharedDataPath}/exports``. The wheel is expected to
already be on ``sys.path`` because the addon's
``_bootstrap.insert_wheels_to_sys_path()`` runs during ``register()``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CONTEXT_KEYFRAME_COUNT = 32
ARMATURE_OBJECT_NAME = "CurveCopilotSmokeArmature"
ARMATURE_DATA_NAME = "CurveCopilotSmokeArmatureData"
ROOT_BONE_NAME = "curve_copilot_smoke_root"
INITIAL_FRAME_COUNT = 36


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--onnx", required=True)
    return parser.parse_args(argv)


def build_minimal_armature():
    import bpy

    armature_data = bpy.data.armatures.new(ARMATURE_DATA_NAME)
    armature_object = bpy.data.objects.new(ARMATURE_OBJECT_NAME, armature_data)
    bpy.context.collection.objects.link(armature_object)
    bpy.context.view_layer.objects.active = armature_object

    bpy.ops.object.mode_set(mode="EDIT")
    edit_bone = armature_data.edit_bones.new(ROOT_BONE_NAME)
    edit_bone.head = (0.0, 0.0, 0.0)
    edit_bone.tail = (0.0, 0.0, 0.5)
    bpy.ops.object.mode_set(mode="POSE")

    pose_bone = armature_object.pose.bones[ROOT_BONE_NAME]
    pose_bone.rotation_mode = "QUATERNION"
    return armature_object


def insert_initial_keyframes(armature_object):
    import bpy

    pose_bone = armature_object.pose.bones[ROOT_BONE_NAME]
    for i in range(INITIAL_FRAME_COUNT):
        frame = i + 1
        bpy.context.scene.frame_set(frame)
        pose_bone.location = (
            float(i) * 0.05,
            float(i) * 0.025,
            float(i) * 0.075,
        )
        pose_bone.keyframe_insert(data_path="location", frame=frame)


def iter_action_fcurves(obj):
    """Enumerate FCurves across Blender's legacy and slotted action APIs."""
    anim = obj.animation_data
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


def select_first_location_x_fcurve(armature_object):
    target_data_path = f'pose.bones["{ROOT_BONE_NAME}"].location'
    for fcurve in iter_action_fcurves(armature_object):
        if fcurve.data_path == target_data_path and fcurve.array_index == 0:
            return fcurve
    raise RuntimeError("curve_copilot smoke: location.x FCurve not found")


def enable_addon():
    """Enable the thyllore_animation add-on regardless of install style.

    Returns the resolved package name so callers can index
    ``bpy.context.preferences.addons[...]``.
    """
    import bpy
    import addon_utils

    candidates = [
        "bl_ext.user_default.thyllore_animation_lite",
        "thyllore_animation_lite",
        "bl_ext.user_default.thyllore_animation",
        "thyllore_animation",
    ]
    for module_name in candidates:
        loaded_default, loaded_state = addon_utils.check(module_name)
        if loaded_default or loaded_state:
            return module_name
        try:
            bpy.ops.preferences.addon_enable(module=module_name)
            return module_name
        except RuntimeError:
            continue
    raise RuntimeError(
        "thyllore_animation add-on is not installed; tried " + ", ".join(candidates)
    )


def configure_preferences(addon_pkg, onnx_path):
    import bpy

    prefs = bpy.context.preferences.addons[addon_pkg].preferences
    prefs.curve_copilot_model_path = onnx_path
    prefs.enable_curve_copilot = True


def run_curve_copilot_operator(armature_object, fcurve):
    import bpy

    pre_count = len(fcurve.keyframe_points)
    operator_status = bpy.ops.thyllore.curve_copilot("EXEC_DEFAULT")
    post_count = len(fcurve.keyframe_points)
    return operator_status, pre_count, post_count


def ghost_is_shown(addon_pkg):
    """Curve Copilot is preview-only: the forecast must populate the ghost overlay."""
    import importlib

    overlay = importlib.import_module(f"{addon_pkg}._ghost_overlay")
    return overlay.has_ghost()


def write_result(path, payload):
    Path(path).write_bytes((json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8"))


def main():
    args = parse_args()

    onnx_path = Path(args.onnx).resolve()
    if not onnx_path.exists():
        write_result(
            args.result,
            {
                "ok": False,
                "stage": "onnx_check",
                "reason": f"onnx not found: {onnx_path}",
            },
        )
        sys.exit(2)

    import bpy

    addon_pkg = enable_addon()

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    armature_object = build_minimal_armature()
    insert_initial_keyframes(armature_object)
    fcurve = select_first_location_x_fcurve(armature_object)
    configure_preferences(addon_pkg, str(onnx_path).replace("\\", "/"))

    try:
        operator_status, pre_count, post_count = run_curve_copilot_operator(
            armature_object, fcurve
        )
    except BaseException as exc:  # noqa: BLE001
        write_result(
            args.result,
            {
                "ok": False,
                "stage": "operator_exception",
                "exception": repr(exc),
            },
        )
        raise

    ghost_shown = ghost_is_shown(addon_pkg)
    payload = {
        "ok": ("FINISHED" in operator_status) and ghost_shown and (post_count == pre_count),
        "operator_status": sorted(operator_status),
        "stage": "operator_result",
        "pre_keyframe_count": pre_count,
        "post_keyframe_count": post_count,
        "inserted_count": post_count - pre_count,
        "ghost_shown": ghost_shown,
    }
    write_result(args.result, payload)

    if not payload["ok"]:
        print(f"curve_copilot smoke: operator status was {sorted(operator_status)}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
