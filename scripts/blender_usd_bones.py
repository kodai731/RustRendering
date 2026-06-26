"""Exports per-frame bone world positions from a USD asset as ground truth.

Mirrors blender_usd_vertices.py but dumps the armature's pose-bone world head
positions (name-keyed) so the engine's bone gizmo math can be compared without
launching the app.

Usage: blender --background --python blender_usd_bones.py -- \
    <model_path> <output_json_path> [frame1,frame2,...]
"""

import bpy
import json
import sys
import os


def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for collection in bpy.data.collections:
        bpy.data.collections.remove(collection)


def import_usd(model_path):
    bpy.ops.wm.usd_import(filepath=model_path)


def pick_main_armature():
    armatures = [o for o in bpy.data.objects if o.type == 'ARMATURE']
    if not armatures:
        return None
    return max(armatures, key=lambda a: len(a.data.bones))


def gather_bone_heads(armature, frame):
    bpy.context.scene.frame_set(frame)
    world = armature.matrix_world
    names = []
    heads = []
    for pose_bone in armature.pose.bones:
        world_head = world @ pose_bone.head
        names.append(pose_bone.name)
        heads.append((world_head.x, world_head.y, world_head.z))
    return names, heads


def get_animation_range():
    scene = bpy.context.scene
    return scene.frame_start, scene.frame_end


def parse_frames(argv, separator_index, frame_start, frame_end):
    if separator_index + 3 < len(argv):
        return [int(f) for f in argv[separator_index + 3].split(',')]
    return [frame_start, (frame_start + frame_end) // 2]


def main():
    argv = sys.argv
    separator_index = argv.index('--') if '--' in argv else -1
    if separator_index < 0 or separator_index + 2 >= len(argv):
        print("Usage: blender --background --python blender_usd_bones.py -- "
              "<model_path> <output_json_path> [frame1,frame2,...]")
        sys.exit(1)

    model_path = os.path.abspath(argv[separator_index + 1])
    output_path = os.path.abspath(argv[separator_index + 2])

    clear_scene()
    import_usd(model_path)

    armature = pick_main_armature()
    if armature is None:
        print("No armature found in USD")
        sys.exit(1)

    frame_start, frame_end = get_animation_range()
    fps = bpy.context.scene.render.fps
    frames = parse_frames(argv, separator_index, frame_start, frame_end)

    bone_names = None
    frames_data = []
    for frame in frames:
        names, heads = gather_bone_heads(armature, frame)
        if bone_names is None:
            bone_names = names
        frames_data.append({
            "frame": frame,
            "heads": [[round(c, 6) for c in h] for h in heads],
        })

    result = {
        "model_path": model_path,
        "armature": armature.name,
        "fps": fps,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "bone_names": bone_names,
        "frames": frames_data,
    }

    with open(output_path, 'w') as f:
        json.dump(result, f)

    print(f"USD bone positions written to: {output_path}")


if __name__ == '__main__':
    main()
