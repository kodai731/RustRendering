import bpy
import json
import sys
import os


MAX_SAMPLES_PER_FRAME = 3000


def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for collection in bpy.data.collections:
        bpy.data.collections.remove(collection)


def import_usd(model_path):
    bpy.ops.wm.usd_import(filepath=model_path)


def mesh_has_armature(obj):
    return any(modifier.type == 'ARMATURE' for modifier in obj.modifiers)


def gather_skinned_vertices(frame):
    bpy.context.scene.frame_set(frame)
    depsgraph = bpy.context.evaluated_depsgraph_get()

    positions = []
    for obj in depsgraph.objects:
        if obj.type != 'MESH' or not mesh_has_armature(obj):
            continue

        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        if mesh is None:
            continue

        world_matrix = eval_obj.matrix_world
        for vert in mesh.vertices:
            world_pos = world_matrix @ vert.co
            positions.append((world_pos.x, world_pos.y, world_pos.z))
        eval_obj.to_mesh_clear()

    return positions


def subsample(positions, max_count):
    if len(positions) <= max_count:
        return positions
    stride = len(positions) // max_count
    return positions[::stride][:max_count]


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
        print("Usage: blender --background --python script.py -- "
              "<model_path> <output_json_path> [frame1,frame2,...]")
        sys.exit(1)

    model_path = os.path.abspath(argv[separator_index + 1])
    output_path = os.path.abspath(argv[separator_index + 2])

    clear_scene()
    import_usd(model_path)

    frame_start, frame_end = get_animation_range()
    fps = bpy.context.scene.render.fps
    frames = parse_frames(argv, separator_index, frame_start, frame_end)

    frames_data = []
    for frame in frames:
        positions = subsample(gather_skinned_vertices(frame), MAX_SAMPLES_PER_FRAME)
        frames_data.append({
            "frame": frame,
            "vertices": [[round(c, 6) for c in p] for p in positions],
        })

    result = {
        "model_path": model_path,
        "fps": fps,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "frames": frames_data,
    }

    with open(output_path, 'w') as f:
        json.dump(result, f)

    print(f"USD vertices written to: {output_path}")


if __name__ == '__main__':
    main()
