import bpy
import json
import sys
import os
import math
import mathutils


def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for collection in bpy.data.collections:
        bpy.data.collections.remove(collection)


def import_usd(model_path):
    bpy.ops.wm.usd_import(filepath=model_path)


def hide_non_skinned_meshes():
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH':
            continue
        has_armature = any(m.type == 'ARMATURE' for m in obj.modifiers)
        obj.hide_render = not has_armature


def look_at_rotation(camera_pos, target):
    direction = mathutils.Vector(target) - mathutils.Vector(camera_pos)
    return direction.to_track_quat('-Z', 'Y').to_euler()


def setup_camera(camera_pos, target, fov_deg):
    cam_data = bpy.data.cameras.new("CompareCamera")
    cam_data.lens_unit = 'FOV'
    cam_data.angle = math.radians(fov_deg)
    cam_obj = bpy.data.objects.new("CompareCamera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    cam_obj.location = camera_pos
    cam_obj.rotation_euler = look_at_rotation(camera_pos, target)
    bpy.context.scene.camera = cam_obj


def set_eevee_engine(scene):
    for name in ('BLENDER_EEVEE_NEXT', 'BLENDER_EEVEE'):
        try:
            scene.render.engine = name
            return
        except TypeError:
            continue


def setup_render(resolution, frame):
    scene = bpy.context.scene
    set_eevee_engine(scene)
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.frame_set(frame)


def parse_vec3(text):
    return tuple(float(v) for v in text.split(','))


def main():
    argv = sys.argv
    separator_index = argv.index('--') if '--' in argv else -1
    if separator_index < 0 or separator_index + 2 >= len(argv):
        print("Usage: blender --background --python script.py -- "
              "<model_path> <output_png> <config_json>")
        sys.exit(1)

    model_path = os.path.abspath(argv[separator_index + 1])
    output_png = os.path.abspath(argv[separator_index + 2])
    config = json.loads(argv[separator_index + 3])

    clear_scene()
    import_usd(model_path)
    hide_non_skinned_meshes()

    setup_camera(
        tuple(config["camera_pos"]),
        tuple(config["camera_target"]),
        config["fov_deg"],
    )
    setup_render(config["resolution"], config["frame"])

    bpy.context.scene.render.filepath = output_png
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.ops.render.render(write_still=True)

    print(f"Render written to: {output_png}")


if __name__ == '__main__':
    main()
