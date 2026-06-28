import bpy
import json
import sys
import os
import math
import mathutils


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


def setup_lighting(camera_pos, target):
    distance = (mathutils.Vector(camera_pos) - mathutils.Vector(target)).length
    light_pos = mathutils.Vector(target) + mathutils.Vector((distance, -distance, distance))

    light_data = bpy.data.lights.new("KeySun", type='SUN')
    light_data.energy = 4.0
    light_obj = bpy.data.objects.new("KeySun", light_data)
    bpy.context.scene.collection.objects.link(light_obj)
    light_obj.location = light_pos
    light_obj.rotation_euler = look_at_rotation(light_pos, target)

    world = bpy.context.scene.world or bpy.data.worlds.new("CompareWorld")
    bpy.context.scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get('Background')
    if background:
        background.inputs['Color'].default_value = (0.05, 0.05, 0.06, 1.0)
        background.inputs['Strength'].default_value = 1.0


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
    scene.render.use_stamp = False
    scene.frame_set(frame)


def main():
    argv = sys.argv
    separator_index = argv.index('--') if '--' in argv else -1
    if separator_index < 0 or separator_index + 2 >= len(argv):
        print("Usage: blender --background <input.blend> --python script.py -- "
              "<output_png> <config_json>")
        sys.exit(1)

    output_png = os.path.abspath(argv[separator_index + 1])
    config = json.loads(argv[separator_index + 2])

    camera_pos = tuple(config["camera_pos"])
    camera_target = tuple(config["camera_target"])
    setup_camera(camera_pos, camera_target, config["fov_deg"])
    setup_lighting(camera_pos, camera_target)
    setup_render(config["resolution"], config["frame"])

    bpy.context.scene.render.filepath = output_png
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.ops.render.render(write_still=True)

    print(f"Render written to: {output_png}")


if __name__ == '__main__':
    main()
