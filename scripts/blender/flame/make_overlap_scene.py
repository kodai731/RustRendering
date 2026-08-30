import sys

import bpy
from mathutils import Vector

FLAME_LOCATION = Vector((0.0, 0.0, 0.0))


def clear_meshes():
    for obj in list(bpy.data.objects):
        if obj.type == "MESH":
            bpy.data.objects.remove(obj, do_unlink=True)


def aim_camera():
    camera = bpy.data.objects["Camera"]
    camera.location = (0.0, -6.0, 1.6)
    target = Vector((0.0, 0.0, 0.8))
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()


def add_flame():
    bpy.ops.thyllore.flame_add()
    flame = bpy.context.active_object
    flame.location = FLAME_LOCATION
    return flame


def add_overlapping_props():
    bpy.ops.mesh.primitive_cube_add(size=0.6, location=(0.3, -0.4, 0.5))
    bpy.context.active_object.name = "CubeIntersectingFlame"
    bpy.ops.mesh.primitive_cube_add(size=0.6, location=(-0.9, -1.4, 0.3))
    bpy.context.active_object.name = "CubeInFront"
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, location=(0.0, 1.5, 0.9))
    bpy.context.active_object.name = "SphereBehind"
    bpy.ops.mesh.primitive_plane_add(size=8.0, location=(0.0, 0.0, 0.0))
    bpy.context.active_object.name = "Ground"


def use_camera_view():
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                area.spaces.active.region_3d.view_perspective = "CAMERA"


clear_meshes()
aim_camera()
add_flame()
add_overlapping_props()
use_camera_view()
bpy.context.scene.frame_end = 120
bpy.ops.wm.save_as_mainfile(filepath=sys.argv[sys.argv.index("--") + 1])
print("[flame/make_overlap_scene] saved", [o.name for o in bpy.data.objects], flush=True)
