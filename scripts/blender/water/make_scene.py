import sys
import bpy
from mathutils import Vector

for obj in list(bpy.data.objects):
    if obj.type == "MESH":
        bpy.data.objects.remove(obj, do_unlink=True)

camera = bpy.data.objects["Camera"]
camera.location = (0.0, -6.0, 2.5)
target = Vector((0.0, 0.0, 0.0))
camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()

bpy.ops.thyllore.water_add()
water = bpy.context.active_object
water.location = (0.0, 0.0, 0.0)

for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type == "VIEW_3D":
            area.spaces.active.region_3d.view_perspective = "CAMERA"

bpy.context.scene.frame_end = 120
bpy.ops.wm.save_as_mainfile(filepath=sys.argv[sys.argv.index("--") + 1])
print("[make_water_blend] saved", [o.name for o in bpy.data.objects], flush=True)
