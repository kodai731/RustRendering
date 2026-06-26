"""Ground-truth Blender screenshots for comparing against the engine bone gizmo.

Works headless (EEVEE, no OpenGL context needed). Produces two PNGs from a
front view (-Y, Z-up):
  <out>_mesh.png  : the character mesh (ground-truth pose/position)
  <out>_bones.png : pose-bone head/tail joints as bright emission spheres
                    (mesh hidden), i.e. the ground-truth skeleton positions

Usage: blender -b <blend> --python blender_bone_screenshot.py -- \
    <out_prefix_png> [resolution] [frame]
"""

import bpy
import bmesh
import sys
import os
from mathutils import Vector


def pick_main_armature():
    armatures = [o for o in bpy.data.objects if o.type == 'ARMATURE']
    return max(armatures, key=lambda a: len(a.data.bones)) if armatures else None


def mesh_objects():
    return [o for o in bpy.context.scene.objects if o.type == 'MESH']


def armature_bounds(armature):
    lo = Vector((1e18, 1e18, 1e18))
    hi = Vector((-1e18, -1e18, -1e18))
    world = armature.matrix_world
    found = False
    for pose_bone in armature.pose.bones:
        for point in (pose_bone.head, pose_bone.tail):
            wp = world @ point
            found = True
            for i in range(3):
                lo[i] = min(lo[i], wp[i])
                hi[i] = max(hi[i], wp[i])
    if not found:
        return Vector((0, 0, 0)), Vector((0, 0, 2))
    return lo, hi


def setup_front_camera(resolution, lo, hi):
    center = (lo + hi) * 0.5
    size = max((hi - lo).x, (hi - lo).y, (hi - lo).z, 0.5)

    cam_data = bpy.data.cameras.new("BoneCompareCam")
    cam_obj = bpy.data.objects.new("BoneCompareCam", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)

    distance = size * 2.0
    cam_obj.location = center + Vector((0.0, -distance, 0.0))
    direction = center - cam_obj.location
    cam_obj.rotation_euler = direction.to_track_quat('-Z', 'Z').to_euler()
    bpy.context.scene.camera = cam_obj

    scene = bpy.context.scene
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.use_stamp = False
    for name in ('BLENDER_EEVEE_NEXT', 'BLENDER_EEVEE'):
        try:
            scene.render.engine = name
            break
        except TypeError:
            continue
    add_fill_light(center, distance)
    return size, center, distance


def add_fill_light(center, distance):
    light_data = bpy.data.lights.new("FillSun", type='SUN')
    light_data.energy = 4.0
    light_obj = bpy.data.objects.new("FillSun", light_data)
    bpy.context.scene.collection.objects.link(light_obj)
    light_obj.location = center + Vector((distance, -distance, distance))
    direction = center - light_obj.location
    light_obj.rotation_euler = direction.to_track_quat('-Z', 'Z').to_euler()
    world = bpy.context.scene.world or bpy.data.worlds.new("W")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get('Background')
    if bg:
        bg.inputs['Color'].default_value = (0.05, 0.05, 0.06, 1.0)


def emission_material(name, color):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    emit = nodes.new('ShaderNodeEmission')
    emit.inputs['Color'].default_value = (*color, 1.0)
    emit.inputs['Strength'].default_value = 5.0
    out = nodes.new('ShaderNodeOutputMaterial')
    mat.node_tree.links.new(emit.outputs['Emission'], out.inputs['Surface'])
    return mat


def build_bone_markers(armature, radius):
    bm = bmesh.new()
    world = armature.matrix_world
    for pose_bone in armature.pose.bones:
        for point in (pose_bone.head, pose_bone.tail):
            wp = world @ point
            sphere = bmesh.new()
            bmesh.ops.create_icosphere(sphere, subdivisions=1, radius=radius)
            bmesh.ops.translate(sphere, verts=sphere.verts, vec=wp)
            bm.from_mesh(_to_mesh(sphere))
            sphere.free()

    mesh = bpy.data.meshes.new("BoneMarkers")
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new("BoneMarkers", mesh)
    obj.data.materials.append(emission_material("BoneEmission", (1.0, 0.1, 0.1)))
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _to_mesh(bm):
    mesh = bpy.data.meshes.new("tmp")
    bm.to_mesh(mesh)
    return mesh


def render_to(path):
    bpy.context.scene.render.filepath = os.path.abspath(path)
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.ops.render.render(write_still=True)


def main():
    argv = sys.argv
    sep = argv.index('--') if '--' in argv else -1
    if sep < 0 or sep + 1 >= len(argv):
        print("Usage: blender -b <blend> --python script.py -- <out_prefix.png> [res] [frame]")
        sys.exit(1)

    out_prefix = argv[sep + 1]
    resolution = int(argv[sep + 2]) if sep + 2 < len(argv) else 512
    base, ext = os.path.splitext(out_prefix)
    ext = ext or '.png'

    armature = pick_main_armature()
    frame = int(argv[sep + 3]) if sep + 3 < len(argv) else bpy.context.scene.frame_start
    bpy.context.scene.frame_set(frame)

    if armature is None:
        print("No armature found; cannot frame character")
        sys.exit(1)

    lo, hi = armature_bounds(armature)
    size, center, distance = setup_front_camera(resolution, lo, hi)

    # Pass 1: mesh only (ground-truth character pose)
    render_to(f"{base}_mesh{ext}")
    print(f"Mesh screenshot written to: {base}_mesh{ext}")

    # Pass 2: bones only (hide meshes, show emission joints)
    if True:
        for obj in mesh_objects():
            obj.hide_render = True
        build_bone_markers(armature, radius=size * 0.02)
        render_to(f"{base}_bones{ext}")
        print(f"Bones screenshot written to: {base}_bones{ext}")
    else:
        print("No armature found; skipped bones pass")


if __name__ == '__main__':
    main()
