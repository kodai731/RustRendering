"""Re-export a .blend to USDC with faithful per-material diffuse colour.

Blender's USD exporter only converts a material to UsdPreviewSurface when its
Material Output Surface resolves (near-)directly to a Principled BSDF. Stylized
materials whose surface is a Mix Shader or nested node group (skin, head, pins,
fannypack, ...) export as empty Material stubs, so the renderer shows them white.
Folds-based materials (denim, shorts, boots) also lose their colour because their
diffuse is an RGB tint multiplied by a greyscale detail map, and only the detail
map survives a naive export.

This script bakes each material's Base Colour (Cycles DIFFUSE colour pass, the
albedo with no lighting) into a per-material texture, rebuilds the material as a
simple Image-Texture -> Principled BSDF, then exports. Baking captures the exact
combined albedo (tint x detail map, mixed shaders, packed textures) uniformly.

Usage:
    blender --background <input.blend> --python scripts/blender_usd_export.py -- <output.usdc> [bake_size]

`bake_size` defaults to 1024.
"""

import bpy
import sys


def parse_args():
    argv = sys.argv
    rest = argv[argv.index("--") + 1:] if "--" in argv else argv[-1:]
    output_path = rest[0]
    bake_size = int(rest[1]) if len(rest) > 1 else 1024
    return output_path, bake_size


def configure_albedo_bake(scene):
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 1
    scene.render.bake.use_pass_direct = False
    scene.render.bake.use_pass_indirect = False
    scene.render.bake.use_pass_color = True
    scene.cycles.bake_type = 'DIFFUSE'


def mesh_objects_with_uv():
    objects = []
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH' or not obj.data.materials:
            continue
        if not obj.data.uv_layers:
            print(f"[usd_export] skip (no UV): {obj.name}")
            continue
        objects.append(obj)
    return objects


def attach_bake_targets(obj, bake_size):
    """Give every material slot a fresh baked image as its active node."""
    targets = {}
    for slot in obj.material_slots:
        material = slot.material
        if material is None or not material.use_nodes:
            continue
        image = bpy.data.images.new(f"albedo_{obj.name}_{material.name}", bake_size, bake_size, alpha=False)
        node_tree = material.node_tree
        tex = node_tree.nodes.new("ShaderNodeTexImage")
        tex.image = image
        node_tree.nodes.active = tex
        targets[material] = image
    return targets


def rebuild_material(material, image):
    material.use_nodes = True
    node_tree = material.node_tree
    node_tree.nodes.clear()
    output = node_tree.nodes.new("ShaderNodeOutputMaterial")
    principled = node_tree.nodes.new("ShaderNodeBsdfPrincipled")
    tex = node_tree.nodes.new("ShaderNodeTexImage")
    tex.image = image
    node_tree.links.new(tex.outputs["Color"], principled.inputs["Base Color"])
    node_tree.links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    image.pack()


def make_bakeable(obj):
    obj.hide_set(False)
    obj.hide_viewport = False
    obj.hide_render = False
    for o in bpy.context.scene.objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def bake_object(obj, bake_size, done):
    make_bakeable(obj)

    targets = attach_bake_targets(obj, bake_size)
    if not targets:
        return
    try:
        bpy.ops.object.bake(type='DIFFUSE')
    except RuntimeError as error:
        print(f"[usd_export] BAKE FAILED {obj.name}: {error}")
        return

    for material, image in targets.items():
        rebuild_material(material, image)
        done.add(material.name)
    print(f"[usd_export] baked {obj.name}: {sorted(t.name for t in targets)}")


def separate_by_material(obj):
    """Split a multi-material mesh into one object per material so each exported
    USD mesh carries a single mesh-level material binding (no GeomSubsets). The
    importer reads one material per mesh, keeping its geometry path untouched."""
    if len(obj.material_slots) <= 1:
        return
    make_bakeable(obj)
    try:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.separate(type='MATERIAL')
        bpy.ops.object.mode_set(mode='OBJECT')
    except RuntimeError as error:
        print(f"[usd_export] SEPARATE FAILED {obj.name}: {error}")


def drop_unused_material_slots(obj):
    make_bakeable(obj)
    try:
        bpy.ops.object.material_slot_remove_unused()
    except RuntimeError:
        pass


def main():
    output_path, bake_size = parse_args()
    configure_albedo_bake(bpy.context.scene)

    for obj in list(mesh_objects_with_uv()):
        separate_by_material(obj)
    for obj in mesh_objects_with_uv():
        drop_unused_material_slots(obj)

    done = set()
    for obj in mesh_objects_with_uv():
        bake_object(obj, bake_size, done)

    print(f"[usd_export] baked {len(done)} materials at {bake_size}px")

    bpy.ops.wm.usd_export(
        filepath=output_path,
        export_materials=True,
        generate_preview_surface=True,
        export_textures=True,
        overwrite_textures=True,
        relative_paths=True,
        root_prim_path="/root",
    )
    print(f"[usd_export] exported {output_path}")


main()
