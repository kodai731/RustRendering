import os

from .render import sequence_path


def flame_node_names(obj_name):
    return {
        "rl": "THYLLORE_FLAME_RL",
        "z_viewer": "THYLLORE_FLAME_Z_VIEWER",
        "img": f"THYLLORE_FLAME_IMG_{obj_name}",
        "ao": f"THYLLORE_FLAME_AO_{obj_name}",
        "composite": "THYLLORE_FLAME_COMPOSITE",
    }


def setup_flame_compositor(scene, obj, sequence_dir, frame_start, frame_end):
    import bpy
    scene.use_nodes = True
    scene.view_layers[0].use_pass_z = True
    if scene.compositing_node_group is None:
        tree = bpy.data.node_groups.new("THYLLORE_FLAME_COMPOSITOR", "CompositorNodeTree")
        scene.compositing_node_group = tree
    else:
        tree = scene.compositing_node_group
    names = flame_node_names(obj.name)
    rl = tree.nodes.get(names["rl"])
    if rl is None:
        rl = tree.nodes.new("CompositorNodeRLayers")
        rl.name = names["rl"]
    z_viewer = tree.nodes.get(names["z_viewer"])
    if z_viewer is None:
        z_viewer = tree.nodes.new("CompositorNodeViewer")
        z_viewer.name = names["z_viewer"]
    img_node = tree.nodes.get(names["img"])
    if img_node is None:
        img_node = tree.nodes.new("CompositorNodeImage")
        img_node.name = names["img"]
    exr_path = sequence_path(sequence_dir, obj.name, frame_start)
    image_name = os.path.basename(exr_path)
    image = bpy.data.images.get(image_name)
    if image is None and os.path.exists(exr_path):
        image = bpy.data.images.load(exr_path)
    if image is not None:
        image.name = image_name
        image.source = "SEQUENCE"
        img_node.image = image
        img_node.frame_duration = frame_end - frame_start + 1
        img_node.frame_start = frame_start
        img_node.use_auto_refresh = True
    ao = tree.nodes.get(names["ao"])
    if ao is None:
        ao = tree.nodes.new("CompositorNodeAlphaOver")
        ao.name = names["ao"]
    out = tree.nodes.get(names["composite"])
    if out is None:
        out = tree.nodes.new("NodeGroupOutput")
        out.name = names["composite"]
    if not any(s.name == "Image" and s.in_out == 'OUTPUT' for s in tree.interface.items_tree if hasattr(s, 'in_out')):
        tree.interface.new_socket(name="Image", in_out='OUTPUT', socket_type='NodeSocketColor')
    existing_links = set()
    for link in tree.links:
        existing_links.add((link.from_socket, link.to_socket))
    def _ensure_link(from_socket, to_socket):
        if (from_socket, to_socket) not in existing_links:
            tree.links.new(from_socket, to_socket)
            existing_links.add((from_socket, to_socket))
    bpy.context.view_layer.update()
    depth_output = rl.outputs.get("Depth")
    if depth_output is not None:
        _ensure_link(depth_output, z_viewer.inputs["Image"])
    _ensure_link(rl.outputs["Image"], ao.inputs["Background"])
    _ensure_link(img_node.outputs["Image"], ao.inputs["Foreground"])
    _ensure_link(ao.outputs["Image"], out.inputs[0])
