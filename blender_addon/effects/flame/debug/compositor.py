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


def _ensure_node(tree, name, node_type):
    node = tree.nodes.get(name)
    if node is None:
        node = tree.nodes.new(node_type)
        node.name = name
    return node


def _ensure_compositor_tree(scene):
    import bpy

    scene.use_nodes = True
    scene.view_layers[0].use_pass_z = True
    if scene.compositing_node_group is None:
        scene.compositing_node_group = bpy.data.node_groups.new("THYLLORE_FLAME_COMPOSITOR", "CompositorNodeTree")
    tree = scene.compositing_node_group
    if not any(getattr(s, "in_out", None) == "OUTPUT" and s.name == "Image" for s in tree.interface.items_tree):
        tree.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    return tree


def _load_sequence_image(img_node, sequence_dir, obj_name, frame_start, frame_end):
    import bpy

    exr_path = sequence_path(sequence_dir, obj_name, frame_start)
    image_name = os.path.basename(exr_path)
    image = bpy.data.images.get(image_name)
    if image is None and os.path.exists(exr_path):
        image = bpy.data.images.load(exr_path)
    if image is None:
        return
    image.name = image_name
    image.source = "SEQUENCE"
    img_node.image = image
    img_node.frame_duration = frame_end - frame_start + 1
    img_node.frame_start = frame_start
    img_node.use_auto_refresh = True


def _link(tree, from_socket, to_socket):
    for link in tree.links:
        if link.from_socket == from_socket and link.to_socket == to_socket:
            return
    tree.links.new(from_socket, to_socket)


def _existing_output_source(tree, out, ao):
    for link in tree.links:
        if link.to_socket == out.inputs[0] and link.from_node != ao:
            return link
    return None


def _insert_alpha_over_before_output(tree, rl, img_node, ao, out):
    upstream_link = _existing_output_source(tree, out, ao)
    if upstream_link is not None:
        background = upstream_link.from_socket
        tree.links.remove(upstream_link)
    elif not ao.inputs["Background"].is_linked:
        background = rl.outputs["Image"]
    else:
        background = None
    if background is not None:
        _link(tree, background, ao.inputs["Background"])
    _link(tree, img_node.outputs["Image"], ao.inputs["Foreground"])
    _link(tree, ao.outputs["Image"], out.inputs[0])


def setup_flame_compositor(scene, obj, sequence_dir, frame_start, frame_end):
    import bpy

    tree = _ensure_compositor_tree(scene)
    names = flame_node_names(obj.name)
    rl = _ensure_node(tree, names["rl"], "CompositorNodeRLayers")
    z_viewer = _ensure_node(tree, names["z_viewer"], "CompositorNodeViewer")
    img_node = _ensure_node(tree, names["img"], "CompositorNodeImage")
    ao = _ensure_node(tree, names["ao"], "CompositorNodeAlphaOver")
    out = _ensure_node(tree, names["composite"], "NodeGroupOutput")
    _load_sequence_image(img_node, sequence_dir, obj.name, frame_start, frame_end)

    bpy.context.view_layer.update()
    depth_output = rl.outputs.get("Depth")
    if depth_output is not None:
        _link(tree, depth_output, z_viewer.inputs["Image"])
    _insert_alpha_over_before_output(tree, rl, img_node, ao, out)
