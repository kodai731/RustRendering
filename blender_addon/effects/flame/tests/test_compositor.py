from blender_addon.effects.flame.debug.compositor import _insert_alpha_over_before_output, flame_node_names


class FakeSocket:
    def __init__(self, node, name):
        self.node = node
        self.name = name

    @property
    def is_linked(self):
        return any(link.to_socket is self for link in self.node.tree.links)


class FakeNode:
    def __init__(self, tree, name, inputs=(), outputs=()):
        self.tree = tree
        self.name = name
        self.inputs = {n: FakeSocket(self, n) for n in inputs}
        self.outputs = {n: FakeSocket(self, n) for n in outputs}
        self.inputs_list = list(self.inputs.values())


class FakeLink:
    def __init__(self, from_socket, to_socket):
        self.from_socket = from_socket
        self.to_socket = to_socket
        self.from_node = from_socket.node


class FakeLinks(list):
    def new(self, from_socket, to_socket):
        self.append(FakeLink(from_socket, to_socket))

    def remove(self, link):
        list.remove(self, link)


class FakeTree:
    def __init__(self):
        self.links = FakeLinks()


class FakeOutput(FakeNode):
    def __init__(self, tree):
        super().__init__(tree, "out", inputs=("Image",))
        self.inputs = self.inputs_list


def _build_tree():
    tree = FakeTree()
    rl = FakeNode(tree, "rl", outputs=("Image", "Depth"))
    img = FakeNode(tree, "img", outputs=("Image",))
    ao = FakeNode(tree, "ao", inputs=("Background", "Foreground"), outputs=("Image",))
    out = FakeOutput(tree)
    return tree, rl, img, ao, out


def _pairs(tree):
    return {(l.from_socket.node.name, l.from_socket.name, l.to_socket.node.name, l.to_socket.name) for l in tree.links}


def test_empty_tree_wires_render_layer_as_background():
    tree, rl, img, ao, out = _build_tree()
    _insert_alpha_over_before_output(tree, rl, img, ao, out)
    assert _pairs(tree) == {
        ("rl", "Image", "ao", "Background"),
        ("img", "Image", "ao", "Foreground"),
        ("ao", "Image", "out", "Image"),
    }


def test_existing_chain_is_kept_upstream_of_alpha_over():
    tree, rl, img, ao, out = _build_tree()
    glare = FakeNode(tree, "glare", inputs=("Image",), outputs=("Image",))
    tree.links.new(rl.outputs["Image"], glare.inputs["Image"])
    tree.links.new(glare.outputs["Image"], out.inputs[0])
    _insert_alpha_over_before_output(tree, rl, img, ao, out)
    assert _pairs(tree) == {
        ("rl", "Image", "glare", "Image"),
        ("glare", "Image", "ao", "Background"),
        ("img", "Image", "ao", "Foreground"),
        ("ao", "Image", "out", "Image"),
    }


def test_setup_is_idempotent():
    tree, rl, img, ao, out = _build_tree()
    _insert_alpha_over_before_output(tree, rl, img, ao, out)
    first = _pairs(tree)
    _insert_alpha_over_before_output(tree, rl, img, ao, out)
    assert _pairs(tree) == first
    assert len(tree.links) == 3


def test_flame_node_names_are_per_object_except_shared_nodes():
    a, b = flame_node_names("A"), flame_node_names("B")
    assert a["img"] != b["img"] and a["ao"] != b["ao"]
    assert a["rl"] == b["rl"] and a["z_viewer"] == b["z_viewer"] and a["composite"] == b["composite"]
