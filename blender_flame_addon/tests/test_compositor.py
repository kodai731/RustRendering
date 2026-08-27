from blender_flame_addon.compositor import flame_node_names


def test_flame_node_names_has_keys():
    names = flame_node_names("Flame")
    assert set(names.keys()) == {"rl", "z_viewer", "img", "ao", "composite"}


def test_flame_node_names_rl_static():
    names = flame_node_names("Flame")
    assert names["rl"] == "THYLLORE_FLAME_RL"


def test_flame_node_names_z_viewer_static():
    names = flame_node_names("Flame")
    assert names["z_viewer"] == "THYLLORE_FLAME_Z_VIEWER"


def test_flame_node_names_composite_static():
    names = flame_node_names("Flame")
    assert names["composite"] == "THYLLORE_FLAME_COMPOSITE"


def test_flame_node_names_img_contains_obj_name():
    names = flame_node_names("MyFlame")
    assert names["img"] == "THYLLORE_FLAME_IMG_MyFlame"


def test_flame_node_names_ao_contains_obj_name():
    names = flame_node_names("MyFlame")
    assert names["ao"] == "THYLLORE_FLAME_AO_MyFlame"


def test_flame_node_names_different_objects():
    names1 = flame_node_names("Flame1")
    names2 = flame_node_names("Flame2")
    assert names1["img"] != names2["img"]
    assert names1["ao"] != names2["ao"]
    assert names1["rl"] == names2["rl"]
    assert names1["z_viewer"] == names2["z_viewer"]
    assert names1["composite"] == names2["composite"]
