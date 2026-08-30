from __future__ import annotations

import sys
import zipfile
from pathlib import Path
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

import bpy

bpy.ops.wm.read_factory_settings(use_empty=True)

wheel_dir = REPO_ROOT / "blender_addon" / "effects" / "flame" / "wheels"
wheels = sorted(wheel_dir.glob("thyllore_effect_core-*.whl"))
if not wheels:
    print("No thyllore_effect_core wheel found", flush=True)
    sys.exit(1)

site_dir = REPO_ROOT / "log" / "blender_flame_probe" / "site"
site_dir.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(wheels[0]) as zf:
    for entry in zf.namelist():
        if entry.startswith("thyllore_effect_core"):
            zf.extract(entry, str(site_dir))

sys.path.insert(0, str(site_dir))

import thyllore_effect_core as fx

import blender_addon.effects.flame as addon
addon.register()

scene = bpy.context.scene
scene.render.engine = "BLENDER_WORKBENCH"

cam_data = bpy.data.cameras.new("Camera")
cam_obj = bpy.data.objects.new("Camera", cam_data)
bpy.context.collection.objects.link(cam_obj)
cam_obj.location = (0.0, -4.0, 1.2)
cam_obj.rotation_euler = (1.5707963, 0.0, 0.0)
scene.camera = cam_obj
bpy.context.view_layer.update()

scene.render.resolution_x = 320
scene.render.resolution_y = 240
scene.render.fps = 24
scene.frame_start = 1
scene.frame_end = 3

obj = bpy.data.objects.new("Flame", None)
scene.collection.objects.link(obj)
obj.thyllore_flame.is_flame = True
obj.thyllore_flame.preset = "campfire"

mesh = bpy.data.meshes.new("Cube")
verts = [
    (-0.6, -0.6, 0.4 - 0.6), (0.6, -0.6, 0.4 - 0.6),
    (0.6, 0.6, 0.4 - 0.6), (-0.6, 0.6, 0.4 - 0.6),
    (-0.6, -0.6, 0.4 + 0.6), (0.6, -0.6, 0.4 + 0.6),
    (0.6, 0.6, 0.4 + 0.6), (-0.6, 0.6, 0.4 + 0.6),
]
faces = [
    (0, 1, 2, 3), (4, 7, 6, 5),
    (0, 3, 7, 4), (1, 5, 6, 2),
    (0, 4, 5, 1), (3, 2, 6, 7),
]
mesh.from_pydata(verts, [], faces)
cube_obj = bpy.data.objects.new("Cube", mesh)
scene.collection.objects.link(cube_obj)

from blender_addon.effects.flame.debug.render import render_flame_sequence
from blender_addon.effects.flame.debug.compositor import setup_flame_compositor

tmp_dir = tempfile.mkdtemp()
setup_flame_compositor(scene, obj, tmp_dir, 1, 3)
node_count_after_first = len(scene.compositing_node_group.nodes)
setup_flame_compositor(scene, obj, tmp_dir, 1, 3)
node_count_after_second = len(scene.compositing_node_group.nodes)
assert node_count_after_first == node_count_after_second, f"compositor not idempotent: {node_count_after_first} vs {node_count_after_second}"
print("COMPOSITOR_SMOKE ok", flush=True)

paths = render_flame_sequence(scene, obj, tmp_dir, 1, 3, write_npy=True, use_scene_depth=False)

assert len(paths) == 3, f"expected 3 paths, got {len(paths)}"

import numpy as np

for i, p in enumerate(paths):
    npy_path = Path(p).with_suffix(".npy")
    assert npy_path.exists(), f"npy not found: {npy_path}"
    arr = np.load(npy_path)
    assert arr.shape == (240, 320, 4)
    alpha_count = int((arr[:, :, 3] > 0.0).sum())
    assert alpha_count > 0, f"frame {i+1} has no alpha>0 pixels"

tmp_dir2 = tempfile.mkdtemp()
paths2 = render_flame_sequence(scene, obj, tmp_dir2, 1, 3, write_npy=True, use_scene_depth=False)
for i in range(3):
    npy1 = Path(paths[i]).with_suffix(".npy")
    npy2 = Path(paths2[i]).with_suffix(".npy")
    bytes1 = npy1.read_bytes()
    bytes2 = npy2.read_bytes()
    assert bytes1 == bytes2, f"idempotency check failed: frame {i+1} npy bytes differ"

print("RENDER_SMOKE ok", flush=True)

tmp_dir_depth = tempfile.mkdtemp()
paths_depth = render_flame_sequence(scene, obj, tmp_dir_depth, 1, 1, write_npy=True, use_scene_depth=True)
npy_depth = Path(paths_depth[0]).with_suffix(".npy")
arr_depth = np.load(npy_depth)
alpha_depth = int((arr_depth[:, :, 3] > 0.0).sum())

npy_no_depth = Path(paths[0]).with_suffix(".npy")
arr_no_depth = np.load(npy_no_depth)
alpha_no_depth = int((arr_no_depth[:, :, 3] > 0.0).sum())

if alpha_depth == 0:
    print("DEPTH_SMOKE skipped (no viewer pixels)", flush=True)
else:
    assert alpha_depth < alpha_no_depth, f"depth alpha {alpha_depth} should be less than no-depth alpha {alpha_no_depth}"
    print("DEPTH_SMOKE ok", flush=True)

addon.unregister()
sys.exit(0)
