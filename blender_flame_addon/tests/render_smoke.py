from __future__ import annotations

import sys
import zipfile
from pathlib import Path
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import bpy

bpy.ops.wm.read_factory_settings(use_empty=True)

wheel_dir = REPO_ROOT / "blender_flame_addon" / "wheels"
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

import blender_flame_addon as addon
addon.register()

cam_data = bpy.data.cameras.new("Camera")
cam_obj = bpy.data.objects.new("Camera", cam_data)
bpy.context.collection.objects.link(cam_obj)
cam_obj.location = (0.0, -4.0, 1.2)
cam_obj.rotation_euler = (1.5707963, 0.0, 0.0)
scene = bpy.context.scene
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

from blender_flame_addon.render import render_flame_sequence

tmp_dir = tempfile.mkdtemp()
paths = render_flame_sequence(scene, obj, tmp_dir, 1, 3, write_npy=True)

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
paths2 = render_flame_sequence(scene, obj, tmp_dir2, 1, 3, write_npy=True)
for i in range(3):
    npy1 = Path(paths[i]).with_suffix(".npy")
    npy2 = Path(paths2[i]).with_suffix(".npy")
    bytes1 = npy1.read_bytes()
    bytes2 = npy2.read_bytes()
    assert bytes1 == bytes2, f"idempotency check failed: frame {i+1} npy bytes differ"

print("RENDER_SMOKE ok", flush=True)

addon.unregister()
sys.exit(0)
