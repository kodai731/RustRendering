from __future__ import annotations

import math
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import bpy

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

bpy.ops.thyllore.flame_add()
obj = bpy.context.active_object

assert abs(obj.thyllore_flame.height - 1.6) < 1e-5, (
    f"campfire height expected 1.6, got {obj.thyllore_flame.height}"
)

obj.thyllore_flame.preset = "candle"
assert abs(obj.thyllore_flame.height - 0.28) < 1e-5, (
    f"candle height expected 0.28, got {obj.thyllore_flame.height}"
)

from blender_flame_addon.properties import collect_params

cls = addon.properties._registered_cls
campfire_params = fx.flame_preset_params("campfire")
collected = collect_params(obj.thyllore_flame, cls.PARAM_NAMES)
assert set(collected.keys()).issubset(set(campfire_params.keys())), (
    f"collected keys {set(collected.keys())} not subset of preset keys {set(campfire_params.keys())}"
)

print("ADDON_SMOKE ok", flush=True)

from math import radians
from blender_flame_addon.draw_handler import FlameViewportRenderer
from blender_flame_addon.coordinates import look_at_view_matrix, engine_projection

view = look_at_view_matrix((0, 1.2, 4.5), (0, 0, -1), (0, 1, 0))
proj = engine_projection(radians(45), 1, 0.1)

renderer = FlameViewportRenderer()
params = collect_params(obj.thyllore_flame, cls.PARAM_NAMES)
position = (0.0, 0.0, 0.0)
rotation = (1.0, 0.0, 0.0, 0.0)
light_pos = (0.0, 2.0, 2.0)
camera_pos = (0.0, 1.2, 4.5)

for i in range(3):
    tex = renderer.render(view, proj, camera_pos, light_pos, params, 1.5, position, rotation, 256, 256)

pixels = tex.read().to_list()
alpha_count = sum(1 for row in pixels for px in row if px[3] > 0.0)
assert alpha_count > 0, f"expected alpha > 0 pixels, got {alpha_count}"
assert renderer.frame_index == 3, f"expected frame_index == 3, got {renderer.frame_index}"

print("DRAW_SMOKE ok", flush=True)

renderer.release()
addon.unregister()
sys.exit(0)
