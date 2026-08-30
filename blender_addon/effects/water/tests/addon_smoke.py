from __future__ import annotations

import math
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

import bpy

wheel_dir = REPO_ROOT / "blender_addon" / "effects" / "water" / "wheels"
wheels = sorted(wheel_dir.glob("thyllore_effect_core-*.whl"))
if not wheels:
    print("No thyllore_effect_core wheel found", flush=True)
    sys.exit(1)

site_dir = REPO_ROOT / "log" / "blender_water_probe" / "site"
site_dir.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(wheels[0]) as zf:
    for entry in zf.namelist():
        if entry.startswith("thyllore_effect_core"):
            zf.extract(entry, str(site_dir))

sys.path.insert(0, str(site_dir))

import thyllore_effect_core as fx

import blender_addon.effects.water as addon
addon.register()

bpy.ops.thyllore.water_add()
obj = bpy.context.active_object

from blender_addon.effects.water.properties import water_render_params

cls = addon.properties._registered_cls
candle_params = fx.water_preset_params(fx.water_preset_names()[0])
collected = water_render_params(obj.thyllore_water)
assert set(collected.keys()) == set(candle_params.keys()), "render params must cover every preset key"

print("ADDON_SMOKE ok", flush=True)

from math import radians
from blender_addon.effects.water.draw_handler import WaterViewportRenderer
from blender_addon.common.coordinates import look_at_view_matrix, engine_projection

view = look_at_view_matrix((0, 1.2, 4.5), (0, 0, -1), (0, 1, 0))
proj = engine_projection(radians(45), 1, 0.1)

renderer = WaterViewportRenderer()
params = water_render_params(obj.thyllore_water)
position = (0.0, 0.0, 0.0)
rotation = (1.0, 0.0, 0.0, 0.0)
light_pos = (0.0, 2.0, 2.0)
camera_pos = (0.0, 1.2, 4.5)

for i in range(3):
    tex = renderer.render(view, proj, camera_pos, light_pos, params, 1.5, position, rotation, 256, 256)

pixels = tex.read().to_list()
alpha_count = sum(1 for row in pixels for px in row if px[3] > 0.0)
assert alpha_count > 0, f"expected alpha > 0 pixels, got {alpha_count}"

print("DRAW_SMOKE ok", flush=True)

renderer.release()
addon.unregister()
sys.exit(0)
