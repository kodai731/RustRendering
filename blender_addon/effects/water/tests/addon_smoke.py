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
    color_tex, depth_tex = renderer.render(view, proj, camera_pos, light_pos, params, 1.5, position, rotation, 256, 256)

pixels = color_tex.read().to_list()
alpha_count = sum(1 for row in pixels for px in row if px[3] > 0.0)
assert alpha_count > 0, f"expected alpha > 0 pixels, got {alpha_count}"

print("DRAW_SMOKE ok", flush=True)

import gpu
import mathutils
from blender_addon.effects.water.draw_handler import composite_tonemapped

window_matrix = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, proj[2][2], proj[2][3]],
    [0.0, 0.0, 1.0, 0.0],
]

color = gpu.types.GPUTexture((256, 256), format="RGBA16F")
depth = gpu.types.GPUTexture((256, 256), format="DEPTH_COMPONENT32F")
fb = gpu.types.GPUFrameBuffer(color_slots=(color,), depth_slot=depth)

# Test Case A: scene_depth all 0.0 (far) -> some alpha > 0
scene_depth_far = gpu.types.GPUTexture((1, 1), format="R32F", data=gpu.types.Buffer("FLOAT", 1, [0.0]))
with fb.bind():
    fb.clear(color=(0, 0, 0, 0), depth=1.0)
    with gpu.matrix.push_pop():
        gpu.matrix.load_identity()
        gpu.matrix.load_projection_matrix(mathutils.Matrix(((2.0 / 256, 0, 0, -1.0), (0, 2.0 / 256, 0, -1.0), (0, 0, -1.0, 0), (0, 0, 0, 1.0))))
        composite_tonemapped(color_tex, depth_tex, 256, 256, window_matrix, scene_depth=scene_depth_far)
pixels_a = color.read().to_list()
alpha_count_a = sum(1 for row in pixels_a for px in row if px[3] > 0.0)
assert alpha_count_a > 0, f"Test Case A: expected alpha > 0 with far scene depth, got {alpha_count_a}"

# Test Case B: scene_depth all 1.0 (near) -> all alpha == 0
scene_depth_near = gpu.types.GPUTexture((1, 1), format="R32F", data=gpu.types.Buffer("FLOAT", 1, [1.0]))
with fb.bind():
    fb.clear(color=(0, 0, 0, 0), depth=1.0)
    with gpu.matrix.push_pop():
        gpu.matrix.load_identity()
        gpu.matrix.load_projection_matrix(mathutils.Matrix(((2.0 / 256, 0, 0, -1.0), (0, 2.0 / 256, 0, -1.0), (0, 0, -1.0, 0), (0, 0, 0, 1.0))))
        composite_tonemapped(color_tex, depth_tex, 256, 256, window_matrix, scene_depth=scene_depth_near)
pixels_b = color.read().to_list()
alpha_count_b = sum(1 for row in pixels_b for px in row if px[3] > 0.0)
assert alpha_count_b == 0, f"Test Case B: expected all alpha == 0 with near scene depth, got {alpha_count_b}"

print("DEPTH_SMOKE ok", flush=True)

renderer.release()
addon.unregister()
sys.exit(0)
