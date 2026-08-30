from __future__ import annotations

import math
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

import bpy

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

bpy.ops.thyllore.flame_add()
obj = bpy.context.active_object

assert abs(obj.thyllore_flame.optical_depth - 1.0) < 1e-5, "initial optical_depth from flame_params.toml"
assert abs(obj.thyllore_flame.height - 1.6) < 1e-5, (
    f"campfire height expected 1.6, got {obj.thyllore_flame.height}"
)

obj.thyllore_flame.preset = "candle"
assert abs(obj.thyllore_flame.height - 0.28) < 1e-5, (
    f"candle height expected 0.28, got {obj.thyllore_flame.height}"
)

from blender_addon.effects.flame.properties import flame_render_params

cls = addon.properties._registered_cls
assert set(cls.PARAM_NAMES) == {
    "height", "radius", "intensity", "optical_depth", "noise_amplitude", "noise_contrast", "noise_aniso_y",
    "color_base", "color_tip", "use_blackbody", "temperature_base_k", "temperature_tip_k",
}, (
    f"unexpected exposed params {cls.PARAM_NAMES}"
)
candle_params = fx.flame_preset_params("candle")
collected = flame_render_params(obj.thyllore_flame)
assert set(collected.keys()) == set(candle_params.keys()), "render params must cover every preset key"
assert collected["height"] == obj.thyllore_flame.height
obj.thyllore_flame.preset = "blue"
assert not obj.thyllore_flame.use_blackbody and abs(obj.thyllore_flame.color_base[2] - 1.0) < 1e-5, "blue preset must reach the color props"
assert flame_render_params(obj.thyllore_flame)["color_base"][2] == obj.thyllore_flame.color_base[2]

print("ADDON_SMOKE ok", flush=True)

from math import radians
from blender_addon.effects.flame.draw_handler import FlameViewportRenderer
from blender_addon.common.coordinates import look_at_view_matrix, engine_projection

view = look_at_view_matrix((0, 1.2, 4.5), (0, 0, -1), (0, 1, 0))
proj = engine_projection(radians(45), 1, 0.1)

renderer = FlameViewportRenderer()
params = flame_render_params(obj.thyllore_flame)
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

import gpu
from gpu_extras.batch import batch_for_shader

from blender_addon.effects.flame.viewport_depth import ViewportDepthCapture

depth_w = depth_h = 64
offscreen = gpu.types.GPUOffScreen(depth_w, depth_h)
gl_window = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, -(100.0 + 0.5) / (100.0 - 0.5), -2.0 * 100.0 * 0.5 / (100.0 - 0.5)],
    [0.0, 0.0, -1.0, 0.0],
]
capture = ViewportDepthCapture()
with offscreen.bind():
    fb = gpu.state.active_framebuffer_get()
    fb.clear(color=(0.0, 0.0, 0.0, 0.0), depth=1.0)
    color_shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    quad = batch_for_shader(color_shader, "TRIS", {"pos": [(-0.5, -0.5, 0.2), (0.5, -0.5, 0.2), (0.5, 0.5, 0.2), (-0.5, -0.5, 0.2), (0.5, 0.5, 0.2), (-0.5, 0.5, 0.2)]})
    gpu.state.depth_test_set("LESS_EQUAL")
    gpu.state.depth_mask_set(True)
    color_shader.bind()
    color_shader.uniform_float("color", (1.0, 1.0, 1.0, 1.0))
    quad.draw(color_shader)
    gpu.state.depth_mask_set(False)
    gpu.state.depth_test_set("NONE")
    engine_depth = capture.capture(depth_w, depth_h, gl_window, 0.1)
assert engine_depth is not None, "viewport depth capture returned None (see failure log above)"
depth_rows = engine_depth.read().to_list()
center = depth_rows[depth_h // 2][depth_w // 2]
corner = depth_rows[1][1]
center_value = center[0] if isinstance(center, (list, tuple)) else center
corner_value = corner[0] if isinstance(corner, (list, tuple)) else corner
assert center_value > 0.0, f"expected occluder depth at center, got {center_value}"
assert corner_value == 0.0, f"expected DEPTH_FAR at empty corner, got {corner_value}"
capture.release()
offscreen.free()
print(f"VIEWPORT_DEPTH_SMOKE ok center={center_value:.4f} corner={corner_value}", flush=True)

renderer.release()
addon.unregister()
sys.exit(0)
