"""Headless smoke test for the Thyllore Flame addon.

Run inside Blender with:
    blender --python tests/addon_smoke.py
"""
from __future__ import annotations

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

addon.unregister()
sys.exit(0)
